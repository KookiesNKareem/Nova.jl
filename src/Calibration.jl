module Calibration

using ..AD: gradient, ForwardDiffBackend, current_backend, ADBackend, ReactantBackend
using ..Models: SABRParams, sabr_implied_vol, sabr_price, HestonParams, heston_price
using LinearAlgebra: norm
using Statistics: mean

# TODO: Investigate if Julia 1.12+ destructuring works with Reactant traced arrays
# FIXME: Heston GPU calibration not supported - Reactant doesn't handle complex AD.
#        Consider implementing real-valued Heston pricing (Carr-Madan with cosine expansion).

const MASK_3_1 = [1.0, 0.0, 0.0]
const MASK_3_2 = [0.0, 1.0, 0.0]
const MASK_3_3 = [0.0, 0.0, 1.0]

const MASK_5_1 = [1.0, 0.0, 0.0, 0.0, 0.0]
const MASK_5_2 = [0.0, 1.0, 0.0, 0.0, 0.0]
const MASK_5_3 = [0.0, 0.0, 1.0, 0.0, 0.0]
const MASK_5_4 = [0.0, 0.0, 0.0, 1.0, 0.0]
const MASK_5_5 = [0.0, 0.0, 0.0, 0.0, 1.0]

"""
    _extract_param(params, mask)

Extract a single parameter from array using dot product with mask.
This avoids scalar indexing which is incompatible with GPU backends (Reactant).
"""
@inline _extract_param(params, mask) = sum(params .* mask)

"""
    _sabr_implied_vol_scalar(F, K, T, α, β, ρ, ν)

SABR implied volatility with scalar parameters (not struct).
Used internally for GPU-compatible calibration.

Note: This version avoids if-statements on traced values for Reactant compatibility.
Uses smooth approximations instead of hard conditionals.
"""
function _sabr_implied_vol_scalar(F, K, T, α, β, ρ, ν)
    # Handle ATM case (F and K are not traced, so this is safe)
    if abs(F - K) < 1e-12
        F_pow = F^(β - 1)
        C1 = ((1-β)^2 / 24) * α^2 * F_pow^2
        C2 = (ρ * β * ν * α * F_pow) / 4
        C3 = (2 - 3*ρ^2) * ν^2 / 24
        return α * F_pow * (1 + (C1 + C2 + C3) * T)
    end

    # Hagan's formula for non-ATM
    logFK = log(F / K)
    FK_mid = (F * K)^((1 - β) / 2)

    z = (ν / α) * FK_mid * logFK

    # Compute x(z) using smooth approximation (GPU-compatible, no if on traced values)
    # For small z: x(z) ≈ 1 + ρz/2 + O(z²)
    # For large z: x(z) = z / log((sqrt(1-2ρz+z²) + z - ρ) / (1-ρ))
    # Use smooth blend to avoid discontinuity
    z_sq = z^2
    sqrt_term = sqrt(1 - 2*ρ*z + z_sq + 1e-12)  # Small epsilon for numerical stability
    log_arg = (sqrt_term + z - ρ) / (1 - ρ + 1e-12)
    # Clamp log_arg to avoid log of negative/zero
    log_arg_safe = max(log_arg, 1e-12)
    x_z_full = z / (log(log_arg_safe) + 1e-12)

    # Smooth transition: use Taylor expansion for small |z|, full formula otherwise
    # Weight: w = z² / (z² + ε²), so w→0 for small z, w→1 for large z
    eps_blend = 1e-6
    w = z_sq / (z_sq + eps_blend)
    x_z_taylor = 1 + ρ * z / 2  # First-order Taylor
    x_z = (1 - w) * x_z_taylor + w * x_z_full

    # Numerator: expansion in logFK
    denom_expansion = 1 + (1-β)^2/24 * logFK^2 + (1-β)^4/1920 * logFK^4
    A = α / (FK_mid * denom_expansion)

    # Correction terms for time
    C1 = ((1-β)^2 / 24) * (α^2 / FK_mid^2)
    C2 = (ρ * β * ν * α) / (4 * FK_mid)
    C3 = (2 - 3*ρ^2) * ν^2 / 24

    return A * x_z * (1 + (C1 + C2 + C3) * T)
end

"""
    OptionQuote

A single option quote from the market.

# Fields
- `strike::Float64` - Strike price
- `expiry::Float64` - Time to expiry (in years)
- `price::Float64` - Market price (can be 0 if only using implied_vol)
- `optiontype::Symbol` - :call or :put
- `implied_vol::Float64` - Market implied volatility
"""
struct OptionQuote
    strike::Float64
    expiry::Float64
    price::Float64
    optiontype::Symbol
    implied_vol::Float64
end

"""
    SmileData

Market data for a single expiry (a volatility smile).

# Fields
- `expiry::Float64` - Time to expiry (in years)
- `forward::Float64` - Forward price
- `rate::Float64` - Risk-free interest rate
- `quotes::Vector{OptionQuote}` - Option quotes at this expiry
"""
struct SmileData
    expiry::Float64
    forward::Float64
    rate::Float64
    quotes::Vector{OptionQuote}
end

"""
    CalibrationResult{P}

Result of model calibration.

# Fields
- `params::P` - Fitted model parameters
- `loss::Float64` - Final objective value
- `converged::Bool` - Whether optimization converged
- `iterations::Int` - Number of iterations taken
- `rmse::Float64` - Root mean squared error (in volatility terms)
"""
struct CalibrationResult{P}
    params::P
    loss::Float64
    converged::Bool
    iterations::Int
    rmse::Float64
end

# TODO: Add support for calibrating beta (currently fixed)
# TODO: Add quote weighting by liquidity/bid-ask spread
# TODO: Consider BFGS or L-BFGS optimizer for faster convergence
# TODO: Add regularization to prevent overfitting on noisy data

"""
    calibrate_sabr(smile::SmileData; beta=0.5, max_iter=1000, tol=1e-8, lr=0.01, backend=current_backend())

Calibrate SABR model to a single expiry volatility smile.

Uses gradient descent with automatic differentiation for optimization.
β is typically fixed (0.5 for rates, 1.0 for equities).

# Arguments
- `smile` - Market smile data for a single expiry
- `beta` - CEV exponent (fixed during calibration)
- `max_iter` - Maximum gradient descent iterations
- `tol` - Convergence tolerance on loss improvement
- `lr` - Learning rate
- `backend` - AD backend for gradient computation (default: current_backend())

# Returns
CalibrationResult with fitted SABRParams.

# Example
```julia
quotes = [OptionQuote(K, 1.0, 0.0, :call, market_vol) for (K, market_vol) in market_data]
smile = SmileData(1.0, 100.0, 0.05, quotes)
result = calibrate_sabr(smile; beta=0.5)

# With explicit GPU backend
result = calibrate_sabr(smile; backend=ReactantBackend())
```
"""
function calibrate_sabr(smile::SmileData;
                        beta::Float64=0.5,
                        max_iter::Int=1000,
                        tol::Float64=1e-8,
                        lr::Float64=0.01,
                        backend::ADBackend=current_backend())

    F, T, r = smile.forward, smile.expiry, smile.rate
    quotes = smile.quotes
    n_quotes = length(quotes)

    # Pre-extract strikes and market vols as arrays (for vectorized GPU computation)
    strikes = Float64[q.strike for q in quotes]
    market_vols = Float64[q.implied_vol for q in quotes]

    # Find ATM quote for initial guess
    atm_idx = argmin([abs(q.strike - F) for q in quotes])
    atm_vol = quotes[atm_idx].implied_vol

    # Initial guess for α from ATM vol relation
    α_init = atm_vol / F^(beta - 1)

    # Estimate initial rho from skew (difference between low and high strike vols)
    low_strike_quotes = filter(q -> q.strike < F * 0.95, quotes)
    high_strike_quotes = filter(q -> q.strike > F * 1.05, quotes)
    ρ_init = 0.0
    if !isempty(low_strike_quotes) && !isempty(high_strike_quotes)
        low_vol = mean(q.implied_vol for q in low_strike_quotes)
        high_vol = mean(q.implied_vol for q in high_strike_quotes)
        # Negative skew (low > high) suggests negative rho
        skew = (low_vol - high_vol) / atm_vol
        ρ_init = clamp(-skew * 2, -0.8, 0.8)  # Heuristic mapping
    end

    # Pack free parameters: [α, ρ_unbounded, log_ν]
    # Transformations ensure constraints:
    #   α > 0 via abs()
    #   -1 < ρ < 1 via tanh()
    #   ν > 0 via exp()
    x = [α_init, atanh(clamp(ρ_init, -0.99, 0.99)), log(0.3)]

    # Check if using GPU backend - use vectorized loss
    use_gpu = backend isa ReactantBackend

    # GPU-compatible loss function using mask-based parameter extraction
    function loss_gpu(params)
        # Extract parameters without scalar indexing (GPU-compatible)
        α_raw = _extract_param(params, MASK_3_1)
        ρ_raw = _extract_param(params, MASK_3_2)
        ν_raw = _extract_param(params, MASK_3_3)

        # Transform to constrained space
        α = abs(α_raw)
        ρ = tanh(ρ_raw)
        ν = exp(ν_raw)

        # Compute model vols for all strikes (vectorized)
        total_sq_error = zero(eltype(params))
        for i in 1:n_quotes
            model_vol = _sabr_implied_vol_scalar(F, strikes[i], T, α, beta, ρ, ν)
            total_sq_error += (model_vol - market_vols[i])^2
        end
        return total_sq_error / n_quotes
    end

    # CPU loss function (original, slightly faster for CPU)
    function loss_cpu(params)
        α = abs(params[1])
        ρ = tanh(params[2])
        ν = exp(params[3])

        sabr = SABRParams(α, beta, ρ, ν)

        total_sq_error = zero(eltype(params))
        for q in quotes
            model_vol = sabr_implied_vol(F, q.strike, T, sabr)
            total_sq_error += (model_vol - q.implied_vol)^2
        end
        return total_sq_error / n_quotes
    end

    # Select loss function based on backend
    loss = use_gpu ? loss_gpu : loss_cpu

    # Gradient descent with AD
    converged = false
    iter = 0
    prev_loss = Inf
    current_lr = lr

    for i in 1:max_iter
        iter = i

        # Compute gradient using AD
        g = gradient(loss, x; backend=backend)

        # Update with gradient descent
        x_new = x - current_lr * g

        current_loss = loss(x_new)

        # Check convergence
        if abs(current_loss - prev_loss) < tol
            converged = true
            x = x_new
            break
        end

        # Adaptive learning rate: reduce if loss increased
        if current_loss > prev_loss * 1.1
            current_lr *= 0.5
        end

        x = x_new
        prev_loss = current_loss
    end

    # Extract final parameters
    α = abs(x[1])
    ρ = tanh(x[2])
    ν = exp(x[3])
    final_params = SABRParams(α, beta, ρ, ν)

    # Compute RMSE
    final_loss = loss(x)
    rmse = sqrt(final_loss)

    return CalibrationResult(final_params, final_loss, converged, iter, rmse)
end

# TODO: Add Feller condition constraint (2κθ > σ²) during optimization
# TODO: Support term-structure of Heston parameters (different kappa per expiry)
# TODO: Add dividend yield support in VolSurface and calibration
# TODO: Consider differential evolution or particle swarm for global optimization

"""
    VolSurface

Market data for multiple expiries (full volatility surface).

# Fields
- `spot::Float64` - Current spot price
- `rate::Float64` - Risk-free interest rate
- `smiles::Vector{SmileData}` - Smile data for each expiry
"""
struct VolSurface
    spot::Float64
    rate::Float64
    smiles::Vector{SmileData}
end

"""
    calibrate_heston(surface::VolSurface; max_iter=2000, tol=1e-8, lr=0.001, backend=current_backend())

Calibrate Heston model to a full volatility surface.

Uses gradient descent with automatic differentiation. Fits a single set of
Heston parameters across all expiries in the surface.

# Arguments
- `surface` - Market volatility surface data
- `max_iter` - Maximum gradient descent iterations
- `tol` - Convergence tolerance on loss improvement
- `lr` - Learning rate
- `backend` - AD backend for gradient computation (default: current_backend())

# Returns
CalibrationResult with fitted HestonParams.

# Example
```julia
smiles = [SmileData(T, F, r, quotes) for (T, F, quotes) in market_data]
surface = VolSurface(100.0, 0.05, smiles)
result = calibrate_heston(surface)

# With explicit GPU backend
result = calibrate_heston(surface; backend=ReactantBackend())
```
"""
function calibrate_heston(surface::VolSurface;
                          max_iter::Int=2000,
                          tol::Float64=1e-8,
                          lr::Float64=0.001,
                          backend::ADBackend=current_backend())

    S, r = surface.spot, surface.rate
    smiles = surface.smiles

    # Count total quotes for normalization
    n_quotes = sum(length(smile.quotes) for smile in smiles)

    # Pre-extract market data as flat arrays (for GPU compatibility)
    all_strikes = Float64[]
    all_expiries = Float64[]
    all_prices = Float64[]
    all_opttypes = Symbol[]
    for smile in smiles
        for q in smile.quotes
            push!(all_strikes, q.strike)
            push!(all_expiries, smile.expiry)
            push!(all_prices, q.price)
            push!(all_opttypes, q.optiontype)
        end
    end

    # Initial guess from ATM volatility
    # Find average ATM vol across expiries
    avg_atm_vol = 0.0
    for smile in smiles
        atm_idx = argmin([abs(q.strike - smile.forward) for q in smile.quotes])
        avg_atm_vol += smile.quotes[atm_idx].implied_vol
    end
    avg_atm_vol /= length(smiles)

    # Initial parameters: v0 ≈ σ_ATM², θ ≈ v0, κ = 1, σ = 0.3, ρ = -0.5
    v0_init = avg_atm_vol^2

    # Pack free parameters with transformations:
    # [log(v0), log(θ), log(κ), log(σ), atanh(ρ)]
    x = [log(v0_init), log(v0_init), log(1.0), log(0.3), atanh(-0.5)]

    # Check if using GPU backend
    use_gpu = backend isa ReactantBackend

    # GPU-compatible loss function using mask-based parameter extraction
    function loss_gpu(params)
        # Extract parameters without scalar indexing (GPU-compatible)
        v0_raw = _extract_param(params, MASK_5_1)
        θ_raw = _extract_param(params, MASK_5_2)
        κ_raw = _extract_param(params, MASK_5_3)
        σ_raw = _extract_param(params, MASK_5_4)
        ρ_raw = _extract_param(params, MASK_5_5)

        # Transform to constrained space
        v0 = exp(v0_raw)
        θ = exp(θ_raw)
        κ = exp(κ_raw)
        σ = exp(σ_raw)
        ρ = tanh(ρ_raw)

        heston = HestonParams(v0, θ, κ, σ, ρ)

        total_sq_error = zero(eltype(params))
        for i in 1:n_quotes
            model_price = heston_price(S, all_strikes[i], all_expiries[i], r, heston, all_opttypes[i])
            rel_error = (model_price - all_prices[i]) / all_strikes[i]
            total_sq_error += rel_error^2
        end
        return total_sq_error / n_quotes
    end

    # CPU loss function (original)
    function loss_cpu(params)
        v0 = exp(params[1])
        θ = exp(params[2])
        κ = exp(params[3])
        σ = exp(params[4])
        ρ = tanh(params[5])

        heston = HestonParams(v0, θ, κ, σ, ρ)

        total_sq_error = zero(eltype(params))
        for smile in smiles
            T = smile.expiry
            for q in smile.quotes
                model_price = heston_price(S, q.strike, T, r, heston, q.optiontype)
                rel_error = (model_price - q.price) / q.strike
                total_sq_error += rel_error^2
            end
        end
        return total_sq_error / n_quotes
    end

    # Select loss function based on backend
    loss = use_gpu ? loss_gpu : loss_cpu

    # Gradient descent with AD
    converged = false
    iter = 0
    prev_loss = Inf
    current_lr = lr

    for i in 1:max_iter
        iter = i

        # Compute gradient using AD
        g = gradient(loss, x; backend=backend)

        # Gradient clipping for stability
        g_norm = norm(g)
        if g_norm > 10.0
            g = g * (10.0 / g_norm)
        end

        # Update with gradient descent
        x_new = x - current_lr * g

        current_loss = loss(x_new)

        # Check convergence
        if abs(current_loss - prev_loss) < tol
            converged = true
            x = x_new
            break
        end

        # Adaptive learning rate
        if current_loss > prev_loss * 1.1
            current_lr *= 0.5
        elseif current_loss < prev_loss * 0.99
            current_lr = min(current_lr * 1.05, lr * 2)
        end

        x = x_new
        prev_loss = current_loss
    end

    # Extract final parameters
    v0 = exp(x[1])
    θ = exp(x[2])
    κ = exp(x[3])
    σ = exp(x[4])
    ρ = tanh(x[5])
    final_params = HestonParams(v0, θ, κ, σ, ρ)

    # Compute RMSE
    final_loss = loss(x)
    rmse = sqrt(final_loss)

    return CalibrationResult(final_params, final_loss, converged, iter, rmse)
end

export OptionQuote, SmileData, CalibrationResult, calibrate_sabr
export VolSurface, calibrate_heston

end
