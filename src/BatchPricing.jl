module BatchPricing

using ..Models: SABRParams, sabr_implied_vol, black76
using ..AD: gradient, current_backend, ADBackend, ReactantBackend, ForwardDiffBackend

# GPU-compatible SABR vol (no branching on traced values)
function _sabr_vol_gpu(F, K, T, α, β, ρ, ν)
    logFK = log(F / K + 1e-12)
    FK_mid = (F * K)^((1 - β) / 2)
    z = (ν / α) * FK_mid * logFK
    z_sq = z^2

    sqrt_term = sqrt(1 - 2*ρ*z + z_sq + 1e-12)
    log_arg = max((sqrt_term + z - ρ) / (1 - ρ + 1e-12), 1e-12)
    x_z_full = z / (log(log_arg) + 1e-12)

    # Smooth blend for small z
    w = z_sq / (z_sq + 1e-6)
    x_z = (1 - w) * (1 + ρ * z / 2) + w * x_z_full

    denom = 1 + (1-β)^2/24 * logFK^2 + (1-β)^4/1920 * logFK^4
    A = α / (FK_mid * denom + 1e-12)
    C1 = ((1-β)^2 / 24) * (α^2 / (FK_mid^2 + 1e-12))
    C2 = (ρ * β * ν * α) / (4 * FK_mid + 1e-12)
    C3 = (2 - 3*ρ^2) * ν^2 / 24

    return A * x_z * (1 + (C1 + C2 + C3) * T)
end

"""
    sabr_vols_batch(F, strikes, T, α, β, ρ, ν) -> Vector{Float64}

Compute SABR implied vols for multiple strikes.
"""
function sabr_vols_batch(F, strikes::AbstractVector, T, α, β, ρ, ν)
    [_sabr_vol_gpu(F, K, T, α, β, ρ, ν) for K in strikes]
end

"""
    sabr_prices_batch(F, strikes, T, r, α, β, ρ, ν, opttypes) -> Vector{Float64}

Compute SABR prices for multiple options.
"""
function sabr_prices_batch(F, strikes::AbstractVector, T, r, α, β, ρ, ν,
                           opttypes::AbstractVector{Symbol})
    vols = sabr_vols_batch(F, strikes, T, α, β, ρ, ν)
    [black76(F, strikes[i], T, r, vols[i], opttypes[i]) for i in eachindex(strikes)]
end

# Masks for GPU parameter extraction
const MASK3_1 = [1.0, 0.0, 0.0]
const MASK3_2 = [0.0, 1.0, 0.0]
const MASK3_3 = [0.0, 0.0, 1.0]

"""
    PrecompiledSABRCalibrator

Pre-compiled SABR calibrator for fast GPU optimization.
Compiles gradient once, runs many iterations without recompilation.
"""
mutable struct PrecompiledSABRCalibrator
    F::Float64
    T::Float64
    β::Float64
    strikes::Vector{Float64}
    market_vols::Vector{Float64}
    n::Int
    compiled_grad::Any
    backend::ADBackend
end

function PrecompiledSABRCalibrator(F::Float64, T::Float64, β::Float64,
                                    strikes::Vector{Float64},
                                    market_vols::Vector{Float64};
                                    backend::ADBackend=ForwardDiffBackend())
    n = length(strikes)
    PrecompiledSABRCalibrator(F, T, β, strikes, market_vols, n, nothing, backend)
end

# Extension hooks - set by QuasarReactantExt when loaded
const _REACTANT_COMPILE_GPU = Ref{Any}(nothing)
const _REACTANT_CALL_GRAD = Ref{Any}(nothing)

"""
    compile_gpu!(cal::PrecompiledSABRCalibrator)

Pre-compile gradient for Reactant. Call once before calibrate!().
Requires Reactant to be loaded.
"""
function compile_gpu!(cal::PrecompiledSABRCalibrator)
    if _REACTANT_COMPILE_GPU[] === nothing
        error("compile_gpu! requires Reactant. Load it with: using Reactant")
    end
    _REACTANT_COMPILE_GPU[](cal)
end

"""
    calibrate!(cal, x0; max_iter=500, lr=0.01, tol=1e-8)

Run calibration using pre-compiled gradient.
"""
function calibrate!(cal::PrecompiledSABRCalibrator, x0::Vector{Float64};
                    max_iter::Int=500, lr::Float64=0.01, tol::Float64=1e-8)
    F, T, β = cal.F, cal.T, cal.β
    strikes, market_vols, n = cal.strikes, cal.market_vols, cal.n

    loss_fn(p) = begin
        α, ρ, ν = abs(p[1]), tanh(p[2]), exp(p[3])
        sum((_sabr_vol_gpu(F, strikes[i], T, α, β, ρ, ν) - market_vols[i])^2 for i in 1:n) / n
    end

    x = copy(x0)
    current_lr, prev_loss = lr, Inf

    for iter in 1:max_iter
        g = if cal.compiled_grad !== nothing
            _call_compiled_grad(cal, x)
        else
            gradient(loss_fn, x; backend=cal.backend)
        end

        x_new = x - current_lr * g
        loss = loss_fn(x_new)

        abs(loss - prev_loss) < tol && return (x=x_new, loss=loss, converged=true, iter=iter)
        loss > prev_loss * 1.1 && (current_lr *= 0.5)

        x, prev_loss = x_new, loss
    end

    (x=x, loss=prev_loss, converged=false, iter=max_iter)
end

function _call_compiled_grad(cal::PrecompiledSABRCalibrator, x)
    if _REACTANT_CALL_GRAD[] === nothing
        error("Compiled gradient not available. Did you call compile_gpu!() with Reactant loaded?")
    end
    _REACTANT_CALL_GRAD[](cal, x)
end

"""
    price_surface_batch(spot, strikes, expiries, r, sabr_params) -> Matrix

Price entire vol surface. sabr_params is Vector of (α,β,ρ,ν) per expiry.
Returns [n_strikes × n_expiries] matrix.
"""
function price_surface_batch(spot::Float64, strikes::Vector{Float64},
                             expiries::Vector{Float64}, r::Float64,
                             sabr_params::Vector{NTuple{4,Float64}})
    prices = Matrix{Float64}(undef, length(strikes), length(expiries))
    for (j, T) in enumerate(expiries)
        F = spot * exp(r * T)
        α, β, ρ, ν = sabr_params[j]
        for (i, K) in enumerate(strikes)
            prices[i,j] = black76(F, K, T, r, _sabr_vol_gpu(F, K, T, α, β, ρ, ν), :call)
        end
    end
    prices
end

# ============================================================================
# General Pre-compiled Calibrator (compile once, run on any data)
# ============================================================================

const _REACTANT_COMPILE_GENERAL = Ref{Any}(nothing)
const _REACTANT_CALL_GENERAL_GRAD = Ref{Any}(nothing)

"""
    GeneralSABRCalibrator

Compile gradient ONCE for a problem structure (n_strikes, F, T, β).
Then run on ANY market data of that shape - massive scale wins.

# Example
```julia
cal = GeneralSABRCalibrator(50, 100.0, 1.0, 0.5)  # 50 strikes
compile_general!(cal)  # One-time: ~18 seconds

# Now calibrate 10,000 different smiles at ~5ms each
for smile in all_smiles
    result = calibrate_general!(cal, smile.strikes, smile.market_vols, x0)
end
```
"""
mutable struct GeneralSABRCalibrator
    n::Int           # Fixed number of strikes
    F::Float64       # Forward price
    T::Float64       # Time to expiry
    β::Float64       # CEV exponent
    compiled_grad::Any
    compiled::Bool
end

function GeneralSABRCalibrator(n::Int, F::Float64, T::Float64, β::Float64)
    GeneralSABRCalibrator(n, F, T, β, nothing, false)
end

"""
    compile_general!(cal::GeneralSABRCalibrator)

Compile gradient that accepts strikes and market_vols as inputs.
Call once, then use with any market data of the same shape.
"""
function compile_general!(cal::GeneralSABRCalibrator)
    if _REACTANT_COMPILE_GENERAL[] === nothing
        error("compile_general! requires Reactant. Load it with: using Reactant")
    end
    _REACTANT_COMPILE_GENERAL[](cal)
    cal.compiled = true
    cal
end

"""
    calibrate_general!(cal, strikes, market_vols, x0; max_iter=500, lr=0.01, tol=1e-8)

Run calibration on NEW market data using pre-compiled gradient.
"""
function calibrate_general!(cal::GeneralSABRCalibrator,
                            strikes::Vector{Float64},
                            market_vols::Vector{Float64},
                            x0::Vector{Float64};
                            max_iter::Int=500, lr::Float64=0.01, tol::Float64=1e-8)
    @assert length(strikes) == cal.n "Expected $(cal.n) strikes, got $(length(strikes))"
    @assert length(market_vols) == cal.n "Expected $(cal.n) market_vols"
    @assert cal.compiled "Call compile_general!(cal) first"

    F, T, β, n = cal.F, cal.T, cal.β, cal.n

    loss_fn(p, K_vec, mv_vec) = begin
        α, ρ, ν = abs(p[1]), tanh(p[2]), exp(p[3])
        err = zero(eltype(p))
        for i in 1:n
            err += (_sabr_vol_gpu(F, K_vec[i], T, α, β, ρ, ν) - mv_vec[i])^2
        end
        err / n
    end

    x = copy(x0)
    current_lr, prev_loss = lr, Inf

    for iter in 1:max_iter
        g = _call_general_grad(cal, x, strikes, market_vols)

        x_new = x - current_lr * g
        loss = loss_fn(x_new, strikes, market_vols)

        abs(loss - prev_loss) < tol && return (x=x_new, loss=loss, converged=true, iter=iter)
        loss > prev_loss * 1.1 && (current_lr *= 0.5)

        x, prev_loss = x_new, loss
    end

    (x=x, loss=prev_loss, converged=false, iter=max_iter)
end

function _call_general_grad(cal::GeneralSABRCalibrator, x, strikes, market_vols)
    if _REACTANT_CALL_GENERAL_GRAD[] === nothing
        error("General compiled gradient not available")
    end
    _REACTANT_CALL_GENERAL_GRAD[](cal, x, strikes, market_vols)
end

# ============================================================================
# Batch Calibrator - Calibrate MANY smiles in ONE GPU call
# ============================================================================

const _REACTANT_COMPILE_BATCH = Ref{Any}(nothing)
const _REACTANT_CALL_BATCH_GRAD = Ref{Any}(nothing)

"""
    BatchSABRCalibrator

Calibrate multiple smiles in a single GPU operation.
This is where GPU truly wins - massive parallelism.

# Example
```julia
cal = BatchSABRCalibrator(30, 100, 100.0, 1.0, 0.5)  # 30 strikes, 100 smiles
compile_batch!(cal)  # One-time

# Pack data: [n_strikes × n_smiles] matrices
all_strikes = ...  # 30×100 matrix
all_vols = ...     # 30×100 matrix
x0 = ...           # 3×100 matrix (initial params for each smile)

results = calibrate_batch!(cal, all_strikes, all_vols, x0; max_iter=100)
# Returns: (params=[3×100], losses=[100], converged=[100])
```
"""
mutable struct BatchSABRCalibrator
    n_strikes::Int
    n_smiles::Int
    F::Float64
    T::Float64
    β::Float64
    compiled_grad::Any
    compiled::Bool
end

function BatchSABRCalibrator(n_strikes::Int, n_smiles::Int, F::Float64, T::Float64, β::Float64)
    BatchSABRCalibrator(n_strikes, n_smiles, F, T, β, nothing, false)
end

"""
    compile_batch!(cal::BatchSABRCalibrator)

Compile gradient for batch calibration.
"""
function compile_batch!(cal::BatchSABRCalibrator)
    if _REACTANT_COMPILE_BATCH[] === nothing
        error("compile_batch! requires Reactant. Load it with: using Reactant")
    end
    _REACTANT_COMPILE_BATCH[](cal)
    cal.compiled = true
    cal
end

"""
    calibrate_batch!(cal, strikes, market_vols, x0; max_iter=100, lr=0.01)

Calibrate all smiles in parallel on GPU.

# Arguments
- `strikes`: [n_strikes × n_smiles] matrix
- `market_vols`: [n_strikes × n_smiles] matrix
- `x0`: [3 × n_smiles] initial parameters

# Returns
Named tuple with `params`, `losses`, `iters`
"""
function calibrate_batch!(cal::BatchSABRCalibrator,
                          strikes::Matrix{Float64},
                          market_vols::Matrix{Float64},
                          x0::Matrix{Float64};
                          max_iter::Int=100, lr::Float64=0.01, tol::Float64=1e-8)
    @assert size(strikes) == (cal.n_strikes, cal.n_smiles)
    @assert size(market_vols) == (cal.n_strikes, cal.n_smiles)
    @assert size(x0) == (3, cal.n_smiles)
    @assert cal.compiled "Call compile_batch!(cal) first"

    if _REACTANT_CALL_BATCH_GRAD[] === nothing
        error("Batch gradient not available")
    end

    F, T, β = cal.F, cal.T, cal.β
    n_smiles = cal.n_smiles

    # Loss for single smile (CPU fallback for loss evaluation)
    function loss_single(p, K_vec, mv_vec)
        α, ρ, ν = abs(p[1]), tanh(p[2]), exp(p[3])
        sum((_sabr_vol_gpu(F, K, T, α, β, ρ, ν) - mv)^2 for (K, mv) in zip(K_vec, mv_vec)) / length(K_vec)
    end

    x = copy(x0)
    prev_losses = fill(Inf, n_smiles)

    for iter in 1:max_iter
        # GPU: compute all gradients in parallel
        g = _REACTANT_CALL_BATCH_GRAD[](cal, x, strikes, market_vols)

        # Update all params
        x_new = x - lr * g

        # Check convergence (on CPU)
        losses = [loss_single(x_new[:, j], strikes[:, j], market_vols[:, j]) for j in 1:n_smiles]

        if maximum(abs.(losses .- prev_losses)) < tol
            return (params=x_new, losses=losses, iters=iter)
        end

        x, prev_losses = x_new, losses
    end

    (params=x, losses=prev_losses, iters=max_iter)
end

export sabr_vols_batch, sabr_prices_batch
export PrecompiledSABRCalibrator, compile_gpu!, calibrate!
export GeneralSABRCalibrator, compile_general!, calibrate_general!
export BatchSABRCalibrator, compile_batch!, calibrate_batch!
export price_surface_batch

end
