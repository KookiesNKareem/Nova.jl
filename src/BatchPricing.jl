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

export sabr_vols_batch, sabr_prices_batch
export PrecompiledSABRCalibrator, compile_gpu!, calibrate!
export price_surface_batch

end
