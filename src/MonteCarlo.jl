module MonteCarlo

# TODO: Add multi-threading for path simulation (parallelization)
# TODO: Add control variates for variance reduction
# TODO: Add importance sampling
# TODO: Add moment matching
# TODO: Add Brownian bridge construction

using Random
using Statistics: mean, std
using Sobol: SobolSeq, next!
using ..Core: ADBackend
using ..AD: gradient, current_backend, ForwardDiffBackend, EnzymeBackend

# ============================================================================
# Path Dynamics
# ============================================================================

abstract type AbstractDynamics end

"""
    GBMDynamics(r, σ)

Geometric Brownian Motion: dS = r*S*dt + σ*S*dW
"""
struct GBMDynamics{T1,T2} <: AbstractDynamics
    r::T1      # risk-free rate
    sigma::T2  # volatility
end

"""
    HestonDynamics(r, v0, κ, θ, ξ, ρ)

Heston stochastic volatility model.
"""
struct HestonDynamics{T} <: AbstractDynamics
    r::T      # risk-free rate
    v0::T     # initial variance
    kappa::T  # mean reversion speed
    theta::T  # long-term variance
    xi::T     # vol of vol
    rho::T    # correlation
end

# ============================================================================
# Payoffs
# ============================================================================

"""
    AbstractPayoff

Abstract base type for all Monte Carlo payoff definitions.

Subtypes must implement `payoff(p::MyPayoff, path::Vector) -> Float64`
where `path` is a simulated price path.
"""
abstract type AbstractPayoff end

"""
    EuropeanCall(K)

European call option payoff: max(S_T - K, 0).

# Arguments
- `K::Float64` - Strike price

# Example
```julia
payoff_fn = EuropeanCall(100.0)
result = mc_price(payoff_fn, dynamics, 1.0, 10000)
```
"""
struct EuropeanCall <: AbstractPayoff
    K::Float64
end

"""
    EuropeanPut(K)

European put option payoff: max(K - S_T, 0).

# Arguments
- `K::Float64` - Strike price

# Example
```julia
payoff_fn = EuropeanPut(100.0)
result = mc_price(payoff_fn, dynamics, 1.0, 10000)
```
"""
struct EuropeanPut <: AbstractPayoff
    K::Float64
end

"""
    AsianCall(K)

Asian call option with arithmetic average: max(avg(S) - K, 0).

The payoff is based on the arithmetic average of the price path,
not just the terminal value.

# Arguments
- `K::Float64` - Strike price

# Example
```julia
payoff_fn = AsianCall(100.0)
result = mc_price(payoff_fn, dynamics, 1.0, 10000; nsteps=252)
```
"""
struct AsianCall <: AbstractPayoff
    K::Float64
end

"""
    AsianPut(K)

Asian put option with arithmetic average: max(K - avg(S), 0).

The payoff is based on the arithmetic average of the price path,
not just the terminal value.

# Arguments
- `K::Float64` - Strike price

# Example
```julia
payoff_fn = AsianPut(100.0)
result = mc_price(payoff_fn, dynamics, 1.0, 10000; nsteps=252)
```
"""
struct AsianPut <: AbstractPayoff
    K::Float64
end

"""
    UpAndOutCall(K, barrier)

Up-and-out barrier call option.

The option becomes worthless (knocks out) if the underlying price
ever touches or exceeds the barrier during the option's life.
Otherwise, pays max(S_T - K, 0) at expiry.

# Arguments
- `K::Float64` - Strike price
- `barrier::Float64` - Upper barrier level (must be > K for typical usage)

# Example
```julia
payoff_fn = UpAndOutCall(100.0, 120.0)  # Knocks out if S >= 120
result = mc_price(payoff_fn, dynamics, 1.0, 10000; nsteps=252)
```
"""
struct UpAndOutCall <: AbstractPayoff
    K::Float64
    barrier::Float64
end

"""
    DownAndOutPut(K, barrier)

Down-and-out barrier put option.

The option becomes worthless (knocks out) if the underlying price
ever touches or falls below the barrier during the option's life.
Otherwise, pays max(K - S_T, 0) at expiry.

# Arguments
- `K::Float64` - Strike price
- `barrier::Float64` - Lower barrier level (must be < K for typical usage)

# Example
```julia
payoff_fn = DownAndOutPut(100.0, 80.0)  # Knocks out if S <= 80
result = mc_price(payoff_fn, dynamics, 1.0, 10000; nsteps=252)
```
"""
struct DownAndOutPut <: AbstractPayoff
    K::Float64
    barrier::Float64
end

# Payoff evaluation
payoff(p::EuropeanCall, path) = max(path[end] - p.K, 0.0)
payoff(p::EuropeanPut, path) = max(p.K - path[end], 0.0)
payoff(p::AsianCall, path) = max(mean(path) - p.K, 0.0)
payoff(p::AsianPut, path) = max(p.K - mean(path), 0.0)

function payoff(p::UpAndOutCall, path)
    maximum(path) >= p.barrier ? 0.0 : max(path[end] - p.K, 0.0)
end

function payoff(p::DownAndOutPut, path)
    minimum(path) <= p.barrier ? 0.0 : max(p.K - path[end], 0.0)
end

# ============================================================================
# Path Generation
# ============================================================================

"""
    simulate_gbm(S0, T, nsteps, dynamics; rng=Random.default_rng())

Generate a single GBM path.
"""
function simulate_gbm(S0, T, nsteps, dynamics::GBMDynamics; rng=Random.default_rng())
    dt = T / nsteps
    path = Vector{typeof(S0 * one(dynamics.r))}(undef, nsteps + 1)
    path[1] = S0

    sqrt_dt = sqrt(dt)
    drift = (dynamics.r - 0.5 * dynamics.sigma^2) * dt

    for i in 1:nsteps
        Z = randn(rng)
        path[i+1] = path[i] * exp(drift + dynamics.sigma * sqrt_dt * Z)
    end

    return path
end

"""
    simulate_gbm_antithetic(S0, T, nsteps, dynamics; rng=Random.default_rng())

Generate two antithetic GBM paths for variance reduction.
"""
function simulate_gbm_antithetic(S0, T, nsteps, dynamics::GBMDynamics; rng=Random.default_rng())
    dt = T / nsteps
    path1 = Vector{typeof(S0 * one(dynamics.r))}(undef, nsteps + 1)
    path2 = Vector{typeof(S0 * one(dynamics.r))}(undef, nsteps + 1)
    path1[1] = S0
    path2[1] = S0

    sqrt_dt = sqrt(dt)
    drift = (dynamics.r - 0.5 * dynamics.sigma^2) * dt

    for i in 1:nsteps
        Z = randn(rng)
        path1[i+1] = path1[i] * exp(drift + dynamics.sigma * sqrt_dt * Z)
        path2[i+1] = path2[i] * exp(drift + dynamics.sigma * sqrt_dt * (-Z))
    end

    return path1, path2
end

"""
    simulate_heston(S0, T, nsteps, dynamics; rng=Random.default_rng())

Generate a single Heston path using full truncation scheme.
"""
# TODO: Document full truncation scheme algorithm
# TODO: Add QE (quadratic exponential) scheme option for better accuracy
# TODO: Add Feller condition check/warning
function simulate_heston(S0, T, nsteps, dynamics::HestonDynamics; rng=Random.default_rng())
    dt = T / nsteps
    sqrt_dt = sqrt(dt)

    path = Vector{typeof(S0 * one(dynamics.r))}(undef, nsteps + 1)
    path[1] = S0
    v = dynamics.v0

    for i in 1:nsteps
        Z1 = randn(rng)
        Z2 = dynamics.rho * Z1 + sqrt(1 - dynamics.rho^2) * randn(rng)

        v_plus = max(v, 0.0)
        sqrt_v = sqrt(v_plus)

        path[i+1] = path[i] * exp((dynamics.r - 0.5 * v_plus) * dt + sqrt_v * sqrt_dt * Z1)
        v = v + dynamics.kappa * (dynamics.theta - v_plus) * dt + dynamics.xi * sqrt_v * sqrt_dt * Z2
    end

    return path
end

# ============================================================================
# Quasi-Monte Carlo (Sobol Sequences)
# ============================================================================

"""
    box_muller(u1, u2)

Transform two uniform [0,1] samples to standard normal using Box-Muller.
Returns two independent N(0,1) samples.
"""
function box_muller(u1, u2)
    # Clamp to avoid log(0)
    u1_safe = clamp(u1, 1e-10, 1.0 - 1e-10)
    r = sqrt(-2 * log(u1_safe))
    theta = 2π * u2
    return r * cos(theta), r * sin(theta)
end

"""
    sobol_normals(dim::Int, n::Int)

Generate n samples of dim-dimensional standard normal vectors using Sobol sequences.
Returns a Matrix{Float64} of size (n, dim).

This is deterministic - same inputs always give same outputs.
"""
function sobol_normals(dim::Int, n::Int)
    # Need 2*dim uniform samples for Box-Muller (pairs)
    uniform_dim = 2 * ((dim + 1) ÷ 2)  # Round up to even
    seq = SobolSeq(uniform_dim)

    result = Matrix{Float64}(undef, n, dim)
    uniform = Vector{Float64}(undef, uniform_dim)

    for i in 1:n
        next!(seq, uniform)
        for j in 1:2:dim
            z1, z2 = box_muller(uniform[j], uniform[j+1])
            result[i, j] = z1
            if j + 1 <= dim
                result[i, j+1] = z2
            end
        end
    end

    return result
end

"""
    simulate_gbm_qmc(S0, T, nsteps, dynamics, Z::AbstractVector)

Generate a single GBM path using pre-computed normal samples Z.
Z should have length nsteps.
"""
function simulate_gbm_qmc(S0, T, nsteps, dynamics::GBMDynamics, Z::AbstractVector)
    dt = T / nsteps
    path = Vector{typeof(S0 * one(dynamics.r) * one(eltype(Z)))}(undef, nsteps + 1)
    path[1] = S0

    sqrt_dt = sqrt(dt)
    drift = (dynamics.r - 0.5 * dynamics.sigma^2) * dt

    for i in 1:nsteps
        path[i+1] = path[i] * exp(drift + dynamics.sigma * sqrt_dt * Z[i])
    end

    return path
end

"""
    mc_price_qmc(S0, T, payoff, dynamics; npaths=10000, nsteps=252)

Price a derivative using Quasi-Monte Carlo (Sobol sequences).

This is deterministic and differentiable with Enzyme.
Better convergence than pseudo-random MC: O(1/N) vs O(1/√N).
"""
function mc_price_qmc(S0, T, pf::AbstractPayoff, dynamics::GBMDynamics;
                      npaths::Int=10000, nsteps::Int=252)

    r = dynamics.r
    df = exp(-r * T)

    # Generate all Sobol normals upfront (deterministic)
    Z = sobol_normals(nsteps, npaths)

    # Compute payoffs
    RT = typeof(S0 * one(r))
    payoffs = Vector{RT}(undef, npaths)

    for i in 1:npaths
        path = simulate_gbm_qmc(S0, T, nsteps, dynamics, @view Z[i, :])
        payoffs[i] = payoff(pf, path)
    end

    price_est = df * mean(payoffs)
    stderr = df * std(payoffs) / sqrt(npaths)

    return MCResult(price_est, stderr, npaths, price_est - 1.96*stderr, price_est + 1.96*stderr)
end

# ============================================================================
# Monte Carlo Pricing
# ============================================================================

"""
    MCResult

Result of Monte Carlo simulation.
"""
# TODO: Add Base.show() for readable output
# TODO: Add confidence level as field (not hardcoded 95%)
struct MCResult{T}
    price::T
    stderr::T
    npaths::Int
    ci_lower::T  # 95% CI
    ci_upper::T
end

"""
    mc_price(S0, T, payoff, dynamics; npaths=10000, nsteps=252, antithetic=true, rng=nothing)

Price a derivative using Monte Carlo simulation.

# Arguments
- `S0` - Initial spot price
- `T` - Time to maturity
- `payoff` - Payoff structure
- `dynamics` - Price dynamics (GBM or Heston)
- `npaths` - Number of simulation paths
- `nsteps` - Time steps per path
- `antithetic` - Use antithetic variates for variance reduction
- `rng` - Random number generator (optional)

# Returns
MCResult with price, standard error, and confidence interval.
"""
function mc_price(S0, T, pf::AbstractPayoff, dynamics::GBMDynamics;
                  npaths::Int=10000, nsteps::Int=252, antithetic::Bool=true,
                  rng=nothing)

    rng = isnothing(rng) ? Random.default_rng() : rng
    r = dynamics.r
    df = exp(-r * T)

    # Use eltype that supports AD (Dual numbers)
    RT = typeof(S0 * one(r))
    payoffs = Vector{RT}(undef, antithetic ? npaths ÷ 2 : npaths)

    if antithetic
        for i in 1:(npaths ÷ 2)
            path1, path2 = simulate_gbm_antithetic(S0, T, nsteps, dynamics; rng=rng)
            payoffs[i] = 0.5 * (payoff(pf, path1) + payoff(pf, path2))
        end
    else
        for i in 1:npaths
            path = simulate_gbm(S0, T, nsteps, dynamics; rng=rng)
            payoffs[i] = payoff(pf, path)
        end
    end

    price_est = df * mean(payoffs)
    stderr = df * std(payoffs) / sqrt(length(payoffs))

    return MCResult(price_est, stderr, npaths, price_est - 1.96*stderr, price_est + 1.96*stderr)
end

function mc_price(S0, T, pf::AbstractPayoff, dynamics::HestonDynamics;
                  npaths::Int=10000, nsteps::Int=252, rng=nothing)

    rng = isnothing(rng) ? Random.default_rng() : rng
    r = dynamics.r
    df = exp(-r * T)

    RT = typeof(S0 * one(r))
    payoffs = Vector{RT}(undef, npaths)

    for i in 1:npaths
        path = simulate_heston(S0, T, nsteps, dynamics; rng=rng)
        payoffs[i] = payoff(pf, path)
    end

    price_est = df * mean(payoffs)
    stderr = df * std(payoffs) / sqrt(npaths)

    return MCResult(price_est, stderr, npaths, price_est - 1.96*stderr, price_est + 1.96*stderr)
end

# ============================================================================
# Monte Carlo Greeks (Pathwise Method)
# ============================================================================

"""
    mc_delta(S0, T, payoff, dynamics; npaths=10000, nsteps=252, backend=current_backend())

Compute delta using pathwise differentiation with AD.

Automatically uses QMC (Sobol sequences) when backend is EnzymeBackend,
since Enzyme cannot differentiate through pseudo-random number generators.
"""
function mc_delta(S0, T, payoff::AbstractPayoff, dynamics::GBMDynamics;
                  npaths::Int=10000, nsteps::Int=252, backend=current_backend())

    if backend isa EnzymeBackend
        # Use QMC for Enzyme (deterministic, no RNG)
        function price_fn_qmc(s0)
            result = mc_price_qmc(s0, T, payoff, dynamics; npaths=npaths, nsteps=nsteps)
            return result.price
        end
        g = gradient(x -> price_fn_qmc(x[1]), [S0]; backend=backend)
        return g[1]
    else
        # Use pseudo-random with fixed seed for ForwardDiff/PureJulia
        function price_fn(s0)
            rng = Random.MersenneTwister(42)
            result = mc_price(s0, T, payoff, dynamics; npaths=npaths, nsteps=nsteps,
                             antithetic=false, rng=rng)
            return result.price
        end
        g = gradient(x -> price_fn(x[1]), [S0]; backend=backend)
        return g[1]
    end
end

"""
    mc_greeks(S0, T, payoff, dynamics; npaths=10000, nsteps=252, backend=current_backend())

Compute delta and vega using pathwise differentiation.

Automatically uses QMC (Sobol sequences) when backend is EnzymeBackend,
since Enzyme cannot differentiate through pseudo-random number generators.
"""
function mc_greeks(S0, T, payoff::AbstractPayoff, dynamics::GBMDynamics;
                   npaths::Int=10000, nsteps::Int=252, backend=current_backend())

    if backend isa EnzymeBackend
        # Use QMC for Enzyme (deterministic, no RNG)
        function price_fn_qmc(params)
            s0, sigma = params
            dyn = GBMDynamics(dynamics.r, sigma)
            result = mc_price_qmc(s0, T, payoff, dyn; npaths=npaths, nsteps=nsteps)
            return result.price
        end
        g = gradient(price_fn_qmc, [S0, dynamics.sigma]; backend=backend)
        return (delta=g[1], vega=g[2])
    else
        # Use pseudo-random with fixed seed for ForwardDiff/PureJulia
        function price_fn(params)
            s0, sigma = params
            dyn = GBMDynamics(dynamics.r, sigma)
            rng = Random.MersenneTwister(42)
            result = mc_price(s0, T, payoff, dyn; npaths=npaths, nsteps=nsteps,
                             antithetic=false, rng=rng)
            return result.price
        end
        g = gradient(price_fn, [S0, dynamics.sigma]; backend=backend)
        return (delta=g[1], vega=g[2])
    end
end

# ============================================================================
# Longstaff-Schwartz American Option Pricing
# ============================================================================

"""
    AmericanPut

American put option for LSM pricing.
"""
struct AmericanPut <: AbstractPayoff
    K::Float64
end

"""
    AmericanCall

American call option for LSM pricing.
"""
struct AmericanCall <: AbstractPayoff
    K::Float64
end

# Intrinsic value
intrinsic(p::AmericanPut, S) = max(p.K - S, 0.0)
intrinsic(p::AmericanCall, S) = max(S - p.K, 0.0)

"""
    lsm_price(S0, T, option, dynamics; npaths=10000, nsteps=50, rng=nothing)

Price an American option using the Longstaff-Schwartz Monte Carlo method.

# Algorithm
1. Simulate paths forward
2. Work backwards from expiry
3. At each step, regress continuation value on polynomial basis of spot
4. Exercise when intrinsic value > continuation value
"""
# TODO: Validate nsteps vs number of exercise dates
# TODO: Add higher-order polynomial basis option (Laguerre, Hermite)
# TODO: Add path degeneracy check (zero variance)
# TODO: Support Heston dynamics (currently GBM only)
"""

# Arguments
- `S0` - Initial spot price
- `T` - Time to maturity
- `option` - AmericanPut or AmericanCall
- `dynamics` - GBMDynamics
- `npaths` - Number of simulation paths
- `nsteps` - Number of exercise dates
- `rng` - Random number generator

# Returns
MCResult with American option price and standard error.

# Reference
Longstaff & Schwartz (2001), "Valuing American Options by Simulation"
"""
function lsm_price(S0, T, option::Union{AmericanPut,AmericanCall}, dynamics::GBMDynamics;
                   npaths::Int=10000, nsteps::Int=50, rng=nothing)

    rng = isnothing(rng) ? Random.default_rng() : rng
    r = dynamics.r
    dt = T / nsteps
    df = exp(-r * dt)  # discount factor per step

    # Generate all paths upfront (npaths x nsteps+1 matrix)
    paths = Matrix{Float64}(undef, npaths, nsteps + 1)
    for i in 1:npaths
        path = simulate_gbm(S0, T, nsteps, dynamics; rng=rng)
        paths[i, :] = path
    end

    # Cash flow matrix: when and how much each path receives
    # Initialize with terminal payoff
    cashflows = zeros(npaths)
    exercise_time = fill(nsteps, npaths)  # when each path exercises

    for i in 1:npaths
        cashflows[i] = intrinsic(option, paths[i, end])
    end

    # Work backwards through time
    for t in (nsteps-1):-1:1
        # Spot prices at time t
        S_t = paths[:, t+1]

        # Find in-the-money paths (candidates for early exercise)
        itm = intrinsic.(Ref(option), S_t) .> 0

        if sum(itm) < 3
            continue  # Need enough points for regression
        end

        # Discounted future cashflows for ITM paths
        # Discount from their exercise time to current time
        Y = zeros(sum(itm))
        itm_indices = findall(itm)
        for (j, i) in enumerate(itm_indices)
            steps_to_exercise = exercise_time[i] - t
            Y[j] = cashflows[i] * df^steps_to_exercise
        end

        # Regression: fit continuation value to polynomial basis
        # Using Laguerre-like polynomials: 1, S, S^2 (simple but effective)
        X_itm = S_t[itm]
        X_mat = hcat(ones(length(X_itm)), X_itm, X_itm.^2)

        # Least squares: β = (X'X)^(-1) X'Y
        β = X_mat \ Y

        # Continuation value estimate for ITM paths
        continuation = X_mat * β

        # Immediate exercise value
        immediate = intrinsic.(Ref(option), X_itm)

        # Exercise if immediate > continuation
        for (j, i) in enumerate(itm_indices)
            if immediate[j] > continuation[j]
                cashflows[i] = immediate[j]
                exercise_time[i] = t
            end
        end
    end

    # Discount all cashflows back to time 0
    discounted = [cashflows[i] * df^exercise_time[i] for i in 1:npaths]

    price_est = mean(discounted)
    stderr = std(discounted) / sqrt(npaths)

    return MCResult(price_est, stderr, npaths, price_est - 1.96*stderr, price_est + 1.96*stderr)
end

# ============================================================================
# Exports
# ============================================================================

export AbstractDynamics, GBMDynamics, HestonDynamics
export AbstractPayoff, EuropeanCall, EuropeanPut, AsianCall, AsianPut
export UpAndOutCall, DownAndOutPut, AmericanPut, AmericanCall
export payoff, simulate_gbm, simulate_heston, simulate_gbm_antithetic
export MCResult, mc_price, mc_delta, mc_greeks
export lsm_price, intrinsic
# QMC exports
export sobol_normals, simulate_gbm_qmc, mc_price_qmc

end
