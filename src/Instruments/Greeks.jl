# Greeks computation - analytical formulas primary, AD for fallback/exotics

using ForwardDiff
using Distributions: Normal, cdf, pdf

"""
    GreeksResult

Container for option Greeks.
"""
struct GreeksResult{T}
    delta::T    # dV/dS
    gamma::T    # d²V/dS²
    vega::T     # dV/dσ (per 1% move, scaled by 0.01)
    theta::T    # -dV/dT (time decay per year)
    rho::T      # dV/dr (per 1% move, scaled by 0.01)
end

# ============================================================================
# Primary Interface - dispatches to analytical when available
# ============================================================================

"""
    compute_greeks(option, market_state)

Compute option Greeks. Uses analytical formulas when available (preferred),
falls back to AD for exotic options without closed-form solutions.
"""
function compute_greeks(opt::EuropeanOption, state::MarketState)
    # European options have closed-form Black-Scholes Greeks
    return _analytical_greeks(opt, state)
end

# Fallback for options without analytical Greeks - use AD
function compute_greeks(opt::AbstractOption, state::MarketState)
    return _ad_greeks(opt, state)
end

# ============================================================================
# Analytical Greeks - Black-Scholes (exact, fast)
# ============================================================================

"""
    _analytical_greeks(option, market_state)

Compute Greeks using analytical Black-Scholes formulas.
Exact closed-form solutions - no numerical approximation.
"""
function _analytical_greeks(opt::EuropeanOption, state::MarketState)
    S = state.prices[opt.underlying]
    K = opt.strike
    T = opt.expiry
    r = first(values(state.rates))
    σ = state.volatilities[opt.underlying]

    d1 = (log(S/K) + (r + σ^2/2)*T) / (σ*sqrt(T))
    d2 = d1 - σ*sqrt(T)

    N = Normal()
    n_d1 = pdf(N, d1)
    N_d1 = cdf(N, d1)
    N_d2 = cdf(N, d2)

    if opt.optiontype == :call
        delta = N_d1
        theta = -S * n_d1 * σ / (2*sqrt(T)) - r * K * exp(-r*T) * N_d2
        rho = K * T * exp(-r*T) * N_d2 * 0.01
    else  # put
        delta = N_d1 - 1
        theta = -S * n_d1 * σ / (2*sqrt(T)) + r * K * exp(-r*T) * cdf(N, -d2)
        rho = -K * T * exp(-r*T) * cdf(N, -d2) * 0.01
    end

    # Gamma and Vega are same for calls and puts
    gamma = n_d1 / (S * σ * sqrt(T))
    vega = S * n_d1 * sqrt(T) * 0.01

    return GreeksResult(delta, gamma, vega, theta, rho)
end

# ============================================================================
# AD Greeks - Fallback for exotics without closed-form solutions
# ============================================================================

"""
    _ad_greeks(option, market_state)

Compute Greeks using automatic differentiation.
Use for exotic options without analytical formulas (Asian, barrier, etc.).
"""
function _ad_greeks(opt::AbstractOption, state::MarketState)
    S = state.prices[opt.underlying]
    K = opt.strike
    T = opt.expiry
    r = first(values(state.rates))
    σ = state.volatilities[opt.underlying]

    # Pack parameters for AD: [S, σ, T, r]
    x = [S, σ, T, r]

    # Price function of packed parameters
    function price_fn(params)
        S_, σ_, T_, r_ = params
        black_scholes(S_, K, T_, r_, σ_, opt.optiontype)
    end

    # First derivatives via AD
    grad = ForwardDiff.gradient(price_fn, x)
    delta = grad[1]
    vega = grad[2] * 0.01   # Scale to per-1% vol move
    theta = -grad[3]        # Negative because we want time decay
    rho = grad[4] * 0.01    # Scale to per-1% rate move

    # Second derivative (gamma) via nested dual
    gamma = ForwardDiff.derivative(s -> ForwardDiff.derivative(
        s_ -> black_scholes(s_, K, T, r, σ, opt.optiontype), s
    ), S)

    return GreeksResult(delta, gamma, vega, theta, rho)
end

export GreeksResult, compute_greeks
