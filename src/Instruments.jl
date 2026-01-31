module Instruments

using ..Core: AbstractEquity, AbstractOption, MarketState, IsPriceable, IsDifferentiable, HasGreeksTrait, priceable, differentiable, greeks_trait
using Distributions: Normal, cdf

# Import to extend
import ..Core: priceable, differentiable, greeks_trait

# Stock
"""
    Stock <: AbstractEquity

A simple equity instrument.

# Fields
- `symbol::String` - Ticker symbol
"""
struct Stock <: AbstractEquity
    symbol::String
end

# Register traits
priceable(::Type{Stock}) = IsPriceable()
differentiable(::Type{Stock}) = IsDifferentiable()

# ============================================================================
# Pricing Interface
# ============================================================================

"""
    price(instrument, market_state)

Compute the current price of an instrument given market state.
"""
function price end

# Stock pricing - just lookup
function price(stock::Stock, state::MarketState)
    return state.prices[stock.symbol]
end

# ============================================================================
# European Option
# ============================================================================

"""
    EuropeanOption <: AbstractOption

A European-style option (exercise only at expiry).

# Fields
- `underlying::String` - Symbol of underlying asset
- `strike::Float64` - Strike price
- `expiry::Float64` - Time to expiration (in years)
- `optiontype::Symbol` - :call or :put
"""
struct EuropeanOption <: AbstractOption
    underlying::String
    strike::Float64
    expiry::Float64
    optiontype::Symbol

    function EuropeanOption(underlying, strike, expiry, optiontype)
        optiontype in (:call, :put) || error("optiontype must be :call or :put")
        new(underlying, strike, expiry, optiontype)
    end
end

# Register traits
priceable(::Type{EuropeanOption}) = IsPriceable()
differentiable(::Type{EuropeanOption}) = IsDifferentiable()
greeks_trait(::Type{EuropeanOption}) = HasGreeksTrait()

# Black-Scholes pricing
function price(opt::EuropeanOption, state::MarketState)
    S = state.prices[opt.underlying]
    K = opt.strike
    T = opt.expiry
    r = first(values(state.rates))  # Assume single rate for now
    σ = state.volatilities[opt.underlying]

    return black_scholes(S, K, T, r, σ, opt.optiontype)
end

"""
    black_scholes(S, K, T, r, σ, optiontype)

Black-Scholes option pricing formula.

# Arguments
- `S` - Current price of underlying
- `K` - Strike price
- `T` - Time to expiration (years)
- `r` - Risk-free rate
- `σ` - Volatility
- `optiontype` - :call or :put
"""
function black_scholes(S, K, T, r, σ, optiontype::Symbol)
    d1 = (log(S/K) + (r + σ^2/2)*T) / (σ*sqrt(T))
    d2 = d1 - σ*sqrt(T)

    N = Normal()

    if optiontype == :call
        return S * cdf(N, d1) - K * exp(-r*T) * cdf(N, d2)
    else  # put
        return K * exp(-r*T) * cdf(N, -d2) - S * cdf(N, -d1)
    end
end

# ============================================================================
# Exports
# ============================================================================

export Stock, EuropeanOption, price, black_scholes

# Greeks computation via AD
include("Greeks.jl")

end
