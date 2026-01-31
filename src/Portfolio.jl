module PortfolioModule

using ..Core
using ..Instruments: price, compute_greeks, GreeksResult, AbstractOption

# Import to extend
import ..Core: priceable, differentiable

# ============================================================================
# Portfolio Type
# ============================================================================

"""
    Portfolio{I<:AbstractInstrument} <: AbstractPortfolio

A collection of financial instruments with associated position weights.

# Fields
- `instruments::Vector{I}` - The instruments in the portfolio
- `weights::Vector{Float64}` - Position sizes (can be shares, contracts, or notional amounts)

# Constructors
- `Portfolio(instruments, weights)` - Create from vectors (type inferred)
- `Portfolio{I}(instruments, weights)` - Create with explicit instrument type

# Example
```julia
# Create a portfolio of options
call = EuropeanOption("AAPL", 150.0, 1.0, :call)
put = EuropeanOption("AAPL", 140.0, 1.0, :put)
portfolio = Portfolio([call, put], [100.0, -50.0])  # Long 100 calls, short 50 puts

# Price the portfolio
state = MarketState(
    prices=Dict("AAPL" => 150.0),
    rates=Dict("USD" => 0.05),
    volatilities=Dict("AAPL" => 0.2),
    timestamp=0.0
)
total_value = portfolio_value(portfolio, state)
```

See also: [`portfolio_value`](@ref), [`portfolio_greeks`](@ref)
"""
struct Portfolio{I<:AbstractInstrument} <: AbstractPortfolio
    instruments::Vector{I}
    weights::Vector{Float64}

    # TODO: Validate weights (no NaN/Inf, optionally normalize)
    function Portfolio{I}(instruments::Vector{I}, weights::Vector{Float64}) where I<:AbstractInstrument
        length(instruments) == length(weights) ||
            error("instruments and weights must have same length")
        new{I}(instruments, weights)
    end
end

# Outer constructor that infers the type parameter
function Portfolio(instruments::Vector{I}, weights::Vector{Float64}) where I<:AbstractInstrument
    Portfolio{I}(instruments, weights)
end

# Convenience constructor for mixed instrument types (untyped Vector)
function Portfolio(instruments::Vector, weights::Vector{Float64})
    converted = convert(Vector{AbstractInstrument}, instruments)
    Portfolio{AbstractInstrument}(converted, weights)
end

Base.length(p::Portfolio) = length(p.instruments)
Base.getindex(p::Portfolio, i) = (p.instruments[i], p.weights[i])

# Register traits
priceable(::Type{<:Portfolio}) = IsPriceable()
differentiable(::Type{<:Portfolio}) = IsDifferentiable()

# ============================================================================
# Portfolio Valuation
# ============================================================================

"""
    value(portfolio, market_state)

Compute total portfolio value.
"""
# TODO: Add parallel computation for large portfolios
# TODO: Return breakdown by instrument for attribution
function value(portfolio::Portfolio, state::MarketState)
    total = 0.0
    for (inst, weight) in zip(portfolio.instruments, portfolio.weights)
        total += price(inst, state) * weight
    end
    return total
end

# ============================================================================
# Portfolio Greeks
# ============================================================================

"""
    portfolio_greeks(portfolio, market_state)

Compute aggregated Greeks for the portfolio.
Only includes instruments that have Greeks (options).
"""
function portfolio_greeks(portfolio::Portfolio, state::MarketState)
    delta = 0.0
    gamma = 0.0
    vega = 0.0
    theta = 0.0
    rho = 0.0

    for (inst, weight) in zip(portfolio.instruments, portfolio.weights)
        if inst isa AbstractOption
            g = compute_greeks(inst, state)
            delta += g.delta * weight
            gamma += g.gamma * weight
            vega += g.vega * weight
            theta += g.theta * weight
            rho += g.rho * weight
        end
    end

    return GreeksResult(delta, gamma, vega, theta, rho)
end

# ============================================================================
# Exports
# ============================================================================

export Portfolio, value, portfolio_greeks

end
