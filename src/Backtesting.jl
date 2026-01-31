module Backtesting

# TODO: Add MomentumStrategy for trend-following
# TODO: Add MeanReversionStrategy for contrarian trades
# TODO: Add CompositeStrategy for combining multiple strategies

using Dates
using ..Simulation: SimulationState, Order, Fill, portfolio_value
using ..Simulation: AbstractDriver, HistoricalDriver, MarketSnapshot
using ..Simulation: AbstractExecutionModel, InstantFill, execute

# ============================================================================
# Strategy Interface
# ============================================================================

abstract type AbstractStrategy end

"""
    generate_orders(strategy, state) -> Vector{Order}

Generate orders based on strategy logic and current state.
"""
function generate_orders end

"""
    should_rebalance(strategy, state) -> Bool

Check if strategy should rebalance at current state.
"""
should_rebalance(::AbstractStrategy, ::SimulationState) = false

# ============================================================================
# Buy and Hold Strategy
# ============================================================================

"""
    BuyAndHoldStrategy <: AbstractStrategy

Invest in target weights once and hold.

# Fields
- `target_weights::Dict{Symbol,Float64}` - Target allocation (must sum to 1.0)
- `invested::Base.RefValue{Bool}` - Track if initial investment made

# Example
```julia
strategy = BuyAndHoldStrategy(Dict(:AAPL => 0.6, :GOOGL => 0.4))
orders = generate_orders(strategy, state)
```
"""
struct BuyAndHoldStrategy <: AbstractStrategy
    target_weights::Dict{Symbol,Float64}
    invested::Base.RefValue{Bool}

    function BuyAndHoldStrategy(target_weights::Dict{Symbol,Float64})
        total = sum(values(target_weights))
        abs(total - 1.0) < 0.01 || error("Target weights must sum to 1.0, got $total")
        new(target_weights, Ref(false))
    end
end

function generate_orders(strategy::BuyAndHoldStrategy, state::SimulationState)
    # Only invest once
    strategy.invested[] && return Order[]

    orders = Order[]
    total_value = portfolio_value(state)

    for (sym, target_weight) in strategy.target_weights
        haskey(state.prices, sym) || continue

        target_value = total_value * target_weight
        current_value = get(state.positions, sym, 0.0) * state.prices[sym]
        diff_value = target_value - current_value

        if abs(diff_value) > 1.0  # Minimum trade threshold
            qty = diff_value / state.prices[sym]
            side = qty > 0 ? :buy : :sell
            push!(orders, Order(sym, abs(qty), side))
        end
    end

    strategy.invested[] = true
    return orders
end

# ============================================================================
# Rebalancing Strategy
# ============================================================================

"""
    RebalancingStrategy <: AbstractStrategy

Periodically rebalance to target weights.

# Fields
- `target_weights::Dict{Symbol,Float64}` - Target allocation (must sum to 1.0)
- `rebalance_frequency::Symbol` - One of :daily, :weekly, :monthly
- `tolerance::Float64` - Rebalance if off by more than this fraction
- `last_rebalance::Base.RefValue{Union{Nothing,DateTime}}` - Last rebalance time

# Example
```julia
strategy = RebalancingStrategy(
    target_weights=Dict(:AAPL => 0.5, :GOOGL => 0.5),
    rebalance_frequency=:monthly,
    tolerance=0.05
)
```
"""
struct RebalancingStrategy <: AbstractStrategy
    target_weights::Dict{Symbol,Float64}
    rebalance_frequency::Symbol  # :daily, :weekly, :monthly
    tolerance::Float64           # Rebalance if off by more than this
    last_rebalance::Base.RefValue{Union{Nothing,DateTime}}

    function RebalancingStrategy(;
        target_weights::Dict{Symbol,Float64},
        rebalance_frequency::Symbol=:monthly,
        tolerance::Float64=0.05
    )
        total = sum(values(target_weights))
        abs(total - 1.0) < 0.01 || error("Target weights must sum to 1.0")
        rebalance_frequency in (:daily, :weekly, :monthly) ||
            error("rebalance_frequency must be :daily, :weekly, or :monthly")
        new(target_weights, rebalance_frequency, tolerance, Ref{Union{Nothing,DateTime}}(nothing))
    end
end

function should_rebalance(strategy::RebalancingStrategy, state::SimulationState)
    # Check time-based trigger
    if !isnothing(strategy.last_rebalance[])
        last = strategy.last_rebalance[]
        current = state.timestamp

        should_by_time = if strategy.rebalance_frequency == :daily
            Date(current) > Date(last)
        elseif strategy.rebalance_frequency == :weekly
            week(current) != week(last) || year(current) != year(last)
        else  # monthly
            month(current) != month(last) || year(current) != year(last)
        end

        !should_by_time && return false
    end

    # Check if weights are off target
    total_value = portfolio_value(state)
    total_value < 1.0 && return false

    for (sym, target) in strategy.target_weights
        current_value = get(state.positions, sym, 0.0) * get(state.prices, sym, 0.0)
        current_weight = current_value / total_value
        if abs(current_weight - target) > strategy.tolerance
            return true
        end
    end

    return false
end

function generate_orders(strategy::RebalancingStrategy, state::SimulationState)
    should_rebalance(strategy, state) || return Order[]

    orders = Order[]
    total_value = portfolio_value(state)

    for (sym, target_weight) in strategy.target_weights
        haskey(state.prices, sym) || continue

        target_value = total_value * target_weight
        current_value = get(state.positions, sym, 0.0) * state.prices[sym]
        diff_value = target_value - current_value

        if abs(diff_value) > 1.0
            qty = diff_value / state.prices[sym]
            side = qty > 0 ? :buy : :sell
            push!(orders, Order(sym, abs(qty), side))
        end
    end

    strategy.last_rebalance[] = state.timestamp
    return orders
end

# ============================================================================
# Exports
# ============================================================================

export AbstractStrategy, generate_orders, should_rebalance
export BuyAndHoldStrategy, RebalancingStrategy

end
