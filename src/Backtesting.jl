module Backtesting

using Dates
using Statistics: mean, std
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
# Strategy Context (Price History for Custom Strategies)
# ============================================================================

"""
    StrategyContext

Tracks price history and provides lookback data for strategies.
This is passed to signal functions to compute indicators.
"""
mutable struct StrategyContext
    symbols::Vector{Symbol}
    price_history::Dict{Symbol, Vector{Float64}}
    return_history::Dict{Symbol, Vector{Float64}}
    timestamps::Vector{DateTime}
    max_lookback::Int

    function StrategyContext(symbols::Vector{Symbol}; max_lookback::Int=252)
        price_history = Dict(s => Float64[] for s in symbols)
        return_history = Dict(s => Float64[] for s in symbols)
        new(symbols, price_history, return_history, DateTime[], max_lookback)
    end
end

function update!(ctx::StrategyContext, state::SimulationState)
    push!(ctx.timestamps, state.timestamp)

    for sym in ctx.symbols
        price = get(state.prices, sym, NaN)
        push!(ctx.price_history[sym], price)

        # Compute return
        if length(ctx.price_history[sym]) > 1
            prev = ctx.price_history[sym][end-1]
            ret = (price - prev) / prev
            push!(ctx.return_history[sym], ret)
        end

        # Trim to max lookback
        if length(ctx.price_history[sym]) > ctx.max_lookback
            popfirst!(ctx.price_history[sym])
        end
        if length(ctx.return_history[sym]) > ctx.max_lookback
            popfirst!(ctx.return_history[sym])
        end
    end

    if length(ctx.timestamps) > ctx.max_lookback
        popfirst!(ctx.timestamps)
    end
end

"""
    get_returns(ctx::StrategyContext, sym::Symbol, lookback::Int) -> Vector{Float64}

Get the last `lookback` returns for a symbol.
"""
function get_returns(ctx::StrategyContext, sym::Symbol, lookback::Int)
    rets = ctx.return_history[sym]
    n = min(lookback, length(rets))
    return rets[end-n+1:end]
end

"""
    get_prices(ctx::StrategyContext, sym::Symbol, lookback::Int) -> Vector{Float64}

Get the last `lookback` prices for a symbol.
"""
function get_prices(ctx::StrategyContext, sym::Symbol, lookback::Int)
    prices = ctx.price_history[sym]
    n = min(lookback, length(prices))
    return prices[end-n+1:end]
end

# ============================================================================
# Signal-Based Strategy (Custom)
# ============================================================================

"""
    SignalStrategy <: AbstractStrategy

A flexible strategy where users provide a signal function that computes target weights.

The signal function receives the StrategyContext and current SimulationState,
and should return a Dict{Symbol,Float64} of target weights.

# Fields
- `signal_fn` - Function `(ctx::StrategyContext, state::SimulationState) -> Dict{Symbol,Float64}`
- `symbols::Vector{Symbol}` - Assets to trade
- `rebalance_frequency::Symbol` - :daily, :weekly, or :monthly
- `min_weight::Float64` - Minimum weight per asset (default: 0.0)
- `max_weight::Float64` - Maximum weight per asset (default: 1.0)
- `lookback::Int` - Price history lookback (default: 252)

# Example
```julia
# Custom momentum signal
function my_signal(ctx, state)
    weights = Dict{Symbol,Float64}()
    for sym in ctx.symbols
        rets = get_returns(ctx, sym, 20)
        if length(rets) >= 20
            momentum = sum(rets)
            weights[sym] = max(0, momentum)  # Long only if positive momentum
        else
            weights[sym] = 0.0
        end
    end
    # Normalize
    total = sum(values(weights))
    if total > 0
        for sym in keys(weights)
            weights[sym] /= total
        end
    end
    return weights
end

strategy = SignalStrategy(my_signal, [:AAPL, :MSFT, :GOOGL])
```
"""
mutable struct SignalStrategy <: AbstractStrategy
    signal_fn::Function
    symbols::Vector{Symbol}
    rebalance_frequency::Symbol
    min_weight::Float64
    max_weight::Float64
    context::StrategyContext
    last_rebalance::Base.RefValue{Union{Nothing,DateTime}}

    function SignalStrategy(
        signal_fn::Function,
        symbols::Vector{Symbol};
        rebalance_frequency::Symbol=:daily,
        min_weight::Float64=0.0,
        max_weight::Float64=1.0,
        lookback::Int=252
    )
        rebalance_frequency in (:daily, :weekly, :monthly) ||
            error("rebalance_frequency must be :daily, :weekly, or :monthly")
        ctx = StrategyContext(symbols; max_lookback=lookback)
        new(signal_fn, symbols, rebalance_frequency, min_weight, max_weight, ctx, Ref{Union{Nothing,DateTime}}(nothing))
    end
end

function should_rebalance(strategy::SignalStrategy, state::SimulationState)
    if isnothing(strategy.last_rebalance[])
        return true
    end

    last = strategy.last_rebalance[]
    current = state.timestamp

    if strategy.rebalance_frequency == :daily
        return Date(current) > Date(last)
    elseif strategy.rebalance_frequency == :weekly
        return week(current) != week(last) || year(current) != year(last)
    else  # monthly
        return month(current) != month(last) || year(current) != year(last)
    end
end

function generate_orders(strategy::SignalStrategy, state::SimulationState)
    # Update context with current prices
    update!(strategy.context, state)

    # Check if we should rebalance
    should_rebalance(strategy, state) || return Order[]

    # Get target weights from signal function
    raw_weights = strategy.signal_fn(strategy.context, state)

    # Apply constraints
    weights = Dict{Symbol,Float64}()
    for sym in strategy.symbols
        w = get(raw_weights, sym, 0.0)
        w = clamp(w, strategy.min_weight, strategy.max_weight)
        weights[sym] = w
    end

    # Normalize to sum to 1 (or less if all weights are small)
    total = sum(values(weights))
    if total > 1.0
        for sym in keys(weights)
            weights[sym] /= total
        end
    end

    # Generate orders
    orders = Order[]
    total_value = portfolio_value(state)

    for (sym, target_weight) in weights
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
# Momentum Strategy
# ============================================================================

"""
    MomentumStrategy <: AbstractStrategy

Trend-following strategy that goes long assets with positive momentum.

Uses past returns over a lookback period to rank assets and allocate
to top performers.

# Fields
- `symbols::Vector{Symbol}` - Assets to trade
- `lookback::Int` - Lookback period for momentum calculation (default: 20)
- `top_n::Int` - Number of top assets to hold (default: all with positive momentum)
- `rebalance_frequency::Symbol` - :daily, :weekly, or :monthly

# Example
```julia
# Hold top 3 momentum stocks, rebalance monthly
strategy = MomentumStrategy(
    [:AAPL, :MSFT, :GOOGL, :AMZN, :META],
    lookback=60,
    top_n=3,
    rebalance_frequency=:monthly
)
```
"""
mutable struct MomentumStrategy <: AbstractStrategy
    symbols::Vector{Symbol}
    lookback::Int
    top_n::Int
    rebalance_frequency::Symbol
    context::StrategyContext
    last_rebalance::Base.RefValue{Union{Nothing,DateTime}}

    function MomentumStrategy(
        symbols::Vector{Symbol};
        lookback::Int=20,
        top_n::Int=0,  # 0 means all with positive momentum
        rebalance_frequency::Symbol=:monthly
    )
        rebalance_frequency in (:daily, :weekly, :monthly) ||
            error("rebalance_frequency must be :daily, :weekly, or :monthly")
        n = top_n > 0 ? top_n : length(symbols)
        ctx = StrategyContext(symbols; max_lookback=lookback + 10)
        new(symbols, lookback, n, rebalance_frequency, ctx, Ref{Union{Nothing,DateTime}}(nothing))
    end
end

function should_rebalance(strategy::MomentumStrategy, state::SimulationState)
    if isnothing(strategy.last_rebalance[])
        return true
    end

    last = strategy.last_rebalance[]
    current = state.timestamp

    if strategy.rebalance_frequency == :daily
        return Date(current) > Date(last)
    elseif strategy.rebalance_frequency == :weekly
        return week(current) != week(last) || year(current) != year(last)
    else
        return month(current) != month(last) || year(current) != year(last)
    end
end

function generate_orders(strategy::MomentumStrategy, state::SimulationState)
    update!(strategy.context, state)
    should_rebalance(strategy, state) || return Order[]

    # Compute momentum scores
    scores = Dict{Symbol,Float64}()
    for sym in strategy.symbols
        rets = get_returns(strategy.context, sym, strategy.lookback)
        if length(rets) >= strategy.lookback
            # Momentum = cumulative return over lookback
            scores[sym] = sum(rets)
        end
    end

    # Not enough history yet
    isempty(scores) && return Order[]

    # Rank and select top N
    sorted_assets = sort(collect(scores), by=x -> -x[2])
    selected = sorted_assets[1:min(strategy.top_n, length(sorted_assets))]

    # Only keep positive momentum (or all if top_n specified)
    if strategy.top_n == 0
        selected = filter(x -> x[2] > 0, selected)
    end

    # Equal weight among selected
    weights = Dict{Symbol,Float64}()
    if !isempty(selected)
        w = 1.0 / length(selected)
        for (sym, _) in selected
            weights[sym] = w
        end
    end

    # Generate orders
    orders = Order[]
    total_value = portfolio_value(state)

    for sym in strategy.symbols
        haskey(state.prices, sym) || continue

        target_weight = get(weights, sym, 0.0)
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
# Mean Reversion Strategy
# ============================================================================

"""
    MeanReversionStrategy <: AbstractStrategy

Contrarian strategy that buys underperformers and sells outperformers.

Uses z-scores of recent returns to identify assets that have deviated
from their mean and are likely to revert.

# Fields
- `symbols::Vector{Symbol}` - Assets to trade
- `lookback::Int` - Lookback for mean/std calculation (default: 20)
- `entry_threshold::Float64` - Z-score threshold for entry (default: 1.5)
- `rebalance_frequency::Symbol` - :daily, :weekly, or :monthly

# Example
```julia
# Mean reversion with 2 std dev entry threshold
strategy = MeanReversionStrategy(
    [:AAPL, :MSFT, :GOOGL],
    lookback=20,
    entry_threshold=2.0,
    rebalance_frequency=:daily
)
```
"""
mutable struct MeanReversionStrategy <: AbstractStrategy
    symbols::Vector{Symbol}
    lookback::Int
    entry_threshold::Float64
    rebalance_frequency::Symbol
    context::StrategyContext
    last_rebalance::Base.RefValue{Union{Nothing,DateTime}}

    function MeanReversionStrategy(
        symbols::Vector{Symbol};
        lookback::Int=20,
        entry_threshold::Float64=1.5,
        rebalance_frequency::Symbol=:daily
    )
        rebalance_frequency in (:daily, :weekly, :monthly) ||
            error("rebalance_frequency must be :daily, :weekly, or :monthly")
        ctx = StrategyContext(symbols; max_lookback=lookback + 10)
        new(symbols, lookback, entry_threshold, rebalance_frequency, ctx, Ref{Union{Nothing,DateTime}}(nothing))
    end
end

function should_rebalance(strategy::MeanReversionStrategy, state::SimulationState)
    if isnothing(strategy.last_rebalance[])
        return true
    end

    last = strategy.last_rebalance[]
    current = state.timestamp

    if strategy.rebalance_frequency == :daily
        return Date(current) > Date(last)
    elseif strategy.rebalance_frequency == :weekly
        return week(current) != week(last) || year(current) != year(last)
    else
        return month(current) != month(last) || year(current) != year(last)
    end
end

function generate_orders(strategy::MeanReversionStrategy, state::SimulationState)
    update!(strategy.context, state)
    should_rebalance(strategy, state) || return Order[]

    # Compute z-scores
    z_scores = Dict{Symbol,Float64}()
    for sym in strategy.symbols
        rets = get_returns(strategy.context, sym, strategy.lookback)
        if length(rets) >= strategy.lookback
            μ = mean(rets)
            σ = std(rets)
            if σ > 1e-10
                # Current return z-score
                current_ret = rets[end]
                z_scores[sym] = (current_ret - μ) / σ
            end
        end
    end

    isempty(z_scores) && return Order[]

    # Compute weights: overweight oversold (negative z), underweight overbought
    weights = Dict{Symbol,Float64}()
    raw_signals = Dict{Symbol,Float64}()

    for (sym, z) in z_scores
        if z < -strategy.entry_threshold
            # Oversold - go long
            raw_signals[sym] = -z  # Positive signal for negative z
        elseif z > strategy.entry_threshold
            # Overbought - reduce/short (but we'll cap at 0 for long-only)
            raw_signals[sym] = 0.0
        else
            # Neutral zone - hold baseline
            raw_signals[sym] = 1.0
        end
    end

    # Normalize
    total = sum(values(raw_signals))
    if total > 0
        for sym in keys(raw_signals)
            weights[sym] = raw_signals[sym] / total
        end
    end

    # Generate orders
    orders = Order[]
    total_value = portfolio_value(state)

    for sym in strategy.symbols
        haskey(state.prices, sym) || continue

        target_weight = get(weights, sym, 0.0)
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
# Composite Strategy
# ============================================================================

"""
    CompositeStrategy <: AbstractStrategy

Combines multiple strategies with specified weights.

Each sub-strategy generates target weights, which are then combined
according to the strategy weights.

# Fields
- `strategies::Vector{<:AbstractStrategy}` - Sub-strategies
- `strategy_weights::Vector{Float64}` - Weight for each strategy (must sum to 1.0)
- `symbols::Vector{Symbol}` - All symbols across strategies

# Example
```julia
# 60% momentum, 40% mean reversion
momentum = MomentumStrategy(symbols, lookback=60)
mean_rev = MeanReversionStrategy(symbols, lookback=20)

strategy = CompositeStrategy(
    [momentum, mean_rev],
    [0.6, 0.4]
)
```
"""
mutable struct CompositeStrategy <: AbstractStrategy
    strategies::Vector{Any}  # AbstractStrategy
    strategy_weights::Vector{Float64}
    symbols::Vector{Symbol}
    last_rebalance::Base.RefValue{Union{Nothing,DateTime}}

    function CompositeStrategy(
        strategies::Vector,
        strategy_weights::Vector{Float64}
    )
        length(strategies) == length(strategy_weights) ||
            error("Number of strategies must match number of weights")
        abs(sum(strategy_weights) - 1.0) < 0.01 ||
            error("Strategy weights must sum to 1.0")

        # Collect all symbols
        all_symbols = Set{Symbol}()
        for s in strategies
            if hasproperty(s, :symbols)
                union!(all_symbols, s.symbols)
            elseif hasproperty(s, :target_weights)
                union!(all_symbols, keys(s.target_weights))
            end
        end

        new(strategies, strategy_weights, collect(all_symbols), Ref{Union{Nothing,DateTime}}(nothing))
    end
end

function generate_orders(strategy::CompositeStrategy, state::SimulationState)
    # Collect weights from each sub-strategy
    combined_weights = Dict{Symbol,Float64}(s => 0.0 for s in strategy.symbols)

    for (i, sub_strategy) in enumerate(strategy.strategies)
        sw = strategy.strategy_weights[i]

        # Generate orders from sub-strategy (updates its internal state)
        sub_orders = generate_orders(sub_strategy, state)

        # Extract target weights from sub-strategy
        sub_weights = Dict{Symbol,Float64}()
        if hasproperty(sub_strategy, :target_weights)
            sub_weights = sub_strategy.target_weights
        elseif hasproperty(sub_strategy, :context)
            # For signal-based strategies, we need to recalculate
            # Actually, let's compute implied weights from orders + current positions
            total_value = portfolio_value(state)
            for sym in strategy.symbols
                current_value = get(state.positions, sym, 0.0) * get(state.prices, sym, 0.0)
                # Find order for this symbol
                order_value = 0.0
                for o in sub_orders
                    if o.symbol == sym
                        price = state.prices[sym]
                        order_value = o.side == :buy ? o.quantity * price : -o.quantity * price
                        break
                    end
                end
                target_value = current_value + order_value
                sub_weights[sym] = total_value > 0 ? target_value / total_value : 0.0
            end
        end

        # Add weighted contribution
        for (sym, w) in sub_weights
            combined_weights[sym] = get(combined_weights, sym, 0.0) + sw * w
        end
    end

    # Normalize
    total = sum(values(combined_weights))
    if total > 0
        for sym in keys(combined_weights)
            combined_weights[sym] /= total
        end
    end

    # Generate orders
    orders = Order[]
    total_value = portfolio_value(state)

    for (sym, target_weight) in combined_weights
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
# Backtest Result
# ============================================================================

"""
    BacktestResult

Complete results from running a backtest.
"""
struct BacktestResult
    initial_value::Float64
    final_value::Float64
    equity_curve::Vector{Float64}
    returns::Vector{Float64}
    timestamps::Vector{DateTime}
    trades::Vector{Fill}
    positions_history::Vector{Dict{Symbol,Float64}}
    metrics::Dict{Symbol,Float64}
end

# ============================================================================
# Backtest Runner
# ============================================================================

"""
    backtest(strategy, timestamps, price_series; kwargs...)

Run a full backtest simulation.

# Arguments
- `strategy::AbstractStrategy` - Trading strategy
- `timestamps::Vector{DateTime}` - Time series dates
- `price_series::Dict{Symbol,Vector{Float64}}` - Price data per asset
- `initial_cash::Float64=100_000.0` - Starting capital
- `execution_model::AbstractExecutionModel=InstantFill()` - How orders execute

# Returns
`BacktestResult` with equity curve, trades, and performance metrics.
"""
function backtest(
    strategy::AbstractStrategy,
    timestamps::Vector{DateTime},
    price_series::Dict{Symbol,Vector{Float64}};
    initial_cash::Float64=100_000.0,
    execution_model::AbstractExecutionModel=InstantFill()
)
    n = length(timestamps)

    # Initialize
    positions = Dict{Symbol,Float64}()
    cash = initial_cash

    equity_curve = Float64[]
    returns_vec = Float64[]
    positions_history = Dict{Symbol,Float64}[]
    all_trades = Fill[]

    prev_value = initial_cash

    for i in 1:n
        # Get current prices
        prices = Dict{Symbol,Float64}()
        for (sym, series) in price_series
            prices[sym] = series[i]
        end

        # Create current state
        state = SimulationState(
            timestamp=timestamps[i],
            cash=cash,
            positions=copy(positions),
            prices=prices
        )

        # Generate and execute orders
        orders = generate_orders(strategy, state)
        for order in orders
            fill = execute(execution_model, order, prices; timestamp=timestamps[i])
            push!(all_trades, fill)

            # Update positions and cash
            sym = fill.symbol
            positions[sym] = get(positions, sym, 0.0) + fill.quantity
            cash -= fill.quantity > 0 ? fill.cost : -fill.cost
        end

        # Record equity
        current_value = cash
        for (sym, qty) in positions
            current_value += qty * prices[sym]
        end
        push!(equity_curve, current_value)
        push!(positions_history, copy(positions))

        # Compute return
        if i > 1
            ret = (current_value - prev_value) / prev_value
            push!(returns_vec, ret)
        end
        prev_value = current_value
    end

    # Compute metrics
    metrics = compute_backtest_metrics(equity_curve, returns_vec)

    BacktestResult(
        initial_cash,
        equity_curve[end],
        equity_curve,
        returns_vec,
        timestamps,
        all_trades,
        positions_history,
        metrics
    )
end

"""
    compute_backtest_metrics(equity_curve, returns)

Compute standard backtest performance metrics.
"""
function compute_backtest_metrics(equity_curve::Vector{Float64}, returns::Vector{Float64})
    metrics = Dict{Symbol,Float64}()

    # Total return
    metrics[:total_return] = (equity_curve[end] - equity_curve[1]) / equity_curve[1]

    # Annualized return (assuming daily data, 252 trading days)
    n_periods = length(returns)
    if n_periods > 0
        metrics[:annualized_return] = (1 + metrics[:total_return])^(252 / n_periods) - 1
    else
        metrics[:annualized_return] = 0.0
    end

    # Volatility (annualized)
    if length(returns) > 1
        metrics[:volatility] = std(returns) * sqrt(252)
    else
        metrics[:volatility] = 0.0
    end

    # Sharpe ratio (assuming 0 risk-free rate)
    if metrics[:volatility] > 0
        metrics[:sharpe_ratio] = metrics[:annualized_return] / metrics[:volatility]
    else
        metrics[:sharpe_ratio] = 0.0
    end

    # Max drawdown
    peak = equity_curve[1]
    max_dd = 0.0
    for v in equity_curve
        peak = max(peak, v)
        dd = (peak - v) / peak
        max_dd = max(max_dd, dd)
    end
    metrics[:max_drawdown] = -max_dd

    # Calmar ratio
    if max_dd > 0
        metrics[:calmar_ratio] = metrics[:annualized_return] / max_dd
    else
        metrics[:calmar_ratio] = 0.0
    end

    # Win rate (if there are returns)
    if !isempty(returns)
        metrics[:win_rate] = count(r -> r > 0, returns) / length(returns)
    else
        metrics[:win_rate] = 0.0
    end

    return metrics
end

# ============================================================================
# Volatility Targeting Strategy
# ============================================================================

"""
    VolatilityTargetStrategy <: AbstractStrategy

Wraps another strategy and scales positions to target a specific volatility level.

Uses exponentially weighted moving average (EWMA) to estimate realized volatility,
then scales all positions to hit the target annualized volatility.

# Fields
- `base_strategy::AbstractStrategy` - The underlying strategy to wrap
- `target_vol::Float64` - Target annualized volatility (e.g., 0.15 for 15%)
- `lookback::Int` - Days for volatility estimation (default: 20)
- `decay::Float64` - EWMA decay factor (default: 0.94, ~20-day half-life)
- `max_leverage::Float64` - Maximum leverage allowed (default: 1.0 = no leverage)
- `min_leverage::Float64` - Minimum leverage (default: 0.1 = 10% invested)
- `rebalance_threshold::Float64` - Only rebalance if leverage changes by this much

# Example
```julia
base = RebalancingStrategy(target_weights=Dict(:AAPL => 0.5, :MSFT => 0.5), ...)
strategy = VolatilityTargetStrategy(base, target_vol=0.15, max_leverage=1.5)
```
"""
mutable struct VolatilityTargetStrategy <: AbstractStrategy
    base_strategy::AbstractStrategy
    target_vol::Float64
    lookback::Int
    decay::Float64
    max_leverage::Float64
    min_leverage::Float64
    rebalance_threshold::Float64
    # State
    return_history::Vector{Float64}
    current_leverage::Float64
    last_portfolio_value::Float64

    function VolatilityTargetStrategy(
        base_strategy::AbstractStrategy;
        target_vol::Float64=0.15,
        lookback::Int=20,
        decay::Float64=0.94,
        max_leverage::Float64=1.0,
        min_leverage::Float64=0.1,
        rebalance_threshold::Float64=0.1
    )
        target_vol > 0 || error("target_vol must be positive")
        0 < decay < 1 || error("decay must be between 0 and 1")
        max_leverage >= min_leverage || error("max_leverage must be >= min_leverage")
        new(base_strategy, target_vol, lookback, decay, max_leverage, min_leverage,
            rebalance_threshold, Float64[], 1.0, 0.0)
    end
end

"""
    estimate_ewma_volatility(returns, decay) -> Float64

Estimate volatility using exponentially weighted moving average.
Returns annualized volatility.
"""
function estimate_ewma_volatility(returns::Vector{Float64}, decay::Float64)
    isempty(returns) && return 0.20  # Default 20% if no history

    # EWMA variance
    variance = 0.0
    weight_sum = 0.0

    for i in length(returns):-1:1
        w = decay^(length(returns) - i)
        variance += w * returns[i]^2
        weight_sum += w
    end

    variance /= weight_sum
    daily_vol = sqrt(variance)

    # Annualize
    return daily_vol * sqrt(252)
end

function generate_orders(strategy::VolatilityTargetStrategy, state::SimulationState)
    # Track returns for volatility estimation
    current_value = portfolio_value(state)
    if strategy.last_portfolio_value > 0
        ret = (current_value - strategy.last_portfolio_value) / strategy.last_portfolio_value
        push!(strategy.return_history, ret)
        # Keep only lookback period
        if length(strategy.return_history) > strategy.lookback * 2
            popfirst!(strategy.return_history)
        end
    end
    strategy.last_portfolio_value = current_value

    # Get base strategy orders
    base_orders = generate_orders(strategy.base_strategy, state)

    # If we don't have enough history, use base orders as-is
    if length(strategy.return_history) < strategy.lookback
        return base_orders
    end

    # Estimate current volatility
    realized_vol = estimate_ewma_volatility(strategy.return_history, strategy.decay)

    # Compute target leverage
    if realized_vol > 0
        raw_leverage = strategy.target_vol / realized_vol
    else
        raw_leverage = 1.0
    end

    # Clamp leverage
    new_leverage = clamp(raw_leverage, strategy.min_leverage, strategy.max_leverage)

    # Check if leverage change is significant enough to rebalance
    leverage_change = abs(new_leverage - strategy.current_leverage) / max(strategy.current_leverage, 0.01)
    if leverage_change < strategy.rebalance_threshold && isempty(base_orders)
        return Order[]
    end

    strategy.current_leverage = new_leverage

    # Scale all orders by leverage factor
    scaled_orders = Order[]
    for order in base_orders
        scaled_qty = order.quantity * new_leverage
        push!(scaled_orders, Order(order.symbol, scaled_qty, order.side))
    end

    # If no base orders but leverage changed significantly, generate rebalancing orders
    if isempty(base_orders) && leverage_change >= strategy.rebalance_threshold
        total_value = portfolio_value(state)
        for (sym, qty) in state.positions
            haskey(state.prices, sym) || continue
            current_value = qty * state.prices[sym]
            target_value = current_value * (new_leverage / strategy.current_leverage)
            diff = target_value - current_value
            if abs(diff) > 1.0
                order_qty = diff / state.prices[sym]
                side = order_qty > 0 ? :buy : :sell
                push!(scaled_orders, Order(sym, abs(order_qty), side))
            end
        end
    end

    return scaled_orders
end

# ============================================================================
# Walk-Forward Backtesting
# ============================================================================

"""
    WalkForwardConfig

Configuration for walk-forward backtesting.

# Fields
- `train_period::Int` - Number of days for training/optimization window
- `test_period::Int` - Number of days for out-of-sample testing
- `step_size::Int` - Days to advance between windows (default: test_period)
- `min_train_periods::Int` - Minimum training periods before first test
- `expanding::Bool` - If true, training window expands; if false, it rolls
"""
struct WalkForwardConfig
    train_period::Int
    test_period::Int
    step_size::Int
    min_train_periods::Int
    expanding::Bool

    function WalkForwardConfig(;
        train_period::Int=252,
        test_period::Int=21,
        step_size::Int=0,  # 0 means use test_period
        min_train_periods::Int=0,
        expanding::Bool=false
    )
        train_period > 0 || error("train_period must be positive")
        test_period > 0 || error("test_period must be positive")
        step = step_size > 0 ? step_size : test_period
        min_train = min_train_periods > 0 ? min_train_periods : train_period
        new(train_period, test_period, step, min_train, expanding)
    end
end

"""
    WalkForwardPeriod

Results from a single walk-forward period.
"""
struct WalkForwardPeriod
    train_start::DateTime
    train_end::DateTime
    test_start::DateTime
    test_end::DateTime
    weights::Dict{Symbol,Float64}
    backtest_result::BacktestResult
end

"""
    WalkForwardResult

Complete results from walk-forward backtesting.
"""
struct WalkForwardResult
    config::WalkForwardConfig
    periods::Vector{WalkForwardPeriod}
    combined_equity_curve::Vector{Float64}
    combined_returns::Vector{Float64}
    combined_timestamps::Vector{DateTime}
    metrics::Dict{Symbol,Float64}
    period_metrics::Vector{Dict{Symbol,Float64}}
end

"""
    walk_forward_backtest(
        optimizer_fn,
        symbols,
        timestamps,
        price_series;
        config=WalkForwardConfig(),
        initial_cash=100_000.0,
        execution_model=InstantFill()
    ) -> WalkForwardResult

Run walk-forward backtesting with rolling optimization windows.

# Arguments
- `optimizer_fn` - Function `(train_returns::Matrix, symbols::Vector) -> Dict{Symbol,Float64}`
                   that takes training returns and returns optimal weights
- `symbols::Vector{String}` - Asset symbols in order
- `timestamps::Vector{DateTime}` - Full timestamp series
- `price_series::Dict{Symbol,Vector{Float64}}` - Price data per asset
- `config::WalkForwardConfig` - Walk-forward configuration
- `initial_cash::Float64` - Starting capital
- `execution_model` - Execution model for backtesting

# Example
```julia
# Define optimizer function
function my_optimizer(returns, symbols)
    μ = vec(mean(returns, dims=1))
    Σ = cov(returns)
    result = optimize(MinimumVariance(Σ))
    return Dict(Symbol(symbols[i]) => result.weights[i] for i in eachindex(symbols))
end

# Run walk-forward
result = walk_forward_backtest(
    my_optimizer,
    ["AAPL", "MSFT", "GOOGL"],
    timestamps,
    prices,
    config=WalkForwardConfig(train_period=252, test_period=21)
)
```
"""
function walk_forward_backtest(
    optimizer_fn,
    symbols::Vector{String},
    timestamps::Vector{DateTime},
    price_series::Dict{Symbol,Vector{Float64}};
    config::WalkForwardConfig=WalkForwardConfig(),
    initial_cash::Float64=100_000.0,
    execution_model::AbstractExecutionModel=InstantFill()
)
    n = length(timestamps)
    n_assets = length(symbols)

    # Validate we have enough data
    min_required = config.min_train_periods + config.test_period
    n >= min_required || error("Need at least $min_required periods, got $n")

    # Build return matrix
    returns_matrix = Matrix{Float64}(undef, n - 1, n_assets)
    for (j, sym) in enumerate(symbols)
        prices = price_series[Symbol(sym)]
        for i in 2:n
            returns_matrix[i-1, j] = (prices[i] - prices[i-1]) / prices[i-1]
        end
    end

    # Generate walk-forward windows
    periods = WalkForwardPeriod[]
    period_metrics = Dict{Symbol,Float64}[]

    combined_equity = Float64[]
    combined_returns = Float64[]
    combined_timestamps = DateTime[]

    current_idx = config.min_train_periods + 1  # First test start (1-indexed for returns)
    current_cash = initial_cash

    while current_idx + config.test_period <= n
        # Determine training window
        if config.expanding
            train_start_idx = 1
        else
            train_start_idx = max(1, current_idx - config.train_period)
        end
        train_end_idx = current_idx - 1

        # Training data (returns are offset by 1 from prices)
        train_returns = returns_matrix[train_start_idx:train_end_idx, :]

        # Get optimal weights from optimizer function
        weights = optimizer_fn(train_returns, symbols)

        # Test window
        test_start_idx = current_idx
        test_end_idx = min(current_idx + config.test_period - 1, n)

        # Extract test data
        test_timestamps = timestamps[test_start_idx:test_end_idx]
        test_prices = Dict{Symbol,Vector{Float64}}()
        for sym in symbols
            test_prices[Symbol(sym)] = price_series[Symbol(sym)][test_start_idx:test_end_idx]
        end

        # Run backtest for this period
        strategy = BuyAndHoldStrategy(weights)
        bt_result = backtest(
            strategy,
            test_timestamps,
            test_prices,
            initial_cash=current_cash,
            execution_model=execution_model
        )

        # Record period
        period = WalkForwardPeriod(
            timestamps[train_start_idx],
            timestamps[train_end_idx],
            timestamps[test_start_idx],
            timestamps[test_end_idx],
            weights,
            bt_result
        )
        push!(periods, period)
        push!(period_metrics, bt_result.metrics)

        # Combine results
        append!(combined_equity, bt_result.equity_curve)
        append!(combined_returns, bt_result.returns)
        append!(combined_timestamps, bt_result.timestamps)

        # Update cash for next period (portfolio value carries forward)
        current_cash = bt_result.final_value

        # Advance window
        current_idx += config.step_size
    end

    # Compute combined metrics
    combined_metrics = compute_backtest_metrics(combined_equity, combined_returns)

    # Add walk-forward specific metrics
    combined_metrics[:n_periods] = Float64(length(periods))
    combined_metrics[:avg_period_return] = mean(m[:total_return] for m in period_metrics)
    combined_metrics[:avg_period_sharpe] = mean(m[:sharpe_ratio] for m in period_metrics)
    combined_metrics[:period_return_std] = length(period_metrics) > 1 ?
        std([m[:total_return] for m in period_metrics]) : 0.0

    # Consistency metrics
    positive_periods = count(m -> m[:total_return] > 0, period_metrics)
    combined_metrics[:period_win_rate] = positive_periods / length(period_metrics)

    WalkForwardResult(
        config,
        periods,
        combined_equity,
        combined_returns,
        combined_timestamps,
        combined_metrics,
        period_metrics
    )
end

# ============================================================================
# Additional Metrics
# ============================================================================

"""
    compute_extended_metrics(returns; rf=0.0, benchmark_returns=nothing)

Compute extended performance metrics including Sortino, Information Ratio, etc.
"""
function compute_extended_metrics(
    returns::Vector{Float64};
    rf::Float64=0.0,
    benchmark_returns::Union{Nothing,Vector{Float64}}=nothing
)
    metrics = Dict{Symbol,Float64}()
    n = length(returns)
    n == 0 && return metrics

    daily_rf = rf / 252

    # Basic stats
    mean_ret = mean(returns)
    metrics[:mean_return] = mean_ret * 252
    metrics[:volatility] = std(returns) * sqrt(252)

    # Sharpe
    excess = returns .- daily_rf
    metrics[:sharpe_ratio] = mean(excess) / std(excess) * sqrt(252)

    # Sortino (downside deviation)
    downside = filter(r -> r < daily_rf, returns)
    if length(downside) > 1
        downside_std = std(downside)
        metrics[:sortino_ratio] = (mean_ret - daily_rf) / downside_std * sqrt(252)
    else
        metrics[:sortino_ratio] = metrics[:sharpe_ratio]
    end

    # Skewness and Kurtosis
    if n > 3
        μ = mean(returns)
        σ = std(returns)
        metrics[:skewness] = mean(((returns .- μ) ./ σ).^3)
        metrics[:kurtosis] = mean(((returns .- μ) ./ σ).^4) - 3  # Excess kurtosis
    end

    # Tail ratio (95th percentile gain / 5th percentile loss)
    sorted = sort(returns)
    p5_idx = max(1, round(Int, 0.05 * n))
    p95_idx = min(n, round(Int, 0.95 * n))
    if sorted[p5_idx] < 0
        metrics[:tail_ratio] = abs(sorted[p95_idx] / sorted[p5_idx])
    else
        metrics[:tail_ratio] = Inf
    end

    # Information ratio (if benchmark provided)
    if benchmark_returns !== nothing && length(benchmark_returns) == n
        active_returns = returns .- benchmark_returns
        tracking_error = std(active_returns) * sqrt(252)
        if tracking_error > 0
            metrics[:information_ratio] = mean(active_returns) * 252 / tracking_error
            metrics[:tracking_error] = tracking_error
        end
    end

    # Profit factor (gross profits / gross losses)
    gains = sum(r for r in returns if r > 0; init=0.0)
    losses = abs(sum(r for r in returns if r < 0; init=0.0))
    if losses > 0
        metrics[:profit_factor] = gains / losses
    else
        metrics[:profit_factor] = gains > 0 ? Inf : 0.0
    end

    # Turnover would need position data - skip here

    return metrics
end

# ============================================================================
# Exports
# ============================================================================

export AbstractStrategy, generate_orders, should_rebalance
export BuyAndHoldStrategy, RebalancingStrategy
export VolatilityTargetStrategy, estimate_ewma_volatility
export BacktestResult, backtest, compute_backtest_metrics
export WalkForwardConfig, WalkForwardPeriod, WalkForwardResult
export walk_forward_backtest
export compute_extended_metrics
# Custom strategy framework
export StrategyContext, update!, get_returns, get_prices
export SignalStrategy
export MomentumStrategy, MeanReversionStrategy
export CompositeStrategy

end
