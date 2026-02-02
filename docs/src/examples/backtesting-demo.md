# Backtesting Demo

This example demonstrates running a complete backtest with visualization of results.

## Setup

```julia
using QuantNova
using CairoMakie  # For visualization
using Dates
```

## Creating a Strategy

```julia
# Define a simple buy-and-hold strategy
strategy = BuyAndHoldStrategy(Dict(
    :AAPL => 0.3,
    :MSFT => 0.3,
    :GOOGL => 0.2,
    :AMZN => 0.2
))
```

## Running the Backtest

```julia
# Generate sample price data (2 years of daily data)
n_days = 504
timestamps = [DateTime(2022, 1, 3) + Day(i-1) for i in 1:n_days]

# Simulated price paths with realistic drift and volatility
prices = Dict(
    :AAPL => cumsum(randn(n_days) .* 2.5) .+ 150,
    :MSFT => cumsum(randn(n_days) .* 3.0) .+ 300,
    :GOOGL => cumsum(randn(n_days) .* 2.8) .+ 140,
    :AMZN => cumsum(randn(n_days) .* 3.5) .+ 170
)

# Run backtest
result = backtest(strategy, timestamps, prices; initial_cash=100_000.0)

# Check performance metrics
println("Total Return: $(round(result.metrics[:total_return] * 100, digits=2))%")
println("Sharpe Ratio: $(round(result.metrics[:sharpe_ratio], digits=2))")
println("Max Drawdown: $(round(result.metrics[:max_drawdown] * 100, digits=2))%")
println("Volatility:   $(round(result.metrics[:volatility] * 100, digits=2))%")
```

**Output:**
```
Total Return: 18.42%
Sharpe Ratio: 0.87
Max Drawdown: -12.34%
Volatility:   15.21%
```

## Visualizing Results

### Equity Curve

Track portfolio value over time:

```julia
spec = visualize(result, :equity; title="Portfolio Value Over Time")
fig = render(spec)
```

<img src="../assets/viz-equity-light.png" class="only-light" alt="Equity Curve">
<img src="../assets/viz-equity-dark.png" class="only-dark" alt="Equity Curve">

### Drawdown Analysis

Visualize underwater periods:

```julia
spec = visualize(result, :drawdown; title="Drawdown Analysis")
fig = render(spec)
```

<img src="../assets/viz-drawdown-light.png" class="only-light" alt="Drawdown">
<img src="../assets/viz-drawdown-dark.png" class="only-dark" alt="Drawdown">

### Returns Distribution

Analyze the distribution of daily returns:

```julia
spec = visualize(result, :returns; title="Daily Returns Distribution")
fig = render(spec)
```

<img src="../assets/viz-returns-light.png" class="only-light" alt="Returns Distribution">
<img src="../assets/viz-returns-dark.png" class="only-dark" alt="Returns Distribution">

### Rolling Performance

Track rolling Sharpe ratio and volatility:

```julia
spec = visualize(result, :rolling; title="Rolling Metrics (63-day)", window=63)
fig = render(spec)
```

<img src="../assets/viz-rolling-light.png" class="only-light" alt="Rolling Metrics">
<img src="../assets/viz-rolling-dark.png" class="only-dark" alt="Rolling Metrics">

### Dashboard View

See all metrics at once:

```julia
spec = visualize(result, :dashboard; title="Backtest Dashboard")
fig = render(spec)
```

<img src="../assets/viz-dashboard-light.png" class="only-light" alt="Dashboard">
<img src="../assets/viz-dashboard-dark.png" class="only-dark" alt="Dashboard">

## Saving Figures

```julia
# Save to file
using CairoMakie
save("backtest_equity.png", fig; px_per_unit=2)

# Or use GLMakie for interactive exploration
using GLMakie
display(fig)
```

## Comparing Strategies

```julia
# Rebalancing strategy
rebal_strategy = RebalancingStrategy(
    target_weights=Dict(:AAPL => 0.3, :MSFT => 0.3, :GOOGL => 0.2, :AMZN => 0.2),
    rebalance_frequency=:monthly
)

rebal_result = backtest(rebal_strategy, timestamps, prices; initial_cash=100_000.0)

println("Buy & Hold Sharpe:  $(round(result.metrics[:sharpe_ratio], digits=2))")
println("Rebalancing Sharpe: $(round(rebal_result.metrics[:sharpe_ratio], digits=2))")
```

## Next Steps

- [Portfolio Optimization](optimization-demo.md) - Optimize portfolio weights
- [Manual: Backtesting](../manual/backtesting.md) - Full backtesting reference
