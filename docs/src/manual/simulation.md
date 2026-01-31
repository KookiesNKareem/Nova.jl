# Simulation Engine

The simulation module provides the core execution engine for backtesting and scenario analysis.

## Core Types

### SimulationState

Point-in-time snapshot of portfolio state:

```julia
state = SimulationState(
    timestamp=DateTime(2024, 1, 1),
    cash=100_000.0,
    positions=Dict(:AAPL => 100.0, :GOOGL => 50.0),
    prices=Dict(:AAPL => 150.0, :GOOGL => 140.0)
)

# Get total portfolio value
total = portfolio_value(state)  # cash + sum(positions * prices)
```

### Execution Models

Control how orders get filled:

```julia
# Instant fill at mid price
instant = InstantFill()

# With bid-ask spread
slippage = SlippageModel(spread_bps=10.0)

# With market impact
impact = MarketImpactModel(spread_bps=10.0, impact_bps_per_unit=0.1)

# Execute an order
order = Order(:AAPL, 100.0, :buy)
fill = execute(slippage, order, prices)
```

### Simulation Drivers

Drive the simulation through time:

```julia
# Historical replay
driver = HistoricalDriver(timestamps, price_series)

# Run simulation
result = simulate(driver, initial_state)
```
