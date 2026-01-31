# Scenario Analysis

Stress testing and forward projections for portfolios.

## Historical Stress Tests

Apply historical crisis scenarios:

```julia
using Quasar

# Portfolio state
state = SimulationState(
    timestamp=now(),
    cash=50_000.0,
    positions=Dict(:SPY => 100.0, :AGG => 200.0),
    prices=Dict(:SPY => 450.0, :AGG => 100.0)
)

# Asset class mapping
asset_classes = Dict(:SPY => :equity, :AGG => :bond)

# Apply 2008 crisis
impact = scenario_impact(CRISIS_SCENARIOS[:financial_crisis_2008], state, asset_classes)
println("Portfolio loss: $(impact.pct_change * 100)%")
```

### Built-in Scenarios

- `:financial_crisis_2008` - 2008 global financial crisis
- `:covid_crash_2020` - March 2020 COVID crash
- `:dot_com_bust_2000` - 2000-2002 tech bubble burst
- `:black_monday_1987` - Single-day 22.6% crash
- `:rate_shock_2022` - 2022 Fed tightening
- `:stagflation_1970s` - 1970s stagflation

## Custom Scenarios

Create your own stress scenarios:

```julia
scenario = StressScenario(
    "Custom Rate Shock",
    "Hypothetical 300bp rate hike",
    Dict(:equity => -0.10, :bond => -0.20, :reit => -0.25),
    90  # duration in days
)
```

## Sensitivity Analysis

Analyze sensitivity to specific shocks:

```julia
results = sensitivity_analysis(
    state,
    asset_classes,
    :equity;
    shock_range=-0.50:0.10:0.50
)

# Plot or analyze results
for r in results
    println("Shock: $(r.shock*100)% => Value: $(r.portfolio_value)")
end
```

## Monte Carlo Projections

Forward-looking simulations:

```julia
projection = monte_carlo_projection(
    state,
    Dict(:SPY => 0.08, :AGG => 0.03),  # Expected annual returns
    Dict(:SPY => 0.18, :AGG => 0.05);  # Annual volatilities
    correlation=0.2,
    horizon_days=252,
    n_simulations=10000
)

println("Expected value: $(projection.mean_value)")
println("95% VaR: $(projection.var_95)")
println("Probability of loss: $(projection.prob_loss * 100)%")
```
