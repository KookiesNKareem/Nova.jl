# Portfolio Risk Management

This example demonstrates building and analyzing a portfolio of options, computing aggregate Greeks, and measuring risk.

## Setup

```julia
using Quasar
using Statistics
```

## Building a Portfolio

### Creating Instruments

```julia
# Define market state
state = MarketState(
    prices = Dict("AAPL" => 150.0, "GOOGL" => 140.0, "MSFT" => 380.0),
    rates = Dict("USD" => 0.05),
    volatilities = Dict("AAPL" => 0.25, "GOOGL" => 0.30, "MSFT" => 0.22),
    timestamp = 0.0
)

# Create various options
options = [
    EuropeanOption("AAPL", 155.0, 0.5, :call),   # AAPL 155 Call
    EuropeanOption("AAPL", 145.0, 0.5, :put),    # AAPL 145 Put
    EuropeanOption("GOOGL", 145.0, 0.5, :call),  # GOOGL 145 Call
    EuropeanOption("MSFT", 390.0, 0.25, :call),  # MSFT 390 Call (shorter dated)
]

# Position sizes (contracts, each representing 100 shares)
positions = [10.0, -5.0, 20.0, 15.0]  # Long 10 AAPL calls, short 5 AAPL puts, etc.

# Create portfolio
portfolio = Portfolio(options, positions)
```

### Portfolio Valuation

```julia
# Total portfolio value
total_value = value(portfolio, state)
println("Portfolio Value: \$$(round(total_value, digits=2))")

# Breakdown by position
println("\nPosition Breakdown:")
for (i, (opt, pos)) in enumerate(zip(portfolio.instruments, portfolio.weights))
    opt_price = price(opt, state)
    pos_value = opt_price * pos
    println("  $(i). $(opt.underlying) $(Int(opt.strike)) $(opt.optiontype): $(Int(pos)) × \$$(round(opt_price, digits=2)) = \$$(round(pos_value, digits=2))")
end
```

**Output:**
```
Portfolio Value: $506.33

Position Breakdown:
  1. AAPL 155 call: 10 × $9.60 = $96.00
  2. AAPL 145 put: -5 × $4.94 = $-24.70
  3. GOOGL 145 call: 20 × $11.28 = $225.60
  4. MSFT 390 call: 15 × $13.96 = $209.43
```

## Aggregate Greeks

```julia
# Portfolio-level Greeks
greeks = portfolio_greeks(portfolio, state)

println("\nPortfolio Greeks:")
println("  Delta: $(round(greeks.delta, digits=2))")
println("  Gamma: $(round(greeks.gamma, digits=4))")
println("  Vega:  $(round(greeks.vega, digits=2))")
println("  Theta: $(round(greeks.theta, digits=2))")
println("  Rho:   $(round(greeks.rho, digits=2))")
```

**Output:**
```
Portfolio Greeks:
  Delta: 24.42
  Gamma: 0.4925
  Vega:  21.52
  Theta: -1027.21
  Rho:   17.22
```

### Greeks by Underlying

```julia
# Decompose Greeks by underlying
function greeks_by_underlying(portfolio, state)
    greeks_map = Dict{String, NamedTuple}()

    for (opt, pos) in zip(portfolio.instruments, portfolio.weights)
        g = compute_greeks(opt, state)
        underlying = opt.underlying

        if haskey(greeks_map, underlying)
            prev = greeks_map[underlying]
            greeks_map[underlying] = (
                delta = prev.delta + g.delta * pos,
                gamma = prev.gamma + g.gamma * pos,
                vega = prev.vega + g.vega * pos,
            )
        else
            greeks_map[underlying] = (
                delta = g.delta * pos,
                gamma = g.gamma * pos,
                vega = g.vega * pos,
            )
        end
    end
    return greeks_map
end

println("\nGreeks by Underlying:")
for (underlying, g) in greeks_by_underlying(portfolio, state)
    println("  $underlying: Δ=$(round(g.delta, digits=2)), Γ=$(round(g.gamma, digits=4)), V=$(round(g.vega, digits=2))")
end
```

**Output:**
```
Greeks by Underlying:
  GOOGL: Δ=10.47, Γ=0.2682, V=7.89
  MSFT: Δ=7.1, Γ=0.1428, V=11.34
  AAPL: Δ=6.86, Γ=0.0815, V=2.29
```

## Risk Metrics

### Value at Risk (VaR)

```julia
# Simulate portfolio returns
function simulate_portfolio_pnl(portfolio, state, n_scenarios;
                                 spot_vol=0.02, vol_change=0.05)
    pnls = zeros(n_scenarios)
    base_value = value(portfolio, state)

    for i in 1:n_scenarios
        # Simulate correlated shocks
        spot_shock_aapl = randn() * spot_vol
        spot_shock_googl = randn() * spot_vol
        spot_shock_msft = randn() * spot_vol
        vol_shock = randn() * vol_change

        # Create shocked state
        shocked_state = MarketState(
            prices = Dict(
                "AAPL" => state.prices["AAPL"] * (1 + spot_shock_aapl),
                "GOOGL" => state.prices["GOOGL"] * (1 + spot_shock_googl),
                "MSFT" => state.prices["MSFT"] * (1 + spot_shock_msft),
            ),
            rates = Dict("USD" => 0.05),
            volatilities = Dict(
                "AAPL" => max(0.05, state.volatilities["AAPL"] * (1 + vol_shock)),
                "GOOGL" => max(0.05, state.volatilities["GOOGL"] * (1 + vol_shock)),
                "MSFT" => max(0.05, state.volatilities["MSFT"] * (1 + vol_shock)),
            ),
            timestamp = 0.0
        )

        pnls[i] = value(portfolio, shocked_state) - base_value
    end
    return pnls
end

# Generate scenarios
pnls = simulate_portfolio_pnl(portfolio, state, 10000)

# Compute risk metrics
var_95 = compute(VaR(0.95), -pnls)  # Note: VaR uses losses, not P&L
var_99 = compute(VaR(0.99), -pnls)
cvar_95 = compute(CVaR(0.95), -pnls)

println("\nRisk Metrics (1-day, simulated):")
println("  95% VaR:  \$$(round(var_95, digits=2))")
println("  99% VaR:  \$$(round(var_99, digits=2))")
println("  95% CVaR: \$$(round(cvar_95, digits=2))")
```

**Output:**
```
Risk Metrics (1-day, simulated):
  95% VaR:  $-125.78
  99% VaR:  $-181.76
  95% CVaR: $-160.04
```

### Greeks-Based VaR

A faster approximation using delta-gamma:

```julia
function delta_gamma_var(portfolio, state; spot_shock=0.02, confidence=0.95)
    greeks = portfolio_greeks(portfolio, state)
    base_value = value(portfolio, state)

    # Approximate portfolio value change using Taylor expansion
    # ΔV ≈ Δ·ΔS + ½·Γ·(ΔS)²

    # For a normal distribution, find the shock at the given percentile
    z = quantile_normal(confidence)  # ≈ 1.645 for 95%

    # Aggregate spot exposure (simplified - assumes single underlying)
    S = state.prices["AAPL"]
    ΔS = -z * spot_shock * S  # Adverse move

    # Delta-gamma approximation
    approx_loss = -(greeks.delta * ΔS + 0.5 * greeks.gamma * ΔS^2)

    return max(0, approx_loss)
end

# Simple normal quantile approximation
function quantile_normal(p)
    # Rational approximation
    t = sqrt(-2 * log(1 - p))
    return t - (2.515517 + 0.802853*t + 0.010328*t^2) /
               (1 + 1.432788*t + 0.189269*t^2 + 0.001308*t^3)
end

dg_var = delta_gamma_var(portfolio, state)
println("\nDelta-Gamma VaR (95%): \$$(round(dg_var, digits=2))")
```

**Output:**
```
Delta-Gamma VaR (95%): $114.54
```

## Stress Testing

```julia
# Define stress scenarios
scenarios = [
    ("Market Crash (-10%)", Dict("AAPL" => 0.90, "GOOGL" => 0.90, "MSFT" => 0.90), 1.5),
    ("Tech Rally (+5%)", Dict("AAPL" => 1.05, "GOOGL" => 1.05, "MSFT" => 1.05), 0.9),
    ("Vol Spike", Dict("AAPL" => 1.0, "GOOGL" => 1.0, "MSFT" => 1.0), 1.5),
    ("Rate Hike", Dict("AAPL" => 0.98, "GOOGL" => 0.98, "MSFT" => 0.98), 1.0),
]

println("\nStress Test Results:")
println("-"^60)
base_value = value(portfolio, state)

for (name, spot_mult, vol_mult) in scenarios
    stressed_state = MarketState(
        prices = Dict(k => state.prices[k] * spot_mult[k] for k in keys(state.prices)),
        rates = Dict("USD" => 0.05),
        volatilities = Dict(k => state.volatilities[k] * vol_mult for k in keys(state.volatilities)),
        timestamp = 0.0
    )

    stressed_value = value(portfolio, stressed_state)
    pnl = stressed_value - base_value
    pct_change = 100 * pnl / base_value

    println("$(rpad(name, 20)) | P&L: \$$(lpad(round(pnl, digits=2), 10)) | $(round(pct_change, digits=1))%")
end
```

**Output:**
```
Stress Test Results:
------------------------------------------------------------
Market Crash (-10%)  | P&L: $   -183.48 | -36.2%
Tech Rally (+5%)     | P&L: $    239.02 | 47.2%
Vol Spike            | P&L: $    271.06 | 53.5%
Rate Hike            | P&L: $    -98.27 | -19.4%
```

## Hedging Strategies

### Delta Hedging

```julia
# Current delta exposure by underlying
delta_exposure = Dict{String, Float64}()
for (opt, pos) in zip(portfolio.instruments, portfolio.weights)
    g = compute_greeks(opt, state)
    underlying = opt.underlying
    delta_exposure[underlying] = get(delta_exposure, underlying, 0.0) + g.delta * pos
end

println("\nDelta Hedge Required:")
for (underlying, delta) in delta_exposure
    shares = -delta * 100  # Convert to shares (100 shares per contract)
    S = state.prices[underlying]
    cost = abs(shares) * S
    println("  $underlying: $(round(shares, digits=0)) shares (\$$(round(cost, digits=0)))")
end
```

**Output:**
```
Delta Hedge Required:
  GOOGL: -1047 shares ($146530)
  MSFT: -710 shares ($269661)
  AAPL: -686 shares ($102901)
```

### Vega Hedging

```julia
# Find portfolio vega
total_vega = portfolio_greeks(portfolio, state).vega

println("\nVega Exposure: $(round(total_vega, digits=2))")

# To hedge, we need an option with known vega
# Example: use a 1-month ATM straddle
hedge_call = EuropeanOption("AAPL", 150.0, 1/12, :call)
hedge_put = EuropeanOption("AAPL", 150.0, 1/12, :put)

hedge_call_vega = compute_greeks(hedge_call, state).vega
hedge_put_vega = compute_greeks(hedge_put, state).vega
straddle_vega = hedge_call_vega + hedge_put_vega

contracts_needed = -total_vega / straddle_vega
println("Hedge with $(round(contracts_needed, digits=1)) AAPL 150 straddles (1M)")
```

**Output:**
```
Vega Exposure: 21.52
Hedge with -62.6 AAPL 150 straddles (1M)
```

## Attribution Analysis

```julia
# P&L attribution after a market move
function pnl_attribution(portfolio, old_state, new_state)
    old_value = value(portfolio, old_state)
    new_value = value(portfolio, new_state)
    total_pnl = new_value - old_value

    greeks = portfolio_greeks(portfolio, old_state)

    # Approximate contributions (simplified for single underlying)
    S_old = old_state.prices["AAPL"]
    S_new = new_state.prices["AAPL"]
    ΔS = S_new - S_old

    σ_old = old_state.volatilities["AAPL"]
    σ_new = new_state.volatilities["AAPL"]
    Δσ = σ_new - σ_old

    delta_pnl = greeks.delta * ΔS
    gamma_pnl = 0.5 * greeks.gamma * ΔS^2
    vega_pnl = greeks.vega * Δσ * 100  # Vega is per 1%
    unexplained = total_pnl - delta_pnl - gamma_pnl - vega_pnl

    return (total=total_pnl, delta=delta_pnl, gamma=gamma_pnl,
            vega=vega_pnl, unexplained=unexplained)
end

# Simulate a market move
new_state = MarketState(
    prices = Dict("AAPL" => 153.0, "GOOGL" => 142.0, "MSFT" => 385.0),
    rates = Dict("USD" => 0.05),
    volatilities = Dict("AAPL" => 0.27, "GOOGL" => 0.32, "MSFT" => 0.24),
    timestamp = 0.0
)

attr = pnl_attribution(portfolio, state, new_state)
println("\nP&L Attribution:")
println("  Total P&L:   \$$(round(attr.total, digits=2))")
println("  Delta:       \$$(round(attr.delta, digits=2))")
println("  Gamma:       \$$(round(attr.gamma, digits=2))")
println("  Vega:        \$$(round(attr.vega, digits=2))")
println("  Unexplained: \$$(round(attr.unexplained, digits=2))")
```

**Output:**
```
P&L Attribution:
  Total P&L:   $123.33
  Delta:       $73.27
  Gamma:       $2.22
  Vega:        $43.04
  Unexplained: $4.8
```

## Next Steps

- [Yield Curve](yield-curve.md) - Interest rate modeling
- [Monte Carlo](monte-carlo-exotic.md) - Path-dependent options
- [Optimization](../manual/optimization.md) - Portfolio optimization
