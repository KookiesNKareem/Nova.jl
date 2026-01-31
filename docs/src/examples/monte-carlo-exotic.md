# Monte Carlo Exotic Options

This example demonstrates pricing exotic options using Monte Carlo simulation with various variance reduction techniques.

## Setup

```julia
using Quasar
```

## Basic Monte Carlo Pricing

### European Options (Benchmark)

Start with European options to verify against Black-Scholes:

```julia
S0 = 100.0   # Initial spot
K = 100.0    # Strike
T = 1.0      # Time to maturity
r = 0.05     # Risk-free rate
σ = 0.20     # Volatility

dynamics = GBMDynamics(r, σ)

# Monte Carlo price
result = mc_price(S0, T, EuropeanCall(K), dynamics;
    npaths = 100000,
    nsteps = 252,
    antithetic = true
)

# Black-Scholes benchmark
bs_price = black_scholes(S0, K, T, r, σ, :call)

println("European Call (K=100):")
println("  Monte Carlo: \$$(round(result.price, digits=4)) ± $(round(result.stderr, digits=4))")
println("  Black-Scholes: \$$(round(bs_price, digits=4))")
println("  Error: $(round(abs(result.price - bs_price), digits=4))")
println("  95% CI: [\$$(round(result.ci_lower, digits=4)), \$$(round(result.ci_upper, digits=4))]")
```

**Output:**
```
European Call (K=100):
  Monte Carlo: $10.4477 ± 0.0329
  Black-Scholes: $10.4506
  Error: 0.0029
  95% CI: [$10.3832, $10.5122]
```

## Asian Options

Asian options depend on the average price over the life of the option:

```julia
# Asian call: max(avg(S) - K, 0)
asian_call = AsianCall(100.0)
asian_put = AsianPut(100.0)

result_call = mc_price(S0, T, asian_call, dynamics;
    npaths = 100000,
    nsteps = 252  # Daily averaging
)

result_put = mc_price(S0, T, asian_put, dynamics;
    npaths = 100000,
    nsteps = 252
)

println("\nAsian Options (K=100, daily averaging):")
println("  Call: \$$(round(result_call.price, digits=4)) ± $(round(result_call.stderr, digits=4))")
println("  Put:  \$$(round(result_put.price, digits=4)) ± $(round(result_put.stderr, digits=4))")

# Compare to European
eu_call = mc_price(S0, T, EuropeanCall(100.0), dynamics; npaths=100000)
println("\n  Asian call is $(round((1 - result_call.price/eu_call.price)*100, digits=1))% cheaper than European")
println("  (Averaging reduces volatility exposure)")
```

**Output:**
```
Asian Options (K=100, daily averaging):
  Call: $5.7374 ± 0.0246
  Put:  $3.2851 ± 0.0171

  Asian call is 45.1% cheaper than European
  (Averaging reduces volatility exposure)
```

### Averaging Frequency Impact

```julia
println("\nImpact of Averaging Frequency:")
for nsteps in [12, 52, 252]
    freq = nsteps == 12 ? "Monthly" : (nsteps == 52 ? "Weekly" : "Daily")
    result = mc_price(S0, T, AsianCall(100.0), dynamics;
        npaths=50000, nsteps=nsteps)
    println("  $freq: \$$(round(result.price, digits=4))")
end
```

**Output:**
```
Impact of Averaging Frequency:
  Monthly: $5.6694
  Weekly: $5.7589
  Daily: $5.7062
```

## Barrier Options

Barrier options are knocked out (or in) when the price crosses a barrier:

```julia
# Up-and-out call: knocked out if S ≥ barrier
K = 100.0
B = 130.0  # Barrier

up_out_call = UpAndOutCall(K, B)

result = mc_price(S0, T, up_out_call, dynamics;
    npaths = 100000,
    nsteps = 252
)

println("\nUp-and-Out Call (K=100, B=130):")
println("  Price: \$$(round(result.price, digits=4)) ± $(round(result.stderr, digits=4))")

# Compare to vanilla
vanilla = mc_price(S0, T, EuropeanCall(K), dynamics; npaths=100000)
println("  European Call: \$$(round(vanilla.price, digits=4))")
println("  Knockout discount: $(round((1 - result.price/vanilla.price)*100, digits=1))%")
```

**Output:**
```
Up-and-Out Call (K=100, B=130):
  Price: $3.5972 ± 0.0289
  European Call: $10.4506
  Knockout discount: 65.6%
```

### Down-and-Out Put

```julia
# Down-and-out put: knocked out if S ≤ barrier
K = 100.0
B = 80.0  # Barrier

down_out_put = DownAndOutPut(K, B)

result = mc_price(S0, T, down_out_put, dynamics;
    npaths = 100000,
    nsteps = 252
)

println("\nDown-and-Out Put (K=100, B=80):")
println("  Price: \$$(round(result.price, digits=4))")
```

**Output:**
```
Down-and-Out Put (K=100, B=80):
  Price: $1.7633
```

### Barrier Monitoring Frequency

```julia
println("\nBarrier Monitoring Frequency Impact:")
println("(More monitoring = more chances to hit barrier = lower price)")
for nsteps in [12, 52, 252, 504]
    result = mc_price(S0, T, UpAndOutCall(100.0, 120.0), dynamics;
        npaths = 50000,
        nsteps = nsteps
    )
    println("  $nsteps steps/year: \$$(round(result.price, digits=4))")
end
```

**Output:**
```
Barrier Monitoring Frequency Impact:
(More monitoring = more chances to hit barrier = lower price)
  12 steps/year: $1.8603
  52 steps/year: $1.4968
  252 steps/year: $1.3657
  504 steps/year: $1.3002
```

## American Options (Longstaff-Schwartz)

American options can be exercised early. The Longstaff-Schwartz algorithm uses regression to estimate continuation values:

```julia
# American put (early exercise is valuable when ITM)
am_put = AmericanPut(100.0)

result = lsm_price(S0, T, am_put, dynamics;
    npaths = 50000,
    nsteps = 50  # Exercise dates
)

println("\nAmerican Put (K=100):")
println("  Price: \$$(round(result.price, digits=4))")

# European put for comparison
eu_put = mc_price(S0, T, EuropeanPut(100.0), dynamics; npaths=50000)
premium = result.price - eu_put.price
println("  European Put: \$$(round(eu_put.price, digits=4))")
println("  Early Exercise Premium: \$$(round(premium, digits=4)) ($(round(premium/eu_put.price*100, digits=1))%)")
```

**Output:**
```
American Put (K=100):
  Price: $6.0647
  European Put: $5.5873
  Early Exercise Premium: $0.4774 (8.5%)
```

### American vs European by Moneyness

```julia
println("\nEarly Exercise Premium by Moneyness:")
println("Strike | American | European | Premium")
println("-"^45)
for K in [80.0, 90.0, 100.0, 110.0, 120.0]
    am = lsm_price(S0, T, AmericanPut(K), dynamics; npaths=30000, nsteps=50)
    eu = mc_price(S0, T, EuropeanPut(K), dynamics; npaths=30000)
    premium = am.price - eu.price
    pct = eu.price > 0.01 ? premium / eu.price * 100 : 0.0
    println("$(Int(K))     | $(round(am.price, digits=2))      | $(round(eu.price, digits=2))      | $(round(premium, digits=2)) ($(round(pct, digits=1))%)")
end
```

## Variance Reduction Techniques

### Antithetic Variates

Uses negatively correlated path pairs to reduce variance:

```julia
# Without antithetic
result_no_anti = mc_price(S0, T, EuropeanCall(100.0), dynamics;
    npaths = 50000,
    antithetic = false
)

# With antithetic
result_anti = mc_price(S0, T, EuropeanCall(100.0), dynamics;
    npaths = 50000,  # Same number of paths
    antithetic = true
)

println("\nAntithetic Variates Impact:")
println("  Without: \$$(round(result_no_anti.price, digits=4)) ± $(round(result_no_anti.stderr, digits=4))")
println("  With:    \$$(round(result_anti.price, digits=4)) ± $(round(result_anti.stderr, digits=4))")
println("  Variance reduction: $(round((1 - (result_anti.stderr/result_no_anti.stderr)^2)*100, digits=1))%")
```

### Quasi-Monte Carlo (Sobol Sequences)

QMC uses low-discrepancy sequences for better convergence:

```julia
# Standard MC
result_mc = mc_price(S0, T, EuropeanCall(100.0), dynamics;
    npaths = 10000,
    nsteps = 50
)

# Quasi-Monte Carlo
result_qmc = mc_price_qmc(S0, T, EuropeanCall(100.0), dynamics;
    npaths = 10000,
    nsteps = 50
)

println("\nQMC vs Standard MC (10,000 paths):")
println("  Standard MC: \$$(round(result_mc.price, digits=4)) ± $(round(result_mc.stderr, digits=4))")
println("  QMC:         \$$(round(result_qmc.price, digits=4))")
```

### Convergence Comparison

```julia
println("\nConvergence Analysis:")
println("Paths  | MC Error  | QMC Error | MC Stderr")
println("-"^50)

bs_price = black_scholes(S0, K, T, r, σ, :call)

for npaths in [1000, 5000, 10000, 50000]
    mc = mc_price(S0, T, EuropeanCall(K), dynamics; npaths=npaths, nsteps=50)
    qmc = mc_price_qmc(S0, T, EuropeanCall(K), dynamics; npaths=npaths, nsteps=50)

    mc_err = abs(mc.price - bs_price)
    qmc_err = abs(qmc.price - bs_price)

    println("$(lpad(npaths, 6)) | $(round(mc_err, digits=4))    | $(round(qmc_err, digits=4))    | $(round(mc.stderr, digits=4))")
end
```

## Stochastic Volatility (Heston)

The Heston model has stochastic variance:

```julia
# Heston parameters
heston = HestonDynamics(
    0.05,   # r - risk-free rate
    0.04,   # v0 - initial variance
    2.0,    # κ - mean reversion speed
    0.04,   # θ - long-term variance
    0.3,    # ξ - vol of vol
    -0.7    # ρ - correlation (negative = leverage effect)
)

# MC price under Heston
result = mc_price(S0, T, EuropeanCall(100.0), heston;
    npaths = 50000,
    nsteps = 252
)

# Compare to semi-analytical Heston
params = HestonParams(0.04, 0.04, 2.0, 0.3, -0.7)
analytical = heston_price(S0, 100.0, T, 0.05, params, :call)

println("\nHeston Model European Call:")
println("  Monte Carlo: \$$(round(result.price, digits=4)) ± $(round(result.stderr, digits=4))")
println("  Semi-analytical: \$$(round(analytical, digits=4))")
```

### Heston Smile

```julia
println("\nHeston Implied Volatility Smile:")
println("Strike | MC Price | Implied Vol")
println("-"^40)

for K in [80.0, 90.0, 100.0, 110.0, 120.0]
    result = mc_price(S0, T, EuropeanCall(K), heston; npaths=30000, nsteps=100)

    # Back out implied vol
    iv = implied_vol_search(result.price, S0, K, T, r)
    println("$(Int(K))     | \$$(round(result.price, digits=3))   | $(round(iv*100, digits=2))%")
end

# Helper function
function implied_vol_search(price, S, K, T, r; tol=1e-6)
    σ_low, σ_high = 0.01, 2.0
    for _ in 1:100
        σ_mid = (σ_low + σ_high) / 2
        model_price = black_scholes(S, K, T, r, σ_mid, :call)
        if abs(model_price - price) < tol
            return σ_mid
        elseif model_price > price
            σ_high = σ_mid
        else
            σ_low = σ_mid
        end
    end
    return (σ_low + σ_high) / 2
end
```

## Monte Carlo Greeks

Compute sensitivities via pathwise differentiation:

```julia
using Enzyme  # Or ForwardDiff

dynamics = GBMDynamics(0.05, 0.20)
payoff = EuropeanCall(100.0)

# Delta only
delta = mc_delta(S0, T, payoff, dynamics;
    npaths = 10000,
    nsteps = 50,
    backend = EnzymeBackend()
)

# Delta and Vega together
greeks = mc_greeks(S0, T, payoff, dynamics;
    npaths = 10000,
    nsteps = 50,
    backend = EnzymeBackend()
)

println("\nMonte Carlo Greeks:")
println("  Delta: $(round(delta, digits=4))")
println("  Vega:  $(round(greeks.vega, digits=4))")

# Compare to analytical
bs_delta, _, bs_vega, _, _ = black_scholes_greeks(S0, K, T, r, σ, :call)
println("\nBlack-Scholes Greeks:")
println("  Delta: $(round(bs_delta, digits=4))")
println("  Vega:  $(round(bs_vega, digits=4))")
```

## Path Simulation

For custom payoffs or analysis, simulate paths directly:

```julia
# Simulate a single GBM path
dynamics = GBMDynamics(0.05, 0.20)
path = Quasar.MonteCarlo.simulate_gbm(S0, T, 252, dynamics)

println("\nSimulated GBM Path:")
println("  Initial: \$$(round(path[1], digits=2))")
println("  Final:   \$$(round(path[end], digits=2))")
println("  Min:     \$$(round(minimum(path), digits=2))")
println("  Max:     \$$(round(maximum(path), digits=2))")

# Antithetic pair
path1, path2 = Quasar.MonteCarlo.simulate_gbm_antithetic(S0, T, 252, dynamics)
println("\nAntithetic Pair:")
println("  Path 1 final: \$$(round(path1[end], digits=2))")
println("  Path 2 final: \$$(round(path2[end], digits=2))")

# QMC path (deterministic)
Z = sobol_normals(252, 1)
path_qmc = Quasar.MonteCarlo.simulate_gbm_qmc(S0, T, 252, dynamics, Z[1, :])
println("\nQMC Path (deterministic):")
println("  Final: \$$(round(path_qmc[end], digits=2))")
```

## Custom Payoffs

Implement custom exotic payoffs:

```julia
# Example: Lookback call (payoff = S_max - K)
function price_lookback_call(S0, T, K, dynamics; npaths=50000, nsteps=252)
    total_payoff = 0.0

    for _ in 1:npaths
        path = Quasar.MonteCarlo.simulate_gbm(S0, T, nsteps, dynamics)
        S_max = maximum(path)
        payoff = max(S_max - K, 0)
        total_payoff += payoff
    end

    r = dynamics.r
    return exp(-r * T) * total_payoff / npaths
end

lookback_price = price_lookback_call(S0, T, 100.0, dynamics)
println("\nLookback Call (K=100):")
println("  Price: \$$(round(lookback_price, digits=4))")

# Compare to vanilla
vanilla = black_scholes(S0, 100.0, T, 0.05, 0.20, :call)
println("  European Call: \$$(round(vanilla, digits=4))")
println("  Lookback premium: $(round((lookback_price/vanilla - 1)*100, digits=1))%")
```

## Next Steps

- [Option Pricing](option-pricing.md) - Black-Scholes basics
- [Portfolio Risk](portfolio-risk.md) - Risk management
- [Monte Carlo Manual](../manual/montecarlo.md) - Full reference
