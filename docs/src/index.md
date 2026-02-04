---
layout: home

hero:
  name: QuantNova.jl
  text: Differentiable Quantitative Finance
  tagline: High-performance derivatives pricing with automatic differentiation
  actions:
    - theme: brand
      text: Get Started
      link: /getting-started/installation
    - theme: alt
      text: View on GitHub
      link: https://github.com/KookiesNKareem/QuantNova.jl
---

<div class="metrics-bar">
  <div class="metric">
    <span class="number">8-139x</span>
    <span class="label">Faster than QuantLib C++</span>
  </div>
  <div class="metric">
    <span class="number">4</span>
    <span class="label">AD Backends</span>
  </div>
  <div class="metric">
    <span class="number">100%</span>
    <span class="label">Differentiable</span>
  </div>
  <div class="metric">
    <span class="number">100%</span>
    <span class="label">Pure Julia</span>
  </div>
</div>

## Price Options in 3 Lines

<div class="code-showcase">
<div class="code-input">

```julia
using QuantNova
S0, K, T, r, σ = 100.0, 100.0, 1.0, 0.05, 0.2
price = black_scholes(S0, K, T, r, σ, :call)
```

</div>
<div class="code-output">

```
price = 10.4506
```

</div>
</div>

## Compute All Greeks Instantly

<div class="code-showcase">
<div class="code-input">

```julia
option = EuropeanOption("AAPL", K, T, :call)
state = MarketState(
    prices = Dict("AAPL" => S0),
    rates = Dict("USD" => r),
    volatilities = Dict("AAPL" => σ)
)
greeks = compute_greeks(option, state)
```

</div>
<div class="code-output">

```
Greeks:
  Δ delta =  0.6179
  Γ gamma =  0.0188
  ν vega  = 39.4478
  θ theta = -6.4140
  ρ rho   = 53.2325
```

</div>
</div>

## Switch AD Backends Seamlessly

<div class="code-showcase">
<div class="code-input">

```julia
# CPU: ForwardDiff (default)
gradient(f, x)

# GPU: Enzyme for large-scale
gradient(f, x; backend=EnzymeBackend())

# XLA: Reactant for acceleration
gradient(f, x; backend=ReactantBackend())
```

</div>
<div class="code-output">

```
Same API, different backends:
├─ ForwardDiff: 1.2ms (CPU)
├─ Enzyme:      0.3ms (GPU)
└─ Reactant:    0.1ms (XLA)
```

</div>
</div>

<div class="features-section">

## Benchmark Methodology

Performance claims and comparisons are fully documented. See the
[Benchmark Methodology](/manual/benchmarks) page for exact scripts,
parameters, and run commands.

## What You Can Build

<div class="feature-grid">
  <div class="feature-card">
    <div class="feature-icon">📈</div>
    <h3>Options Pricing</h3>
    <p>Black-Scholes, Heston, SABR models with analytical and Monte Carlo methods</p>
  </div>
  <div class="feature-card">
    <div class="feature-icon">🎯</div>
    <h3>Greeks & Sensitivities</h3>
    <p>First and second-order Greeks via AD — no finite differences needed</p>
  </div>
  <div class="feature-card">
    <div class="feature-icon">📊</div>
    <h3>Portfolio Optimization</h3>
    <p>Mean-variance, Sharpe maximization, risk parity with constraints</p>
  </div>
  <div class="feature-card">
    <div class="feature-icon">⚠️</div>
    <h3>Risk Management</h3>
    <p>VaR, CVaR, volatility, drawdown — all differentiable</p>
  </div>
  <div class="feature-card">
    <div class="feature-icon">💹</div>
    <h3>Interest Rates</h3>
    <p>Yield curves, bonds, caps, floors, swaptions, short-rate models</p>
  </div>
  <div class="feature-card">
    <div class="feature-icon">🔧</div>
    <h3>Model Calibration</h3>
    <p>SABR and Heston calibration with gradient-based optimization</p>
  </div>
</div>

</div>
