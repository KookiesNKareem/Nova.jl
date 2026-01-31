# Quasar.jl

*Differentiable Quantitative Finance for Julia*

Quasar is a quantitative finance library for Julia that puts automatic differentiation at the center of everything. Whether you're pricing exotic derivatives, computing Greeks, calibrating models, or optimizing portfolios, Quasar provides a unified, differentiable API.

## Quick Example

```julia
using Quasar

# Price a European call option
S0, K, T, r, σ = 100.0, 100.0, 1.0, 0.05, 0.2
price = black_scholes(S0, K, T, r, σ, :call)

# Compute Greeks via AD
option = EuropeanOption("AAPL", K, T, :call)
state = MarketState(prices=Dict("AAPL" => S0), rates=Dict("USD" => r), volatilities=Dict("AAPL" => σ))
greeks = compute_greeks(option, state)

# Monte Carlo with pathwise Greeks
dynamics = GBMDynamics(r, σ)
delta = mc_delta(S0, T, EuropeanCall(K), dynamics; backend=EnzymeBackend())
```

## Key Features

| Feature | Description |
|---------|-------------|
| **Options Pricing** | Black-Scholes, Heston, SABR models |
| **Monte Carlo** | European, Asian, barrier, American (LSM) |
| **Greeks** | Analytical and AD-based sensitivities |
| **Calibration** | SABR and Heston model calibration |
| **Optimization** | Mean-variance, Sharpe, CVaR objectives |
| **Risk Measures** | VaR, CVaR, volatility, max drawdown |

## AD Backends

Choose the right backend for your workload:

| Backend | Best For |
|---------|----------|
| `ForwardDiffBackend()` | Default, reliable, low-dimensional |
| `EnzymeBackend()` | Large-scale, reverse-mode, GPU |
| `ReactantBackend()` | XLA compilation, GPU acceleration |
| `PureJuliaBackend()` | Debugging, testing |

```julia
# Switch backends easily
gradient(f, x; backend=EnzymeBackend())

# Or use scoped switching
with_backend(ReactantBackend()) do
    optimize(objective, x0)
end
```

## Getting Started

- [Installation](@ref) - How to install Quasar
- [Quick Start](@ref) - Get up and running quickly

## Manual

- [AD Backends](@ref) - Guide to automatic differentiation backends
- [Monte Carlo](@ref) - Monte Carlo simulation engine
- [Portfolio Optimization](@ref) - Portfolio optimization with AD

## API Reference

- [API Reference](@ref) - Complete API documentation
