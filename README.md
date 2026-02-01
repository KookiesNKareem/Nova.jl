# QuantNova

[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://KookiesNKareem.github.io/QuantNova.jl/dev/)
[![Build Status](https://github.com/KookiesNKareem/QuantNova.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/KookiesNKareem/QuantNova.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/KookiesNKareem/QuantNova.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/KookiesNKareem/QuantNova.jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Julia](https://img.shields.io/badge/Julia-1.11+-9558B2.svg?logo=julia)](https://julialang.org/)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

A differentiable quantitative finance library for Julia.

## Performance

### Option Pricing (vs QuantLib C++ v1.41)

| Benchmark | QuantNova.jl | QuantLib C++ | |
|-----------|---------|--------------|---------|
| European option pricing | 0.04 μs | 5.7 μs | **139x faster** |
| Greeks (all 5 via AD) | 0.08 μs | 5.7 μs | **71x faster** |
| American option (100-step binomial) | 8.5 μs | 67 μs | **8x faster** |
| SABR implied vol | 0.04 μs | 0.8 μs | **20x faster** |
| Batch pricing (1000 options) | 40 μs | 5.7 ms | **142x faster** |

### Factor Models & Statistics (vs Python)

| Benchmark | QuantNova.jl | Python | |
|-----------|---------|--------|---------|
| CAPM regression | 21 μs | 450 μs (statsmodels) | **21x faster** |
| Fama-French 3-factor | 23 μs | 550 μs (statsmodels) | **24x faster** |
| Rolling beta (60-day, 5yr) | 376 μs | 12 ms (pandas) | **32x faster** |
| Information coefficient | 0.6 μs | 25 μs (scipy) | **40x faster** |
| Sharpe ratio | 0.96 μs | 2 μs (numpy) | **2x faster** |

### Backtesting (vs Python pandas)

| Benchmark | QuantNova.jl | Python (pandas) | |
|-----------|---------|-----------------|---------|
| SMA crossover (5yr daily) | 104 μs | 2.5 ms | **24x faster** |
| Rolling Sharpe (252-day) | 313 μs | 800 μs | **3x faster** |
| Full metrics (Sharpe, MaxDD, etc.) | 7 μs | 150 μs | **21x faster** |

*Benchmarks on Apple M1. See `benchmarks/comparison/` for methodology and reproducibility.*

**Why the difference?** Julia compiles specialized native code via JIT, eliminating Python's interpreter overhead. QuantLib's object graph (`Handle`, `Quote`, `PricingEngine`) provides flexibility for complex structures but adds construction cost. QuantNova's pure-function design enables aggressive compiler optimization.

## Features

- **Differentiable by default**: Every computation flows through a unified AD system. Gradients are first-class outputs.
- **Backend-agnostic**: Same code runs on CPU (ForwardDiff) or GPU (Reactant+Enzyme). Write once, deploy anywhere.
- **Research to production**: Pure Julia reference implementations for debugging, optimized backends for production.
- **Composable abstractions**: Small, focused types that combine naturally.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/KookiesNKareem/QuantNova.jl")
```

## Quick Start

### Market State

QuantNova separates instruments from market data. A `MarketState` holds current prices, rates, and volatilities:

```julia
using QuantNova

state = MarketState(
    prices = Dict("AAPL" => 150.0, "GOOGL" => 140.0),
    rates = Dict("USD" => 0.05),
    volatilities = Dict("AAPL" => 0.2, "GOOGL" => 0.25),
    timestamp = now()
)
```

### Pricing Options

```julia
# Create instruments (just contracts, no market data)
stock = Stock("AAPL")
option = EuropeanOption("AAPL", 155.0, 0.5, :call)  # underlying, strike, expiry, type

# Price using Black-Scholes (pass market state)
price(option, state)

# Or use Black-Scholes directly
black_scholes(150.0, 155.0, 0.5, 0.05, 0.2, :call)  # S, K, T, r, σ, type
```

### Computing Greeks

```julia
# Compute all Greeks via automatic differentiation
greeks = compute_greeks(option, state)

greeks.delta  # Price sensitivity to underlying
greeks.gamma  # Delta sensitivity to underlying
greeks.vega   # Price sensitivity to volatility
greeks.theta  # Price sensitivity to time
greeks.rho    # Price sensitivity to interest rate
```

### Portfolio Management

```julia
# Create a portfolio of options
options = [
    EuropeanOption("AAPL", 155.0, 0.5, :call),
    EuropeanOption("AAPL", 145.0, 0.5, :put)
]
portfolio = Portfolio(options, [100.0, 50.0])  # instruments, positions

# Get portfolio value
value(portfolio, state)

# Compute aggregated Greeks
portfolio_greeks(portfolio, state)
```

### Risk Measures

```julia
returns = randn(1000) * 0.02  # Simulated daily returns

# Value at Risk (95% confidence)
compute(VaR(0.95), returns)

# Conditional VaR (Expected Shortfall)
compute(CVaR(0.95), returns)

# Volatility
compute(Volatility(), returns)

# Sharpe Ratio
compute(Sharpe(; rf=0.02), returns)

# Maximum Drawdown
compute(MaxDrawdown(), returns)
```

### Portfolio Optimization

```julia
# Expected returns and covariance matrix
mu = [0.10, 0.12, 0.08]
Sigma = [0.04 0.01 0.005;
         0.01 0.05 0.01;
         0.005 0.01 0.03]

# Mean-variance optimization
mv = MeanVariance(mu, Sigma)
result = optimize(mv; target_return=0.10)
result.weights      # Optimal portfolio weights
result.objective    # Achieved variance

# Sharpe ratio maximization
sm = SharpeMaximizer(mu, Sigma; rf=0.02)
result = optimize(sm)
```

### Monte Carlo Pricing

```julia
# GBM dynamics for European/Asian/Barrier options
dynamics = GBMDynamics(0.05, 0.2)  # r=5%, σ=20%

# European options with variance reduction
result = mc_price(100.0, 1.0, EuropeanCall(105.0), dynamics;
                  npaths=50000, antithetic=true)
result.price   # Monte Carlo estimate
result.stderr  # Standard error

# Asian options (path-dependent)
mc_price(100.0, 1.0, AsianCall(100.0), dynamics; npaths=50000)

# Barrier options
mc_price(100.0, 1.0, UpAndOutCall(100.0, 130.0), dynamics; npaths=50000)

# Heston stochastic volatility
heston = HestonDynamics(0.05, 0.04, 2.0, 0.04, 0.3, -0.7)
mc_price(100.0, 1.0, EuropeanCall(100.0), heston; npaths=50000)
```

### American Options (Longstaff-Schwartz)

```julia
# Price American options using LSM algorithm
dynamics = GBMDynamics(0.05, 0.2)

am_put = lsm_price(100.0, 1.0, AmericanPut(100.0), dynamics;
                   npaths=50000, nsteps=50)
am_put.price   # American option price
am_put.stderr  # Standard error

# Compare to European (American >= European for puts)
eu_put = mc_price(100.0, 1.0, EuropeanPut(100.0), dynamics; npaths=50000)
early_exercise_premium = am_put.price - eu_put.price
```

### Model Calibration

```julia
# Calibrate SABR model to market smile
quotes = [OptionQuote(K, T, 0.0, :call, market_vol) for (K, market_vol) in data]
smile = SmileData(T, forward, rate, quotes)

result = calibrate_sabr(smile; beta=1.0)
result.params.alpha  # Fitted α
result.params.rho    # Fitted ρ (skew)
result.params.nu     # Fitted ν (smile curvature)
result.rmse          # Calibration error

# Price with calibrated SABR
sabr_implied_vol(F, K, T, result.params)
sabr_price(F, K, T, r, result.params, :call)

# Heston calibration to term structure
surface = VolSurface([smile1, smile2, smile3])
result = calibrate_heston(surface)
```

## AD Backends

QuantNova supports multiple automatic differentiation backends with a unified API:

| Backend | Engine | Best For |
|---------|--------|----------|
| `ForwardDiffBackend()` | ForwardDiff.jl | Default, low-dim, reliable |
| `PureJuliaBackend()` | Finite differences | Debugging, testing |
| `EnzymeBackend()` | Enzyme.jl (LLVM) | Large-scale, reverse-mode |
| `ReactantBackend()` | Reactant.jl (XLA) | GPU acceleration |

```julia
using QuantNova

# Default: ForwardDiff
gradient(f, x)

# Per-call override
gradient(f, x; backend=EnzymeBackend())

# Global switch
set_backend!(EnzymeBackend())

# Scoped (recommended)
with_backend(ReactantBackend()) do
    gradient(f, x)
    hessian(f, x)
end  # Original backend restored

# GPU backends require loading first
using Enzyme    # For EnzymeBackend
using Reactant  # For ReactantBackend
```

### Monte Carlo Greeks with Enzyme

Enzyme can't differentiate through RNGs, so QuantNova automatically uses Quasi-Monte Carlo (Sobol sequences):

```julia
using Enzyme
using QuantNova

# This works! Uses QMC internally for Enzyme
delta = mc_delta(S0, T, payoff, dynamics; backend=EnzymeBackend())
greeks = mc_greeks(S0, T, payoff, dynamics; backend=EnzymeBackend())
```

See [full documentation](https://KookiesNKareem.github.io/QuantNova.jl/dev/backends/) for backend details, limitations, and performance tips.

## Type Hierarchy

```
AbstractInstrument
├── AbstractEquity
│   └── Stock
├── AbstractDerivative
│   ├── AbstractOption
│   │   └── EuropeanOption
│   └── AbstractFuture
└── AbstractPortfolio
    └── Portfolio{I<:AbstractInstrument}
```

## Traits

QuantNova uses Julia's Holy Traits pattern for capability dispatch:

- `Priceable` - Can compute present value given market state
- `Differentiable` - Participates in AD
- `HasGreeks` - Can compute sensitivities (Delta, Gamma, Vega, etc.)
- `Simulatable` - Can be included in Monte Carlo paths

## Notebooks

Interactive tutorials demonstrating key features:

| Notebook | Description |
|----------|-------------|
| [Volatility Smile Calibration](notebooks/volatility_smile_calibration.ipynb) | Calibrate SABR model to market smiles using AD-powered optimization |
| [American Options with LSM](notebooks/american_options_lsm.ipynb) | Price American options with Longstaff-Schwartz Monte Carlo |

## License

MIT License - see LICENSE file for details.
