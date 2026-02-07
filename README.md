# QuantNova

[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://KookiesNKareem.github.io/QuantNova.jl/dev/)
[![Build Status](https://github.com/KookiesNKareem/QuantNova.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/KookiesNKareem/QuantNova.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/KookiesNKareem/QuantNova.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/KookiesNKareem/QuantNova.jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Julia](https://img.shields.io/badge/Julia-1.11+-9558B2.svg?logo=julia)](https://julialang.org/)

A differentiable quantitative finance library for Julia. **8-142x faster than QuantLib C++**.

## Performance

### vs QuantLib C++ (v1.41)

| Benchmark | QuantNova | QuantLib C++ | Speedup |
|-----------|-----------|--------------|---------|
| European option | 0.04 μs | 5.7 μs | **139x** |
| Greeks (all 5 via AD) | 0.08 μs | 5.7 μs | **71x** |
| American (100-step) | 8.5 μs | 67 μs | **8x** |
| SABR implied vol | 0.04 μs | 0.8 μs | **20x** |
| Batch (1000 options) | 40 μs | 5.7 ms | **142x** |

### vs Python

| Benchmark | QuantNova | Python | Speedup |
|-----------|-----------|--------|---------|
| CAPM regression | 21 μs | 450 μs (statsmodels) | **21x** |
| Fama-French 3-factor | 23 μs | 550 μs (statsmodels) | **24x** |
| Rolling beta (5yr) | 376 μs | 12 ms (pandas) | **32x** |
| Information coefficient | 0.6 μs | 25 μs (scipy) | **40x** |
| SMA crossover backtest | 104 μs | 2.5 ms (pandas) | **24x** |

*Benchmarks on Apple M1. Full methodology: see the [Benchmark Methodology](https://KookiesNKareem.github.io/QuantNova.jl/dev/manual/benchmarks.html) page.*

## Installation

```julia
using Pkg
Pkg.add("QuantNova")
```

## Quick Start

```julia
using QuantNova

# Price an option
price = black_scholes(100.0, 100.0, 1.0, 0.05, 0.2, :call)  # $10.45

# Compute Greeks via AD (not finite differences)
state = MarketState(
    prices = Dict("SPX" => 100.0),
    rates = Dict("USD" => 0.05),
    volatilities = Dict("SPX" => 0.2)
)
option = EuropeanOption("SPX", 100.0, 1.0, :call)
greeks = compute_greeks(option, state)
# greeks.delta = 0.637, greeks.gamma = 0.019, greeks.vega = 0.375

# Calibrate SABR to market smile
quotes = [OptionQuote(K, 1.0, 0.0, :call, vol) for (K, vol) in market_data]
result = calibrate_sabr(SmileData(1.0, 100.0, 0.05, quotes))
# result.rmse < 0.3%, result.converged = true

# Monte Carlo for exotics
mc_price(100.0, 1.0, AsianCall(100.0), GBMDynamics(0.05, 0.2); npaths=50000)
lsm_price(100.0, 1.0, AmericanPut(100.0), GBMDynamics(0.05, 0.2); npaths=50000)
```

## Features

- **Options**: Black-Scholes, SABR, Heston, Monte Carlo (European, Asian, Barrier, American)
- **Greeks**: All sensitivities via automatic differentiation (ForwardDiff, Enzyme, Reactant)
- **Calibration**: SABR and Heston with multi-start Adam optimizer
- **Risk**: VaR, CVaR, Sharpe, drawdown, factor models (CAPM, Fama-French)
- **Interest Rates**: Yield curves, bonds, swaps, caps/floors, short-rate models
- **Backtesting**: Strategy signals, position management, transaction costs

## Demo

Run the full demo to see pricing, Greeks, SABR calibration, and Monte Carlo in action:

```bash
julia --project=. demos/options_pricing_demo.jl
```

See the [pricing & calibration demo](https://KookiesNKareem.github.io/QuantNova.jl/dev/examples/pricing-calibration-demo) in the docs.

## AD Backends

```julia
gradient(f, x)                              # ForwardDiff (default)
gradient(f, x; backend=EnzymeBackend())     # Enzyme (GPU)
gradient(f, x; backend=ReactantBackend())   # Reactant (XLA)
```

## Documentation

Full docs at [KookiesNKareem.github.io/QuantNova.jl](https://KookiesNKareem.github.io/QuantNova.jl/dev/)

## License

MIT
