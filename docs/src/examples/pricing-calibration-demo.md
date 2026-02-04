# Pricing and Calibration Case Study

This case study walks through a realistic derivatives workflow:

1. Price vanilla options and validate correctness
2. Compute Greeks with automatic differentiation
3. Calibrate SABR to a volatility smile
4. Price exotics with Monte Carlo and sanity-check results
5. Connect results to the benchmark methodology

The goal is not just to show features, but to demonstrate trustworthy outputs and reproducible performance.

## Reproduce This Case Study

```bash
julia --project=. demos/options_pricing_demo.jl
```

## Scenario Setup

We use a simple equity-style setup to keep the math interpretable:

- Spot S = 100
- Strike K = 100
- Time to expiry T = 1 year
- Risk-free rate r = 5%
- Volatility sigma = 20%

## 1) Analytical Pricing and Validation

We price a European call/put using Black-Scholes and verify put-call parity.

```julia
using QuantNova

S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20

call_price = black_scholes(S, K, T, r, sigma, :call)
put_price  = black_scholes(S, K, T, r, sigma, :put)

parity = call_price - put_price - S + K * exp(-r * T)

println("Call Price: $call_price")
println("Put Price:  $put_price")
println("Put-Call Parity: $parity")
```

Example output:
```
Call Price: 10.4506
Put Price:  5.5735
Put-Call Parity: 0.000000
```

Validation check: parity approximately 0 confirms consistent pricing.

## 2) Greeks via Automatic Differentiation

We compute all major Greeks in a single AD pass.

```julia
state = MarketState(
    prices = Dict("SPX" => S),
    rates = Dict("USD" => r),
    volatilities = Dict("SPX" => sigma),
    timestamp = 0.0
)
option = EuropeanOption("SPX", K, T, :call)

# AD-based Greeks
greeks = compute_greeks(option, state)
println(greeks)
```

Example output:
```
Greeks:
  delta =  0.637
  gamma =  0.019
  vega  =  0.375
  theta = -6.414
  rho   =  0.532
```

Validation check: values align with analytical Black-Scholes Greeks for the same inputs.

## 3) SABR Calibration to a Smile

We calibrate SABR to a synthetic volatility smile with realistic downside skew.

```julia
strikes = [85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0]
vols    = [0.28, 0.25, 0.22, 0.20, 0.19, 0.185, 0.18]

quotes = [OptionQuote(k, T, 0.0, :call, v) for (k, v) in zip(strikes, vols)]
smile_data = SmileData(T, 100.0, r, quotes)

result = calibrate_sabr(smile_data; beta=0.5)
println(result)
```

Example output:
```
Calibrated Parameters:
  alpha = 1.8578
  beta  = 0.5 (fixed)
  rho   = -0.4514
  nu    = 1.3211
  RMSE  = 0.28%
```

Validation check: low RMSE indicates the model fits the smile well.

## 4) Monte Carlo Pricing and Sanity Checks

We price exotics and validate MC results against analytic prices where possible.

```julia
dynamics = GBMDynamics(r, sigma)

# European call
mc = mc_price(S, T, EuropeanCall(K), dynamics; npaths=50000, nsteps=50)
bs = black_scholes(S, K, T, r, sigma, :call)

println("MC: $mc.price +/- $mc.stderr")
println("BS: $bs")
```

Example output:
```
MC: 10.46 +/- 0.04
BS: 10.45
```

Validation check: MC estimate should be within a few standard errors of the analytic price.

We also price an American option with Longstaff-Schwartz and verify it exceeds the European price.

## 5) Performance Benchmarks

Performance claims are based on reproducible benchmark scripts. The full methodology,
parameters, and run commands are documented here:

- [Benchmark Methodology](/manual/benchmarks)

## Takeaways

- Analytical pricing matches known identities (put-call parity).
- AD-based Greeks agree with analytical values.
- SABR calibration achieves low error on a realistic smile.
- Monte Carlo results are consistent with analytic prices within error bounds.
- Performance methodology is explicit and reproducible.
