# Option Pricing Walkthrough

This example demonstrates the complete workflow for pricing European options and computing Greeks.

## Setup

```julia
using Quasar
```

## Black-Scholes Pricing

Let's price a European call option on Apple stock.

### Direct Black-Scholes

The simplest approach uses the `black_scholes` function directly:

```julia
# Parameters
S = 150.0   # Current stock price
K = 155.0   # Strike price
T = 0.5     # Time to expiry (6 months)
r = 0.05    # Risk-free rate (5%)
σ = 0.20    # Volatility (20%)

# Price call and put
call_price = black_scholes(S, K, T, r, σ, :call)
put_price = black_scholes(S, K, T, r, σ, :put)

println("Call price: \$$(round(call_price, digits=4))")
println("Put price: \$$(round(put_price, digits=4))")

# Verify put-call parity: C - P = S - K*exp(-rT)
parity_lhs = call_price - put_price
parity_rhs = S - K * exp(-r * T)
println("Put-call parity check: $(round(parity_lhs, digits=6)) ≈ $(round(parity_rhs, digits=6))")
```

**Output:**
```
Call price: $7.9152
Put price: $9.0882
Put-call parity check: -1.173036 ≈ -1.173036
```

### Using Instruments and Market State

For portfolio management, use the object-oriented approach:

```julia
# Define market conditions
state = MarketState(
    prices = Dict("AAPL" => 150.0),
    rates = Dict("USD" => 0.05),
    volatilities = Dict("AAPL" => 0.20),
    timestamp = 0.0
)

# Create option instruments
call = EuropeanOption("AAPL", 155.0, 0.5, :call)
put = EuropeanOption("AAPL", 155.0, 0.5, :put)

# Price them
println("Call: \$$(round(price(call, state), digits=4))")
println("Put: \$$(round(price(put, state), digits=4))")
```

**Output:**
```
Call: $7.9152
Put: $9.0882
```

## Computing Greeks

### First-Order Greeks

Greeks measure option sensitivity to various market parameters:

```julia
# Compute all Greeks
greeks = compute_greeks(call, state)

println("Delta: $(round(greeks.delta, digits=4))")   # Sensitivity to spot
println("Gamma: $(round(greeks.gamma, digits=4))")   # Delta sensitivity
println("Vega: $(round(greeks.vega, digits=4))")     # Sensitivity to vol (per 1%)
println("Theta: $(round(greeks.theta, digits=4))")   # Time decay (per day)
println("Rho: $(round(greeks.rho, digits=4))")       # Sensitivity to rate (per 1%)
```

**Output:**
```
Delta: 0.5062
Gamma: 0.0188
Vega: 0.4231
Theta: -11.8628
Rho: 0.3401
```

### Second-Order Greeks

For advanced risk management, use second-order sensitivities:

```julia
println("Vanna: $(round(greeks.vanna, digits=4))")   # ∂Delta/∂σ
println("Volga: $(round(greeks.volga, digits=4))")   # ∂Vega/∂σ
println("Charm: $(round(greeks.charm, digits=4))")   # ∂Delta/∂T
```

**Output:**
```
Vanna: 0.2509
Volga: -0.0042
Charm: -0.1912
```

### Delta Hedging Example

Use delta to construct a hedged portfolio:

```julia
# Long 100 calls
num_calls = 100
option_value = price(call, state) * num_calls
delta_exposure = greeks.delta * num_calls

# Hedge with stock
shares_to_short = -delta_exposure

println("Portfolio value: \$$(round(option_value, digits=2))")
println("Shares to short for delta hedge: $(round(shares_to_short, digits=1))")

# Verify hedge - if stock moves up $1
S_new = 151.0
state_new = MarketState(
    prices = Dict("AAPL" => S_new),
    rates = Dict("USD" => 0.05),
    volatilities = Dict("AAPL" => 0.20),
    timestamp = 0.0
)

# P&L breakdown
option_pnl = (price(call, state_new) - price(call, state)) * num_calls
stock_pnl = shares_to_short * (S_new - 150.0)
total_pnl = option_pnl + stock_pnl

println("\nFor \$1 spot move:")
println("Option P&L: \$$(round(option_pnl, digits=2))")
println("Stock hedge P&L: \$$(round(stock_pnl, digits=2))")
println("Net P&L (should be ≈0): \$$(round(total_pnl, digits=2))")
```

**Output:**
```
Portfolio value: $791.52
Shares to short for delta hedge: -50.6

For $1 spot move:
Option P&L: $51.56
Stock hedge P&L: $-50.62
Net P&L (should be ≈0): $0.94
```

## Implied Volatility

Back out the volatility from a market price:

```julia
# Given a market price, find implied volatility
market_price = 8.50  # Observed option price
S, K, T, r = 150.0, 155.0, 0.5, 0.05

# Bisection search for implied vol
function implied_vol(market_price, S, K, T, r, opttype; tol=1e-6)
    σ_low, σ_high = 0.01, 2.0

    for _ in 1:100
        σ_mid = (σ_low + σ_high) / 2
        model_price = black_scholes(S, K, T, r, σ_mid, opttype)

        if abs(model_price - market_price) < tol
            return σ_mid
        elseif model_price > market_price
            σ_high = σ_mid
        else
            σ_low = σ_mid
        end
    end
    return (σ_low + σ_high) / 2
end

iv = implied_vol(market_price, S, K, T, r, :call)
println("Implied volatility: $(round(iv * 100, digits=2))%")

# Verify
model_price = black_scholes(S, K, T, r, iv, :call)
println("Model price at IV: \$$(round(model_price, digits=4))")
```

**Output:**
```
Implied volatility: 21.38%
Model price at IV: $8.5
```

## Volatility Smile

Plot implied volatility across strikes:

```julia
# Generate a volatility smile
strikes = 130.0:5.0:170.0
spot = 150.0

# Simulate market prices with skew (in practice, use real quotes)
# Higher vol for low strikes (downside skew)
function skewed_vol(K, S, base_vol=0.20)
    moneyness = log(K/S)
    skew = -0.15 * moneyness  # Negative skew
    smile = 0.05 * moneyness^2  # Smile curvature
    return base_vol + skew + smile
end

println("Strike | True Vol | Market Price | Implied Vol")
println("-"^50)
for K in strikes
    true_vol = skewed_vol(K, spot)
    market_price = black_scholes(spot, K, 0.5, 0.05, true_vol, :call)
    iv = implied_vol(market_price, spot, K, 0.5, 0.05, :call)
    println("$(Int(K))    | $(round(true_vol*100, digits=1))%      | \$$(round(market_price, digits=2))       | $(round(iv*100, digits=1))%")
end
```

**Output:**
```
Strike | True Vol | Market Price | Implied Vol
--------------------------------------------------
130    | 22.2%      | $24.79       | 22.2%
135    | 21.6%      | $20.68       | 21.6%
140    | 21.1%      | $16.87       | 21.1%
145    | 20.5%      | $13.4        | 20.5%
150    | 20.0%      | $10.33       | 20.0%
155    | 19.5%      | $7.71        | 19.5%
160    | 19.1%      | $5.54        | 19.1%
165    | 18.6%      | $3.83        | 18.6%
170    | 18.2%      | $2.53        | 18.2%
```

## Edge Cases

Quasar handles edge cases gracefully:

```julia
# At expiry (T = 0)
expired = black_scholes(150.0, 155.0, 0.0, 0.05, 0.2, :call)
println("Expired OTM call: \$$(expired)")  # Should be 0

expired_itm = black_scholes(160.0, 155.0, 0.0, 0.05, 0.2, :call)
println("Expired ITM call: \$$(expired_itm)")  # Should be S - K = 5

# Zero volatility (deterministic forward)
deterministic = black_scholes(150.0, 140.0, 1.0, 0.05, 0.0, :call)
forward = 150.0 * exp(0.05)  # S * exp(rT)
intrinsic = max(forward - 140.0, 0) * exp(-0.05)  # Discounted payoff
println("Zero vol call: \$$(round(deterministic, digits=4)) (should be ≈\$$(round(intrinsic, digits=4)))")
```

**Output:**
```
Expired OTM call: $0.0
Expired ITM call: $5.0
Zero vol call: $16.8279 (should be ≈$16.8279)
```

## Next Steps

- [Portfolio Risk](portfolio-risk.md) - Combine options into portfolios
- [Monte Carlo Pricing](monte-carlo-exotic.md) - Price exotic options
- [Interest Rates](yield-curve.md) - Build yield curves for discounting
