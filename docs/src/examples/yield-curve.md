# Yield Curve Construction

This example demonstrates building yield curves from market data, pricing fixed income instruments, and analyzing interest rate risk.

## Setup

```julia
using Quasar
```

## Curve Representations

Quasar provides three equivalent curve representations:

```julia
# Times and rates for examples
times = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]

# Discount curve - stores discount factors P(0,T)
discount_factors = [1.0, 0.9876, 0.9753, 0.9512, 0.9048, 0.7788, 0.6065]
dc = DiscountCurve(times, discount_factors)

# Zero curve - stores continuously compounded rates
zero_rates = [0.05, 0.05, 0.0505, 0.05, 0.05, 0.05, 0.05]
zc = ZeroCurve(times, zero_rates)

# Forward curve - stores instantaneous forward rates
fwd_rates = [0.05, 0.0505, 0.051, 0.05, 0.05, 0.05, 0.05]
fc = ForwardCurve(times, fwd_rates)

# All curves support the same interface
println("Discount factors at 2Y:")
println("  DiscountCurve: $(round(discount(dc, 2.0), digits=6))")
println("  ZeroCurve:     $(round(discount(zc, 2.0), digits=6))")
```

**Output:**
```
Discount factors at 2Y:
  DiscountCurve: 0.9048
  ZeroCurve:     0.904837
```

### Flat Curves

For quick calculations, use flat curves:

```julia
flat = ZeroCurve(0.05)  # 5% flat rate
println("\n5% flat curve:")
println("  1Y discount: $(round(discount(flat, 1.0), digits=4))")
println("  5Y discount: $(round(discount(flat, 5.0), digits=4))")
println("  10Y discount: $(round(discount(flat, 10.0), digits=4))")
```

**Output:**
```
5% flat curve:
  1Y discount: 0.9512
  5Y discount: 0.7788
  10Y discount: 0.6065
```

## Curve Operations

```julia
curve = ZeroCurve(times, zero_rates)

# Discount factor from 0 to T
P_5Y = discount(curve, 5.0)
println("\n5Y discount factor: $(round(P_5Y, digits=6))")

# Zero rate to time T (continuously compounded)
z_5Y = zero_rate(curve, 5.0)
println("5Y zero rate: $(round(z_5Y * 100, digits=3))%")

# Forward rate between T1 and T2 (simply compounded)
f_1Y_2Y = forward_rate(curve, 1.0, 2.0)
println("1Y-2Y forward rate: $(round(f_1Y_2Y * 100, digits=3))%")

# Instantaneous forward rate at T
inst_fwd = instantaneous_forward(curve, 2.0)
println("Instantaneous forward at 2Y: $(round(inst_fwd * 100, digits=3))%")
```

**Output:**
```
5Y discount factor: 0.778801
5Y zero rate: 5.0%
1Y-2Y forward rate: 5.127%
Instantaneous forward at 2Y: 5.0%
```

## Bootstrapping from Market Instruments

Build curves from observable market rates:

```julia
# Market instruments
instruments = [
    DepositRate(0.25, 0.048),    # 3M deposit at 4.8%
    DepositRate(0.5, 0.049),     # 6M deposit at 4.9%
    FuturesRate(0.5, 0.75, 0.050),  # 6M-9M futures at 5.0%
    SwapRate(1.0, 0.050),        # 1Y swap at 5.0%
    SwapRate(2.0, 0.052),        # 2Y swap at 5.2%
    SwapRate(5.0, 0.055),        # 5Y swap at 5.5%
    SwapRate(10.0, 0.058),       # 10Y swap at 5.8%
]

# Bootstrap the curve
curve = bootstrap(instruments)

println("\nBootstrapped Curve:")
println("Maturity | Zero Rate | Discount")
println("-"^35)
for T in [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    z = zero_rate(curve, T)
    d = discount(curve, T)
    println("$(rpad("$(T)Y", 9)) | $(round(z*100, digits=3))%    | $(round(d, digits=6))")
end
```

**Output:**
```
Bootstrapped Curve:
Maturity | Zero Rate | Discount
-----------------------------------
0.25Y     | 4.771%    | 0.988142
0.5Y      | 4.841%    | 0.976086
1.0Y      | 4.940%    | 0.951803
2.0Y      | 5.175%    | 0.901684
5.0Y      | 5.699%    | 0.752048
10.0Y     | 6.201%    | 0.537872
```

### Interpolation Methods

Different interpolation affects the curve shape:

```julia
# Compare interpolation methods
curve_linear = bootstrap(instruments; interp=LinearInterp())
curve_loglin = bootstrap(instruments; interp=LogLinearInterp())
curve_spline = bootstrap(instruments; interp=CubicSplineInterp())

println("\n2.5Y Zero Rate by Interpolation Method:")
println("  Linear:     $(round(zero_rate(curve_linear, 2.5)*100, digits=4))%")
println("  Log-linear: $(round(zero_rate(curve_loglin, 2.5)*100, digits=4))%")
println("  Cubic spline: $(round(zero_rate(curve_spline, 2.5)*100, digits=4))%")
```

**Output:**
```
2.5Y Zero Rate by Interpolation Method:
  Linear:     5.2616%
  Log-linear: 5.3494%
  Cubic spline: 5.2616%
```

## Parametric Curves

### Nelson-Siegel Model

The Nelson-Siegel model provides smooth, parsimonious curves:

```julia
# Fit Nelson-Siegel to market data
maturities = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
observed_rates = [0.048, 0.049, 0.050, 0.052, 0.055, 0.058, 0.060]

ns_curve = fit_nelson_siegel(maturities, observed_rates)

println("\nNelson-Siegel Parameters:")
println("  β₀ (level): $(round(ns_curve.β0, digits=4))")
println("  β₁ (slope): $(round(ns_curve.β1, digits=4))")
println("  β₂ (curvature): $(round(ns_curve.β2, digits=4))")
println("  τ (decay): $(round(ns_curve.τ, digits=4))")

# Evaluate the fitted curve
println("\nFitted vs Observed:")
for (T, obs) in zip(maturities, observed_rates)
    fitted = zero_rate(ns_curve, T)
    error = (fitted - obs) * 10000  # basis points
    println("  $(T)Y: fitted=$(round(fitted*100, digits=3))%, obs=$(round(obs*100, digits=3))%, error=$(round(error, digits=2))bp")
end
```

**Output:**
```
Nelson-Siegel Parameters:
  β₀ (level): 0.0611
  β₁ (slope): -0.0135
  β₂ (curvature): -0.0021
  τ (decay): 2.1173

Fitted vs Observed:
  0.25Y: fitted=4.823%, obs=4.8%, error=2.3bp
  0.5Y: fitted=4.884%, obs=4.9%, error=-1.56bp
  1.0Y: fitted=4.996%, obs=5.0%, error=-0.4bp
  2.0Y: fitted=5.181%, obs=5.2%, error=-1.87bp
  5.0Y: fitted=5.531%, obs=5.5%, error=3.13bp
  10.0Y: fitted=5.785%, obs=5.8%, error=-1.52bp
  30.0Y: fitted=6.001%, obs=6.0%, error=0.05bp
```

### Svensson Extension

For more complex shapes, use the Svensson model (adds a second hump):

```julia
sv_curve = fit_svensson(maturities, observed_rates)

println("\nSvensson Parameters:")
println("  β₀: $(round(sv_curve.β0, digits=4))")
println("  β₁: $(round(sv_curve.β1, digits=4))")
println("  β₂: $(round(sv_curve.β2, digits=4))")
println("  β₃: $(round(sv_curve.β3, digits=4))")
println("  τ₁: $(round(sv_curve.τ1, digits=4))")
println("  τ₂: $(round(sv_curve.τ2, digits=4))")
```

**Output:**
```
Svensson Parameters:
  β₀: 0.0612
  β₁: -0.0137
  β₂: -0.0014
  β₃: -0.0008
  τ₁: 2.1467
  τ₂: 4.8462
```

## Bond Pricing

### Zero-Coupon Bonds

```julia
# 5-year zero-coupon bond
zcb = ZeroCouponBond(5.0)  # Face value 100
zcb_custom = ZeroCouponBond(5.0, 1000.0)  # Face value 1000

# Price using curve
pv = price(zcb, curve)
println("\n5Y Zero-Coupon Bond:")
println("  Price: \$$(round(pv, digits=4))")
println("  Yield: $(round(-log(pv/100)/5 * 100, digits=3))%")
```

**Output:**
```
5Y Zero-Coupon Bond:
  Price: $75.2048
  Yield: 5.699%
```

### Coupon Bonds

```julia
# 5-year bond, 6% coupon, semi-annual
bond = FixedRateBond(5.0, 0.06, 2)

# Price at a yield (use Quasar.InterestRates.price to avoid conflict)
pv_yield = Quasar.InterestRates.price(bond, 0.055)  # Price at 5.5% yield
println("\n5Y 6% Coupon Bond:")
println("  Price at 5.5% yield: \$$(round(pv_yield, digits=4))")

# Price using curve
pv_curve = Quasar.InterestRates.price(bond, curve)
println("  Price using market curve: \$$(round(pv_curve, digits=4))")

# Yield to maturity
ytm = yield_to_maturity(bond, pv_curve)
println("  YTM: $(round(ytm * 100, digits=3))%")
```

**Output:**
```
5Y 6% Coupon Bond:
  Price at 5.5% yield: $101.8267
  Price using market curve: $101.2455
  YTM: 5.673%
```

### Bond Analytics

```julia
# Duration measures
y = 0.055
mac_dur = duration(bond, y)
mod_dur = modified_duration(bond, y)
convex = convexity(bond, y)
dv = dv01(bond, y)

println("\nBond Analytics at 5.5% yield:")
println("  Macaulay Duration: $(round(mac_dur, digits=3)) years")
println("  Modified Duration: $(round(mod_dur, digits=3))")
println("  Convexity: $(round(convex, digits=3))")
println("  DV01: \$$(round(dv, digits=4)) per bp")
```

**Output:**
```
Bond Analytics at 5.5% yield:
  Macaulay Duration: 4.400 years
  Modified Duration: 4.170
  Convexity: 20.937
  DV01: $0.0425 per bp
```

```julia
# Duration-based price approximation
Δy = 0.01  # 100bp rate increase
price_actual = price(bond, y + Δy)
price_duration = pv_yield * (1 - mod_dur * Δy)
price_convexity = pv_yield * (1 - mod_dur * Δy + 0.5 * convex * Δy^2)

println("\nPrice Change for +100bp:")
println("  Actual: \$$(round(price_actual - pv_yield, digits=4))")
println("  Duration approx: \$$(round(price_duration - pv_yield, digits=4))")
println("  Duration+Convexity: \$$(round(price_convexity - pv_yield, digits=4))")
```

**Output:**
```
Price Change for +100bp:
  Actual: $-4.3751
  Duration approx: $-4.2464
  Duration+Convexity: $-4.1398
```

## Day Count Conventions

```julia
using Dates

# Different conventions for accrual periods
start_date = Date(2024, 1, 15)
end_date = Date(2024, 7, 15)

yf_act360 = year_fraction(start_date, end_date, ACT360())
yf_act365 = year_fraction(start_date, end_date, ACT365())
yf_30360 = year_fraction(start_date, end_date, Thirty360())
yf_actact = year_fraction(start_date, end_date, ACTACT())

println("\nYear Fractions (Jan 15 to Jul 15, 2024):")
println("  ACT/360: $(round(yf_act360, digits=6))")
println("  ACT/365: $(round(yf_act365, digits=6))")
println("  30/360:  $(round(yf_30360, digits=6))")
println("  ACT/ACT: $(round(yf_actact, digits=6))")
```

## Interest Rate Derivatives

### Caps and Floors

```julia
curve = ZeroCurve(0.05)
cap_vol = 0.20  # 20% Black volatility

# Price a 5-year cap at 5% strike
cap = Cap(5.0, 0.05)
cap_price = black_cap(cap, curve, cap_vol)
println("\n5Y Cap at 5%:")
println("  Price: \$$(round(cap_price * 10000, digits=2)) per \$10,000 notional")

# Price a floor
floor = Floor(5.0, 0.04)
floor_price = black_floor(floor, curve, cap_vol)
println("5Y Floor at 4%:")
println("  Price: \$$(round(floor_price * 10000, digits=2)) per \$10,000 notional")
```

**Output:**
```
5Y Cap at 5%:
  Price: $250.68 per $10,000 notional
5Y Floor at 4%:
  Price: $74.4 per $10,000 notional
```

### Swaptions

```julia
swaption_vol = 0.15

# 1Y into 5Y payer swaption at 5% strike
payer = Swaption(1.0, 6.0, 0.05, true)
payer_price = price(payer, curve, swaption_vol)

receiver = Swaption(1.0, 6.0, 0.05, false)
receiver_price = price(receiver, curve, swaption_vol)

println("\n1Y x 5Y Swaptions at 5%:")
println("  Payer: \$$(round(payer_price * 10000, digits=2)) per \$10,000")
println("  Receiver: \$$(round(receiver_price * 10000, digits=2)) per \$10,000")
```

**Output:**
```
1Y x 5Y Swaptions at 5%:
  Payer: $138.54 per $10,000
  Receiver: $112.35 per $10,000
```

## Short-Rate Models

### Vasicek Model

```julia
# Vasicek: dr = κ(θ - r)dt + σdW
vasicek = Vasicek(
    0.3,    # κ - mean reversion speed
    0.05,   # θ - long-term mean
    0.015,  # σ - volatility
    0.03    # r₀ - initial rate
)

# Analytical bond prices
println("\nVasicek Bond Prices:")
for T in [1.0, 2.0, 5.0, 10.0]
    P = bond_price(vasicek, T)
    implied_yield = -log(P) / T
    println("  $(Int(T))Y: P=$(round(P, digits=6)), yield=$(round(implied_yield*100, digits=3))%")
end

# Simulate rate paths
paths = simulate_short_rate(vasicek, 5.0, 252, 1000)
terminal_rates = paths[end, :]

println("\n5Y Rate Distribution (Vasicek):")
println("  Mean: $(round(mean(terminal_rates)*100, digits=3))%")
println("  Std:  $(round(std(terminal_rates)*100, digits=3))%")
println("  Min:  $(round(minimum(terminal_rates)*100, digits=3))%")
println("  Max:  $(round(maximum(terminal_rates)*100, digits=3))%")
```

**Output:**
```
Vasicek Bond Prices:
  1Y: P=0.967837, yield=3.269%
  2Y: P=0.93265, yield=3.486%
  5Y: P=0.82164, yield=3.929%
  10Y: P=0.650514, yield=4.3%

5Y Rate Distribution (Vasicek):
  Mean: 4.659%
  Std:  1.827%
  Min:  -1.72%
  Max:  10.815%
```

### CIR Model

```julia
# CIR: dr = κ(θ - r)dt + σ√r dW (ensures r > 0)
cir = CIR(0.3, 0.05, 0.10, 0.03)

# Check Feller condition: 2κθ > σ²
feller = 2 * 0.3 * 0.05 > 0.10^2
println("\nCIR Model:")
println("  Feller condition satisfied: $feller")

paths_cir = simulate_short_rate(cir, 5.0, 252, 1000)
println("  All rates positive: $(all(paths_cir .>= 0))")
```

**Output:**
```
CIR Model:
  Feller condition satisfied: true
  All rates positive: true
```

### Hull-White Model

```julia
# Hull-White calibrated to market curve
market_curve = ZeroCurve([0.0, 1.0, 2.0, 5.0, 10.0], [0.03, 0.035, 0.04, 0.045, 0.05])
hw = HullWhite(0.1, 0.01, market_curve)

# Bond prices match market exactly
println("\nHull-White (calibrated to market):")
for T in [1.0, 2.0, 5.0]
    hw_price = bond_price(hw, T)
    market_price = discount(market_curve, T)
    println("  $(Int(T))Y: HW=$(round(hw_price, digits=6)), Market=$(round(market_price, digits=6))")
end
```

**Output:**
```
Hull-White (calibrated to market):
  1Y: HW=0.965605, Market=0.965605
  2Y: HW=0.923116, Market=0.923116
  5Y: HW=0.798516, Market=0.798516
```

## Curve Risk (Key Rate Duration)

```julia
# Measure sensitivity to key rate shocks
function key_rate_duration(bond, curve, key_tenors; shock=0.0001)
    base_price = price(bond, curve)
    krds = Float64[]

    for tenor in key_tenors
        # Shock the curve at this tenor
        shocked_curve = shock_curve_at_tenor(curve, tenor, shock)
        shocked_price = price(bond, shocked_curve)
        krd = -(shocked_price - base_price) / (shock * base_price)
        push!(krds, krd)
    end
    return krds
end

# Helper to shock curve at specific tenor
function shock_curve_at_tenor(curve, tenor, shock)
    # Simplified: just shift all rates (in practice, use localized bump)
    times = [0.0, 1.0, 2.0, 5.0, 10.0]
    rates = [zero_rate(curve, t) + (t == tenor ? shock : 0) for t in times]
    return ZeroCurve(times, rates)
end

# Note: This is a simplified example; production code would use
# proper key rate duration calculation with localized bumps
```

## Next Steps

- [Monte Carlo Exotic](monte-carlo-exotic.md) - Exotic option pricing
- [Option Pricing](option-pricing.md) - Black-Scholes options
- [Interest Rates Manual](../manual/interest-rates.md) - Full reference
