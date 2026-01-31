module InterestRates

using LinearAlgebra
using Distributions: Normal, cdf, pdf

export
    # Day count conventions
    DayCountConvention, ACT360, ACT365, Thirty360, ACTACT,
    year_fraction,
    # Curve types
    RateCurve, DiscountCurve, ZeroCurve, ForwardCurve,
    # Parametric curves
    NelsonSiegelCurve, SvenssonCurve, fit_nelson_siegel, fit_svensson,
    # Curve operations
    discount, zero_rate, forward_rate, instantaneous_forward,
    # Interpolation
    LinearInterp, LogLinearInterp, CubicSplineInterp,
    # Bootstrapping
    DepositRate, FuturesRate, SwapRate, bootstrap,
    # Bonds
    Bond, ZeroCouponBond, FixedRateBond, FloatingRateBond,
    # Note: price() not exported to avoid collision with Instruments.price
    # Use Quasar.InterestRates.price() for bond pricing
    yield_to_maturity, duration, modified_duration, convexity, dv01,
    accrued_interest, clean_price, dirty_price,
    # Short rate models
    ShortRateModel, Vasicek, CIR, HullWhite,
    bond_price, short_rate, simulate_short_rate,
    # IR Derivatives
    Caplet, Floorlet, Cap, Floor, Swaption,
    black_caplet, black_cap

# ============================================================================
# Day Count Conventions
# ============================================================================

"""
    DayCountConvention

Abstract type for day count conventions used to calculate year fractions
between two dates.
"""
abstract type DayCountConvention end

"""
    ACT360 <: DayCountConvention

Actual/360 day count: actual days divided by 360.
Common for money market instruments (LIBOR, EURIBOR).
"""
struct ACT360 <: DayCountConvention end

"""
    ACT365 <: DayCountConvention

Actual/365 (Fixed) day count: actual days divided by 365.
Common for GBP and JPY markets.
"""
struct ACT365 <: DayCountConvention end

"""
    Thirty360 <: DayCountConvention

30/360 (Bond Basis) day count: assumes 30-day months and 360-day years.
Common for US corporate bonds.
"""
struct Thirty360 <: DayCountConvention end

"""
    ACTACT <: DayCountConvention

Actual/Actual (ISDA) day count: actual days in each period.
Standard for government bonds in many markets.
"""
struct ACTACT <: DayCountConvention end

"""
    year_fraction(start_date, end_date, convention::DayCountConvention) -> Float64

Calculate the year fraction between two dates using the specified day count convention.

# Arguments
- `start_date` - Start date (as Date or day count from reference)
- `end_date` - End date (as Date or day count from reference)
- `convention` - Day count convention to use

# Example
```julia
using Dates
year_fraction(Date(2024,1,1), Date(2024,7,1), ACT360())  # ≈ 0.5028
year_fraction(Date(2024,1,1), Date(2024,7,1), ACT365())  # ≈ 0.4959
year_fraction(Date(2024,1,1), Date(2024,7,1), Thirty360())  # = 0.5
```
"""
function year_fraction end

# For simple numeric time (already in years), just return the difference
# These take priority over the date-based implementations when inputs are numbers
year_fraction(t1::Real, t2::Real, ::ACT360) = Float64(t2 - t1)
year_fraction(t1::Real, t2::Real, ::ACT365) = Float64(t2 - t1)
year_fraction(t1::Real, t2::Real, ::Thirty360) = Float64(t2 - t1)
year_fraction(t1::Real, t2::Real, ::ACTACT) = Float64(t2 - t1)

# Date-based implementations (requires Dates to be available)
# These are for actual Date objects, not numeric types
function _year_fraction_act360(d1, d2)
    days = _day_count(d1, d2)
    days / 360.0
end

function _year_fraction_act365(d1, d2)
    days = _day_count(d1, d2)
    days / 365.0
end

function _year_fraction_30360(d1, d2)
    y1, m1, day1 = _year_month_day(d1)
    y2, m2, day2 = _year_month_day(d2)

    # Adjust day counts per 30/360 convention
    day1 = min(day1, 30)
    if day1 == 30
        day2 = min(day2, 30)
    end

    (360 * (y2 - y1) + 30 * (m2 - m1) + (day2 - day1)) / 360.0
end

function _year_fraction_actact(d1, d2)
    # Simplified ISDA implementation: actual days / average year length
    days = _day_count(d1, d2)
    y1, _, _ = _year_month_day(d1)
    y2, _, _ = _year_month_day(d2)

    # Handle same year case
    if y1 == y2
        return days / (_is_leap_year(y1) ? 366.0 : 365.0)
    end

    # For multi-year periods, weight by actual days in each year
    total = 0.0
    for y in y1:y2
        year_days = _is_leap_year(y) ? 366.0 : 365.0
        if y == y1
            # Days remaining in first year
            year_end = _make_date(y + 1, 1, 1)
            total += _day_count(d1, year_end) / year_days
        elseif y == y2
            # Days in final year
            year_start = _make_date(y, 1, 1)
            total += _day_count(year_start, d2) / year_days
        else
            # Full year
            total += 1.0
        end
    end
    total
end

# Helper functions for date handling
# These work with both Date objects and simple (year, month, day) tuples

function _day_count(d1, d2)
    # Try to use Dates if available, otherwise assume numeric
    if isdefined(Main, :Dates)
        return Int(Main.Dates.value(d2) - Main.Dates.value(d1))
    else
        # Assume d1, d2 are already day numbers
        return Int(d2 - d1)
    end
end

function _year_month_day(d)
    if isdefined(Main, :Dates)
        return Main.Dates.year(d), Main.Dates.month(d), Main.Dates.day(d)
    else
        # Assume d is a tuple (y, m, d)
        return d[1], d[2], d[3]
    end
end

function _make_date(y, m, d)
    if isdefined(Main, :Dates)
        return Main.Dates.Date(y, m, d)
    else
        return (y, m, d)
    end
end

function _is_leap_year(y)
    (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0)
end

# ============================================================================
# Interpolation Methods
# ============================================================================

"""
    InterpolationMethod

Abstract base type for interpolation methods used in yield curve construction.

Subtypes: [`LinearInterp`](@ref), [`LogLinearInterp`](@ref), [`CubicSplineInterp`](@ref)
"""
abstract type InterpolationMethod end

"""
    LinearInterp <: InterpolationMethod

Linear interpolation between data points.

Simple and stable, but can produce kinks in forward rates.
Best for zero rate curves where smoothness is less critical.

# Example
```julia
curve = ZeroCurve(times, rates; interp=LinearInterp())
```
"""
struct LinearInterp <: InterpolationMethod end

"""
    LogLinearInterp <: InterpolationMethod

Log-linear interpolation (linear in log-space).

The default for discount curves. Ensures discount factors remain positive
and produces smoother forward rates than linear interpolation.

# Example
```julia
curve = DiscountCurve(times, dfs; interp=LogLinearInterp())
```
"""
struct LogLinearInterp <: InterpolationMethod end

"""
    CubicSplineInterp <: InterpolationMethod

Natural cubic spline interpolation.

Produces the smoothest curves with continuous first and second derivatives.
Best for applications requiring smooth forward rates (e.g., HJM models).

# Fields
- `coeffs::Vector{NTuple{4,Float64}}` - Spline coefficients (a, b, c, d) per segment

# Example
```julia
curve = ZeroCurve(times, rates; interp=CubicSplineInterp())
```
"""
struct CubicSplineInterp <: InterpolationMethod
    coeffs::Vector{NTuple{4,Float64}}  # (a, b, c, d) for each segment
end

function CubicSplineInterp()
    CubicSplineInterp(NTuple{4,Float64}[])
end

# Cubic spline coefficient computation
function compute_spline_coeffs(x::Vector{Float64}, y::Vector{Float64})
    n = length(x) - 1
    h = diff(x)

    # Set up tridiagonal system for second derivatives
    A = zeros(n + 1, n + 1)
    b = zeros(n + 1)

    A[1, 1] = 1.0
    A[end, end] = 1.0

    for i in 2:n
        A[i, i-1] = h[i-1]
        A[i, i] = 2(h[i-1] + h[i])
        A[i, i+1] = h[i]
        b[i] = 3((y[i+1] - y[i])/h[i] - (y[i] - y[i-1])/h[i-1])
    end

    c = A \ b

    coeffs = NTuple{4,Float64}[]
    for i in 1:n
        a = y[i]
        b_i = (y[i+1] - y[i])/h[i] - h[i]*(2c[i] + c[i+1])/3
        d = (c[i+1] - c[i])/(3h[i])
        push!(coeffs, (a, b_i, c[i], d))
    end

    coeffs
end

function interp_value(::LinearInterp, x::Vector{Float64}, y::Vector{Float64}, t::Float64)
    t <= x[1] && return y[1]
    t >= x[end] && return y[end]

    i = searchsortedlast(x, t)
    i = clamp(i, 1, length(x) - 1)

    w = (t - x[i]) / (x[i+1] - x[i])
    y[i] * (1 - w) + y[i+1] * w
end

function interp_value(::LogLinearInterp, x::Vector{Float64}, y::Vector{Float64}, t::Float64)
    t <= x[1] && return y[1]
    t >= x[end] && return y[end]

    i = searchsortedlast(x, t)
    i = clamp(i, 1, length(x) - 1)

    w = (t - x[i]) / (x[i+1] - x[i])
    exp(log(y[i]) * (1 - w) + log(y[i+1]) * w)
end

function interp_value(interp::CubicSplineInterp, x::Vector{Float64}, y::Vector{Float64}, t::Float64)
    t <= x[1] && return y[1]
    t >= x[end] && return y[end]

    i = searchsortedlast(x, t)
    i = clamp(i, 1, length(x) - 1)

    if isempty(interp.coeffs)
        # Fall back to linear if coeffs not computed
        return interp_value(LinearInterp(), x, y, t)
    end

    a, b, c, d = interp.coeffs[i]
    dx = t - x[i]
    a + b*dx + c*dx^2 + d*dx^3
end

# ============================================================================
# Yield Curves
# ============================================================================

"""
    RateCurve

Abstract base type for interest rate curves.

All rate curves support the following operations:
- `discount(curve, T)` - Get discount factor to time T
- `zero_rate(curve, T)` - Get zero rate to time T
- `forward_rate(curve, T1, T2)` - Get forward rate between T1 and T2
- `instantaneous_forward(curve, T)` - Get instantaneous forward rate at T

# Subtypes
- [`DiscountCurve`](@ref) - Curve of discount factors
- [`ZeroCurve`](@ref) - Curve of zero rates
- [`ForwardCurve`](@ref) - Curve of instantaneous forward rates
- [`NelsonSiegelCurve`](@ref) - Parametric Nelson-Siegel curve
- [`SvenssonCurve`](@ref) - Parametric Svensson curve

# Example
```julia
# Create a flat 5% curve
curve = ZeroCurve(0.05)

# Get discount factor and zero rate at 2 years
df = discount(curve, 2.0)      # ≈ 0.9048
r = zero_rate(curve, 2.0)      # = 0.05
```
"""
abstract type RateCurve end

"""
    DiscountCurve(times, discount_factors; interp=LogLinearInterp())

Curve of discount factors P(0,T). Interpolates in log-space by default
to ensure discount factors remain positive.

# Arguments
- `times::Vector{Float64}` - Maturities in years
- `discount_factors::Vector{Float64}` - Discount factors (must be positive)
- `interp::InterpolationMethod` - Interpolation method (default: LogLinearInterp)

# Constructors
- `DiscountCurve(times, dfs)` - From vectors
- `DiscountCurve(rate)` - Flat curve at given rate

# Example
```julia
curve = DiscountCurve([0.0, 1.0, 5.0], [1.0, 0.95, 0.78])
df = discount(curve, 2.5)  # Interpolated discount factor
```
"""
struct DiscountCurve <: RateCurve
    times::Vector{Float64}
    values::Vector{Float64}
    interp::InterpolationMethod

    # TODO: Check that discount factors are decreasing with time
    function DiscountCurve(times, values; interp::InterpolationMethod=LogLinearInterp())
        @assert length(times) == length(values)
        @assert all(values .> 0) "Discount factors must be positive"
        idx = sortperm(times)
        new(times[idx], values[idx], interp)
    end
end

"""
    ZeroCurve(times, zero_rates; interp=LinearInterp())

Curve of continuously compounded zero rates.
"""
struct ZeroCurve <: RateCurve
    times::Vector{Float64}
    values::Vector{Float64}
    interp::InterpolationMethod

    function ZeroCurve(times, values; interp::InterpolationMethod=LinearInterp())
        @assert length(times) == length(values)
        idx = sortperm(times)
        new(times[idx], values[idx], interp)
    end
end

"""
    ForwardCurve(times, forward_rates; interp=LinearInterp())

Curve of instantaneous forward rates.
"""
struct ForwardCurve <: RateCurve
    times::Vector{Float64}
    values::Vector{Float64}
    interp::InterpolationMethod

    function ForwardCurve(times, values; interp::InterpolationMethod=LinearInterp())
        @assert length(times) == length(values)
        idx = sortperm(times)
        new(times[idx], values[idx], interp)
    end
end

# Flat curve constructor
DiscountCurve(rate::Float64, max_t::Float64=30.0) =
    DiscountCurve([0.0, max_t], [1.0, exp(-rate * max_t)])
ZeroCurve(rate::Float64, max_t::Float64=30.0) =
    ZeroCurve([0.0, max_t], [rate, rate])

# Core curve operations
"""
    discount(curve, T) -> Float64

Discount factor from time 0 to time T.
"""
function discount(curve::DiscountCurve, T::Float64)
    T <= 0 && return 1.0
    interp_value(curve.interp, curve.times, curve.values, T)
end

function discount(curve::ZeroCurve, T::Float64)
    T <= 0 && return 1.0
    r = interp_value(curve.interp, curve.times, curve.values, T)
    exp(-r * T)
end

function discount(curve::ForwardCurve, T::Float64)
    T <= 0 && return 1.0
    # Integrate forward rates: DF = exp(-∫f(s)ds)
    n = 100
    dt = T / n
    integral = sum(interp_value(curve.interp, curve.times, curve.values, i*dt) * dt for i in 0:n-1)
    exp(-integral)
end

"""
    zero_rate(curve, T) -> Float64

Continuously compounded zero rate to time T.
"""
function zero_rate(curve::DiscountCurve, T::Float64)
    T <= 0 && return curve.values[1] > 0 ? -log(curve.values[1]) : 0.0
    df = discount(curve, T)
    -log(df) / T
end

function zero_rate(curve::ZeroCurve, T::Float64)
    T <= 0 && return curve.values[1]
    interp_value(curve.interp, curve.times, curve.values, T)
end

function zero_rate(curve::ForwardCurve, T::Float64)
    T <= 0 && return curve.values[1]
    df = discount(curve, T)
    -log(df) / T
end

"""
    forward_rate(curve, T1, T2) -> Float64

Simply compounded forward rate between T1 and T2.
"""
function forward_rate(curve::RateCurve, T1::Float64, T2::Float64)
    @assert T2 > T1 "T2 must be greater than T1"
    df1 = discount(curve, T1)
    df2 = discount(curve, T2)
    (df1 / df2 - 1) / (T2 - T1)
end

"""
    instantaneous_forward(curve, T) -> Float64

Instantaneous forward rate at time T: f(T) = -d/dT ln(P(0,T))
"""
function instantaneous_forward(curve::RateCurve, T::Float64)
    ε = 1e-6
    df_plus = discount(curve, T + ε)
    df_minus = discount(curve, T - ε)
    -log(df_plus / df_minus) / (2ε)
end

function instantaneous_forward(curve::ForwardCurve, T::Float64)
    interp_value(curve.interp, curve.times, curve.values, T)
end

# Curve conversions
# Sample many points to preserve interpolation accuracy across methods
function DiscountCurve(zc::ZeroCurve)
    t_max = zc.times[end]
    n_points = max(100, length(zc.times) * 20)
    times = vcat([0.0], collect(range(1e-6, t_max, length=n_points)))
    dfs = [discount(zc, t) for t in times]
    DiscountCurve(times, dfs; interp=LogLinearInterp())
end

function ZeroCurve(dc::DiscountCurve)
    t_max = dc.times[end]
    n_points = max(100, length(dc.times) * 20)
    times = vcat([0.0], collect(range(1e-6, t_max, length=n_points)))
    rates = [zero_rate(dc, t) for t in times]
    ZeroCurve(times, rates; interp=LinearInterp())
end

# ============================================================================
# Parametric Yield Curves (Nelson-Siegel, Svensson)
# ============================================================================

"""
    NelsonSiegelCurve(β0, β1, β2, τ)

Nelson-Siegel parametric yield curve model.

The zero rate at maturity T is given by:
```
r(T) = β₀ + β₁ * (1 - exp(-T/τ)) / (T/τ) + β₂ * ((1 - exp(-T/τ)) / (T/τ) - exp(-T/τ))
```

# Parameters
- `β0` - Long-term rate level (asymptotic rate as T → ∞)
- `β1` - Short-term component (controls slope at origin)
- `β2` - Medium-term component (controls curvature/hump)
- `τ` - Decay factor (time at which medium-term component reaches maximum)

# Example
```julia
curve = NelsonSiegelCurve(0.05, -0.02, 0.01, 2.0)
zero_rate(curve, 5.0)  # Get 5-year zero rate
```
"""
struct NelsonSiegelCurve <: RateCurve
    β0::Float64
    β1::Float64
    β2::Float64
    τ::Float64

    function NelsonSiegelCurve(β0, β1, β2, τ)
        τ > 0 || throw(ArgumentError("τ must be positive, got $τ"))
        new(β0, β1, β2, τ)
    end
end

"""
    SvenssonCurve(β0, β1, β2, β3, τ1, τ2)

Svensson extension of Nelson-Siegel with an additional hump component.

The zero rate at maturity T is given by:
```
r(T) = β₀ + β₁ * g1(T/τ1) + β₂ * h1(T/τ1) + β₃ * h2(T/τ2)
```
where:
- g1(x) = (1 - exp(-x)) / x
- h1(x) = g1(x) - exp(-x)
- h2(x) = g1(x, τ2) - exp(-T/τ2)

# Parameters
- `β0` - Long-term rate level
- `β1` - Short-term component
- `β2` - First medium-term component (hump at τ1)
- `β3` - Second medium-term component (hump at τ2)
- `τ1` - First decay factor
- `τ2` - Second decay factor

# Example
```julia
curve = SvenssonCurve(0.05, -0.02, 0.01, 0.005, 2.0, 5.0)
zero_rate(curve, 10.0)  # Get 10-year zero rate
```
"""
struct SvenssonCurve <: RateCurve
    β0::Float64
    β1::Float64
    β2::Float64
    β3::Float64
    τ1::Float64
    τ2::Float64

    function SvenssonCurve(β0, β1, β2, β3, τ1, τ2)
        τ1 > 0 || throw(ArgumentError("τ1 must be positive, got $τ1"))
        τ2 > 0 || throw(ArgumentError("τ2 must be positive, got $τ2"))
        new(β0, β1, β2, β3, τ1, τ2)
    end
end

# Nelson-Siegel loading functions
function _ns_g1(x::Float64)
    # g1(x) = (1 - exp(-x)) / x, with limit g1(0) = 1
    abs(x) < 1e-10 ? 1.0 - x/2 + x^2/6 : (1 - exp(-x)) / x
end

function _ns_h1(x::Float64)
    # h1(x) = g1(x) - exp(-x)
    _ns_g1(x) - exp(-x)
end

# Zero rate implementations
function zero_rate(curve::NelsonSiegelCurve, T::Float64)
    T <= 0 && return curve.β0 + curve.β1  # Instantaneous rate
    x = T / curve.τ
    curve.β0 + curve.β1 * _ns_g1(x) + curve.β2 * _ns_h1(x)
end

function zero_rate(curve::SvenssonCurve, T::Float64)
    T <= 0 && return curve.β0 + curve.β1  # Instantaneous rate
    x1 = T / curve.τ1
    x2 = T / curve.τ2
    curve.β0 + curve.β1 * _ns_g1(x1) + curve.β2 * _ns_h1(x1) + curve.β3 * _ns_h1(x2)
end

# Discount factors
function discount(curve::NelsonSiegelCurve, T::Float64)
    T <= 0 && return 1.0
    exp(-zero_rate(curve, T) * T)
end

function discount(curve::SvenssonCurve, T::Float64)
    T <= 0 && return 1.0
    exp(-zero_rate(curve, T) * T)
end

# Instantaneous forward rate: f(T) = -d/dT ln(P(0,T)) = r(T) + T * dr/dT
function instantaneous_forward(curve::NelsonSiegelCurve, T::Float64)
    T <= 0 && return curve.β0 + curve.β1
    x = T / curve.τ
    ex = exp(-x)

    # r(T) = β0 + β1 * g1 + β2 * h1
    # f(T) = r(T) + T * dr/dT
    # dr/dT = (β1/τ) * dg1/dx + (β2/τ) * dh1/dx
    # where dg1/dx = (x*exp(-x) - (1-exp(-x)))/x² = (exp(-x) - g1)/x
    # and dh1/dx = dg1/dx + exp(-x) = (exp(-x) - g1)/x + exp(-x)

    g1 = _ns_g1(x)
    dg1_dx = abs(x) < 1e-10 ? -0.5 + x/3 : (ex - g1) / x
    dh1_dx = dg1_dx + ex

    r = curve.β0 + curve.β1 * g1 + curve.β2 * _ns_h1(x)
    dr_dT = (curve.β1 * dg1_dx + curve.β2 * dh1_dx) / curve.τ

    r + T * dr_dT
end

function instantaneous_forward(curve::SvenssonCurve, T::Float64)
    # Use numerical differentiation for Svensson
    ε = 1e-6
    df_plus = discount(curve, T + ε)
    df_minus = discount(curve, max(0.0, T - ε))
    -log(df_plus / df_minus) / (2ε)
end

"""
    fit_nelson_siegel(maturities, rates; initial_guess=nothing) -> NelsonSiegelCurve

Fit a Nelson-Siegel curve to observed zero rates using least squares.

# Arguments
- `maturities` - Vector of maturities (in years)
- `rates` - Vector of observed zero rates
- `initial_guess` - Optional (β0, β1, β2, τ) starting point

# Returns
A fitted NelsonSiegelCurve

# Example
```julia
mats = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
rates = [0.02, 0.022, 0.025, 0.028, 0.032, 0.035, 0.037]
curve = fit_nelson_siegel(mats, rates)
```
"""
function fit_nelson_siegel(maturities::Vector{Float64}, rates::Vector{Float64};
                           initial_guess::Union{Nothing, NTuple{4,Float64}}=nothing)
    @assert length(maturities) == length(rates)
    @assert length(maturities) >= 4 "Need at least 4 data points to fit Nelson-Siegel"

    # Default initial guess
    if initial_guess === nothing
        β0_init = rates[end]  # Long rate
        β1_init = rates[1] - rates[end]  # Slope
        β2_init = 0.0  # Curvature
        τ_init = 2.0  # Typical decay
    else
        β0_init, β1_init, β2_init, τ_init = initial_guess
    end

    # Simple gradient descent optimization
    params = [β0_init, β1_init, β2_init, τ_init]

    function loss(p)
        β0, β1, β2, τ = p
        τ = max(τ, 0.01)  # Keep τ positive
        curve = NelsonSiegelCurve(β0, β1, β2, τ)
        sum((zero_rate(curve, t) - r)^2 for (t, r) in zip(maturities, rates))
    end

    # Gradient-free Nelder-Mead simplex optimization
    params = _nelder_mead(loss, params, 1000, 1e-10)

    NelsonSiegelCurve(params[1], params[2], params[3], max(params[4], 0.01))
end

"""
    fit_svensson(maturities, rates; initial_guess=nothing) -> SvenssonCurve

Fit a Svensson curve to observed zero rates using least squares.

# Arguments
- `maturities` - Vector of maturities (in years)
- `rates` - Vector of observed zero rates
- `initial_guess` - Optional (β0, β1, β2, β3, τ1, τ2) starting point

# Returns
A fitted SvenssonCurve
"""
function fit_svensson(maturities::Vector{Float64}, rates::Vector{Float64};
                      initial_guess::Union{Nothing, NTuple{6,Float64}}=nothing)
    @assert length(maturities) == length(rates)
    @assert length(maturities) >= 6 "Need at least 6 data points to fit Svensson"

    # Default initial guess
    if initial_guess === nothing
        β0_init = rates[end]
        β1_init = rates[1] - rates[end]
        β2_init = 0.0
        β3_init = 0.0
        τ1_init = 2.0
        τ2_init = 5.0
    else
        β0_init, β1_init, β2_init, β3_init, τ1_init, τ2_init = initial_guess
    end

    params = [β0_init, β1_init, β2_init, β3_init, τ1_init, τ2_init]

    function loss(p)
        β0, β1, β2, β3, τ1, τ2 = p
        τ1 = max(τ1, 0.01)
        τ2 = max(τ2, 0.01)
        curve = SvenssonCurve(β0, β1, β2, β3, τ1, τ2)
        sum((zero_rate(curve, t) - r)^2 for (t, r) in zip(maturities, rates))
    end

    params = _nelder_mead(loss, params, 2000, 1e-10)

    SvenssonCurve(params[1], params[2], params[3], params[4],
                  max(params[5], 0.01), max(params[6], 0.01))
end

# Simple Nelder-Mead optimizer (no external dependencies)
function _nelder_mead(f, x0::Vector{Float64}, max_iter::Int, tol::Float64)
    n = length(x0)

    # Initialize simplex
    simplex = Vector{Vector{Float64}}(undef, n + 1)
    simplex[1] = copy(x0)
    for i in 2:n+1
        simplex[i] = copy(x0)
        simplex[i][i-1] += 0.1 * max(abs(x0[i-1]), 1.0)
    end

    # Evaluate function at all vertices
    fvals = [f(v) for v in simplex]

    α, γ, ρ, σ = 1.0, 2.0, 0.5, 0.5  # Standard parameters

    for _ in 1:max_iter
        # Sort vertices by function value
        order = sortperm(fvals)
        simplex = simplex[order]
        fvals = fvals[order]

        # Check convergence
        if maximum(abs(fvals[i] - fvals[1]) for i in 2:n+1) < tol
            break
        end

        # Centroid of best n points
        x0_new = sum(simplex[1:n]) / n

        # Reflection
        xr = x0_new + α * (x0_new - simplex[end])
        fr = f(xr)

        if fvals[1] <= fr < fvals[n]
            simplex[end] = xr
            fvals[end] = fr
        elseif fr < fvals[1]
            # Expansion
            xe = x0_new + γ * (xr - x0_new)
            fe = f(xe)
            if fe < fr
                simplex[end] = xe
                fvals[end] = fe
            else
                simplex[end] = xr
                fvals[end] = fr
            end
        else
            # Contraction
            xc = x0_new + ρ * (simplex[end] - x0_new)
            fc = f(xc)
            if fc < fvals[end]
                simplex[end] = xc
                fvals[end] = fc
            else
                # Shrink
                for i in 2:n+1
                    simplex[i] = simplex[1] + σ * (simplex[i] - simplex[1])
                    fvals[i] = f(simplex[i])
                end
            end
        end
    end

    simplex[argmin(fvals)]
end

# ============================================================================
# Curve Bootstrapping
# ============================================================================

abstract type MarketInstrument end

"""Deposit rate: simple rate for short maturities"""
struct DepositRate <: MarketInstrument
    maturity::Float64
    rate::Float64
end

"""Futures rate: convexity-adjusted forward rate"""
struct FuturesRate <: MarketInstrument
    start::Float64
    maturity::Float64
    rate::Float64
    convexity_adj::Float64
end
FuturesRate(start, mat, rate) = FuturesRate(start, mat, rate, 0.0)

"""Swap rate: par swap rate"""
struct SwapRate <: MarketInstrument
    maturity::Float64
    rate::Float64
    frequency::Int  # payments per year
end
SwapRate(mat, rate) = SwapRate(mat, rate, 2)  # semi-annual default

"""
    bootstrap(instruments; interp=LogLinearInterp()) -> DiscountCurve

Bootstrap a discount curve from market instruments using sequential stripping.

The function iteratively solves for discount factors that reprice each
instrument, starting from the shortest maturity. Instruments should be
provided in order of increasing maturity.

# Arguments
- `instruments::Vector{<:MarketInstrument}` - Market quotes (sorted by maturity)
- `interp::InterpolationMethod` - Interpolation for intermediate points

# Supported Instruments
- [`DepositRate`](@ref) - Money market deposits (short end)
- [`FuturesRate`](@ref) - Interest rate futures (middle)
- [`SwapRate`](@ref) - Par swap rates (long end)

# Returns
A [`DiscountCurve`](@ref) that reprices all input instruments.

# Example
```julia
instruments = [
    DepositRate(0.25, 0.02),   # 3-month deposit at 2%
    DepositRate(0.5, 0.022),   # 6-month deposit at 2.2%
    SwapRate(2.0, 0.028),      # 2-year swap at 2.8%
    SwapRate(5.0, 0.032),      # 5-year swap at 3.2%
]
curve = bootstrap(instruments)
```

See also: [`DepositRate`](@ref), [`FuturesRate`](@ref), [`SwapRate`](@ref)
"""
function bootstrap(instruments::Vector{<:MarketInstrument}; interp::InterpolationMethod=LogLinearInterp())
    times = Float64[0.0]
    dfs = Float64[1.0]

    for inst in instruments
        df = bootstrap_instrument(inst, times, dfs)
        push!(times, get_maturity(inst))
        push!(dfs, df)
    end

    DiscountCurve(times, dfs; interp=interp)
end

get_maturity(d::DepositRate) = d.maturity
get_maturity(f::FuturesRate) = f.maturity
get_maturity(s::SwapRate) = s.maturity

function bootstrap_instrument(d::DepositRate, times, dfs)
    # DF = 1 / (1 + r * T) for simple rate
    1.0 / (1.0 + d.rate * d.maturity)
end

function bootstrap_instrument(f::FuturesRate, times, dfs)
    # Need DF to start date
    df_start = interp_value(LogLinearInterp(), times, dfs, f.start)
    # Forward rate (adjusted)
    fwd = f.rate - f.convexity_adj
    tau = f.maturity - f.start
    df_start / (1.0 + fwd * tau)
end

function bootstrap_instrument(s::SwapRate, times, dfs)
    # Par swap: sum of DF * tau = (1 - DF_N) / rate
    # Solve for DF_N
    tau = 1.0 / s.frequency
    n_periods = Int(s.maturity * s.frequency)

    pv_fixed = 0.0
    for i in 1:(n_periods - 1)
        t = i * tau
        df = interp_value(LogLinearInterp(), times, dfs, t)
        pv_fixed += df * tau
    end

    # DF_N = (1 - rate * pv_fixed) / (1 + rate * tau)
    (1.0 - s.rate * pv_fixed) / (1.0 + s.rate * tau)
end

# ============================================================================
# Bonds
# ============================================================================

"""
    Bond

Abstract base type for fixed income instruments.

All bonds support the following operations:
- `price(bond, curve)` - Present value using a discount curve
- `price(bond, yield)` - Present value at a given yield
- `yield_to_maturity(bond, price)` - Solve for yield given price
- `duration(bond, yield)` - Macaulay duration
- `modified_duration(bond, yield)` - Modified duration
- `convexity(bond, yield)` - Convexity
- `dv01(bond, yield)` - Dollar value of 1 basis point

# Subtypes
- [`ZeroCouponBond`](@ref) - Zero-coupon (discount) bond
- [`FixedRateBond`](@ref) - Fixed-rate coupon bond
- [`FloatingRateBond`](@ref) - Floating-rate bond

# Example
```julia
bond = FixedRateBond(5.0, 0.04, 2)  # 5-year, 4% semi-annual
curve = ZeroCurve(0.05)
pv = price(bond, curve)
ytm = yield_to_maturity(bond, 95.0)
dur = duration(bond, ytm)
```
"""
abstract type Bond end

"""
    ZeroCouponBond(maturity, face_value=100.0)

Zero-coupon bond paying face value at maturity.

# Arguments
- `maturity::Float64` - Time to maturity in years
- `face_value::Float64` - Face (par) value (default: 100.0)

# Example
```julia
zcb = ZeroCouponBond(5.0)  # 5-year zero
price(zcb, 0.05)  # ≈ 78.12 at 5% yield
```
"""
struct ZeroCouponBond <: Bond
    maturity::Float64
    face_value::Float64
end
ZeroCouponBond(maturity) = ZeroCouponBond(maturity, 100.0)

"""
    FixedRateBond(maturity, coupon_rate, frequency=2, face_value=100.0)

Fixed-rate coupon bond. Coupon rate is annual, paid at given frequency.
"""
struct FixedRateBond <: Bond
    maturity::Float64
    coupon_rate::Float64
    frequency::Int
    face_value::Float64
end
FixedRateBond(mat, cpn) = FixedRateBond(mat, cpn, 2, 100.0)
FixedRateBond(mat, cpn, freq) = FixedRateBond(mat, cpn, freq, 100.0)

"""
    FloatingRateBond(maturity, spread, frequency=4, face_value=100.0)

Floating-rate bond paying reference rate + spread.
"""
struct FloatingRateBond <: Bond
    maturity::Float64
    spread::Float64
    frequency::Int
    face_value::Float64
end
FloatingRateBond(mat, spread) = FloatingRateBond(mat, spread, 4, 100.0)

# Cash flows
function cash_flows(bond::ZeroCouponBond)
    [(bond.maturity, bond.face_value)]
end

function cash_flows(bond::FixedRateBond)
    tau = 1.0 / bond.frequency
    n = Int(ceil(bond.maturity * bond.frequency))
    coupon = bond.face_value * bond.coupon_rate * tau

    flows = Tuple{Float64,Float64}[]
    for i in 1:n
        t = i * tau
        cf = i == n ? coupon + bond.face_value : coupon
        push!(flows, (t, cf))
    end
    flows
end

"""
    price(bond, curve) -> Float64

Present value of bond using discount curve.
"""
function price(bond::Bond, curve::RateCurve)
    sum(cf * discount(curve, t) for (t, cf) in cash_flows(bond))
end

"""
    price(bond, yield) -> Float64

Present value of bond at given yield (continuously compounded).
"""
function price(bond::Bond, yield::Float64)
    sum(cf * exp(-yield * t) for (t, cf) in cash_flows(bond))
end

"""
    yield_to_maturity(bond, market_price; tol=1e-10) -> Float64

Solve for yield given market price using Newton-Raphson.
"""
function yield_to_maturity(bond::Bond, market_price::Float64; tol::Float64=1e-10, max_iter::Int=100)
    # Initial guess from simple yield
    y = bond isa ZeroCouponBond ?
        -log(market_price / bond.face_value) / bond.maturity :
        (bond.coupon_rate * bond.face_value + (bond.face_value - market_price) / bond.maturity) / market_price

    for _ in 1:max_iter
        p = price(bond, y)
        if abs(p - market_price) < tol
            return y
        end
        # dp/dy = -sum(t * cf * exp(-y*t))
        dp = -sum(t * cf * exp(-y * t) for (t, cf) in cash_flows(bond))
        y -= (p - market_price) / dp
    end

    y
end

"""
    duration(bond, yield) -> Float64

Macaulay duration: weighted average time to cash flows.
"""
function duration(bond::Bond, yield::Float64)
    p = price(bond, yield)
    sum(t * cf * exp(-yield * t) for (t, cf) in cash_flows(bond)) / p
end

"""
    modified_duration(bond, yield) -> Float64

Modified duration: -1/P * dP/dy
"""
function modified_duration(bond::Bond, yield::Float64)
    duration(bond, yield) / (1 + yield)
end

"""
    convexity(bond, yield) -> Float64

Convexity: 1/P * d²P/dy²
"""
function convexity(bond::Bond, yield::Float64)
    p = price(bond, yield)
    sum(t^2 * cf * exp(-yield * t) for (t, cf) in cash_flows(bond)) / p
end

"""
    dv01(bond, yield) -> Float64

Dollar value of 1 basis point: price change for 1bp yield move.
"""
function dv01(bond::Bond, yield::Float64)
    p = price(bond, yield)
    modified_duration(bond, yield) * p * 0.0001
end

"""
    accrued_interest(bond, settlement_time) -> Float64

Accrued interest from last coupon to settlement.
"""
function accrued_interest(bond::FixedRateBond, settlement_time::Float64)
    tau = 1.0 / bond.frequency
    last_coupon = floor(settlement_time / tau) * tau
    accrual_fraction = (settlement_time - last_coupon) / tau
    bond.face_value * bond.coupon_rate * tau * accrual_fraction
end
accrued_interest(::ZeroCouponBond, ::Float64) = 0.0

"""Clean price = dirty price - accrued interest"""
clean_price(bond::Bond, curve::RateCurve, settlement::Float64=0.0) =
    price(bond, curve) - accrued_interest(bond, settlement)

"""Dirty price = full price including accrued"""
dirty_price(bond::Bond, curve::RateCurve) = price(bond, curve)

# ============================================================================
# Short Rate Models
# ============================================================================

abstract type ShortRateModel end

"""
    Vasicek(κ, θ, σ, r0)

Vasicek model: dr = κ(θ - r)dt + σdW

Parameters:
- κ: mean reversion speed
- θ: long-term mean rate
- σ: volatility
- r0: initial short rate
"""
struct Vasicek <: ShortRateModel
    κ::Float64
    θ::Float64
    σ::Float64
    r0::Float64
end

"""
    CIR(κ, θ, σ, r0)

Cox-Ingersoll-Ross model: dr = κ(θ - r)dt + σ√r dW

Feller condition: 2κθ > σ² ensures r stays positive.
"""
struct CIR <: ShortRateModel
    κ::Float64
    θ::Float64
    σ::Float64
    r0::Float64

    function CIR(κ, θ, σ, r0)
        if 2κ * θ <= σ^2
            @warn "Feller condition violated: 2κθ = $(2κ*θ) ≤ σ² = $(σ^2). Rate may hit zero."
        end
        new(κ, θ, σ, r0)
    end
end

"""
    HullWhite(κ, σ, curve)

Hull-White model: dr = (θ(t) - κr)dt + σdW

Time-dependent θ(t) calibrated to fit initial term structure.
"""
struct HullWhite <: ShortRateModel
    κ::Float64
    σ::Float64
    curve::RateCurve  # Initial term structure to fit
end

# Analytical bond prices under short rate models

"""
    bond_price(model, T) -> Float64

Zero-coupon bond price P(0,T) under the short rate model.
"""
function bond_price(m::Vasicek, T::Float64)
    κ, θ, σ, r0 = m.κ, m.θ, m.σ, m.r0

    B = (1 - exp(-κ * T)) / κ
    A = exp((θ - σ^2 / (2κ^2)) * (B - T) - σ^2 * B^2 / (4κ))

    A * exp(-B * r0)
end

function bond_price(m::CIR, T::Float64)
    κ, θ, σ, r0 = m.κ, m.θ, m.σ, m.r0

    γ = sqrt(κ^2 + 2σ^2)

    num = 2γ * exp((κ + γ) * T / 2)
    den = (κ + γ) * (exp(γ * T) - 1) + 2γ

    A = (num / den)^(2κ * θ / σ^2)
    B = 2(exp(γ * T) - 1) / den

    A * exp(-B * r0)
end

function bond_price(m::HullWhite, T::Float64)
    κ, σ = m.κ, m.σ

    # Use market discount factor and adjust
    P_market = discount(m.curve, T)
    f0 = instantaneous_forward(m.curve, 0.0)

    B = (1 - exp(-κ * T)) / κ

    # Under HW, P(0,T) = P_market(0,T) * exp(-B*r0 + B*f0 + σ²B²(1-exp(-2κT))/(4κ))
    # Simplified: we fit to market, so return market price
    P_market
end

"""
    short_rate(model, t) -> (mean, variance)

Expected short rate and variance at time t.
"""
function short_rate(m::Vasicek, t::Float64)
    κ, θ, σ, r0 = m.κ, m.θ, m.σ, m.r0

    mean = θ + (r0 - θ) * exp(-κ * t)
    var = σ^2 * (1 - exp(-2κ * t)) / (2κ)

    (mean, var)
end

function short_rate(m::CIR, t::Float64)
    κ, θ, σ, r0 = m.κ, m.θ, m.σ, m.r0

    mean = θ + (r0 - θ) * exp(-κ * t)
    var = r0 * σ^2 * exp(-κ * t) * (1 - exp(-κ * t)) / κ +
          θ * σ^2 * (1 - exp(-κ * t))^2 / (2κ)

    (mean, var)
end

"""
    simulate_short_rate(model, T, n_steps, n_paths) -> Matrix

Simulate short rate paths. Returns [n_steps+1 × n_paths] matrix.
"""
function simulate_short_rate(m::Vasicek, T::Float64, n_steps::Int, n_paths::Int)
    κ, θ, σ, r0 = m.κ, m.θ, m.σ, m.r0
    dt = T / n_steps

    paths = zeros(n_steps + 1, n_paths)
    paths[1, :] .= r0

    sqrt_dt = sqrt(dt)
    for i in 1:n_steps
        dW = randn(n_paths) * sqrt_dt
        paths[i+1, :] = paths[i, :] + κ * (θ .- paths[i, :]) * dt + σ * dW
    end

    paths
end

function simulate_short_rate(m::CIR, T::Float64, n_steps::Int, n_paths::Int)
    κ, θ, σ, r0 = m.κ, m.θ, m.σ, m.r0
    dt = T / n_steps

    paths = zeros(n_steps + 1, n_paths)
    paths[1, :] .= r0

    sqrt_dt = sqrt(dt)
    for i in 1:n_steps
        dW = randn(n_paths) * sqrt_dt
        r = paths[i, :]
        # Full truncation scheme
        r_pos = max.(r, 0.0)
        paths[i+1, :] = r + κ * (θ .- r_pos) * dt + σ * sqrt.(r_pos) .* dW
        paths[i+1, :] = max.(paths[i+1, :], 0.0)
    end

    paths
end

function simulate_short_rate(m::HullWhite, T::Float64, n_steps::Int, n_paths::Int)
    κ, σ = m.κ, m.σ
    dt = T / n_steps

    # θ(t) chosen to fit initial curve
    function theta(t)
        f = instantaneous_forward(m.curve, t)
        df_dt = (instantaneous_forward(m.curve, t + 1e-6) - f) / 1e-6
        df_dt + κ * f + σ^2 * (1 - exp(-2κ * t)) / (2κ)
    end

    r0 = instantaneous_forward(m.curve, 0.0)
    paths = zeros(n_steps + 1, n_paths)
    paths[1, :] .= r0

    sqrt_dt = sqrt(dt)
    for i in 1:n_steps
        t = (i - 1) * dt
        dW = randn(n_paths) * sqrt_dt
        paths[i+1, :] = paths[i, :] + (theta(t) .- κ * paths[i, :]) * dt + σ * dW
    end

    paths
end

# ============================================================================
# Interest Rate Derivatives
# ============================================================================

"""Caplet: call option on forward rate"""
struct Caplet
    start::Float64      # Start of rate period
    maturity::Float64   # End of rate period (payment date)
    strike::Float64     # Strike rate
    notional::Float64
end
Caplet(start, mat, strike) = Caplet(start, mat, strike, 1.0)

"""Floorlet: put option on forward rate"""
struct Floorlet
    start::Float64
    maturity::Float64
    strike::Float64
    notional::Float64
end
Floorlet(start, mat, strike) = Floorlet(start, mat, strike, 1.0)

"""Cap: portfolio of caplets"""
struct Cap
    maturity::Float64
    strike::Float64
    frequency::Int
    notional::Float64
end
Cap(mat, strike) = Cap(mat, strike, 4, 1.0)

"""Floor: portfolio of floorlets"""
struct Floor
    maturity::Float64
    strike::Float64
    frequency::Int
    notional::Float64
end
Floor(mat, strike) = Floor(mat, strike, 4, 1.0)

"""
    Swaption(expiry, swap_maturity, strike, is_payer, notional)

European swaption - option to enter a swap.
"""
# TODO: Export black_swaption function (currently only price() method exists)
struct Swaption
    expiry::Float64         # Option expiry
    swap_maturity::Float64  # Underlying swap maturity
    strike::Float64         # Strike swap rate
    is_payer::Bool          # true = payer swaption
    frequency::Int
    notional::Float64
end
Swaption(exp, mat, strike, is_payer) = Swaption(exp, mat, strike, is_payer, 2, 1.0)

"""
    black_caplet(caplet, curve, volatility) -> Float64

Price a caplet using Black's formula.
"""
function black_caplet(c::Caplet, curve::RateCurve, σ::Float64)
    τ = c.maturity - c.start
    F = forward_rate(curve, c.start, c.maturity)
    df = discount(curve, c.maturity)

    if c.start <= 0
        # Already started, intrinsic value only
        return c.notional * τ * df * max(F - c.strike, 0)
    end

    d1 = (log(F / c.strike) + 0.5 * σ^2 * c.start) / (σ * sqrt(c.start))
    d2 = d1 - σ * sqrt(c.start)

    N = Normal()
    c.notional * τ * df * (F * cdf(N, d1) - c.strike * cdf(N, d2))
end

"""
    black_floorlet(floorlet, curve, volatility) -> Float64

Price a floorlet using Black's formula.
"""
function black_floorlet(f::Floorlet, curve::RateCurve, σ::Float64)
    τ = f.maturity - f.start
    F = forward_rate(curve, f.start, f.maturity)
    df = discount(curve, f.maturity)

    if f.start <= 0
        return f.notional * τ * df * max(f.strike - F, 0)
    end

    d1 = (log(F / f.strike) + 0.5 * σ^2 * f.start) / (σ * sqrt(f.start))
    d2 = d1 - σ * sqrt(f.start)

    N = Normal()
    f.notional * τ * df * (f.strike * cdf(N, -d2) - F * cdf(N, -d1))
end

"""
    black_cap(cap, curve, volatilities) -> Float64

Price a cap as sum of caplets.
volatilities can be scalar (flat) or vector (per caplet).
"""
function black_cap(cap::Cap, curve::RateCurve, σ)
    τ = 1.0 / cap.frequency
    n = Int(cap.maturity * cap.frequency)

    total = 0.0
    for i in 1:n
        start = (i - 1) * τ
        mat = i * τ
        vol = σ isa Vector ? σ[i] : σ
        caplet = Caplet(start, mat, cap.strike, cap.notional)
        total += black_caplet(caplet, curve, vol)
    end
    total
end

"""
    black_floor(floor, curve, volatilities) -> Float64

Price a floor as sum of floorlets.
"""
function black_floor(floor::Floor, curve::RateCurve, σ)
    τ = 1.0 / floor.frequency
    n = Int(floor.maturity * floor.frequency)

    total = 0.0
    for i in 1:n
        start = (i - 1) * τ
        mat = i * τ
        vol = σ isa Vector ? σ[i] : σ
        floorlet = Floorlet(start, mat, floor.strike, floor.notional)
        total += black_floorlet(floorlet, curve, vol)
    end
    total
end

"""
    price(swaption, curve, volatility) -> Float64

Price a European swaption using Black's formula.
"""
function price(s::Swaption, curve::RateCurve, σ::Float64)
    # Annuity factor
    τ = 1.0 / s.frequency
    n = Int((s.swap_maturity - s.expiry) * s.frequency)

    A = sum(τ * discount(curve, s.expiry + i * τ) for i in 1:n)

    # Forward swap rate
    df_start = discount(curve, s.expiry)
    df_end = discount(curve, s.swap_maturity)
    S = (df_start - df_end) / A

    # Black's formula
    d1 = (log(S / s.strike) + 0.5 * σ^2 * s.expiry) / (σ * sqrt(s.expiry))
    d2 = d1 - σ * sqrt(s.expiry)

    N = Normal()
    if s.is_payer
        s.notional * A * (S * cdf(N, d1) - s.strike * cdf(N, d2))
    else
        s.notional * A * (s.strike * cdf(N, -d2) - S * cdf(N, -d1))
    end
end

end # module
