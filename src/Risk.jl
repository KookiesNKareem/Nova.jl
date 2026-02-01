module Risk

# TODO: Add historical simulation VaR
# TODO: Add Monte Carlo VaR
# TODO: Add Cornish-Fisher expansion for non-normal VaR
# TODO: Add correlation/covariance analytics
# TODO: Add stress testing framework

using ..Core: AbstractRiskMeasure
using ..PortfolioModule
using Statistics

"""
    VaR <: AbstractRiskMeasure

Value at Risk at specified confidence level.
"""
struct VaR <: AbstractRiskMeasure
    confidence::Float64

    function VaR(confidence::Float64=0.95)
        0 < confidence < 1 || error("confidence must be in (0, 1)")
        new(confidence)
    end
end

"""
    CVaR <: AbstractRiskMeasure

Conditional Value at Risk (Expected Shortfall) at specified confidence level.
"""
struct CVaR <: AbstractRiskMeasure
    confidence::Float64

    function CVaR(confidence::Float64=0.95)
        0 < confidence < 1 || error("confidence must be in (0, 1)")
        new(confidence)
    end
end

"""
    Volatility <: AbstractRiskMeasure

Standard deviation of returns.
"""
struct Volatility <: AbstractRiskMeasure end

"""
    Sharpe <: AbstractRiskMeasure

Sharpe ratio (excess return / volatility).

# Fields
- `rf::Float64` - Annualized risk-free rate (e.g., 0.05 for 5%)
- `periods_per_year::Int` - Number of return periods per year (252 for daily, 52 for weekly, 12 for monthly)
"""
struct Sharpe <: AbstractRiskMeasure
    rf::Float64
    periods_per_year::Int

    Sharpe(; rf::Float64=0.0, periods_per_year::Int=252) = new(rf, periods_per_year)
end

"""
    MaxDrawdown <: AbstractRiskMeasure

Maximum peak-to-trough decline.
"""
struct MaxDrawdown <: AbstractRiskMeasure end

"""
    compute(measure::AbstractRiskMeasure, returns)

Compute the risk measure for given returns.
"""
function compute end

# VaR implementation
function compute(var::VaR, returns::AbstractVector)
    sorted = sort(returns)
    idx = ceil(Int, (1 - var.confidence) * length(sorted))
    idx = max(1, idx)
    return sorted[idx]
end

# CVaR implementation
function compute(cvar::CVaR, returns::AbstractVector)
    var_threshold = compute(VaR(cvar.confidence), returns)
    tail_returns = filter(r -> r <= var_threshold, returns)
    return mean(tail_returns)
end

# Volatility implementation
function compute(::Volatility, returns::AbstractVector)
    return std(returns)
end

# Sharpe implementation
function compute(sharpe::Sharpe, returns::AbstractVector)
    rf_per_period = sharpe.rf / sharpe.periods_per_year
    excess_return = mean(returns) - rf_per_period
    vol = std(returns)
    return excess_return / vol
end

# MaxDrawdown implementation
function compute(::MaxDrawdown, returns::AbstractVector)
    # Convert returns to price series (starting at 1)
    prices = cumprod(1 .+ returns)

    # Track running maximum
    peak = prices[1]
    max_dd = 0.0

    for p in prices
        peak = max(peak, p)
        drawdown = (p - peak) / peak
        max_dd = min(max_dd, drawdown)
    end

    return max_dd
end

export VaR, CVaR, Volatility, Sharpe, MaxDrawdown
export compute

end
