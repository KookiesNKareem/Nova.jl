module Quasar

# Core must come first - defines abstract types
include("Core/Core.jl")
using .Core
export AbstractInstrument, AbstractEquity, AbstractDerivative, AbstractOption
export AbstractPortfolio, AbstractRiskMeasure, AbstractADBackend
export MarketState, Priceable, Differentiable, HasGreeks, Simulatable

# AD backend system
include("AD/AD.jl")
using .AD
export PureJuliaBackend, ForwardDiffBackend, ReactantBackend
export gradient, hessian, jacobian, current_backend, set_backend!

# Instruments
include("Instruments/Instruments.jl")
using .Instruments
export Stock, EuropeanOption, AmericanOption, AsianOption

# Portfolio
include("Portfolio/Portfolio.jl")
using .Portfolio
export Portfolio

# Risk measures
include("Risk/Risk.jl")
using .Risk
export VaR, CVaR, Volatility, Sharpe, MaxDrawdown, compute

# Optimization
include("Optimization/Optimization.jl")
using .Optimization
export optimize, MeanVariance, CVaRObjective, KellyCriterion

# Market data
include("MarketData/MarketData.jl")
using .MarketData
export AbstractMarketData, AbstractPriceHistory, CSVAdapter, ParquetAdapter

end
