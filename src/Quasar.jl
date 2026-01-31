module Quasar

# Core must come first - defines abstract types
include("Core/Core.jl")
using .Core
export AbstractInstrument, AbstractEquity, AbstractDerivative, AbstractOption, AbstractFuture
export AbstractPortfolio, AbstractRiskMeasure, AbstractADBackend
export MarketState
export Priceable, IsPriceable, NotPriceable
export Differentiable, IsDifferentiable, NotDifferentiable
export HasGreeks, HasGreeksTrait, NoGreeksTrait
export Simulatable, IsSimulatable, NotSimulatable
export priceable, ispriceable, differentiable, isdifferentiable
export greeks_trait, hasgreeks, simulatable, issimulatable

# AD backend system
include("AD/AD.jl")
using .AD
export PureJuliaBackend, ForwardDiffBackend, ReactantBackend
export gradient, hessian, jacobian, current_backend, set_backend!

# Instruments
include("Instruments/Instruments.jl")
using .Instruments
export Stock, EuropeanOption, AmericanOption, AsianOption, price, black_scholes
export GreeksResult, compute_greeks

# Portfolio
include("Portfolio/Portfolio.jl")
using .PortfolioModule
export Portfolio, value, portfolio_greeks

# Risk measures
include("Risk/Risk.jl")
using .Risk
export VaR, CVaR, Volatility, Sharpe, MaxDrawdown, compute

# Optimization
include("Optimization/Optimization.jl")
using .Optimization
export optimize, MeanVariance, SharpeMaximizer, CVaRObjective, KellyCriterion, OptimizationResult

# Market data
include("MarketData/MarketData.jl")
using .MarketData
export AbstractMarketData, AbstractPriceHistory, CSVAdapter, ParquetAdapter

end
