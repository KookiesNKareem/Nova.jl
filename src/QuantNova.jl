module QuantNova

# TODO: Add precompilation workloads for faster load times
# TODO: Consider SnoopCompile for latency analysis

# Core must come first - defines abstract types
include("Core.jl")
using .Core
export AbstractInstrument, AbstractEquity, AbstractDerivative, AbstractOption, AbstractFuture
export AbstractPortfolio, AbstractRiskMeasure, ADBackend
export MarketState
export Priceable, IsPriceable, NotPriceable
export Differentiable, IsDifferentiable, NotDifferentiable
export HasGreeks, HasGreeksTrait, NoGreeksTrait
export Simulatable, IsSimulatable, NotSimulatable
export priceable, ispriceable, differentiable, isdifferentiable
export greeks_trait, hasgreeks, simulatable, issimulatable

# AD backend system
include("AD.jl")
using .AD
export PureJuliaBackend, ForwardDiffBackend, ReactantBackend, EnzymeBackend
export gradient, hessian, jacobian, value_and_gradient, current_backend, set_backend!, with_backend, enable_gpu!

# Instruments
include("Instruments.jl")
using .Instruments
export Stock, EuropeanOption, AmericanOption, AsianOption, price, black_scholes
export american_binomial, asian_geometric, asian_arithmetic_approx
export GreeksResult, compute_greeks

# Portfolio
include("Portfolio.jl")
using .PortfolioModule
export Portfolio, value, portfolio_greeks

# Risk measures
include("Risk.jl")
using .Risk
export VaR, CVaR, Volatility, Sharpe, MaxDrawdown, compute

# Optimization
include("Optimization.jl")
using .Optimization
export optimize, MeanVariance, SharpeMaximizer, CVaRObjective, KellyCriterion, OptimizationResult
export AbstractOptimizationObjective, AbstractConstraint, AbstractSolver, AbstractCovarianceEstimator
export FullInvestmentConstraint, LongOnlyConstraint, BoxConstraint
export GroupConstraint, TurnoverConstraint, CardinalityConstraint
export standard_constraints, check_constraint_violation, check_all_constraints
export QPSolver, LBFGSSolver, ProjectedGradientSolver
export project_simplex, project_constraints, solve_qp, solve_min_variance_qp
export solve_lbfgs, solve_projected_gradient
export MinimumVariance, RiskParity, MaximumDiversification, BlackLitterman
export compute_risk_contributions, compute_fractional_risk_contributions
export compute_marginal_risk, compute_component_risk, compute_beta
export compute_tracking_error, compute_active_risk_contributions
export PortfolioAnalytics, analyze_portfolio
export risk_parity_objective, diversification_ratio
export bl_equilibrium_returns, bl_posterior_returns, bl_posterior_covariance
export EfficientFrontier, compute_efficient_frontier
export get_tangency_portfolio, get_min_variance_portfolio, interpolate_frontier
export SampleCovariance, LedoitWolfShrinkage, ExponentialWeighted
export estimate_covariance, estimate_expected_returns, estimate_parameters
export PortfolioSpec
export min_variance_portfolio, max_sharpe_portfolio, risk_parity_portfolio, target_return_portfolio

# Stochastic volatility models (SABR, Heston)
include("Models.jl")
using .Models
export SABRParams, sabr_implied_vol, sabr_price, black76
export HestonParams, heston_price, heston_characteristic

# Batch pricing and GPU-optimized calibration
include("BatchPricing.jl")
using .BatchPricing
export sabr_vols_batch, sabr_prices_batch
export PrecompiledSABRCalibrator, compile_gpu!, calibrate!
export price_surface_batch

# Model calibration
include("Calibration.jl")
using .Calibration
export OptionQuote, SmileData, VolSurface, CalibrationResult
export calibrate_sabr, calibrate_heston

# Monte Carlo simulation
include("MonteCarlo.jl")
using .MonteCarlo
export GBMDynamics, HestonDynamics
export EuropeanCall, EuropeanPut, AsianCall, AsianPut, UpAndOutCall, DownAndOutPut
export AmericanPut, AmericanCall
export MCResult, mc_price, mc_delta, mc_greeks, lsm_price
export mc_price_qmc, sobol_normals, simulate_gbm_qmc

# Market data
include("MarketData.jl")
using .MarketData
export AbstractMarketData, AbstractPriceHistory, AbstractDataAdapter
export PriceHistory, returns, resample, align
export CSVAdapter, ParquetAdapter, YAHOO_ADAPTER
export fetch_prices, fetch_multiple, fetch_returns, fetch_return_matrix
export to_backtest_format
import .MarketData: load, save

# Interest rates
include("InterestRates.jl")
using .InterestRates
export RateCurve, DiscountCurve, ZeroCurve, ForwardCurve
export NelsonSiegelCurve, SvenssonCurve, fit_nelson_siegel, fit_svensson
export discount, zero_rate, forward_rate, instantaneous_forward
export LinearInterp, LogLinearInterp, CubicSplineInterp
export DepositRate, FuturesRate, SwapRate, bootstrap
export Bond, ZeroCouponBond, FixedRateBond, FloatingRateBond
export yield_to_maturity, duration, modified_duration, convexity, dv01
export accrued_interest, clean_price, dirty_price
export ShortRateModel, Vasicek, CIR, HullWhite
export bond_price, short_rate, simulate_short_rate
export Caplet, Floorlet, Cap, Floor, Swaption
export black_caplet, black_floorlet, black_cap, black_floor

# Simulation engine
include("Simulation.jl")
using .Simulation
export SimulationState, portfolio_value
export Order, Fill, AbstractExecutionModel, InstantFill, SlippageModel, MarketImpactModel
export execute
export MarketSnapshot, AbstractDriver, HistoricalDriver
export SimulationResult, simulate

# Transaction costs
include("TransactionCosts.jl")
using .TransactionCosts
export AbstractCostModel, compute_cost
export FixedCostModel, ProportionalCostModel, TieredCostModel
export SpreadCostModel, AlmgrenChrissModel, CompositeCostModel
export TradeCostBreakdown, CostTracker, record_trade!, cost_summary
export compute_turnover
export CostAwareExecutionModel, execute_with_costs
export RETAIL_COSTS, INSTITUTIONAL_COSTS, HFT_COSTS, create_cost_model
export compute_net_returns, estimate_break_even_sharpe

# Backtesting
include("Backtesting.jl")
using .Backtesting
export AbstractStrategy, generate_orders, should_rebalance
export BuyAndHoldStrategy, RebalancingStrategy
export VolatilityTargetStrategy, estimate_ewma_volatility
export BacktestResult, backtest, compute_backtest_metrics
export WalkForwardConfig, WalkForwardPeriod, WalkForwardResult
export walk_forward_backtest, compute_extended_metrics
# Custom strategy framework
export StrategyContext, get_returns, get_prices
export SignalStrategy
export MomentumStrategy, MeanReversionStrategy
export CompositeStrategy

# Scenario Analysis
include("ScenarioAnalysis.jl")
using .ScenarioAnalysis
export StressScenario, ScenarioImpact, CRISIS_SCENARIOS
export apply_scenario, scenario_impact
export compare_scenarios, worst_case_scenario
export SensitivityResult, sensitivity_analysis
export ProjectionResult, monte_carlo_projection

# Statistical Testing
include("Statistics.jl")
using .Statistics
export sharpe_ratio, sharpe_std_error, sharpe_confidence_interval
export sharpe_t_stat, sharpe_pvalue
export probabilistic_sharpe_ratio, deflated_sharpe_ratio
export compare_sharpe_ratios
export minimum_backtest_length, probability_of_backtest_overfitting
export combinatorial_purged_cv_pbo
export information_coefficient, hit_rate, hit_rate_significance

# Factor Models
include("FactorModels.jl")
using .FactorModels
export RegressionResult, factor_regression
export capm_regression
export FamaFrenchResult, fama_french_regression
export construct_market_factor, construct_long_short_factor
export AttributionResult, return_attribution
export rolling_beta, rolling_alpha
export StyleAnalysisResult, style_analysis
export tracking_error, information_ratio
export up_capture_ratio, down_capture_ratio, capture_ratio

# Visualization
include("Visualization.jl")
using .Visualization
export AbstractVisualization, VisualizationSpec
export visualize, set_theme!, get_theme, available_views
export LIGHT_THEME, DARK_THEME, COLORS
export render

end
