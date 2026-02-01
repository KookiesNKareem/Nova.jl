# Comprehensive QuantNova.jl Benchmark Suite
#
# Benchmarks all core areas:
# 1. Option Pricing (European, American, Asian)
# 2. Greeks (AD vs analytical)
# 3. Monte Carlo (paths, exotics)
# 4. Calibration (SABR, Heston)
# 5. Backtesting (strategies, metrics)
# 6. Factor Models (CAPM, FF3, rolling)
# 7. Statistics (Sharpe, confidence intervals)
#
# Run: julia --project=../.. comprehensive_benchmark.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))

using QuantNova
using Statistics
using LinearAlgebra
using Random
using Printf
using Dates

# =============================================================================
# Timing Utilities
# =============================================================================

function benchmark(f; n_runs=1000, n_warmup=100)
    # Warmup
    for _ in 1:n_warmup
        f()
    end

    # Timed runs
    times = Float64[]
    for _ in 1:n_runs
        t = @elapsed f()
        push!(times, t * 1e6)  # Convert to μs
    end

    return (
        median = median(times),
        mean = mean(times),
        std = std(times),
        min = minimum(times),
        max = maximum(times)
    )
end

# =============================================================================
# Option Pricing Benchmarks
# =============================================================================

function run_pricing_benchmarks()
    println("\n" * "=" ^ 70)
    println("OPTION PRICING BENCHMARKS")
    println("=" ^ 70)

    S, K, T, r, σ = 100.0, 100.0, 1.0, 0.05, 0.2
    results = Dict{String, Float64}()

    # -------------------------------------------------------------------------
    # European Option (Black-Scholes)
    # -------------------------------------------------------------------------
    println("\n[1/4] European Option (Black-Scholes analytic)")

    price = black_scholes(S, K, T, r, σ, :call)
    @printf("  Price: %.6f\n", price)

    t = benchmark(() -> black_scholes(S, K, T, r, σ, :call); n_runs=10000, n_warmup=1000)
    @printf("  Timing: %.3f μs (median)\n", t.median)
    results["european_bs"] = t.median

    # -------------------------------------------------------------------------
    # American Option (Binomial 100-step)
    # -------------------------------------------------------------------------
    println("\n[2/4] American Option (100-step binomial)")

    am_price = american_binomial(S, K, T, r, σ, :put, 100)
    @printf("  Price: %.6f\n", am_price)

    t = benchmark(() -> american_binomial(S, K, T, r, σ, :put, 100); n_runs=500, n_warmup=50)
    @printf("  Timing: %.2f μs (median)\n", t.median)
    results["american_100"] = t.median

    # -------------------------------------------------------------------------
    # SABR Implied Vol
    # -------------------------------------------------------------------------
    println("\n[3/4] SABR Implied Volatility")

    params = SABRParams(0.2, 0.5, -0.3, 0.4)
    sabr_iv = sabr_implied_vol(100.0, 100.0, 1.0, params)
    @printf("  Implied Vol: %.6f\n", sabr_iv)

    t = benchmark(() -> sabr_implied_vol(100.0, 100.0, 1.0, params); n_runs=10000, n_warmup=1000)
    @printf("  Timing: %.3f μs (median)\n", t.median)
    results["sabr_vol"] = t.median

    # -------------------------------------------------------------------------
    # Batch Pricing (1000 options)
    # -------------------------------------------------------------------------
    println("\n[4/4] Batch Pricing (1,000 European options)")

    Random.seed!(42)
    Ks = 80.0 .+ 40.0 .* rand(1000)
    Ts = 0.1 .+ 1.9 .* rand(1000)
    σs = 0.1 .+ 0.4 .* rand(1000)

    function batch_price()
        prices = Vector{Float64}(undef, 1000)
        for i in 1:1000
            prices[i] = black_scholes(S, Ks[i], Ts[i], r, σs[i], :call)
        end
        return prices
    end

    t = benchmark(batch_price; n_runs=100, n_warmup=10)
    @printf("  Timing: %.2f ms (median)\n", t.median / 1000)
    @printf("  Per-option: %.3f μs\n", t.median / 1000)
    results["batch_1000"] = t.median

    return results
end

# =============================================================================
# Greeks Benchmarks
# =============================================================================

function run_greeks_benchmarks()
    println("\n" * "=" ^ 70)
    println("GREEKS BENCHMARKS")
    println("=" ^ 70)

    S, K, T, r, σ = 100.0, 100.0, 1.0, 0.05, 0.2
    results = Dict{String, Float64}()

    state = MarketState(
        prices = Dict("TEST" => S),
        rates = Dict("USD" => r),
        volatilities = Dict("TEST" => σ),
        timestamp = 0.0
    )
    option = EuropeanOption("TEST", K, T, :call)

    # -------------------------------------------------------------------------
    # All Greeks (AD)
    # -------------------------------------------------------------------------
    println("\n[1/2] All Greeks via AD (ForwardDiff)")

    greeks = compute_greeks(option, state)
    @printf("  Delta: %.6f\n", greeks.delta)
    @printf("  Gamma: %.6f\n", greeks.gamma)
    @printf("  Vega:  %.6f\n", greeks.vega)
    @printf("  Theta: %.6f\n", greeks.theta)
    @printf("  Rho:   %.6f\n", greeks.rho)

    t = benchmark(() -> compute_greeks(option, state); n_runs=1000, n_warmup=100)
    @printf("  Timing: %.2f μs (median)\n", t.median)
    results["greeks_ad"] = t.median

    # -------------------------------------------------------------------------
    # Batch Greeks (100 options)
    # -------------------------------------------------------------------------
    println("\n[2/2] Batch Greeks (100 options)")

    options = [EuropeanOption("TEST", 80.0 + 0.4 * i, T, :call) for i in 1:100]

    function batch_greeks()
        return [compute_greeks(opt, state) for opt in options]
    end

    t = benchmark(batch_greeks; n_runs=100, n_warmup=10)
    @printf("  Timing: %.2f ms (median)\n", t.median / 1000)
    @printf("  Per-option: %.2f μs\n", t.median / 100)
    results["batch_greeks_100"] = t.median

    return results
end

# =============================================================================
# Monte Carlo Benchmarks
# =============================================================================

function run_mc_benchmarks()
    println("\n" * "=" ^ 70)
    println("MONTE CARLO BENCHMARKS")
    println("=" ^ 70)

    S, K, T, r, σ = 100.0, 100.0, 1.0, 0.05, 0.2
    dynamics = GBMDynamics(r, σ)
    results = Dict{String, Float64}()

    # -------------------------------------------------------------------------
    # European MC (10k paths)
    # -------------------------------------------------------------------------
    println("\n[1/3] MC European (10,000 paths)")

    result = mc_price(S, T, EuropeanCall(K), dynamics; npaths=10000, nsteps=50)
    @printf("  Price: %.4f ± %.4f\n", result.price, result.stderr)

    t = benchmark(() -> mc_price(S, T, EuropeanCall(K), dynamics; npaths=10000, nsteps=50);
                  n_runs=20, n_warmup=2)
    @printf("  Timing: %.2f ms (median)\n", t.median / 1000)
    results["mc_european_10k"] = t.median

    # -------------------------------------------------------------------------
    # Asian MC (10k paths)
    # -------------------------------------------------------------------------
    println("\n[2/3] MC Asian (10,000 paths)")

    result = mc_price(S, T, AsianCall(K), dynamics; npaths=10000, nsteps=50)
    @printf("  Price: %.4f ± %.4f\n", result.price, result.stderr)

    t = benchmark(() -> mc_price(S, T, AsianCall(K), dynamics; npaths=10000, nsteps=50);
                  n_runs=20, n_warmup=2)
    @printf("  Timing: %.2f ms (median)\n", t.median / 1000)
    results["mc_asian_10k"] = t.median

    # -------------------------------------------------------------------------
    # American LSM (10k paths)
    # -------------------------------------------------------------------------
    println("\n[3/3] American LSM (10,000 paths, 50 steps)")

    result = lsm_price(S, T, AmericanPut(K), dynamics; npaths=10000, nsteps=50)
    @printf("  Price: %.4f ± %.4f\n", result.price, result.stderr)

    t = benchmark(() -> lsm_price(S, T, AmericanPut(K), dynamics; npaths=10000, nsteps=50);
                  n_runs=10, n_warmup=2)
    @printf("  Timing: %.2f ms (median)\n", t.median / 1000)
    results["lsm_10k"] = t.median

    return results
end

# =============================================================================
# Backtesting Benchmarks
# =============================================================================

function run_backtest_benchmarks()
    println("\n" * "=" ^ 70)
    println("BACKTESTING BENCHMARKS")
    println("=" ^ 70)

    Random.seed!(42)
    results = Dict{String, Float64}()

    # Generate synthetic data (5 years daily)
    n_days = 252 * 5
    dates = Date(2019, 1, 1) .+ Day.(0:n_days-1)
    returns_data = 0.0004 .+ 0.02 .* randn(n_days)
    prices = 100.0 .* cumprod(1 .+ returns_data)

    # -------------------------------------------------------------------------
    # SMA Crossover Strategy
    # -------------------------------------------------------------------------
    println("\n[1/4] SMA Crossover Backtest (5 years)")

    function sma_crossover()
        fast = [i < 20 ? NaN : mean(prices[i-19:i]) for i in 1:n_days]
        slow = [i < 50 ? NaN : mean(prices[i-49:i]) for i in 1:n_days]
        signal = fast .> slow
        strat_returns = returns_data .* [i == 1 ? false : signal[i-1] for i in 1:n_days]
        return prod(1 .+ strat_returns)
    end

    t = benchmark(sma_crossover; n_runs=100, n_warmup=10)
    @printf("  Timing: %.1f μs (median)\n", t.median)
    results["sma_crossover"] = t.median

    # -------------------------------------------------------------------------
    # Rolling Sharpe
    # -------------------------------------------------------------------------
    println("\n[2/4] Rolling Sharpe (252-day window)")

    function rolling_sharpe()
        window = 252
        sharpes = fill(NaN, n_days)
        for i in window:n_days
            r = returns_data[i-window+1:i]
            sharpes[i] = mean(r) / std(r) * sqrt(252)
        end
        return sharpes
    end

    t = benchmark(rolling_sharpe; n_runs=100, n_warmup=10)
    @printf("  Timing: %.1f μs (median)\n", t.median)
    results["rolling_sharpe"] = t.median

    # -------------------------------------------------------------------------
    # Full Metrics
    # -------------------------------------------------------------------------
    println("\n[3/4] Full Backtest Metrics")

    function compute_all_metrics()
        sr = mean(returns_data) / std(returns_data) * sqrt(252)
        cum = cumprod(1 .+ returns_data)
        dd = cum ./ accumulate(max, cum) .- 1
        max_dd = minimum(dd)
        vol = std(returns_data) * sqrt(252)
        total_ret = cum[end] - 1
        return (sharpe=sr, max_dd=max_dd, vol=vol, total_ret=total_ret)
    end

    t = benchmark(compute_all_metrics; n_runs=1000, n_warmup=100)
    @printf("  Timing: %.1f μs (median)\n", t.median)
    results["full_metrics"] = t.median

    # -------------------------------------------------------------------------
    # Portfolio Rebalancing Simulation
    # -------------------------------------------------------------------------
    println("\n[4/4] Portfolio Rebalancing (10 assets)")

    # Multi-asset returns
    n_assets = 10
    multi_returns = 0.0004 .+ 0.02 .* randn(n_days, n_assets)

    function portfolio_rebalance()
        weights = fill(1.0 / n_assets, n_assets)
        port_returns = multi_returns * weights
        cum = cumprod(1 .+ port_returns)
        return cum[end]
    end

    t = benchmark(portfolio_rebalance; n_runs=1000, n_warmup=100)
    @printf("  Timing: %.1f μs (median)\n", t.median)
    results["portfolio_rebalance"] = t.median

    return results
end

# =============================================================================
# Factor Model Benchmarks
# =============================================================================

function run_factor_benchmarks()
    println("\n" * "=" ^ 70)
    println("FACTOR MODEL BENCHMARKS")
    println("=" ^ 70)

    Random.seed!(42)
    results = Dict{String, Float64}()

    # Generate synthetic factor data (5 years)
    n = 252 * 5

    mkt = 0.0003 .+ 0.01 .* randn(n)
    smb = 0.0001 .+ 0.005 .* randn(n)
    hml = 0.00005 .+ 0.005 .* randn(n)

    # Strategy returns with known loadings
    alpha = 0.0002
    returns = alpha .+ 1.1 .* mkt .+ 0.3 .* smb .- 0.2 .* hml .+ 0.005 .* randn(n)

    # -------------------------------------------------------------------------
    # CAPM Regression
    # -------------------------------------------------------------------------
    println("\n[1/4] CAPM Regression")

    result = capm_regression(returns, mkt)
    @printf("  Alpha: %.4f, Beta: %.4f, R²: %.4f\n", result.alpha, result.beta, result.r_squared)

    t = benchmark(() -> capm_regression(returns, mkt); n_runs=1000, n_warmup=100)
    @printf("  Timing: %.1f μs (median)\n", t.median)
    results["capm"] = t.median

    # -------------------------------------------------------------------------
    # Fama-French 3-Factor
    # -------------------------------------------------------------------------
    println("\n[2/4] Fama-French 3-Factor Regression")

    ff_result = fama_french_regression(returns, mkt, smb, hml)
    @printf("  Alpha: %.4f, MKT: %.4f, SMB: %.4f, HML: %.4f\n",
            ff_result.alpha, ff_result.market_beta, ff_result.smb_beta, ff_result.hml_beta)
    @printf("  R²: %.4f\n", ff_result.r_squared)

    t = benchmark(() -> fama_french_regression(returns, mkt, smb, hml); n_runs=1000, n_warmup=100)
    @printf("  Timing: %.1f μs (median)\n", t.median)
    results["ff3"] = t.median

    # -------------------------------------------------------------------------
    # Rolling Beta
    # -------------------------------------------------------------------------
    println("\n[3/4] Rolling Beta (60-day window)")

    t = benchmark(() -> rolling_beta(returns, mkt; window=60); n_runs=100, n_warmup=10)
    @printf("  Timing: %.1f μs (median)\n", t.median)
    results["rolling_beta"] = t.median

    # -------------------------------------------------------------------------
    # Information Coefficient
    # -------------------------------------------------------------------------
    println("\n[4/4] Information Coefficient")

    predictions = randn(1000)
    outcomes = 0.3 .* predictions .+ 0.7 .* randn(1000)

    ic = information_coefficient(predictions, outcomes)
    @printf("  IC: %.4f\n", ic)

    t = benchmark(() -> information_coefficient(predictions, outcomes); n_runs=10000, n_warmup=1000)
    @printf("  Timing: %.2f μs (median)\n", t.median)
    results["ic"] = t.median

    return results
end

# =============================================================================
# Statistics Benchmarks
# =============================================================================

function run_statistics_benchmarks()
    println("\n" * "=" ^ 70)
    println("STATISTICAL TESTING BENCHMARKS")
    println("=" ^ 70)

    Random.seed!(42)
    returns = 0.0004 .+ 0.01 .* randn(252 * 5)
    results = Dict{String, Float64}()

    # -------------------------------------------------------------------------
    # Sharpe Ratio
    # -------------------------------------------------------------------------
    println("\n[1/3] Sharpe Ratio")

    sr = sharpe_ratio(returns)
    @printf("  Sharpe: %.4f\n", sr)

    t = benchmark(() -> sharpe_ratio(returns); n_runs=10000, n_warmup=1000)
    @printf("  Timing: %.3f μs (median)\n", t.median)
    results["sharpe"] = t.median

    # -------------------------------------------------------------------------
    # Sharpe Confidence Interval
    # -------------------------------------------------------------------------
    println("\n[2/3] Sharpe Confidence Interval")

    ci = sharpe_confidence_interval(returns)
    @printf("  Sharpe: %.4f [%.4f, %.4f]\n", ci.sharpe, ci.lower, ci.upper)

    t = benchmark(() -> sharpe_confidence_interval(returns); n_runs=10000, n_warmup=1000)
    @printf("  Timing: %.3f μs (median)\n", t.median)
    results["sharpe_ci"] = t.median

    # -------------------------------------------------------------------------
    # Probabilistic Sharpe Ratio
    # -------------------------------------------------------------------------
    println("\n[3/3] Probabilistic Sharpe Ratio")

    psr = probabilistic_sharpe_ratio(returns, 0.5)
    @printf("  PSR (vs 0.5 benchmark): %.4f\n", psr)

    t = benchmark(() -> probabilistic_sharpe_ratio(returns, 0.5); n_runs=10000, n_warmup=1000)
    @printf("  Timing: %.3f μs (median)\n", t.median)
    results["psr"] = t.median

    return results
end

# =============================================================================
# Main
# =============================================================================

function main()
    println("=" ^ 70)
    println("QUANTNOVA.JL COMPREHENSIVE BENCHMARK SUITE")
    println("=" ^ 70)
    println("\nJulia version: $(VERSION)")
    println("QuantNova.jl: Differentiable Quantitative Finance\n")

    all_results = Dict{String, Float64}()

    merge!(all_results, run_pricing_benchmarks())
    merge!(all_results, run_greeks_benchmarks())
    merge!(all_results, run_mc_benchmarks())
    merge!(all_results, run_backtest_benchmarks())
    merge!(all_results, run_factor_benchmarks())
    merge!(all_results, run_statistics_benchmarks())

    # Summary
    println("\n" * "=" ^ 70)
    println("SUMMARY")
    println("=" ^ 70)
    println("\n  Benchmark                      Time")
    println("  " * "─" ^ 50)

    for (name, time) in sort(collect(all_results))
        if time > 1000
            @printf("  %-30s %10.2f ms\n", name, time / 1000)
        else
            @printf("  %-30s %10.2f μs\n", name, time)
        end
    end

    println("=" ^ 70)

    return all_results
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
