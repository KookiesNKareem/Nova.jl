# Live Data Integration Tests
#
# These tests require network access to Yahoo Finance.
# They are skipped by default in CI but can be run manually:
#   julia --project=. -e 'ENV["QUANTNOVA_TEST_LIVE_DATA"] = "1"; using Pkg; Pkg.test()'
#
# Or run this file directly:
#   julia --project=. test/live_data.jl

using Test
using QuantNova
using Statistics
using Dates
using LinearAlgebra

const RUN_LIVE_TESTS = get(ENV, "QUANTNOVA_TEST_LIVE_DATA", "0") == "1"

@testset "Live Data Integration" begin

    if !RUN_LIVE_TESTS
        @info "Skipping live data tests. Set QUANTNOVA_TEST_LIVE_DATA=1 to enable."
        @test true  # Placeholder so the testset isn't empty
        return  # Exit early - skip all live tests
    end

        @testset "fetch_prices" begin
            # Fetch 1 month of AAPL data
            prices = fetch_prices("AAPL", range="1mo")

            @test prices isa PriceHistory
            @test prices.symbol == "AAPL"
            @test length(prices) > 15  # At least 15 trading days in a month
            @test all(prices.close .> 0)
            @test all(prices.volume .>= 0)
            @test prices.timestamps[end] > prices.timestamps[1]

            # Test with date range
            start = Date(2024, 1, 1)
            stop = Date(2024, 6, 30)
            prices_range = fetch_prices("AAPL", startdt=start, enddt=stop)
            @test Date(prices_range.timestamps[1]) >= start
            @test Date(prices_range.timestamps[end]) <= stop
        end

        @testset "fetch_multiple" begin
            symbols = ["AAPL", "MSFT", "GOOGL"]
            histories = fetch_multiple(symbols, range="1mo")

            @test length(histories) == 3
            @test all(h -> h isa PriceHistory, histories)

            # Check alignment - all should have same timestamps
            @test all(h -> h.timestamps == histories[1].timestamps, histories)

            # Check symbols preserved
            fetched_symbols = [h.symbol for h in histories]
            @test Set(fetched_symbols) == Set(symbols)
        end

        @testset "fetch_returns" begin
            rets = fetch_returns("AAPL", range="3mo", type=:simple)

            @test rets isa Vector{Float64}
            @test length(rets) > 50  # ~60 trading days in 3 months
            @test all(isfinite, rets)

            # Log returns
            log_rets = fetch_returns("AAPL", range="3mo", type=:log)
            @test length(log_rets) == length(rets)
        end

        @testset "fetch_return_matrix" begin
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
            R = fetch_return_matrix(symbols, range="6mo")

            @test R isa Matrix{Float64}
            @test size(R, 2) == 4  # 4 assets
            @test size(R, 1) > 100  # ~125 trading days in 6 months
            @test all(isfinite, R)

            # Check correlation is reasonable
            corr = cor(R)
            @test all(diag(corr) .≈ 1.0)
            @test all(-1 .<= corr .<= 1)
        end

        @testset "to_backtest_format" begin
            histories = fetch_multiple(["AAPL", "MSFT"], range="1mo")
            timestamps, prices = to_backtest_format(histories)

            @test timestamps isa Vector{DateTime}
            @test prices isa Dict{Symbol,Vector{Float64}}
            @test haskey(prices, :AAPL)
            @test haskey(prices, :MSFT)
            @test length(prices[:AAPL]) == length(timestamps)
        end

        @testset "Portfolio Optimization Pipeline" begin
            # Fetch data
            symbols = ["AAPL", "MSFT", "GOOGL"]
            R = fetch_return_matrix(symbols, range="1y")

            # Estimate parameters
            μ = estimate_expected_returns(R)
            Σ = estimate_covariance(R, LedoitWolfShrinkage())

            @test length(μ) == 3
            @test size(Σ) == (3, 3)
            @test issymmetric(Σ)

            # Optimize
            result = optimize(MinimumVariance(Σ))

            @test result.converged
            @test length(result.weights) == 3
            @test sum(result.weights) ≈ 1.0 atol=1e-6
            @test all(result.weights .>= -1e-6)  # Long-only (within tolerance)

            # Analyze
            analytics = analyze_portfolio(result.weights, μ, Σ)
            @test analytics.expected_return isa Float64
            @test analytics.volatility > 0
        end

        @testset "Backtest Pipeline" begin
            # Fetch data
            symbols = ["AAPL", "MSFT"]
            histories = fetch_multiple(symbols, range="6mo")
            timestamps, prices = to_backtest_format(histories)

            # Create strategy
            target_weights = Dict(:AAPL => 0.6, :MSFT => 0.4)
            strategy = BuyAndHoldStrategy(target_weights)

            # Run backtest
            result = backtest(strategy, timestamps, prices, initial_cash=100_000.0)

            @test result isa BacktestResult
            @test result.initial_value == 100_000.0
            @test result.final_value > 0
            @test length(result.equity_curve) == length(timestamps)
            @test length(result.trades) >= 2  # At least bought both stocks

            # Check metrics
            @test haskey(result.metrics, :total_return)
            @test haskey(result.metrics, :sharpe_ratio)
            @test haskey(result.metrics, :max_drawdown)
        end

        @testset "Rebalancing Strategy" begin
            histories = fetch_multiple(["AAPL", "MSFT", "GOOGL"], range="1y")
            timestamps, prices = to_backtest_format(histories)

            target = Dict(:AAPL => 0.4, :MSFT => 0.3, :GOOGL => 0.3)
            strategy = RebalancingStrategy(
                target_weights=target,
                rebalance_frequency=:monthly,
                tolerance=0.05
            )

            result = backtest(strategy, timestamps, prices, initial_cash=100_000.0)

            @test result.final_value > 0
            # Monthly rebalancing should generate more trades than buy-and-hold
            @test length(result.trades) >= 3
        end

        @testset "Train/Test Split Optimization" begin
            # Fetch 2 years of data
            symbols = ["AAPL", "MSFT", "GOOGL"]
            histories = fetch_multiple(symbols, range="2y")

            n = length(histories[1])
            split_idx = div(n, 2)

            # Training data
            train_returns = Matrix{Float64}(undef, split_idx-1, length(symbols))
            for (j, ph) in enumerate(histories)
                train_ph = PriceHistory(ph.symbol, ph.timestamps[1:split_idx], ph.close[1:split_idx])
                train_returns[:, j] = returns(train_ph)
            end

            # Optimize on training data
            μ = estimate_expected_returns(train_returns)
            Σ = estimate_covariance(train_returns, LedoitWolfShrinkage())
            opt_result = optimize(MinimumVariance(Σ))

            @test opt_result.converged
            @test sum(opt_result.weights) ≈ 1.0 atol=1e-6

            # Backtest on test data
            test_timestamps = histories[1].timestamps[split_idx+1:end]
            test_prices = Dict{Symbol,Vector{Float64}}()
            for ph in histories
                test_prices[Symbol(ph.symbol)] = ph.close[split_idx+1:end]
            end

            optimal_weights = Dict(Symbol(symbols[i]) => opt_result.weights[i] for i in 1:length(symbols))
            strategy = BuyAndHoldStrategy(optimal_weights)
            bt_result = backtest(strategy, test_timestamps, test_prices)

            @test bt_result.final_value > 0
            @test haskey(bt_result.metrics, :sharpe_ratio)
        end

end  # testset

# Note: To run this file directly with live tests enabled:
#   QUANTNOVA_TEST_LIVE_DATA=1 julia --project=. test/live_data.jl
