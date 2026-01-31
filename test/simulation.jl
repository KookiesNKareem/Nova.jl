using Test
using Quasar
using Dates

@testset "Simulation" begin
    @testset "SimulationState" begin
        state = SimulationState(
            timestamp=DateTime(2024, 1, 1),
            cash=100_000.0,
            positions=Dict(:AAPL => 100.0, :GOOGL => 50.0),
            prices=Dict(:AAPL => 150.0, :GOOGL => 140.0)
        )

        @test state.timestamp == DateTime(2024, 1, 1)
        @test state.cash == 100_000.0
        @test state.positions[:AAPL] == 100.0
        @test state.prices[:AAPL] == 150.0

        # Portfolio value computation
        @test portfolio_value(state) == 100_000.0 + 100*150 + 50*140  # 122,000
    end

    @testset "Execution Models" begin
        # Instant fill - no slippage
        instant = InstantFill()
        order = Order(:AAPL, 10.0, :buy)  # Buy 10 shares
        prices = Dict(:AAPL => 150.0)

        fill = execute(instant, order, prices)
        @test fill.quantity == 10.0
        @test fill.price == 150.0
        @test fill.cost == 1500.0

        # Slippage model
        slippage = SlippageModel(spread_bps=10.0)  # 10 bps spread
        fill_slip = execute(slippage, order, prices)
        @test fill_slip.price > 150.0  # Buy at higher price due to spread
        @test fill_slip.price ≈ 150.0 * 1.001 atol=0.01  # ~10 bps higher

        # Market impact model
        impact = MarketImpactModel(spread_bps=10.0, impact_bps_per_unit=0.1)
        large_order = Order(:AAPL, 1000.0, :buy)
        fill_impact = execute(impact, large_order, prices)
        @test fill_impact.price > fill_slip.price  # More slippage due to size
    end

    @testset "Simulation Driver" begin
        # Create a simple historical driver
        timestamps = [DateTime(2024, 1, i) for i in 1:5]
        price_series = Dict(
            :AAPL => [150.0, 152.0, 148.0, 155.0, 153.0],
            :GOOGL => [140.0, 142.0, 141.0, 145.0, 144.0]
        )

        driver = HistoricalDriver(timestamps, price_series)

        # Driver should be iterable
        states = collect(driver)
        @test length(states) == 5
        @test states[1].timestamp == DateTime(2024, 1, 1)
        @test states[1].prices[:AAPL] == 150.0
        @test states[3].prices[:GOOGL] == 141.0
    end

    @testset "SimulationResult" begin
        # Run a simple simulation
        timestamps = [DateTime(2024, 1, i) for i in 1:5]
        price_series = Dict(
            :AAPL => [150.0, 152.0, 148.0, 155.0, 153.0]
        )

        driver = HistoricalDriver(timestamps, price_series)
        initial_state = SimulationState(
            timestamp=DateTime(2024, 1, 1),
            cash=10_000.0,
            positions=Dict(:AAPL => 100.0),
            prices=Dict(:AAPL => 150.0)
        )

        result = simulate(driver, initial_state)

        @test length(result.states) == 5
        @test result.initial_value == 10_000.0 + 100 * 150.0
        @test result.final_value == 10_000.0 + 100 * 153.0
        @test length(result.returns) == 4  # n-1 returns
    end
end
