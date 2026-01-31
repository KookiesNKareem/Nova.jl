using Test
using Quasar
using Dates

@testset "Backtesting" begin
    @testset "Strategy Interface" begin
        # Simple buy-and-hold strategy
        strategy = BuyAndHoldStrategy(Dict(:AAPL => 0.6, :GOOGL => 0.4))

        state = SimulationState(
            timestamp=DateTime(2024, 1, 1),
            cash=100_000.0,
            positions=Dict{Symbol,Float64}(),
            prices=Dict(:AAPL => 150.0, :GOOGL => 140.0)
        )

        # Strategy should generate orders to reach target allocation
        orders = generate_orders(strategy, state)
        @test length(orders) == 2

        # After executing, should have target weights
        @test any(o -> o.symbol == :AAPL && o.side == :buy, orders)
        @test any(o -> o.symbol == :GOOGL && o.side == :buy, orders)
    end

    @testset "BuyAndHoldStrategy validation" begin
        # Weights must sum to 1.0
        @test_throws ErrorException BuyAndHoldStrategy(Dict(:AAPL => 0.5, :GOOGL => 0.6))

        # Valid strategy
        strategy = BuyAndHoldStrategy(Dict(:AAPL => 0.7, :GOOGL => 0.3))
        @test strategy.target_weights[:AAPL] == 0.7
        @test strategy.target_weights[:GOOGL] == 0.3
    end

    @testset "BuyAndHoldStrategy invests once" begin
        strategy = BuyAndHoldStrategy(Dict(:AAPL => 1.0))

        state = SimulationState(
            timestamp=DateTime(2024, 1, 1),
            cash=10_000.0,
            positions=Dict{Symbol,Float64}(),
            prices=Dict(:AAPL => 100.0)
        )

        # First call generates orders
        orders1 = generate_orders(strategy, state)
        @test length(orders1) == 1
        @test orders1[1].symbol == :AAPL

        # Second call returns no orders (already invested)
        orders2 = generate_orders(strategy, state)
        @test isempty(orders2)
    end

    @testset "Rebalancing Strategy" begin
        # Periodic rebalancing
        strategy = RebalancingStrategy(
            target_weights=Dict(:AAPL => 0.5, :GOOGL => 0.5),
            rebalance_frequency=:monthly
        )

        # State where we need rebalancing (off target)
        state = SimulationState(
            timestamp=DateTime(2024, 1, 1),  # First of month
            cash=10_000.0,
            positions=Dict(:AAPL => 100.0, :GOOGL => 50.0),  # Unbalanced
            prices=Dict(:AAPL => 150.0, :GOOGL => 140.0)
        )

        # Should detect rebalance needed
        @test should_rebalance(strategy, state)

        orders = generate_orders(strategy, state)
        @test !isempty(orders)
    end

    @testset "RebalancingStrategy validation" begin
        # Weights must sum to 1.0
        @test_throws ErrorException RebalancingStrategy(
            target_weights=Dict(:AAPL => 0.3, :GOOGL => 0.3),
            rebalance_frequency=:monthly
        )

        # Invalid frequency
        @test_throws ErrorException RebalancingStrategy(
            target_weights=Dict(:AAPL => 0.5, :GOOGL => 0.5),
            rebalance_frequency=:yearly
        )
    end

    @testset "RebalancingStrategy frequency" begin
        strategy = RebalancingStrategy(
            target_weights=Dict(:AAPL => 0.5, :GOOGL => 0.5),
            rebalance_frequency=:monthly,
            tolerance=0.01
        )

        # Initial state - should rebalance since no positions
        state1 = SimulationState(
            timestamp=DateTime(2024, 1, 1),
            cash=10_000.0,
            positions=Dict{Symbol,Float64}(),
            prices=Dict(:AAPL => 100.0, :GOOGL => 100.0)
        )
        @test should_rebalance(strategy, state1)

        # Trigger first rebalance
        orders1 = generate_orders(strategy, state1)
        @test length(orders1) == 2

        # Same month - should NOT rebalance even if unbalanced
        state2 = SimulationState(
            timestamp=DateTime(2024, 1, 15),
            cash=0.0,
            positions=Dict(:AAPL => 60.0, :GOOGL => 40.0),  # Off target
            prices=Dict(:AAPL => 100.0, :GOOGL => 100.0)
        )
        @test !should_rebalance(strategy, state2)

        # New month - should rebalance
        state3 = SimulationState(
            timestamp=DateTime(2024, 2, 1),
            cash=0.0,
            positions=Dict(:AAPL => 60.0, :GOOGL => 40.0),
            prices=Dict(:AAPL => 100.0, :GOOGL => 100.0)
        )
        @test should_rebalance(strategy, state3)
    end

    @testset "RebalancingStrategy tolerance" begin
        # High tolerance - won't rebalance small drifts
        strategy = RebalancingStrategy(
            target_weights=Dict(:AAPL => 0.5, :GOOGL => 0.5),
            rebalance_frequency=:daily,
            tolerance=0.20
        )

        # Slightly off target but within tolerance
        state = SimulationState(
            timestamp=DateTime(2024, 1, 1),
            cash=0.0,
            positions=Dict(:AAPL => 55.0, :GOOGL => 45.0),  # 55%/45%
            prices=Dict(:AAPL => 100.0, :GOOGL => 100.0)
        )
        @test !should_rebalance(strategy, state)

        # Far off target - outside tolerance
        state2 = SimulationState(
            timestamp=DateTime(2024, 1, 1),
            cash=0.0,
            positions=Dict(:AAPL => 80.0, :GOOGL => 20.0),  # 80%/20%
            prices=Dict(:AAPL => 100.0, :GOOGL => 100.0)
        )
        @test should_rebalance(strategy, state2)
    end

    @testset "RebalancingStrategy sell orders" begin
        strategy = RebalancingStrategy(
            target_weights=Dict(:AAPL => 0.5, :GOOGL => 0.5),
            rebalance_frequency=:daily,
            tolerance=0.01
        )

        # AAPL overweight, need to sell
        state = SimulationState(
            timestamp=DateTime(2024, 1, 1),
            cash=0.0,
            positions=Dict(:AAPL => 80.0, :GOOGL => 20.0),
            prices=Dict(:AAPL => 100.0, :GOOGL => 100.0)
        )

        orders = generate_orders(strategy, state)
        @test length(orders) == 2

        # Find AAPL order - should be sell
        aapl_order = filter(o -> o.symbol == :AAPL, orders)[1]
        @test aapl_order.side == :sell

        # Find GOOGL order - should be buy
        googl_order = filter(o -> o.symbol == :GOOGL, orders)[1]
        @test googl_order.side == :buy
    end

    @testset "AbstractStrategy default behavior" begin
        # Custom strategy for testing
        struct DummyStrategy <: AbstractStrategy end

        state = SimulationState(
            timestamp=DateTime(2024, 1, 1),
            cash=10_000.0,
            positions=Dict{Symbol,Float64}(),
            prices=Dict(:AAPL => 100.0)
        )

        # Default should_rebalance returns false
        @test !should_rebalance(DummyStrategy(), state)
    end
end
