using Test
using Quasar
using Dates
using Statistics: std

@testset "Scenario Analysis" begin
    @testset "Historical Stress Scenarios" begin
        # Built-in crisis scenarios
        @test haskey(CRISIS_SCENARIOS, :financial_crisis_2008)
        @test haskey(CRISIS_SCENARIOS, :covid_crash_2020)
        @test haskey(CRISIS_SCENARIOS, :dot_com_bust_2000)

        scenario = CRISIS_SCENARIOS[:financial_crisis_2008]
        @test scenario isa StressScenario
        @test haskey(scenario.shocks, :equity)
        @test scenario.shocks[:equity] < 0  # Equities dropped
    end

    @testset "Apply Stress Scenario" begin
        # Create a portfolio state
        state = SimulationState(
            timestamp=DateTime(2024, 1, 1),
            cash=50_000.0,
            positions=Dict(:SPY => 100.0, :TLT => 50.0),
            prices=Dict(:SPY => 450.0, :TLT => 100.0)
        )

        # Define asset classes
        asset_classes = Dict(:SPY => :equity, :TLT => :bond)

        # Apply 2008 crisis scenario
        scenario = CRISIS_SCENARIOS[:financial_crisis_2008]
        stressed_state = apply_scenario(scenario, state, asset_classes)

        # Equity should drop, bonds may rise
        @test stressed_state.prices[:SPY] < state.prices[:SPY]

        # Compute impact
        impact = scenario_impact(scenario, state, asset_classes)
        @test impact.pnl < 0  # Loss during crisis
        @test impact.pct_change < 0
    end

    @testset "All Crisis Scenarios" begin
        # Verify all built-in scenarios exist and have valid structure
        expected_scenarios = [
            :financial_crisis_2008,
            :covid_crash_2020,
            :dot_com_bust_2000,
            :black_monday_1987,
            :rate_shock_2022,
            :stagflation_1970s
        ]

        for scenario_key in expected_scenarios
            @test haskey(CRISIS_SCENARIOS, scenario_key)
            scenario = CRISIS_SCENARIOS[scenario_key]
            @test scenario isa StressScenario
            @test !isempty(scenario.name)
            @test !isempty(scenario.description)
            @test !isempty(scenario.shocks)
            @test scenario.duration_days > 0
        end
    end

    @testset "Scenario Impact Details" begin
        # Create a diversified portfolio
        state = SimulationState(
            timestamp=DateTime(2024, 1, 1),
            cash=10_000.0,
            positions=Dict(:SPY => 100.0, :TLT => 200.0, :GLD => 50.0),
            prices=Dict(:SPY => 450.0, :TLT => 100.0, :GLD => 180.0)
        )

        # SPY = equity, TLT = bond, GLD = gold
        asset_classes = Dict(:SPY => :equity, :TLT => :bond, :GLD => :gold)

        # Test 2008 scenario where bonds and gold rise
        scenario = CRISIS_SCENARIOS[:financial_crisis_2008]
        impact = scenario_impact(scenario, state, asset_classes)

        # Verify impact structure
        @test impact.scenario_name == "2008 Financial Crisis"
        @test impact.initial_value > 0
        @test impact.stressed_value > 0

        # Equity should have negative impact
        @test impact.asset_impacts[:SPY] < 0

        # Bond should have positive impact (2008 scenario has +10% bond shock)
        @test impact.asset_impacts[:TLT] > 0

        # Gold should have positive impact (2008 scenario has +5% gold shock)
        @test impact.asset_impacts[:GLD] > 0

        # Total P&L should match sum of asset impacts
        total_asset_pnl = sum(values(impact.asset_impacts))
        @test impact.pnl ≈ total_asset_pnl atol=1e-10
    end

    @testset "Default Asset Class" begin
        # Test that unknown assets default to equity class
        state = SimulationState(
            timestamp=DateTime(2024, 1, 1),
            cash=10_000.0,
            positions=Dict(:UNKNOWN => 100.0),
            prices=Dict(:UNKNOWN => 100.0)
        )

        # Empty asset class mapping - should default to equity
        asset_classes = Dict{Symbol,Symbol}()

        scenario = CRISIS_SCENARIOS[:financial_crisis_2008]
        stressed_state = apply_scenario(scenario, state, asset_classes)

        # Should apply equity shock (-50%) to unknown asset
        @test stressed_state.prices[:UNKNOWN] ≈ 50.0 atol=1e-10
    end

    @testset "Cash Preservation" begin
        # Verify that cash is not affected by stress scenarios
        state = SimulationState(
            timestamp=DateTime(2024, 1, 1),
            cash=100_000.0,
            positions=Dict(:SPY => 10.0),
            prices=Dict(:SPY => 450.0)
        )

        asset_classes = Dict(:SPY => :equity)
        scenario = CRISIS_SCENARIOS[:financial_crisis_2008]
        stressed_state = apply_scenario(scenario, state, asset_classes)

        @test stressed_state.cash == state.cash
    end

    @testset "Multiple Scenarios Comparison" begin
        state = SimulationState(
            timestamp=DateTime(2024, 1, 1),
            cash=50_000.0,
            positions=Dict(:SPY => 100.0),
            prices=Dict(:SPY => 450.0)
        )
        asset_classes = Dict(:SPY => :equity)

        # Compare different crisis severities
        impact_2008 = scenario_impact(CRISIS_SCENARIOS[:financial_crisis_2008], state, asset_classes)
        impact_covid = scenario_impact(CRISIS_SCENARIOS[:covid_crash_2020], state, asset_classes)
        impact_black_monday = scenario_impact(CRISIS_SCENARIOS[:black_monday_1987], state, asset_classes)

        # 2008 was worse than COVID which was worse than Black Monday for equities
        @test impact_2008.pct_change < impact_covid.pct_change
        @test impact_covid.pct_change < impact_black_monday.pct_change
    end

    @testset "Custom Hypothetical Scenarios" begin
        # Create custom scenario
        scenario = StressScenario(
            "Rate Hike Shock",
            "Hypothetical 500bp rate hike",
            Dict(:equity => -0.15, :bond => -0.25, :reit => -0.30),
            90
        )

        state = SimulationState(
            timestamp=DateTime(2024, 1, 1),
            cash=10_000.0,
            positions=Dict(:SPY => 50.0, :AGG => 100.0, :VNQ => 30.0),
            prices=Dict(:SPY => 450.0, :AGG => 100.0, :VNQ => 80.0)
        )

        asset_classes = Dict(:SPY => :equity, :AGG => :bond, :VNQ => :reit)

        impact = scenario_impact(scenario, state, asset_classes)
        @test impact.pct_change < 0

        # Multi-scenario comparison
        scenarios = [
            scenario,
            CRISIS_SCENARIOS[:financial_crisis_2008],
            CRISIS_SCENARIOS[:covid_crash_2020]
        ]

        comparison = compare_scenarios(scenarios, state, asset_classes)
        @test length(comparison) == 3
        @test all(c -> c isa ScenarioImpact, comparison)

        # Worst case
        worst = worst_case_scenario(scenarios, state, asset_classes)
        @test worst.pct_change == minimum(c.pct_change for c in comparison)
    end

    @testset "Sensitivity Analysis" begin
        state = SimulationState(
            timestamp=DateTime(2024, 1, 1),
            cash=10_000.0,
            positions=Dict(:SPY => 100.0),
            prices=Dict(:SPY => 450.0)
        )

        asset_classes = Dict(:SPY => :equity)

        # Sensitivity to equity shocks
        results = sensitivity_analysis(
            state,
            asset_classes,
            :equity;
            shock_range=-0.50:0.10:0.50
        )

        @test length(results) == 11  # -50% to +50% in 10% steps
        @test results[1].shock < results[end].shock
        @test results[1].portfolio_value < results[end].portfolio_value
    end

    @testset "Monte Carlo Projections" begin
        state = SimulationState(
            timestamp=DateTime(2024, 1, 1),
            cash=10_000.0,
            positions=Dict(:SPY => 100.0, :AGG => 50.0),
            prices=Dict(:SPY => 450.0, :AGG => 100.0)
        )

        # Define expected returns and covariance
        expected_returns = Dict(:SPY => 0.08, :AGG => 0.03)  # Annual
        volatilities = Dict(:SPY => 0.18, :AGG => 0.05)
        correlation = 0.2

        # Run Monte Carlo projection
        projection = monte_carlo_projection(
            state,
            expected_returns,
            volatilities;
            correlation=correlation,
            horizon_days=252,
            n_simulations=1000
        )

        @test projection isa ProjectionResult
        @test length(projection.terminal_values) == 1000
        @test projection.horizon_days == 252

        # VaR and CVaR
        @test projection.var_95 < portfolio_value(state)  # VaR is a loss threshold
        @test projection.cvar_95 < projection.var_95      # CVaR is worse than VaR

        # Percentiles
        @test projection.percentile_5 < projection.percentile_50
        @test projection.percentile_50 < projection.percentile_95
    end

    @testset "Projection with Correlated Assets" begin
        state = SimulationState(
            timestamp=DateTime(2024, 1, 1),
            cash=0.0,
            positions=Dict(:A => 100.0, :B => 100.0),
            prices=Dict(:A => 100.0, :B => 100.0)
        )

        expected_returns = Dict(:A => 0.10, :B => 0.10)
        volatilities = Dict(:A => 0.20, :B => 0.20)

        # Perfectly correlated
        proj_corr = monte_carlo_projection(
            state, expected_returns, volatilities;
            correlation=0.99, n_simulations=500
        )

        # Negatively correlated
        proj_neg = monte_carlo_projection(
            state, expected_returns, volatilities;
            correlation=-0.99, n_simulations=500
        )

        # Negative correlation should reduce volatility
        @test std(proj_neg.terminal_values) < std(proj_corr.terminal_values)
    end
end
