using Test
using QuantNova
using Random
using Dates

Random.seed!(42)

@testset "Visualization" begin
    @testset "Theme Management" begin
        # Default theme is light
        theme = get_theme()
        @test theme[:backgroundcolor] == :white

        # Switch to dark theme
        set_theme!(:dark)
        theme = get_theme()
        @test theme[:backgroundcolor] == "#0d1117"

        # Switch back to light
        set_theme!(:light)
        theme = get_theme()
        @test theme[:backgroundcolor] == :white

        # Invalid theme throws error
        @test_throws ErrorException set_theme!(:invalid)
    end

    @testset "VisualizationSpec Creation" begin
        # Create a mock BacktestResult
        result = BacktestResult(
            10000.0,
            12000.0,
            [10000.0, 10500.0, 11000.0, 11500.0, 12000.0],
            [0.0, 0.05, 0.0476, 0.0454, 0.0435],
            [DateTime(2024, 1, i) for i in 1:5],
            Fill[],
            [Dict{Symbol,Float64}() for _ in 1:5],
            Dict{Symbol,Float64}(:sharpe => 1.5, :max_drawdown => 0.05)
        )

        # Default visualization
        spec = visualize(result)
        @test spec isa VisualizationSpec
        @test spec.view == :dashboard
        @test spec.data === result

        # Specific view
        spec = visualize(result, :drawdown)
        @test spec.view == :drawdown

        # Multiple views
        specs = visualize(result, [:equity, :drawdown])
        @test length(specs) == 2
        @test specs[1].view == :equity
        @test specs[2].view == :drawdown

        # With options
        spec = visualize(result, :equity; theme=:dark, title="My Chart")
        @test spec.options[:theme] == DARK_THEME
        @test spec.options[:title] == "My Chart"
    end

    @testset "Available Views" begin
        result = BacktestResult(
            10000.0, 12000.0,
            Float64[], Float64[], DateTime[], Fill[],
            Dict{Symbol,Float64}[], Dict{Symbol,Float64}()
        )
        views = available_views(result)
        @test :equity in views
        @test :drawdown in views
        @test :dashboard in views
    end

    @testset "LinkedContext" begin
        ctx = LinkedContext()
        @test ctx.time_range == (0.0, 1.0)
        @test ctx.cursor_time === nothing
        @test ctx.selected_asset === nothing
        @test ctx.zoom_level == 1.0

        # Modify context
        ctx.time_range = (0.5, 1.0)
        @test ctx.time_range == (0.5, 1.0)

        ctx.cursor_time = 0.75
        @test ctx.cursor_time == 0.75

        ctx.selected_asset = :AAPL
        @test ctx.selected_asset == :AAPL

        ctx.zoom_level = 2.0
        @test ctx.zoom_level == 2.0

        # Reset cursor
        ctx.cursor_time = nothing
        @test ctx.cursor_time === nothing
    end
end
