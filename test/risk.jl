using Test
using Quasar
using Statistics
using Random

@testset "Risk Measures" begin
    Random.seed!(42)  # Fixed seed for reproducibility
    # Sample returns data
    returns = randn(1000) * 0.02  # Daily returns, ~2% vol

    @testset "VaR" begin
        var95 = compute(VaR(0.95), returns)
        var99 = compute(VaR(0.99), returns)

        # Higher confidence = higher VaR (more negative)
        @test var99 < var95

        # VaR should be negative (loss)
        @test var95 < 0
    end

    @testset "CVaR (Expected Shortfall)" begin
        cvar95 = compute(CVaR(0.95), returns)
        var95 = compute(VaR(0.95), returns)

        # CVaR >= VaR (expected loss beyond VaR)
        @test cvar95 <= var95
    end

    @testset "Volatility" begin
        vol = compute(Volatility(), returns)
        @test vol ≈ std(returns) atol=1e-10
        @test vol > 0
    end

    @testset "Sharpe Ratio" begin
        # Use returns with clear positive mean
        good_returns = 0.001 .+ randn(1000) * 0.02
        sharpe = compute(Sharpe(rf=0.0), good_returns)

        @test sharpe > 0  # Positive excess returns
    end

    @testset "Max Drawdown" begin
        # Create returns with known drawdown
        prices = cumprod(1 .+ returns)
        mdd = compute(MaxDrawdown(), returns)

        @test mdd <= 0  # Drawdown is negative
        @test mdd >= -1  # Can't lose more than 100%
    end

    @testset "Custom Risk Measure" begin
        # User-defined risk measure
        struct DownsideVol <: AbstractRiskMeasure end

        function Quasar.compute(::DownsideVol, returns)
            negative_returns = filter(r -> r < 0, returns)
            return std(negative_returns)
        end

        dv = compute(DownsideVol(), returns)
        @test dv > 0
    end
end
