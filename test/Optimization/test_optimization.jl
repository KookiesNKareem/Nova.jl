using Test
using Quasar
using LinearAlgebra

@testset "Optimization" begin
    @testset "Mean-Variance" begin
        # Simple 2-asset case
        expected_returns = [0.10, 0.15]  # 10% and 15% expected returns
        cov_matrix = [0.04 0.01; 0.01 0.09]  # 20% and 30% vol, 0.167 correlation

        # Optimize for minimum variance
        result = optimize(
            MeanVariance(expected_returns, cov_matrix),
            target_return=0.12
        )

        @test length(result.weights) == 2
        @test sum(result.weights) ≈ 1.0 atol=1e-8  # Weights sum to 1
        @test all(result.weights .>= -1e-8)  # No short selling (within tolerance)

        # Check return constraint is met (within numerical tolerance for gradient descent)
        @test dot(result.weights, expected_returns) ≈ 0.12 atol=1e-4

        # Check it's on efficient frontier (higher return = higher risk beyond min-variance point)
        result_high = optimize(
            MeanVariance(expected_returns, cov_matrix),
            target_return=0.14
        )

        var_12 = result.weights' * cov_matrix * result.weights
        var_14 = result_high.weights' * cov_matrix * result_high.weights

        @test var_14 > var_12  # Higher return = higher variance on efficient frontier
    end

    @testset "Gradient-based optimization" begin
        # Non-convex objective: maximize Sharpe ratio
        expected_returns = [0.10, 0.15, 0.12]
        cov_matrix = [
            0.04 0.01 0.02;
            0.01 0.09 0.01;
            0.02 0.01 0.05
        ]

        result = optimize(
            SharpeMaximizer(expected_returns, cov_matrix, rf=0.02)
        )

        @test length(result.weights) == 3
        @test sum(result.weights) ≈ 1.0 atol=1e-6
        @test result.objective > 0  # Positive Sharpe ratio
    end
end
