module Optimization

using ..Core: AbstractADBackend
using ..AD: gradient, current_backend, ForwardDiffBackend
using LinearAlgebra

# ============================================================================
# Objective Types
# ============================================================================

"""
    MeanVariance

Mean-variance optimization objective (Markowitz).
"""
struct MeanVariance
    expected_returns::Vector{Float64}
    cov_matrix::Matrix{Float64}
end

"""
    SharpeMaximizer

Maximize Sharpe ratio (non-convex).
"""
struct SharpeMaximizer
    expected_returns::Vector{Float64}
    cov_matrix::Matrix{Float64}
    rf::Float64

    SharpeMaximizer(μ, Σ; rf=0.0) = new(μ, Σ, rf)
end

"""
    CVaRObjective

Conditional Value at Risk optimization objective.
"""
struct CVaRObjective
    expected_returns::Vector{Float64}
    cov_matrix::Matrix{Float64}
    alpha::Float64

    CVaRObjective(μ, Σ; alpha=0.95) = new(μ, Σ, alpha)
end

"""
    KellyCriterion

Kelly criterion for optimal position sizing.
"""
struct KellyCriterion
    expected_returns::Vector{Float64}
    cov_matrix::Matrix{Float64}
end

"""
    OptimizationResult

Result of portfolio optimization.
"""
struct OptimizationResult
    weights::Vector{Float64}
    objective::Float64
    converged::Bool
    iterations::Int
end

# ============================================================================
# Optimize Interface
# ============================================================================

"""
    optimize(objective; kwargs...)

Optimize portfolio weights for given objective.
"""
function optimize end

# Mean-Variance with target return (gradient descent with penalties)
function optimize(mv::MeanVariance; target_return::Float64, max_iter=5000, tol=1e-10, lr=0.005)
    μ = mv.expected_returns
    Σ = mv.cov_matrix
    n = length(μ)

    # Initialize with equal weights
    w = ones(n) / n

    # Use higher penalty and adaptive learning rate
    penalty = 10000.0

    for i in 1:max_iter
        function obj(weights)
            # Variance
            var = weights' * Σ * weights
            # Return constraint penalty (squared)
            ret_penalty = penalty * (dot(weights, μ) - target_return)^2
            # Sum to 1 penalty
            sum_penalty = penalty * (sum(weights) - 1)^2
            # Non-negativity penalty
            neg_penalty = penalty * sum(max.(-weights, 0).^2)
            return var + ret_penalty + sum_penalty + neg_penalty
        end

        g = gradient(obj, w; backend=ForwardDiffBackend())

        # Adaptive learning rate (decrease over time)
        current_lr = lr / (1 + i * 0.0001)
        w_new = w - current_lr * g

        # Project to simplex (non-negative, sum to 1)
        w_new = max.(w_new, 0.0)
        if sum(w_new) > 0
            w_new = w_new / sum(w_new)
        else
            w_new = ones(n) / n
        end

        if norm(w_new - w) < tol
            variance = w_new' * Σ * w_new
            return OptimizationResult(w_new, variance, true, i)
        end

        w = w_new
    end

    variance = w' * Σ * w
    return OptimizationResult(w, variance, false, max_iter)
end

# Sharpe Maximizer (gradient-based)
function optimize(sm::SharpeMaximizer; max_iter=1000, tol=1e-8, lr=0.1)
    μ = sm.expected_returns
    Σ = sm.cov_matrix
    rf = sm.rf
    n = length(μ)

    # Initialize with equal weights
    w = ones(n) / n

    for i in 1:max_iter
        # Negative Sharpe (we minimize)
        function neg_sharpe(weights)
            ret = dot(weights, μ)
            vol = sqrt(weights' * Σ * weights)
            # Add small epsilon to avoid division by zero
            sharpe = (ret - rf) / (vol + 1e-10)

            # Penalties for constraints
            penalty = 100.0
            sum_penalty = penalty * (sum(weights) - 1)^2
            neg_penalty = penalty * sum(max.(-weights, 0).^2)

            return -sharpe + sum_penalty + neg_penalty
        end

        g = gradient(neg_sharpe, w; backend=ForwardDiffBackend())
        w_new = w - lr * g

        # Project to simplex
        w_new = max.(w_new, 0.0)
        if sum(w_new) > 0
            w_new = w_new / sum(w_new)
        else
            w_new = ones(n) / n
        end

        if norm(w_new - w) < tol
            ret = dot(w_new, μ)
            vol = sqrt(w_new' * Σ * w_new)
            sharpe = (ret - rf) / vol
            return OptimizationResult(w_new, sharpe, true, i)
        end

        w = w_new
    end

    ret = dot(w, μ)
    vol = sqrt(w' * Σ * w)
    sharpe = (ret - rf) / vol
    return OptimizationResult(w, sharpe, false, max_iter)
end

# ============================================================================
# Exports
# ============================================================================

export MeanVariance, SharpeMaximizer, CVaRObjective, KellyCriterion, OptimizationResult
export optimize

end
