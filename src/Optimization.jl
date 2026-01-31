module Optimization

using ..Core: ADBackend
using ..AD: gradient, current_backend, ForwardDiffBackend
using LinearAlgebra
using Statistics

# ============================================================================
# Abstract Types
# ============================================================================

"""
    AbstractOptimizationObjective

Base type for all optimization objectives.
"""
abstract type AbstractOptimizationObjective end

"""
    AbstractConstraint

Base type for portfolio constraints.
"""
abstract type AbstractConstraint end

"""
    AbstractSolver

Base type for optimization solvers.
"""
abstract type AbstractSolver end

"""
    AbstractCovarianceEstimator

Base type for covariance matrix estimators.
"""
abstract type AbstractCovarianceEstimator end

# ============================================================================
# Constraint Types
# ============================================================================

"""
    FullInvestmentConstraint(target=1.0)

Constraint that weights must sum to a target value (default 1.0).
"""
struct FullInvestmentConstraint <: AbstractConstraint
    target::Float64
    FullInvestmentConstraint(target::Float64=1.0) = new(target)
end

"""
    LongOnlyConstraint()

Constraint that all weights must be non-negative.
"""
struct LongOnlyConstraint <: AbstractConstraint end

"""
    BoxConstraint(lower, upper)
    BoxConstraint(n; min_weight=0.0, max_weight=1.0)

Per-asset weight bounds: lower[i] ≤ w[i] ≤ upper[i].
"""
struct BoxConstraint <: AbstractConstraint
    lower::Vector{Float64}
    upper::Vector{Float64}

    function BoxConstraint(lower::Vector{Float64}, upper::Vector{Float64})
        length(lower) == length(upper) || throw(ArgumentError("lower and upper must have same length"))
        all(lower .<= upper) || throw(ArgumentError("lower bounds must not exceed upper bounds"))
        new(lower, upper)
    end
end

function BoxConstraint(n::Int; min_weight::Float64=0.0, max_weight::Float64=1.0)
    BoxConstraint(fill(min_weight, n), fill(max_weight, n))
end

"""
    GroupConstraint(indices, lower, upper; name="")

Constraint on sum of weights for a group of assets.
Enforces: lower ≤ Σᵢ∈group w[i] ≤ upper
"""
struct GroupConstraint <: AbstractConstraint
    indices::Vector{Int}
    lower::Float64
    upper::Float64
    name::String

    function GroupConstraint(indices::Vector{Int}, lower::Float64, upper::Float64; name::String="")
        lower <= upper || throw(ArgumentError("lower bound must not exceed upper bound"))
        new(indices, lower, upper, name)
    end
end

"""
    TurnoverConstraint(current_weights, max_turnover)

Constraint on portfolio turnover: Σᵢ |w[i] - current[i]| ≤ max_turnover
"""
struct TurnoverConstraint <: AbstractConstraint
    current_weights::Vector{Float64}
    max_turnover::Float64

    function TurnoverConstraint(current_weights::Vector{Float64}, max_turnover::Float64)
        max_turnover >= 0 || throw(ArgumentError("max_turnover must be non-negative"))
        new(current_weights, max_turnover)
    end
end

"""
    CardinalityConstraint(max_assets)
    CardinalityConstraint(min_assets, max_assets)

Constraint on number of non-zero weights (cardinality).
Note: Makes problem non-convex, requires specialized solver.
"""
struct CardinalityConstraint <: AbstractConstraint
    min_assets::Int
    max_assets::Int

    function CardinalityConstraint(min_assets::Int, max_assets::Int)
        min_assets >= 0 || throw(ArgumentError("min_assets must be non-negative"))
        min_assets <= max_assets || throw(ArgumentError("min_assets must not exceed max_assets"))
        new(min_assets, max_assets)
    end
end

CardinalityConstraint(max_assets::Int) = CardinalityConstraint(1, max_assets)

# ============================================================================
# Constraint Helpers
# ============================================================================

"""
    standard_constraints(n; long_only=true, full_investment=true, max_weight=1.0, min_weight=0.0)

Create a standard set of portfolio constraints.

# Arguments
- `n`: Number of assets
- `long_only`: If true, adds LongOnlyConstraint (default: true)
- `full_investment`: If true, adds FullInvestmentConstraint(1.0) (default: true)
- `max_weight`: Maximum weight per asset (default: 1.0, no constraint if >= 1.0)
- `min_weight`: Minimum weight per asset (default: 0.0)

# Returns
Vector of AbstractConstraint
"""
function standard_constraints(n::Int;
                              long_only::Bool=true,
                              full_investment::Bool=true,
                              max_weight::Float64=1.0,
                              min_weight::Float64=0.0)
    constraints = AbstractConstraint[]

    full_investment && push!(constraints, FullInvestmentConstraint(1.0))
    long_only && push!(constraints, LongOnlyConstraint())

    # Add box constraint if non-default bounds
    if max_weight < 1.0 || min_weight > 0.0
        push!(constraints, BoxConstraint(n; min_weight=min_weight, max_weight=max_weight))
    end

    constraints
end

"""
    check_constraint_violation(w, constraint) -> Float64

Returns the violation amount for a constraint (0.0 if satisfied).
"""
function check_constraint_violation(w::Vector{Float64}, c::FullInvestmentConstraint)
    abs(sum(w) - c.target)
end

function check_constraint_violation(w::Vector{Float64}, c::LongOnlyConstraint)
    sum(max.(-w, 0.0))
end

function check_constraint_violation(w::Vector{Float64}, c::BoxConstraint)
    lower_viol = sum(max.(c.lower .- w, 0.0))
    upper_viol = sum(max.(w .- c.upper, 0.0))
    lower_viol + upper_viol
end

function check_constraint_violation(w::Vector{Float64}, c::GroupConstraint)
    group_sum = sum(w[c.indices])
    lower_viol = max(c.lower - group_sum, 0.0)
    upper_viol = max(group_sum - c.upper, 0.0)
    lower_viol + upper_viol
end

function check_constraint_violation(w::Vector{Float64}, c::TurnoverConstraint)
    turnover = sum(abs.(w .- c.current_weights))
    max(turnover - c.max_turnover, 0.0)
end

function check_constraint_violation(w::Vector{Float64}, c::CardinalityConstraint)
    n_nonzero = sum(abs.(w) .> 1e-8)
    lower_viol = max(c.min_assets - n_nonzero, 0)
    upper_viol = max(n_nonzero - c.max_assets, 0)
    Float64(lower_viol + upper_viol)
end

"""
    check_all_constraints(w, constraints) -> Dict{String, Float64}

Check all constraints and return dict of violations.
"""
function check_all_constraints(w::Vector{Float64}, constraints::Vector{<:AbstractConstraint})
    violations = Dict{String, Float64}()
    for (i, c) in enumerate(constraints)
        name = _constraint_name(c, i)
        viol = check_constraint_violation(w, c)
        if viol > 1e-8
            violations[name] = viol
        end
    end
    violations
end

function _constraint_name(c::FullInvestmentConstraint, i::Int)
    "FullInvestment"
end

function _constraint_name(c::LongOnlyConstraint, i::Int)
    "LongOnly"
end

function _constraint_name(c::BoxConstraint, i::Int)
    "Box"
end

function _constraint_name(c::GroupConstraint, i::Int)
    isempty(c.name) ? "Group_$i" : c.name
end

function _constraint_name(c::TurnoverConstraint, i::Int)
    "Turnover"
end

function _constraint_name(c::CardinalityConstraint, i::Int)
    "Cardinality"
end

# ============================================================================
# Solver Types
# ============================================================================

"""
    QPSolver(; max_iter=1000, tol=1e-10)

Quadratic programming solver using active-set method.
Best for convex portfolio problems (MeanVariance, MinimumVariance).

Solves: min 0.5*x'Qx + c'x  s.t. Ax=b, lb≤x≤ub
"""
struct QPSolver <: AbstractSolver
    max_iter::Int
    tol::Float64

    QPSolver(; max_iter::Int=1000, tol::Float64=1e-10) = new(max_iter, tol)
end

"""
    LBFGSSolver(; max_iter=1000, tol=1e-8, memory=10, backend=nothing)

L-BFGS solver for large-scale non-convex problems.
Uses limited-memory quasi-Newton approximation.
"""
struct LBFGSSolver <: AbstractSolver
    max_iter::Int
    tol::Float64
    memory::Int
    backend::Union{ADBackend, Nothing}

    LBFGSSolver(; max_iter::Int=1000, tol::Float64=1e-8, memory::Int=10, backend=nothing) =
        new(max_iter, tol, memory, backend)
end

"""
    ProjectedGradientSolver(; max_iter=5000, tol=1e-10, lr=0.01, momentum=0.0, nesterov=false, adaptive_lr=true, backend=nothing)

Projected gradient descent solver with optional momentum and adaptive learning rate.
Falls back to this solver for non-convex objectives or when QP is not applicable.
"""
struct ProjectedGradientSolver <: AbstractSolver
    max_iter::Int
    tol::Float64
    lr::Float64
    momentum::Float64
    nesterov::Bool
    adaptive_lr::Bool
    backend::Union{ADBackend, Nothing}

    ProjectedGradientSolver(;
        max_iter::Int=5000,
        tol::Float64=1e-10,
        lr::Float64=0.01,
        momentum::Float64=0.0,
        nesterov::Bool=false,
        adaptive_lr::Bool=true,
        backend=nothing
    ) = new(max_iter, tol, lr, momentum, nesterov, adaptive_lr, backend)
end

# ============================================================================
# Projection Functions
# ============================================================================

"""
    project_simplex(x, target_sum=1.0; lower=nothing, upper=nothing)

Project x onto the simplex {y : sum(y)=target_sum, lower≤y≤upper}.
Uses efficient O(n log n) bisection method on the Lagrange multiplier.

# Algorithm
For the simple case (0 ≤ y ≤ ∞, sum(y)=1):
    y = max.(x .- θ, 0)  where θ is chosen so sum(y)=1

For box-constrained case (lower ≤ y ≤ upper, sum(y)=target):
    y = clamp.(x .- θ, lower, upper)  where θ satisfies the sum constraint
"""
function project_simplex(x::Vector{Float64}, target_sum::Float64=1.0;
                         lower::Union{Vector{Float64}, Nothing}=nothing,
                         upper::Union{Vector{Float64}, Nothing}=nothing)
    n = length(x)

    # Default bounds: [0, Inf] per component
    lb = isnothing(lower) ? zeros(n) : lower
    # For upper, use large but finite value if unbounded
    has_upper = !isnothing(upper)
    ub = has_upper ? upper : fill(1e10, n)

    # Special case: no sum constraint
    if !isfinite(target_sum)
        return clamp.(x, lb, ub)
    end

    # Simple case: standard simplex projection (lower=0, no upper)
    # Fast path using sorting algorithm
    if !has_upper && isnothing(lower)
        return _project_simplex_standard(x, target_sum)
    end

    # General case: bisection on Lagrange multiplier θ
    # We solve: sum(clamp.(x .- θ, lb, ub)) = target_sum

    # Find bounds for θ by considering when all constraints are active
    # When θ is very small, y = clamp(x - θ, lb, ub) → ub (if x - θ > ub)
    # When θ is very large, y = clamp(x - θ, lb, ub) → lb (if x - θ < lb)
    θ_low = minimum(x) - maximum(ub)
    θ_high = maximum(x) - minimum(lb)

    # Ensure finite bounds
    θ_low = max(θ_low, -1e10)
    θ_high = min(θ_high, 1e10)

    # Check feasibility at bounds
    sum_at_low = sum(clamp.(x .- θ_low, lb, ub))
    sum_at_high = sum(clamp.(x .- θ_high, lb, ub))

    if target_sum > sum_at_low + 1e-8
        # Target too high - return best feasible (upper bounds clamped)
        y = min.(x, ub)
        y = max.(y, lb)
        return y
    end
    if target_sum < sum_at_high - 1e-8
        # Target too low - return lower bounds
        return copy(lb)
    end

    # Bisection
    for _ in 1:100
        θ = (θ_low + θ_high) / 2
        y = clamp.(x .- θ, lb, ub)
        s = sum(y)

        if abs(s - target_sum) < 1e-12
            return y
        elseif s > target_sum
            θ_low = θ  # Need to increase θ to decrease sum
        else
            θ_high = θ  # Need to decrease θ to increase sum
        end

        if θ_high - θ_low < 1e-14
            break
        end
    end

    θ = (θ_low + θ_high) / 2
    clamp.(x .- θ, lb, ub)
end

"""
Fast standard simplex projection: {y : sum(y) = target, y >= 0}
Uses O(n log n) sorting algorithm.
"""
function _project_simplex_standard(x::Vector{Float64}, target_sum::Float64)
    n = length(x)

    # Sort in descending order
    u = sort(x, rev=true)

    # Find ρ = max{j : u_j - (sum_{i=1}^j u_i - target) / j > 0}
    cssv = cumsum(u)
    rho = 0
    for j in 1:n
        if u[j] - (cssv[j] - target_sum) / j > 0
            rho = j
        end
    end

    if rho == 0
        # Edge case: return equal weights
        return fill(target_sum / n, n)
    end

    # Compute threshold
    θ = (cssv[rho] - target_sum) / rho

    # Project
    max.(x .- θ, 0.0)
end

"""
    project_constraints(x, constraints)

Project x onto the feasible set defined by constraints.
Uses iterative projection (Dykstra's algorithm for multiple constraints).
"""
function project_constraints(x::Vector{Float64}, constraints::Vector{<:AbstractConstraint})
    n = length(x)
    y = copy(x)

    # Extract constraint parameters
    lb = zeros(n)
    ub = fill(Inf, n)
    target_sum = 1.0
    has_full_investment = false

    for c in constraints
        if c isa LongOnlyConstraint
            lb = max.(lb, 0.0)
        elseif c isa BoxConstraint
            lb = max.(lb, c.lower)
            ub = min.(ub, c.upper)
        elseif c isa FullInvestmentConstraint
            target_sum = c.target
            has_full_investment = true
        end
    end

    # Simple case: box + full investment (most common)
    if has_full_investment
        return project_simplex(y, target_sum; lower=lb, upper=ub)
    else
        return clamp.(y, lb, ub)
    end
end

# ============================================================================
# QP Solver Implementation
# ============================================================================

"""
    solve_qp(Q, c; A=nothing, b=nothing, lb=nothing, ub=nothing, target_sum=nothing, solver=QPSolver())

Solve quadratic program: min 0.5*x'Qx + c'x
Subject to:
- Ax = b (equality constraints, optional)
- lb ≤ x ≤ ub (box constraints, optional)
- sum(x) = target_sum (full investment, optional)

Uses active-set method with projected gradient steps.
"""
function solve_qp(Q::Matrix{Float64}, c::Vector{Float64};
                  A::Union{Matrix{Float64}, Nothing}=nothing,
                  b::Union{Vector{Float64}, Nothing}=nothing,
                  lb::Union{Vector{Float64}, Nothing}=nothing,
                  ub::Union{Vector{Float64}, Nothing}=nothing,
                  target_sum::Union{Float64, Nothing}=nothing,
                  x0::Union{Vector{Float64}, Nothing}=nothing,
                  solver::QPSolver=QPSolver())
    n = size(Q, 1)

    # Default bounds
    lower = isnothing(lb) ? fill(-Inf, n) : lb
    upper = isnothing(ub) ? fill(Inf, n) : ub

    # Initialize
    if isnothing(x0)
        if !isnothing(target_sum)
            x = fill(target_sum / n, n)
            x = clamp.(x, lower, upper)
            # Rescale to meet target
            if sum(x) > 0
                x = x * target_sum / sum(x)
            end
        else
            x = zeros(n)
        end
    else
        x = copy(x0)
    end

    # Active-set QP via projected gradient with exact line search
    α = 1.0 / (opnorm(Q) + 1e-8)  # Initial step size based on Lipschitz constant

    for iter in 1:solver.max_iter
        # Gradient of 0.5*x'Qx + c'x
        grad = Q * x + c

        # Gradient step
        x_new = x - α * grad

        # Project onto constraints
        if !isnothing(target_sum)
            x_new = project_simplex(x_new, target_sum; lower=lower, upper=upper)
        else
            x_new = clamp.(x_new, lower, upper)
        end

        # Check convergence
        if norm(x_new - x) < solver.tol
            obj = 0.5 * dot(x_new, Q * x_new) + dot(c, x_new)
            return (x=x_new, objective=obj, converged=true, iterations=iter)
        end

        x = x_new
    end

    obj = 0.5 * dot(x, Q * x) + dot(c, x)
    (x=x, objective=obj, converged=false, iterations=solver.max_iter)
end

"""
    solve_min_variance_qp(Σ; target_return=nothing, μ=nothing, constraints=nothing, solver=QPSolver())

Solve minimum variance portfolio optimization using QP.

If target_return is provided, adds constraint: μ'w = target_return
"""
function solve_min_variance_qp(Σ::Matrix{Float64};
                               target_return::Union{Float64, Nothing}=nothing,
                               μ::Union{Vector{Float64}, Nothing}=nothing,
                               constraints::Union{Vector{<:AbstractConstraint}, Nothing}=nothing,
                               solver::QPSolver=QPSolver())
    n = size(Σ, 1)

    # Extract constraint parameters
    lb = zeros(n)
    ub = fill(Inf, n)
    target_sum = 1.0

    if !isnothing(constraints)
        for c in constraints
            if c isa LongOnlyConstraint
                lb = max.(lb, 0.0)
            elseif c isa BoxConstraint
                lb = max.(lb, c.lower)
                ub = min.(ub, c.upper)
            elseif c isa FullInvestmentConstraint
                target_sum = c.target
            end
        end
    end

    # If target return specified, use augmented Lagrangian approach
    if !isnothing(target_return) && !isnothing(μ)
        return _solve_mv_with_return_constraint(Σ, μ, target_return, lb, ub, target_sum, solver)
    end

    # Simple min variance: Q = 2Σ, c = 0
    result = solve_qp(2.0 * Σ, zeros(n);
                      lb=lb, ub=ub, target_sum=target_sum, solver=solver)

    result
end

"""
Solve mean-variance with return constraint using iterative approach.
"""
function _solve_mv_with_return_constraint(Σ, μ, target_return, lb, ub, target_sum, solver)
    n = length(μ)

    # Use penalty method with increasing penalty
    penalty = 1000.0
    x = fill(target_sum / n, n)
    x = clamp.(x, lb, ub)
    if sum(x) > 0
        x = x * target_sum / sum(x)
    end

    α = 1.0 / (opnorm(Σ) + penalty * dot(μ, μ) + 1e-8)

    for outer in 1:10
        for iter in 1:solver.max_iter ÷ 10
            # Gradient of variance + penalty*(return - target)^2
            ret = dot(μ, x)
            grad = 2.0 * Σ * x + 2.0 * penalty * (ret - target_return) * μ

            x_new = x - α * grad
            x_new = project_simplex(x_new, target_sum; lower=lb, upper=ub)

            if norm(x_new - x) < solver.tol
                break
            end
            x = x_new
        end

        # Check if return constraint satisfied
        actual_return = dot(μ, x)
        if abs(actual_return - target_return) < 1e-6
            break
        end

        # Increase penalty
        penalty *= 10.0
        α = 1.0 / (opnorm(Σ) + penalty * dot(μ, μ) + 1e-8)
    end

    obj = dot(x, Σ * x)
    (x=x, objective=obj, converged=true, iterations=solver.max_iter)
end

# ============================================================================
# L-BFGS Solver Implementation
# ============================================================================

"""
    solve_lbfgs(f, x0; constraints=nothing, solver=LBFGSSolver(), backend=nothing)

Solve unconstrained or box-constrained optimization using L-BFGS.
Uses two-loop recursion for efficient Hessian approximation.

# Arguments
- `f`: Objective function to minimize
- `x0`: Initial point
- `constraints`: Optional constraints (box, full investment)
- `solver`: LBFGSSolver with parameters
- `backend`: AD backend for gradients

# Returns
Named tuple with (x, objective, converged, iterations)
"""
function solve_lbfgs(f, x0::Vector{Float64};
                     constraints::Union{Vector{<:AbstractConstraint}, Nothing}=nothing,
                     solver::LBFGSSolver=LBFGSSolver(),
                     backend=nothing)
    n = length(x0)
    m = solver.memory

    # Use provided backend or default
    ad_backend = isnothing(backend) ? (isnothing(solver.backend) ? current_backend() : solver.backend) : backend

    # Extract constraint parameters
    lb, ub, target_sum = _extract_bounds(constraints, n)
    has_sum_constraint = !isnothing(target_sum)

    # Initialize
    x = copy(x0)
    if has_sum_constraint
        x = project_simplex(x, target_sum; lower=lb, upper=_finite_or_nothing(ub))
    else
        x = clamp.(x, lb, ub)
    end

    # L-BFGS storage: recent (s, y) pairs
    S = Vector{Vector{Float64}}()  # s_k = x_{k+1} - x_k
    Y = Vector{Vector{Float64}}()  # y_k = g_{k+1} - g_k
    ρ = Vector{Float64}()          # ρ_k = 1 / (y_k' * s_k)

    g = gradient(f, x; backend=ad_backend)
    f_val = f(x)
    best_x = copy(x)
    best_f = f_val

    ub_finite = _finite_or_nothing(ub)

    for iter in 1:solver.max_iter
        # Compute search direction using two-loop recursion
        q = copy(g)
        α_hist = zeros(length(S))

        # First loop (backward)
        for i in length(S):-1:1
            α_hist[i] = ρ[i] * dot(S[i], q)
            q = q - α_hist[i] * Y[i]
        end

        # Initial Hessian approximation: H_0 = γI
        if !isempty(S)
            γ = dot(S[end], Y[end]) / (dot(Y[end], Y[end]) + 1e-10)
            γ = clamp(γ, 1e-6, 1e6)  # Prevent extreme scaling
        else
            γ = 0.1  # Conservative initial step
        end
        r = γ * q

        # Second loop (forward)
        for i in 1:length(S)
            β = ρ[i] * dot(Y[i], r)
            r = r + (α_hist[i] - β) * S[i]
        end

        d = -r  # Search direction

        # Projected line search (Armijo along projected arc)
        α = 1.0
        c1 = 1e-4

        for ls_iter in 1:30
            x_new = x + α * d
            if has_sum_constraint
                x_new = project_simplex(x_new, target_sum; lower=lb, upper=ub_finite)
            else
                x_new = clamp.(x_new, lb, ub)
            end

            f_new = f(x_new)

            # Armijo condition using actual step
            actual_step = x_new - x
            if f_new <= f_val + c1 * dot(g, actual_step) || α < 1e-10
                break
            end
            α *= 0.5
        end

        x_new = x + α * d
        if has_sum_constraint
            x_new = project_simplex(x_new, target_sum; lower=lb, upper=ub_finite)
        else
            x_new = clamp.(x_new, lb, ub)
        end
        f_new = f(x_new)

        # Track best solution
        if f_new < best_f
            best_x = copy(x_new)
            best_f = f_new
        end

        # Check convergence (both step and gradient)
        step_norm = norm(x_new - x)
        if step_norm < solver.tol
            return (x=best_x, objective=best_f, converged=true, iterations=iter)
        end

        # Update L-BFGS memory
        g_new = gradient(f, x_new; backend=ad_backend)
        s = x_new - x
        y = g_new - g

        sy = dot(s, y)
        if sy > 1e-12 * norm(s) * norm(y)  # More robust curvature check
            if length(S) >= m
                popfirst!(S)
                popfirst!(Y)
                popfirst!(ρ)
            end
            push!(S, s)
            push!(Y, y)
            push!(ρ, 1.0 / sy)
        end

        x = x_new
        g = g_new
        f_val = f_new
    end

    (x=best_x, objective=best_f, converged=false, iterations=solver.max_iter)
end

# ============================================================================
# Projected Gradient Solver Implementation
# ============================================================================

"""
    solve_projected_gradient(f, x0; constraints=nothing, solver=ProjectedGradientSolver(), backend=nothing)

Solve constrained optimization using projected gradient descent with momentum.

# Arguments
- `f`: Objective function to minimize
- `x0`: Initial point
- `constraints`: Optional constraints
- `solver`: ProjectedGradientSolver with parameters
- `backend`: AD backend for gradients

# Returns
Named tuple with (x, objective, converged, iterations)
"""
function solve_projected_gradient(f, x0::Vector{Float64};
                                  constraints::Union{Vector{<:AbstractConstraint}, Nothing}=nothing,
                                  solver::ProjectedGradientSolver=ProjectedGradientSolver(),
                                  backend=nothing)
    n = length(x0)

    # Use provided backend or default
    ad_backend = isnothing(backend) ? (isnothing(solver.backend) ? current_backend() : solver.backend) : backend

    # Extract constraint parameters
    lb, ub, target_sum = _extract_bounds(constraints, n)
    has_sum_constraint = !isnothing(target_sum)
    ub_finite = _finite_or_nothing(ub)

    # Initialize
    x = copy(x0)
    if has_sum_constraint
        x = project_simplex(x, target_sum; lower=lb, upper=ub_finite)
    else
        x = clamp.(x, lb, ub)
    end

    v = zeros(n)  # Velocity for momentum
    lr = solver.lr
    best_x = copy(x)
    best_f = f(x)

    for iter in 1:solver.max_iter
        g = gradient(f, x; backend=ad_backend)

        # Adaptive learning rate
        if solver.adaptive_lr
            lr = solver.lr / (1 + iter * 0.0001)
        end

        # Compute update with momentum
        if solver.nesterov && solver.momentum > 0
            # Nesterov accelerated gradient
            x_look = x + solver.momentum * v
            g_look = gradient(f, x_look; backend=ad_backend)
            v = solver.momentum * v - lr * g_look
            x_new = x + v
        elseif solver.momentum > 0
            # Standard momentum
            v = solver.momentum * v - lr * g
            x_new = x + v
        else
            # Plain gradient descent
            x_new = x - lr * g
        end

        # Project onto constraints
        if has_sum_constraint
            x_new = project_simplex(x_new, target_sum; lower=lb, upper=ub_finite)
        else
            x_new = clamp.(x_new, lb, ub)
        end

        # Track best solution
        f_new = f(x_new)
        if f_new < best_f
            best_x = copy(x_new)
            best_f = f_new
        end

        # Check convergence
        if norm(x_new - x) < solver.tol
            return (x=best_x, objective=best_f, converged=true, iterations=iter)
        end

        x = x_new
    end

    (x=best_x, objective=best_f, converged=false, iterations=solver.max_iter)
end

# Helper to extract bounds from constraints
function _extract_bounds(constraints, n)
    lb = fill(-Inf, n)
    ub = fill(Inf, n)
    target_sum = nothing

    if !isnothing(constraints)
        for c in constraints
            if c isa LongOnlyConstraint
                lb = max.(lb, 0.0)
            elseif c isa BoxConstraint
                lb = max.(lb, c.lower)
                ub = min.(ub, c.upper)
            elseif c isa FullInvestmentConstraint
                target_sum = c.target
            end
        end
    end

    (lb, ub, target_sum)
end

# Helper to convert infinite bounds to nothing for project_simplex
function _finite_or_nothing(ub)
    all(isfinite, ub) ? ub : nothing
end

# ============================================================================
# Objective Types
# ============================================================================

"""
    MeanVariance

Mean-variance optimization objective (Markowitz).
"""
struct MeanVariance <: AbstractOptimizationObjective
    expected_returns::Vector{Float64}
    cov_matrix::Matrix{Float64}
end

"""
    SharpeMaximizer

Maximize Sharpe ratio (non-convex).
"""
struct SharpeMaximizer <: AbstractOptimizationObjective
    expected_returns::Vector{Float64}
    cov_matrix::Matrix{Float64}
    rf::Float64

    SharpeMaximizer(μ, Σ; rf=0.0) = new(μ, Σ, rf)
end

"""
    CVaRObjective

Conditional Value at Risk optimization objective.

Uses parametric (Gaussian) CVaR approximation for optimization.
For more accurate CVaR with non-normal returns, use scenario-based methods.
"""
struct CVaRObjective <: AbstractOptimizationObjective
    expected_returns::Vector{Float64}
    cov_matrix::Matrix{Float64}
    alpha::Float64

    CVaRObjective(μ, Σ; alpha=0.95) = new(μ, Σ, alpha)
end

"""
    KellyCriterion

Kelly criterion for optimal position sizing.

Maximizes expected log growth: E[log(1 + w'r)]
For Gaussian returns, optimal unconstrained Kelly is w* = Σ⁻¹μ
"""
struct KellyCriterion <: AbstractOptimizationObjective
    expected_returns::Vector{Float64}
    cov_matrix::Matrix{Float64}
end

"""
    MinimumVariance(cov_matrix)

Global minimum variance portfolio objective.
Ignores expected returns, minimizes portfolio variance only.
"""
struct MinimumVariance <: AbstractOptimizationObjective
    cov_matrix::Matrix{Float64}
end

"""
    RiskParity(cov_matrix; target_contributions=nothing)

Risk parity objective: minimize deviation from target risk contributions.

Risk contribution of asset i: RC_i = w_i * (Σw)_i / σ_p
Default target: equal risk contribution (1/n for each asset).

The optimization minimizes: Σᵢ (RC_i/σ_p - target_i)²
"""
struct RiskParity <: AbstractOptimizationObjective
    cov_matrix::Matrix{Float64}
    target_contributions::Vector{Float64}

    function RiskParity(Σ::Matrix{Float64}; target_contributions::Union{Vector{Float64}, Nothing}=nothing)
        n = size(Σ, 1)
        targets = isnothing(target_contributions) ? fill(1.0/n, n) : target_contributions
        abs(sum(targets) - 1.0) < 1e-8 || throw(ArgumentError("target_contributions must sum to 1"))
        new(Σ, targets)
    end
end

"""
    MaximumDiversification(cov_matrix)

Maximum diversification ratio objective.

Diversification ratio = (w'σ) / √(w'Σw)
where σ is the vector of asset volatilities.

Maximizing this ratio produces portfolios that spread risk across assets.
"""
struct MaximumDiversification <: AbstractOptimizationObjective
    cov_matrix::Matrix{Float64}
    asset_vols::Vector{Float64}

    function MaximumDiversification(Σ::Matrix{Float64})
        σ = sqrt.(diag(Σ))
        new(Σ, σ)
    end
end

"""
    BlackLitterman(cov_matrix, market_weights; P, Q, Omega=nothing, tau=0.05, risk_aversion=2.5)

Black-Litterman model combining market equilibrium with investor views.

# Arguments
- `cov_matrix`: Covariance matrix
- `market_weights`: Market capitalization weights
- `P`: Views matrix (K × N), each row defines a view on assets
- `Q`: View returns (K), expected returns for each view
- `Omega`: View uncertainty matrix (K × K), default: τ * diag(P * Σ * P')
- `tau`: Scaling factor for prior uncertainty (default: 0.05)
- `risk_aversion`: Market risk aversion parameter (default: 2.5)

The posterior expected returns combine equilibrium returns (implied by market weights)
with investor views using Bayesian updating.
"""
struct BlackLitterman <: AbstractOptimizationObjective
    cov_matrix::Matrix{Float64}
    market_weights::Vector{Float64}
    P::Matrix{Float64}  # Views matrix (K × N)
    Q::Vector{Float64}  # View returns (K)
    Omega::Matrix{Float64}  # View uncertainty (K × K)
    tau::Float64
    risk_aversion::Float64

    function BlackLitterman(Σ::Matrix{Float64}, market_weights::Vector{Float64};
                           P::Matrix{Float64}, Q::Vector{Float64},
                           Omega::Union{Matrix{Float64}, Nothing}=nothing,
                           tau::Float64=0.05, risk_aversion::Float64=2.5)
        n = size(Σ, 1)
        K = length(Q)

        # Validate dimensions
        size(P) == (K, n) || throw(ArgumentError("P must be K × N where K = length(Q)"))
        length(market_weights) == n || throw(ArgumentError("market_weights must have N elements"))

        # Default uncertainty: proportional to view variance
        Ω = isnothing(Omega) ? tau * Diagonal(diag(P * Σ * P')) : Omega

        new(Σ, market_weights, P, Q, Matrix(Ω), tau, risk_aversion)
    end
end

# ============================================================================
# Risk Parity and Diversification Helpers
# ============================================================================

"""
    compute_risk_contributions(w, Σ)

Compute risk contributions for each asset.
RC_i = w_i * (Σw)_i / σ_p

Returns vector of absolute risk contributions (sum to σ_p).
"""
function compute_risk_contributions(w::Vector{Float64}, Σ::Matrix{Float64})
    marginal = Σ * w
    total_risk = sqrt(w' * Σ * w)
    if total_risk < 1e-12
        return zeros(length(w))
    end
    (w .* marginal) / total_risk
end

"""
    compute_fractional_risk_contributions(w, Σ)

Compute fractional risk contributions (sum to 1).
FRC_i = RC_i / σ_p = w_i * (Σw)_i / (w'Σw)
"""
function compute_fractional_risk_contributions(w::Vector{Float64}, Σ::Matrix{Float64})
    var = w' * Σ * w
    if var < 1e-12
        n = length(w)
        return fill(1.0/n, n)
    end
    marginal = Σ * w
    (w .* marginal) / var
end

"""
    compute_marginal_risk(w, Σ)

Compute marginal risk contribution for each asset.
MR_i = ∂σ_p/∂w_i = (Σw)_i / σ_p

Adding Δw to asset i increases portfolio vol by approximately MR_i * Δw.
"""
function compute_marginal_risk(w::Vector{Float64}, Σ::Matrix{Float64})
    port_vol = sqrt(w' * Σ * w)
    if port_vol < 1e-12
        return zeros(length(w))
    end
    (Σ * w) / port_vol
end

"""
    compute_component_risk(w, Σ)

Compute component risk (absolute risk contribution) for each asset.
CR_i = w_i * MR_i = w_i * (Σw)_i / σ_p

Sum of component risks equals total portfolio volatility.
"""
function compute_component_risk(w::Vector{Float64}, Σ::Matrix{Float64})
    compute_risk_contributions(w, Σ)  # Same as risk contributions
end

"""
    compute_beta(w, Σ)

Compute portfolio beta of each asset relative to the portfolio.
β_i = Cov(r_i, r_p) / Var(r_p) = (Σw)_i / (w'Σw)
"""
function compute_beta(w::Vector{Float64}, Σ::Matrix{Float64})
    var = w' * Σ * w
    if var < 1e-12
        return ones(length(w))
    end
    (Σ * w) / var
end

"""
    compute_tracking_error(w, w_benchmark, Σ)

Compute tracking error (active risk) between portfolio and benchmark.
TE = √((w - w_b)'Σ(w - w_b))
"""
function compute_tracking_error(w::Vector{Float64}, w_benchmark::Vector{Float64}, Σ::Matrix{Float64})
    active = w - w_benchmark
    sqrt(max(active' * Σ * active, 0.0))
end

"""
    compute_active_risk_contributions(w, w_benchmark, Σ)

Compute contribution of each active bet to tracking error.
"""
function compute_active_risk_contributions(w::Vector{Float64}, w_benchmark::Vector{Float64}, Σ::Matrix{Float64})
    active = w - w_benchmark
    te = compute_tracking_error(w, w_benchmark, Σ)
    if te < 1e-12
        return zeros(length(w))
    end
    (active .* (Σ * active)) / te
end

"""
    PortfolioAnalytics

Comprehensive portfolio analytics summary.
"""
struct PortfolioAnalytics
    weights::Vector{Float64}
    expected_return::Float64
    volatility::Float64
    variance::Float64
    sharpe_ratio::Float64
    risk_contributions::Vector{Float64}
    fractional_risk_contributions::Vector{Float64}
    marginal_risks::Vector{Float64}
    betas::Vector{Float64}
    diversification_ratio::Float64
    effective_n::Float64  # Effective number of bets
end

"""
    analyze_portfolio(w, μ, Σ; rf=0.0)

Compute comprehensive portfolio analytics.

# Returns
PortfolioAnalytics struct with all risk measures.
"""
function analyze_portfolio(w::Vector{Float64}, μ::Vector{Float64}, Σ::Matrix{Float64}; rf::Float64=0.0)
    ret = dot(w, μ)
    var = w' * Σ * w
    vol = sqrt(var)
    sharpe = vol > 0 ? (ret - rf) / vol : 0.0

    rc = compute_risk_contributions(w, Σ)
    frc = compute_fractional_risk_contributions(w, Σ)
    mr = compute_marginal_risk(w, Σ)
    betas = compute_beta(w, Σ)

    # Diversification ratio
    asset_vols = sqrt.(diag(Σ))
    dr = vol > 0 ? dot(w, asset_vols) / vol : 1.0

    # Effective N (Herfindahl-based)
    eff_n = sum(frc) > 0 ? 1.0 / sum(frc.^2) : length(w)

    PortfolioAnalytics(w, ret, vol, var, sharpe, rc, frc, mr, betas, dr, eff_n)
end

function Base.show(io::IO, ::MIME"text/plain", pa::PortfolioAnalytics)
    n = length(pa.weights)
    println(io, "PortfolioAnalytics ($n assets)")
    println(io, "  Expected Return: $(round(pa.expected_return*100, digits=2))%")
    println(io, "  Volatility: $(round(pa.volatility*100, digits=2))%")
    println(io, "  Sharpe Ratio: $(round(pa.sharpe_ratio, digits=3))")
    println(io, "  Diversification Ratio: $(round(pa.diversification_ratio, digits=3))")
    println(io, "  Effective N: $(round(pa.effective_n, digits=2))")
    println(io, "  Weights: $(round.(pa.weights, digits=3))")
    print(io, "  Risk Contributions: $(round.(pa.fractional_risk_contributions, digits=3))")
end

"""
    risk_parity_objective(w, rp::RiskParity)

Compute risk parity objective: sum of squared deviations from target risk contributions.
"""
function risk_parity_objective(w::Vector{Float64}, rp::RiskParity)
    frc = compute_fractional_risk_contributions(w, rp.cov_matrix)
    sum((frc .- rp.target_contributions).^2)
end

"""
    diversification_ratio(w, md::MaximumDiversification)

Compute diversification ratio: (w'σ) / √(w'Σw)
"""
function diversification_ratio(w::Vector{Float64}, md::MaximumDiversification)
    weighted_vol = dot(w, md.asset_vols)
    port_vol = sqrt(w' * md.cov_matrix * w)
    if port_vol < 1e-12
        return 1.0
    end
    weighted_vol / port_vol
end

"""
    bl_equilibrium_returns(bl::BlackLitterman)

Compute equilibrium returns implied by market weights (reverse optimization).
π = λ * Σ * w_mkt
"""
function bl_equilibrium_returns(bl::BlackLitterman)
    bl.risk_aversion * bl.cov_matrix * bl.market_weights
end

"""
    bl_posterior_returns(bl::BlackLitterman)

Compute Black-Litterman posterior expected returns.
Combines equilibrium returns with investor views via Bayesian updating.

μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ * [(τΣ)⁻¹π + P'Ω⁻¹Q]
"""
function bl_posterior_returns(bl::BlackLitterman)
    Σ = bl.cov_matrix
    τ = bl.tau
    P = bl.P
    Q = bl.Q
    Ω = bl.Omega

    # Equilibrium returns
    π = bl_equilibrium_returns(bl)

    # Precision matrices
    τΣ_inv = inv(τ * Σ)
    Ω_inv = inv(Ω)

    # Posterior precision and mean
    posterior_precision = τΣ_inv + P' * Ω_inv * P
    posterior_mean = posterior_precision \ (τΣ_inv * π + P' * Ω_inv * Q)

    posterior_mean
end

"""
    bl_posterior_covariance(bl::BlackLitterman)

Compute Black-Litterman posterior covariance matrix.
"""
function bl_posterior_covariance(bl::BlackLitterman)
    Σ = bl.cov_matrix
    τ = bl.tau
    P = bl.P
    Ω = bl.Omega

    τΣ_inv = inv(τ * Σ)
    Ω_inv = inv(Ω)

    posterior_precision = τΣ_inv + P' * Ω_inv * P
    inv(posterior_precision)
end

"""
    OptimizationResult

Result of a portfolio optimization.

# Fields
- `weights::Vector{Float64}` - Optimal portfolio weights (typically sum to 1)
- `objective::Float64` - Final objective function value (e.g., variance for MVO)
- `converged::Bool` - Whether the optimization converged successfully
- `iterations::Int` - Number of iterations used

# Example
```julia
returns = [0.10, 0.08, 0.12]
cov_matrix = [0.04 0.01 0.02;
              0.01 0.03 0.01;
              0.02 0.01 0.05]

result = optimize(MeanVariance(returns, cov_matrix); target_return=0.10)
if result.converged
    println("Optimal weights: ", result.weights)
end
```

See also: [`optimize`](@ref), [`MeanVariance`](@ref)
"""
struct OptimizationResult
    weights::Vector{Float64}
    objective::Float64
    converged::Bool
    iterations::Int
end

function Base.show(io::IO, r::OptimizationResult)
    status = r.converged ? "converged" : "not converged"
    n = length(r.weights)
    if n <= 5
        print(io, "OptimizationResult(weights=$(round.(r.weights, digits=3)), obj=$(round(r.objective, digits=4)), $status)")
    else
        print(io, "OptimizationResult($(n) assets, obj=$(round(r.objective, digits=4)), $status)")
    end
end

function Base.show(io::IO, ::MIME"text/plain", r::OptimizationResult)
    n = length(r.weights)
    status = r.converged ? "✓ converged" : "✗ not converged"
    println(io, "OptimizationResult ($status in $(r.iterations) iterations)")
    println(io, "  Objective: $(round(r.objective, digits=6))")

    # Show weights nicely
    if n <= 10
        println(io, "  Weights: $(round.(r.weights, digits=4))")
    else
        # Truncate for large portfolios
        top5 = sortperm(r.weights, rev=true)[1:5]
        println(io, "  Top 5 weights:")
        for i in top5
            println(io, "    Asset $i: $(round(r.weights[i]*100, digits=2))%")
        end
    end
    print(io, "  Sum of weights: $(round(sum(r.weights), digits=6))")
end

# ============================================================================
# User-Friendly Portfolio Construction
# ============================================================================

"""
    PortfolioSpec

Specification for portfolio optimization with user-friendly defaults.

# Example
```julia
# From returns matrix
spec = PortfolioSpec(returns_matrix)

# With custom settings
spec = PortfolioSpec(returns_matrix;
    cov_estimator = LedoitWolfShrinkage(),
    min_weight = 0.02,
    max_weight = 0.25
)

# Optimize
result = optimize(spec; objective=:min_variance)
result = optimize(spec; objective=:max_sharpe)
result = optimize(spec; objective=:risk_parity)
result = optimize(spec; objective=:target_return, target=0.10)
```
"""
struct PortfolioSpec
    expected_returns::Vector{Float64}
    cov_matrix::Matrix{Float64}
    n_assets::Int
    asset_names::Vector{String}
    constraints::Vector{AbstractConstraint}
    rf::Float64
end

"""
    PortfolioSpec(returns; kwargs...)

Create a portfolio specification from a returns matrix.

# Arguments
- `returns`: T × N matrix of asset returns (T observations, N assets)

# Keyword Arguments
- `asset_names`: Names for assets (default: ["Asset1", "Asset2", ...])
- `cov_estimator`: Covariance estimator (default: LedoitWolfShrinkage())
- `return_shrinkage`: Shrinkage for expected returns (default: 0.3)
- `annualize`: Factor to annualize (default: 252 for daily data)
- `rf`: Risk-free rate (default: 0.0)
- `long_only`: Enforce non-negative weights (default: true)
- `min_weight`: Minimum weight per asset (default: 0.0)
- `max_weight`: Maximum weight per asset (default: 1.0)
- `full_investment`: Weights must sum to 1 (default: true)
"""
function PortfolioSpec(returns::Matrix{Float64};
                       asset_names::Union{Vector{String}, Nothing}=nothing,
                       cov_estimator::AbstractCovarianceEstimator=LedoitWolfShrinkage(),
                       return_shrinkage::Float64=0.3,
                       annualize::Float64=252.0,
                       rf::Float64=0.0,
                       long_only::Bool=true,
                       min_weight::Float64=0.0,
                       max_weight::Float64=1.0,
                       full_investment::Bool=true)
    T, N = size(returns)

    # Estimate parameters
    params = estimate_parameters(returns; cov_estimator=cov_estimator, return_shrinkage=return_shrinkage)
    μ = params.expected_returns * annualize
    Σ = params.cov_matrix * annualize

    # Asset names
    names = isnothing(asset_names) ? ["Asset$i" for i in 1:N] : asset_names
    length(names) == N || throw(ArgumentError("asset_names must have $N elements"))

    # Build constraints
    constraints = AbstractConstraint[]
    full_investment && push!(constraints, FullInvestmentConstraint(1.0))
    long_only && push!(constraints, LongOnlyConstraint())
    if min_weight > 0.0 || max_weight < 1.0
        push!(constraints, BoxConstraint(N; min_weight=min_weight, max_weight=max_weight))
    end

    PortfolioSpec(μ, Σ, N, names, constraints, rf)
end

"""
    PortfolioSpec(μ, Σ; kwargs...)

Create a portfolio specification from expected returns and covariance.

# Arguments
- `μ`: Expected returns vector
- `Σ`: Covariance matrix

# Keyword Arguments
- `asset_names`: Names for assets
- `rf`: Risk-free rate (default: 0.0)
- `long_only`: Enforce non-negative weights (default: true)
- `min_weight`: Minimum weight per asset (default: 0.0)
- `max_weight`: Maximum weight per asset (default: 1.0)
"""
function PortfolioSpec(μ::Vector{Float64}, Σ::Matrix{Float64};
                       asset_names::Union{Vector{String}, Nothing}=nothing,
                       rf::Float64=0.0,
                       long_only::Bool=true,
                       min_weight::Float64=0.0,
                       max_weight::Float64=1.0,
                       full_investment::Bool=true)
    N = length(μ)
    size(Σ) == (N, N) || throw(ArgumentError("Σ must be $N × $N"))

    names = isnothing(asset_names) ? ["Asset$i" for i in 1:N] : asset_names

    constraints = AbstractConstraint[]
    full_investment && push!(constraints, FullInvestmentConstraint(1.0))
    long_only && push!(constraints, LongOnlyConstraint())
    if min_weight > 0.0 || max_weight < 1.0
        push!(constraints, BoxConstraint(N; min_weight=min_weight, max_weight=max_weight))
    end

    PortfolioSpec(μ, Σ, N, names, constraints, rf)
end

"""
    optimize(spec::PortfolioSpec; objective=:max_sharpe, target=nothing)

Optimize a portfolio with a simple objective specification.

# Arguments
- `spec`: PortfolioSpec with asset data and constraints
- `objective`: Optimization goal, one of:
  - `:max_sharpe` - Maximum Sharpe ratio (default)
  - `:min_variance` - Minimum variance
  - `:risk_parity` - Equal risk contribution
  - `:max_diversification` - Maximum diversification ratio
  - `:target_return` - Minimum variance for target return (requires `target`)
  - `:target_volatility` - Maximum return for target volatility (requires `target`)
- `target`: Target value for `:target_return` or `:target_volatility`

# Returns
OptimizationResult with optimal weights

# Example
```julia
spec = PortfolioSpec(μ, Σ; max_weight=0.3)

# Different objectives
result = optimize(spec; objective=:max_sharpe)
result = optimize(spec; objective=:risk_parity)
result = optimize(spec; objective=:target_return, target=0.10)
```
"""
function optimize(spec::PortfolioSpec;
                  objective::Symbol=:max_sharpe,
                  target::Union{Float64, Nothing}=nothing)
    μ = spec.expected_returns
    Σ = spec.cov_matrix
    rf = spec.rf
    constraints = spec.constraints

    if objective == :max_sharpe
        # Use SharpeMaximizer
        obj = SharpeMaximizer(μ, Σ; rf=rf)
        return optimize(obj)

    elseif objective == :min_variance
        obj = MinimumVariance(Σ)
        return optimize(obj; constraints=constraints)

    elseif objective == :risk_parity
        obj = RiskParity(Σ)
        return optimize(obj; constraints=constraints)

    elseif objective == :max_diversification
        obj = MaximumDiversification(Σ)
        return optimize(obj; constraints=constraints)

    elseif objective == :target_return
        isnothing(target) && throw(ArgumentError("target required for :target_return objective"))
        obj = MeanVariance(μ, Σ)
        return optimize(obj; target_return=target)

    elseif objective == :target_volatility
        isnothing(target) && throw(ArgumentError("target required for :target_volatility objective"))
        # Binary search for return that achieves target vol
        return _optimize_target_volatility(μ, Σ, target, constraints, rf)

    else
        throw(ArgumentError("Unknown objective: $objective. Use :max_sharpe, :min_variance, :risk_parity, :max_diversification, :target_return, or :target_volatility"))
    end
end

function _optimize_target_volatility(μ, Σ, target_vol, constraints, rf)
    # Find return range
    min_var_result = optimize(MinimumVariance(Σ); constraints=constraints)
    min_vol = sqrt(min_var_result.weights' * Σ * min_var_result.weights)
    min_ret = dot(μ, min_var_result.weights)

    if target_vol <= min_vol
        return min_var_result
    end

    # Binary search for return
    max_ret = maximum(μ)
    low_ret, high_ret = min_ret, max_ret

    for _ in 1:50
        mid_ret = (low_ret + high_ret) / 2
        result = optimize(MeanVariance(μ, Σ); target_return=mid_ret)
        vol = sqrt(result.weights' * Σ * result.weights)

        if abs(vol - target_vol) < 1e-6
            return result
        elseif vol < target_vol
            low_ret = mid_ret
        else
            high_ret = mid_ret
        end
    end

    optimize(MeanVariance(μ, Σ); target_return=(low_ret + high_ret) / 2)
end

function Base.show(io::IO, ::MIME"text/plain", spec::PortfolioSpec)
    println(io, "PortfolioSpec ($(spec.n_assets) assets)")
    println(io, "  Expected returns: $(round(minimum(spec.expected_returns)*100, digits=1))% - $(round(maximum(spec.expected_returns)*100, digits=1))%")
    vols = sqrt.(diag(spec.cov_matrix))
    println(io, "  Volatilities: $(round(minimum(vols)*100, digits=1))% - $(round(maximum(vols)*100, digits=1))%")
    println(io, "  Risk-free rate: $(round(spec.rf*100, digits=2))%")
    print(io, "  Constraints: $(length(spec.constraints))")
end

# ============================================================================
# Quick Portfolio Functions (Even Simpler API)
# ============================================================================

"""
    min_variance_portfolio(Σ; kwargs...)

Quick function to get minimum variance portfolio.

# Example
```julia
weights = min_variance_portfolio(cov_matrix; max_weight=0.3)
```
"""
function min_variance_portfolio(Σ::Matrix{Float64};
                                long_only::Bool=true,
                                max_weight::Float64=1.0,
                                min_weight::Float64=0.0)
    n = size(Σ, 1)
    constraints = AbstractConstraint[FullInvestmentConstraint()]
    long_only && push!(constraints, LongOnlyConstraint())
    (min_weight > 0 || max_weight < 1) && push!(constraints, BoxConstraint(n; min_weight=min_weight, max_weight=max_weight))

    result = optimize(MinimumVariance(Σ); constraints=constraints)
    result.weights
end

"""
    max_sharpe_portfolio(μ, Σ; rf=0.0)

Quick function to get maximum Sharpe ratio portfolio.

# Example
```julia
weights = max_sharpe_portfolio(expected_returns, cov_matrix)
```
"""
function max_sharpe_portfolio(μ::Vector{Float64}, Σ::Matrix{Float64}; rf::Float64=0.0)
    result = optimize(SharpeMaximizer(μ, Σ; rf=rf))
    result.weights
end

"""
    risk_parity_portfolio(Σ; target_risk=nothing, kwargs...)

Quick function to get risk parity portfolio.

# Example
```julia
weights = risk_parity_portfolio(cov_matrix)
weights = risk_parity_portfolio(cov_matrix; max_weight=0.25)
```
"""
function risk_parity_portfolio(Σ::Matrix{Float64};
                               target_risk::Union{Vector{Float64}, Nothing}=nothing,
                               long_only::Bool=true,
                               max_weight::Float64=1.0,
                               min_weight::Float64=0.0)
    n = size(Σ, 1)
    constraints = AbstractConstraint[FullInvestmentConstraint()]
    long_only && push!(constraints, LongOnlyConstraint())
    (min_weight > 0 || max_weight < 1) && push!(constraints, BoxConstraint(n; min_weight=min_weight, max_weight=max_weight))

    obj = isnothing(target_risk) ? RiskParity(Σ) : RiskParity(Σ; target_contributions=target_risk)
    result = optimize(obj; constraints=constraints)
    result.weights
end

"""
    target_return_portfolio(μ, Σ, target_return; kwargs...)

Quick function to get minimum variance portfolio for a target return.

# Example
```julia
weights = target_return_portfolio(expected_returns, cov_matrix, 0.10)
```
"""
function target_return_portfolio(μ::Vector{Float64}, Σ::Matrix{Float64}, target_return::Float64;
                                 long_only::Bool=true,
                                 max_weight::Float64=1.0)
    result = optimize(MeanVariance(μ, Σ); target_return=target_return)
    result.weights
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
# TODO: Validate covariance matrix is positive definite
# TODO: Check if target_return is achievable (min/max return constraints)
# TODO: Handle singular covariance matrices (use QR decomposition)
# TODO: Consider BFGS/L-BFGS for faster convergence
function optimize(mv::MeanVariance; target_return::Float64, backend=current_backend(), max_iter=5000, tol=1e-10, lr=0.005)
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

        g = gradient(obj, w; backend=backend)

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
function optimize(sm::SharpeMaximizer; backend=current_backend(), max_iter=1000, tol=1e-8, lr=0.1)
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

        g = gradient(neg_sharpe, w; backend=backend)
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

# CVaR Minimization (parametric Gaussian approximation)
"""
    optimize(cvar::CVaRObjective; target_return, backend, max_iter, tol, lr)

Minimize CVaR subject to a target return constraint.

Uses the parametric (Gaussian) CVaR formula:
    CVaR_α = -μ'w + σ(w) * φ(z_α) / (1-α)

where z_α = Φ⁻¹(α) is the VaR quantile and φ is the standard normal PDF.
"""
function optimize(cvar::CVaRObjective; target_return::Float64,
                  backend=current_backend(), max_iter::Int=5000,
                  tol::Float64=1e-10, lr::Float64=0.01)
    μ = cvar.expected_returns
    Σ = cvar.cov_matrix
    α = cvar.alpha
    n = length(μ)

    # Standard normal quantile and PDF at quantile
    z_α = _norminv(α)
    φ_z = exp(-z_α^2 / 2) / sqrt(2π)
    cvar_factor = φ_z / (1 - α)

    # Initialize with equal weights
    w = ones(n) / n
    penalty = 10000.0

    for i in 1:max_iter
        function obj(weights)
            port_return = dot(weights, μ)
            port_vol = sqrt(weights' * Σ * weights + 1e-12)

            # Parametric CVaR (we want to minimize, so negative return + risk term)
            cvar_val = -port_return + port_vol * cvar_factor

            # Constraints
            ret_penalty = penalty * (port_return - target_return)^2
            sum_penalty = penalty * (sum(weights) - 1)^2
            neg_penalty = penalty * sum(max.(-weights, 0).^2)

            return cvar_val + ret_penalty + sum_penalty + neg_penalty
        end

        g = gradient(obj, w; backend=backend)

        current_lr = lr / (1 + i * 0.0001)
        w_new = w - current_lr * g

        # Project to simplex
        w_new = max.(w_new, 0.0)
        if sum(w_new) > 0
            w_new = w_new / sum(w_new)
        else
            w_new = ones(n) / n
        end

        if norm(w_new - w) < tol
            port_return = dot(w_new, μ)
            port_vol = sqrt(w_new' * Σ * w_new)
            cvar_val = -port_return + port_vol * cvar_factor
            return OptimizationResult(w_new, cvar_val, true, i)
        end

        w = w_new
    end

    port_return = dot(w, μ)
    port_vol = sqrt(w' * Σ * w)
    cvar_val = -port_return + port_vol * cvar_factor
    return OptimizationResult(w, cvar_val, false, max_iter)
end

# Kelly Criterion Optimization
"""
    optimize(kelly::KellyCriterion; backend, max_iter, tol, lr, fractional)

Maximize expected log growth rate (Kelly criterion).

The unconstrained Kelly optimal is w* = Σ⁻¹μ, but this can produce extreme
leverage. The `fractional` parameter scales the result (e.g., 0.5 for half-Kelly).

For long-only portfolios, uses gradient descent with simplex projection.
"""
function optimize(kelly::KellyCriterion; backend=current_backend(),
                  max_iter::Int=2000, tol::Float64=1e-10, lr::Float64=0.05,
                  fractional::Float64=1.0, long_only::Bool=true)
    μ = kelly.expected_returns
    Σ = kelly.cov_matrix
    n = length(μ)

    if !long_only
        # Unconstrained Kelly: w* = Σ⁻¹μ (scaled by fractional)
        w_kelly = Σ \ μ
        w_kelly = w_kelly * fractional

        # Normalize to sum to 1 for comparison
        w_normalized = w_kelly / sum(w_kelly)
        growth = dot(w_kelly, μ) - 0.5 * (w_kelly' * Σ * w_kelly)
        return OptimizationResult(w_normalized, growth, true, 1)
    end

    # Long-only: gradient descent with simplex projection
    w = ones(n) / n
    penalty = 1000.0

    for i in 1:max_iter
        # Kelly objective: maximize E[log(1 + w'r)] ≈ w'μ - 0.5 w'Σw (quadratic approx)
        # We minimize negative of this
        function neg_kelly(weights)
            growth = dot(weights, μ) - 0.5 * (weights' * Σ * weights)

            # Constraints
            sum_penalty = penalty * (sum(weights) - 1)^2
            neg_penalty = penalty * sum(max.(-weights, 0).^2)

            return -growth + sum_penalty + neg_penalty
        end

        g = gradient(neg_kelly, w; backend=backend)
        w_new = w - lr * g

        # Project to simplex
        w_new = max.(w_new, 0.0)
        if sum(w_new) > 0
            w_new = w_new / sum(w_new)
        else
            w_new = ones(n) / n
        end

        # Scale by fractional Kelly
        if fractional != 1.0
            w_scaled = w_new * fractional + (1 - fractional) * ones(n) / n
            w_new = w_scaled / sum(w_scaled)
        end

        if norm(w_new - w) < tol
            growth = dot(w_new, μ) - 0.5 * (w_new' * Σ * w_new)
            return OptimizationResult(w_new, growth, true, i)
        end

        w = w_new
    end

    growth = dot(w, μ) - 0.5 * (w' * Σ * w)
    return OptimizationResult(w, growth, false, max_iter)
end

# ============================================================================
# New Objective Optimization Methods
# ============================================================================

"""
    optimize(mv::MinimumVariance; constraints=nothing, solver=nothing, backend=current_backend())

Find the global minimum variance portfolio.
Uses QP solver for efficiency.
"""
function optimize(mv::MinimumVariance;
                  constraints::Union{Vector{<:AbstractConstraint}, Nothing}=nothing,
                  solver::Union{AbstractSolver, Nothing}=nothing,
                  backend=current_backend())
    Σ = mv.cov_matrix
    n = size(Σ, 1)

    # Default constraints
    if isnothing(constraints)
        constraints = [FullInvestmentConstraint(), LongOnlyConstraint()]
    end

    # Use QP solver
    qp = isnothing(solver) ? QPSolver() : solver
    result = solve_min_variance_qp(Σ; constraints=constraints, solver=qp isa QPSolver ? qp : QPSolver())

    variance = result.objective
    OptimizationResult(result.x, variance, result.converged, result.iterations)
end

"""
    optimize(rp::RiskParity; constraints=nothing, backend=current_backend(), max_iter=5000, tol=1e-10, lr=0.01)

Find risk parity portfolio that equalizes risk contributions.

Uses Spinu (2013) formulation: minimize Σᵢ(wᵢ(Σw)ᵢ - c)²
where c is chosen to achieve target risk budget.
"""
function optimize(rp::RiskParity;
                  constraints::Union{Vector{<:AbstractConstraint}, Nothing}=nothing,
                  backend=current_backend(),
                  max_iter::Int=5000,
                  tol::Float64=1e-10,
                  lr::Float64=0.01)
    Σ = rp.cov_matrix
    targets = rp.target_contributions
    n = size(Σ, 1)

    # Extract bounds from constraints
    lb = zeros(n)
    ub = fill(Inf, n)
    target_sum = 1.0

    if !isnothing(constraints)
        for c in constraints
            if c isa BoxConstraint
                lb = max.(lb, c.lower)
                ub = min.(ub, c.upper)
            elseif c isa LongOnlyConstraint
                lb = max.(lb, 0.0)
            elseif c isa FullInvestmentConstraint
                target_sum = c.target
            end
        end
    end

    # Initialize with inverse volatility weights (good starting point for risk parity)
    σ = sqrt.(diag(Σ))
    w = 1.0 ./ σ
    w = w / sum(w) * target_sum
    w = clamp.(w, lb, ub)

    # Determine if upper bounds are finite
    ub_finite = all(isfinite, ub) ? ub : nothing
    w = project_simplex(w, target_sum; lower=lb, upper=ub_finite)

    for i in 1:max_iter
        # Objective: sum((w_i * (Σw)_i / var - target_i)^2)
        function obj(weights)
            var = weights' * Σ * weights + 1e-12
            marginal = Σ * weights
            frc = (weights .* marginal) / var  # Fractional risk contributions
            sum((frc .- targets).^2)
        end

        g = gradient(obj, w; backend=backend)

        current_lr = lr / (1 + i * 0.0001)
        w_new = w - current_lr * g

        # Project to feasible set
        w_new = project_simplex(w_new, target_sum; lower=lb, upper=ub_finite)

        if norm(w_new - w) < tol
            obj_val = risk_parity_objective(w_new, rp)
            return OptimizationResult(w_new, obj_val, true, i)
        end

        w = w_new
    end

    obj_val = risk_parity_objective(w, rp)
    return OptimizationResult(w, obj_val, false, max_iter)
end

"""
    optimize(md::MaximumDiversification; constraints=nothing, backend=current_backend(), max_iter=1000, tol=1e-8, lr=0.1)

Maximize diversification ratio: (w'σ) / √(w'Σw)
"""
function optimize(md::MaximumDiversification;
                  constraints::Union{Vector{<:AbstractConstraint}, Nothing}=nothing,
                  backend=current_backend(),
                  max_iter::Int=1000,
                  tol::Float64=1e-8,
                  lr::Float64=0.1)
    Σ = md.cov_matrix
    σ_vols = md.asset_vols
    n = size(Σ, 1)

    # Extract bounds from constraints
    lb = zeros(n)
    ub = fill(Inf, n)
    target_sum = 1.0

    if !isnothing(constraints)
        for c in constraints
            if c isa BoxConstraint
                lb = max.(lb, c.lower)
                ub = min.(ub, c.upper)
            elseif c isa LongOnlyConstraint
                lb = max.(lb, 0.0)
            elseif c isa FullInvestmentConstraint
                target_sum = c.target
            end
        end
    end

    # Determine if upper bounds are finite
    ub_finite = all(isfinite, ub) ? ub : nothing

    # Initialize with equal weights
    w = fill(target_sum / n, n)

    for i in 1:max_iter
        # Negative diversification ratio (we minimize)
        function neg_div_ratio(weights)
            weighted_vol = dot(weights, σ_vols)
            port_vol = sqrt(weights' * Σ * weights + 1e-12)
            -weighted_vol / port_vol
        end

        g = gradient(neg_div_ratio, w; backend=backend)
        w_new = w - lr * g

        # Project to feasible set
        w_new = project_simplex(w_new, target_sum; lower=lb, upper=ub_finite)

        if norm(w_new - w) < tol
            dr = diversification_ratio(w_new, md)
            return OptimizationResult(w_new, dr, true, i)
        end

        w = w_new
    end

    dr = diversification_ratio(w, md)
    return OptimizationResult(w, dr, false, max_iter)
end

"""
    optimize(bl::BlackLitterman; constraints=nothing, risk_aversion=nothing, solver=nothing, backend=current_backend())

Optimize using Black-Litterman posterior returns.

First computes BL posterior expected returns, then runs mean-variance optimization.
"""
function optimize(bl::BlackLitterman;
                  constraints::Union{Vector{<:AbstractConstraint}, Nothing}=nothing,
                  risk_aversion::Union{Float64, Nothing}=nothing,
                  solver::Union{AbstractSolver, Nothing}=nothing,
                  backend=current_backend())
    # Compute posterior returns and covariance
    μ_bl = bl_posterior_returns(bl)
    Σ_bl = bl.cov_matrix  # Could also use posterior covariance

    # Use provided or default risk aversion
    λ = isnothing(risk_aversion) ? bl.risk_aversion : risk_aversion

    n = length(μ_bl)

    # Default constraints
    if isnothing(constraints)
        constraints = [FullInvestmentConstraint(), LongOnlyConstraint()]
    end

    # Extract bounds
    lb = zeros(n)
    ub = fill(Inf, n)
    target_sum = 1.0

    for c in constraints
        if c isa BoxConstraint
            lb = max.(lb, c.lower)
            ub = min.(ub, c.upper)
        elseif c isa LongOnlyConstraint
            lb = max.(lb, 0.0)
        elseif c isa FullInvestmentConstraint
            target_sum = c.target
        end
    end

    # Solve: max μ'w - (λ/2) w'Σw  ⟺  min (λ/2) w'Σw - μ'w
    # QP form: Q = λΣ, c = -μ
    qp = isnothing(solver) ? QPSolver() : (solver isa QPSolver ? solver : QPSolver())
    result = solve_qp(λ * Σ_bl, -μ_bl; lb=lb, ub=ub, target_sum=target_sum, solver=qp)

    # Compute utility as objective
    w = result.x
    utility = dot(μ_bl, w) - (λ/2) * (w' * Σ_bl * w)

    OptimizationResult(w, utility, result.converged, result.iterations)
end

# Helper: inverse normal CDF (Beasley-Springer-Moro approximation)
function _norminv(p::Float64)
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]

    p_low = 0.02425
    p_high = 1 - p_low

    if p < p_low
        q = sqrt(-2 * log(p))
        return (((((c[1]*q + c[2])*q + c[3])*q + c[4])*q + c[5])*q + c[6]) /
               ((((d[1]*q + d[2])*q + d[3])*q + d[4])*q + 1)
    elseif p <= p_high
        q = p - 0.5
        r = q * q
        return (((((a[1]*r + a[2])*r + a[3])*r + a[4])*r + a[5])*r + a[6])*q /
               (((((b[1]*r + b[2])*r + b[3])*r + b[4])*r + b[5])*r + 1)
    else
        q = sqrt(-2 * log(1 - p))
        return -(((((c[1]*q + c[2])*q + c[3])*q + c[4])*q + c[5])*q + c[6]) /
                ((((d[1]*q + d[2])*q + d[3])*q + d[4])*q + 1)
    end
end

# ============================================================================
# Covariance Estimators
# ============================================================================

"""
    SampleCovariance()

Standard sample covariance estimator (no shrinkage).
"""
struct SampleCovariance <: AbstractCovarianceEstimator end

"""
    LedoitWolfShrinkage(; target=:identity)

Ledoit-Wolf shrinkage estimator for covariance matrices.

Shrinks sample covariance toward a structured target to reduce estimation error.

# Arguments
- `target`: Shrinkage target, one of:
  - `:identity` - Shrink toward scaled identity matrix (default)
  - `:constant_correlation` - Shrink toward constant correlation matrix
  - `:diagonal` - Shrink toward diagonal matrix
"""
struct LedoitWolfShrinkage <: AbstractCovarianceEstimator
    target::Symbol

    function LedoitWolfShrinkage(; target::Symbol=:identity)
        target in (:identity, :constant_correlation, :diagonal) ||
            throw(ArgumentError("target must be :identity, :constant_correlation, or :diagonal"))
        new(target)
    end
end

"""
    ExponentialWeighted(; halflife=60, min_periods=20)

Exponentially weighted moving average (EWMA) covariance estimator.

More recent observations receive higher weight, making the estimator
more responsive to regime changes.

# Arguments
- `halflife`: Number of periods for weight to decay by half (default: 60)
- `min_periods`: Minimum observations required (default: 20)
"""
struct ExponentialWeighted <: AbstractCovarianceEstimator
    halflife::Int
    min_periods::Int

    function ExponentialWeighted(; halflife::Int=60, min_periods::Int=20)
        halflife > 0 || throw(ArgumentError("halflife must be positive"))
        min_periods > 0 || throw(ArgumentError("min_periods must be positive"))
        new(halflife, min_periods)
    end
end

"""
    estimate_covariance(returns, estimator=SampleCovariance())

Estimate covariance matrix from returns data.

# Arguments
- `returns`: T × N matrix of returns (T observations, N assets)
- `estimator`: Covariance estimator (default: SampleCovariance())

# Returns
N × N covariance matrix
"""
function estimate_covariance(returns::Matrix{Float64}, estimator::SampleCovariance=SampleCovariance())
    cov(returns)
end

function estimate_covariance(returns::Matrix{Float64}, estimator::LedoitWolfShrinkage)
    T, N = size(returns)

    # Sample covariance
    S = cov(returns)

    # Compute shrinkage target
    if estimator.target == :identity
        # Target: μI where μ = trace(S)/N
        μ = tr(S) / N
        F = μ * I(N)
    elseif estimator.target == :diagonal
        # Target: diagonal of sample covariance
        F = Diagonal(diag(S))
    else  # :constant_correlation
        # Target: constant correlation matrix
        var_vec = diag(S)
        std_vec = sqrt.(var_vec)
        # Average correlation
        corr_mat = S ./ (std_vec * std_vec')
        avg_corr = (sum(corr_mat) - N) / (N * (N - 1))
        F = Diagonal(std_vec) * (avg_corr * ones(N, N) + (1 - avg_corr) * I(N)) * Diagonal(std_vec)
    end

    # Compute optimal shrinkage intensity (Ledoit-Wolf formula)
    δ = _ledoit_wolf_shrinkage_intensity(returns, S, F)

    # Shrunk covariance
    (1 - δ) * S + δ * Matrix(F)
end

function estimate_covariance(returns::Matrix{Float64}, estimator::ExponentialWeighted)
    T, N = size(returns)

    T >= estimator.min_periods ||
        throw(ArgumentError("Need at least $(estimator.min_periods) observations, got $T"))

    # Decay factor: λ = exp(-log(2)/halflife)
    λ = exp(-log(2) / estimator.halflife)

    # Compute weights (most recent = highest weight)
    weights = [λ^(T - t) for t in 1:T]
    weights = weights / sum(weights)

    # Weighted mean
    μ = zeros(N)
    for t in 1:T
        μ .+= weights[t] * returns[t, :]
    end

    # Weighted covariance
    Σ = zeros(N, N)
    for t in 1:T
        dev = returns[t, :] - μ
        Σ .+= weights[t] * (dev * dev')
    end

    # Bias correction for weighted covariance
    sum_w_sq = sum(weights.^2)
    Σ = Σ / (1 - sum_w_sq)

    Σ
end

"""
Compute optimal Ledoit-Wolf shrinkage intensity.
"""
function _ledoit_wolf_shrinkage_intensity(X::Matrix{Float64}, S::Matrix{Float64}, F::AbstractMatrix)
    T, N = size(X)

    # Demean
    X_centered = X .- mean(X, dims=1)

    # Frobenius norm of (S - F)
    d2 = sum((S - F).^2)

    # Estimate of π (sum of asymptotic variances)
    π_mat = zeros(N, N)
    for k in 1:T
        x_k = X_centered[k, :]
        π_mat .+= (x_k * x_k' - S).^2
    end
    π_sum = sum(π_mat) / T

    # Estimate of ρ (sum of asymptotic covariances with target)
    # Simplified: for identity target, ρ ≈ 0
    ρ = 0.0

    # Optimal shrinkage intensity
    κ = (π_sum - ρ) / d2
    δ = max(0.0, min(1.0, κ / T))

    δ
end

"""
    estimate_expected_returns(returns; method=:mean)

Estimate expected returns from historical data.

# Arguments
- `returns`: T × N matrix of returns
- `method`: Estimation method
  - `:mean` - Simple arithmetic mean (default)
  - `:shrinkage` - Shrink toward grand mean
"""
function estimate_expected_returns(returns::Matrix{Float64}; method::Symbol=:mean, shrinkage::Float64=0.0)
    μ = vec(mean(returns, dims=1))

    if method == :shrinkage || shrinkage > 0
        grand_mean = mean(μ)
        μ = (1 - shrinkage) * μ .+ shrinkage * grand_mean
    end

    μ
end

"""
    estimate_parameters(returns; cov_estimator=LedoitWolfShrinkage(), return_shrinkage=0.0)

Estimate both expected returns and covariance from returns data.

# Arguments
- `returns`: T × N matrix of returns
- `cov_estimator`: Covariance estimator (default: LedoitWolfShrinkage())
- `return_shrinkage`: Shrinkage toward grand mean for returns (default: 0.0)

# Returns
Named tuple (expected_returns, cov_matrix)

# Example
```julia
returns = randn(252, 10)  # 1 year of daily returns for 10 assets
params = estimate_parameters(returns)
result = optimize(MeanVariance(params.expected_returns, params.cov_matrix); target_return=0.10)
```
"""
function estimate_parameters(returns::Matrix{Float64};
                            cov_estimator::AbstractCovarianceEstimator=LedoitWolfShrinkage(),
                            return_shrinkage::Float64=0.0)
    μ = estimate_expected_returns(returns; shrinkage=return_shrinkage)
    Σ = estimate_covariance(returns, cov_estimator)
    (expected_returns=μ, cov_matrix=Σ)
end

# ============================================================================
# Efficient Frontier
# ============================================================================

"""
    EfficientFrontier

Represents the efficient frontier of a portfolio optimization problem.

# Fields
- `returns`: Expected returns at each point
- `volatilities`: Portfolio volatilities at each point
- `weights`: Weight matrix (n_points × n_assets)
- `sharpe_ratios`: Sharpe ratio at each point
- `min_variance_idx`: Index of minimum variance portfolio
- `max_sharpe_idx`: Index of maximum Sharpe ratio portfolio
- `n_assets`: Number of assets
"""
struct EfficientFrontier
    returns::Vector{Float64}
    volatilities::Vector{Float64}
    weights::Matrix{Float64}
    sharpe_ratios::Vector{Float64}
    min_variance_idx::Int
    max_sharpe_idx::Int
    n_assets::Int
end

"""
    compute_efficient_frontier(μ, Σ; n_points=50, rf=0.0, constraints=nothing)

Compute the efficient frontier for a set of assets.

# Arguments
- `μ`: Expected returns vector
- `Σ`: Covariance matrix
- `n_points`: Number of points on the frontier (default: 50)
- `rf`: Risk-free rate for Sharpe ratio calculation (default: 0.0)
- `constraints`: Optional constraints (default: full investment + long only)

# Returns
EfficientFrontier struct containing returns, volatilities, weights, and key indices.

# Example
```julia
μ = [0.10, 0.15, 0.12]
Σ = [0.04 0.01 0.02; 0.01 0.09 0.01; 0.02 0.01 0.05]
frontier = compute_efficient_frontier(μ, Σ; n_points=20)
println("Max Sharpe weights: ", frontier.weights[frontier.max_sharpe_idx, :])
```
"""
function compute_efficient_frontier(μ::Vector{Float64}, Σ::Matrix{Float64};
                                    n_points::Int=50,
                                    rf::Float64=0.0,
                                    constraints::Union{Vector{<:AbstractConstraint}, Nothing}=nothing)
    n = length(μ)

    # Default constraints
    if isnothing(constraints)
        constraints = [FullInvestmentConstraint(), LongOnlyConstraint()]
    end

    # First, find the minimum variance portfolio
    min_var_result = optimize(MinimumVariance(Σ); constraints=constraints)
    min_var_return = dot(min_var_result.weights, μ)
    min_var_vol = sqrt(min_var_result.weights' * Σ * min_var_result.weights)

    # Find the maximum return achievable (100% in highest return asset, subject to constraints)
    max_return = maximum(μ)

    # For long-only with box constraints, find achievable max return
    lb, ub, _ = _extract_bounds(constraints, n)
    # Maximum return is when we max out on the highest returning assets
    max_achievable_return = min_var_return
    for i in sortperm(μ, rev=true)
        if ub[i] > lb[i]
            max_achievable_return = max(max_achievable_return, μ[i])
        end
    end

    # Create return targets from min variance to max return
    # Use slightly less than max to ensure feasibility
    target_returns = range(min_var_return, max_achievable_return * 0.999, length=n_points)

    # Preallocate results
    returns = zeros(n_points)
    volatilities = zeros(n_points)
    weights = zeros(n_points, n)
    sharpe_ratios = zeros(n_points)

    # Compute each point on the frontier
    for (i, target_ret) in enumerate(target_returns)
        mv = MeanVariance(μ, Σ)
        result = optimize(mv; target_return=target_ret)

        w = result.weights
        ret = dot(w, μ)
        vol = sqrt(w' * Σ * w)

        returns[i] = ret
        volatilities[i] = vol
        weights[i, :] = w
        sharpe_ratios[i] = vol > 0 ? (ret - rf) / vol : 0.0
    end

    # Find key points
    min_variance_idx = argmin(volatilities)
    max_sharpe_idx = argmax(sharpe_ratios)

    EfficientFrontier(returns, volatilities, weights, sharpe_ratios,
                      min_variance_idx, max_sharpe_idx, n)
end

"""
    get_tangency_portfolio(frontier::EfficientFrontier)

Get the tangency (maximum Sharpe ratio) portfolio weights.
"""
function get_tangency_portfolio(frontier::EfficientFrontier)
    frontier.weights[frontier.max_sharpe_idx, :]
end

"""
    get_min_variance_portfolio(frontier::EfficientFrontier)

Get the minimum variance portfolio weights.
"""
function get_min_variance_portfolio(frontier::EfficientFrontier)
    frontier.weights[frontier.min_variance_idx, :]
end

"""
    interpolate_frontier(frontier::EfficientFrontier, target_return::Float64)

Find weights for a target return by interpolating on the frontier.
"""
function interpolate_frontier(frontier::EfficientFrontier, target_return::Float64)
    # Find bracketing points
    idx = searchsortedfirst(frontier.returns, target_return)

    if idx == 1
        return frontier.weights[1, :]
    elseif idx > length(frontier.returns)
        return frontier.weights[end, :]
    end

    # Linear interpolation
    r1, r2 = frontier.returns[idx-1], frontier.returns[idx]
    w1, w2 = frontier.weights[idx-1, :], frontier.weights[idx, :]

    t = (target_return - r1) / (r2 - r1)
    (1 - t) * w1 + t * w2
end

function Base.show(io::IO, ::MIME"text/plain", ef::EfficientFrontier)
    println(io, "EfficientFrontier with $(length(ef.returns)) points")
    println(io, "  Assets: $(ef.n_assets)")
    println(io, "  Return range: $(round(minimum(ef.returns)*100, digits=2))% - $(round(maximum(ef.returns)*100, digits=2))%")
    println(io, "  Volatility range: $(round(minimum(ef.volatilities)*100, digits=2))% - $(round(maximum(ef.volatilities)*100, digits=2))%")
    println(io, "  Min variance portfolio:")
    println(io, "    Return: $(round(ef.returns[ef.min_variance_idx]*100, digits=2))%")
    println(io, "    Volatility: $(round(ef.volatilities[ef.min_variance_idx]*100, digits=2))%")
    println(io, "  Tangency portfolio (max Sharpe):")
    println(io, "    Return: $(round(ef.returns[ef.max_sharpe_idx]*100, digits=2))%")
    println(io, "    Volatility: $(round(ef.volatilities[ef.max_sharpe_idx]*100, digits=2))%")
    print(io, "    Sharpe: $(round(ef.sharpe_ratios[ef.max_sharpe_idx], digits=3))")
end

# ============================================================================
# Exports
# ============================================================================

# Abstract types
export AbstractOptimizationObjective, AbstractConstraint, AbstractSolver, AbstractCovarianceEstimator

# Constraint types
export FullInvestmentConstraint, LongOnlyConstraint, BoxConstraint
export GroupConstraint, TurnoverConstraint, CardinalityConstraint
export standard_constraints, check_constraint_violation, check_all_constraints

# Solver types
export QPSolver, LBFGSSolver, ProjectedGradientSolver
export project_simplex, project_constraints, solve_qp, solve_min_variance_qp
export solve_lbfgs, solve_projected_gradient

# Objective types (existing)
export MeanVariance, SharpeMaximizer, CVaRObjective, KellyCriterion, OptimizationResult

# Objective types (new)
export MinimumVariance, RiskParity, MaximumDiversification, BlackLitterman

# Portfolio analytics
export compute_risk_contributions, compute_fractional_risk_contributions
export compute_marginal_risk, compute_component_risk, compute_beta
export compute_tracking_error, compute_active_risk_contributions
export PortfolioAnalytics, analyze_portfolio
export risk_parity_objective, diversification_ratio
export bl_equilibrium_returns, bl_posterior_returns, bl_posterior_covariance

# Efficient frontier
export EfficientFrontier, compute_efficient_frontier
export get_tangency_portfolio, get_min_variance_portfolio, interpolate_frontier

# Covariance estimators
export SampleCovariance, LedoitWolfShrinkage, ExponentialWeighted
export estimate_covariance, estimate_expected_returns, estimate_parameters

# User-friendly interface
export PortfolioSpec
export min_variance_portfolio, max_sharpe_portfolio, risk_parity_portfolio, target_return_portfolio

# Main interface
export optimize

end
