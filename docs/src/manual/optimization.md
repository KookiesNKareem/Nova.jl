# Portfolio Optimization

QuantNova provides differentiable portfolio optimization with multiple objective functions.

<img src="../assets/viz-frontier-light.png" class="only-light" alt="Efficient Frontier">
<img src="../assets/viz-frontier-dark.png" class="only-dark" alt="Efficient Frontier">

## Objectives

### Mean-Variance

Classic Markowitz optimization minimizing variance for a target return:

```julia
mu = [0.10, 0.12, 0.08]  # Expected returns
Sigma = [0.04 0.01 0.005;
         0.01 0.05 0.01;
         0.005 0.01 0.03]  # Covariance matrix

mv = MeanVariance(mu, Sigma)
result = optimize(mv; target_return=0.10)

result.weights     # Optimal portfolio weights
result.objective   # Achieved variance
result.converged   # Optimization success
```

### Sharpe Maximization

Maximize risk-adjusted return:

```julia
sm = SharpeMaximizer(mu, Sigma; rf=0.02)
result = optimize(sm)

# Sharpe ratio of result
sharpe = (dot(result.weights, mu) - 0.02) / sqrt(result.weights' * Sigma * result.weights)
```

### CVaR Minimization (Planned)

Minimize tail risk (Conditional Value at Risk):

```julia
# Coming soon - CVaR optimization is planned but not yet implemented
# cvar_obj = CVaRObjective(mu, Sigma; alpha=0.95)
# result = optimize(cvar_obj)
```

### Kelly Criterion (Planned)

Maximize long-term growth rate:

```julia
# Coming soon - Kelly criterion optimization is planned but not yet implemented
# kelly = KellyCriterion(mu, Sigma)
# result = optimize(kelly)
```

## Using AD Backends

Optimization uses the current AD backend for gradient computation:

```julia
using Enzyme
using QuantNova

# Use Enzyme for faster gradients on large portfolios
with_backend(EnzymeBackend()) do
    result = optimize(MeanVariance(mu, Sigma); target_return=0.10)
end
```

## Constraints

All optimizers enforce:
- Weights sum to 1 (fully invested)
- Weights ≥ 0 (long-only, by default)

## Custom Optimization

Access the underlying gradient functions:

```julia
# Portfolio variance as a function of weights
f(w) = w' * Sigma * w

# Gradient via AD
g = gradient(f, w0; backend=EnzymeBackend())

# Hessian
H = hessian(f, w0; backend=EnzymeBackend())
```

## Model Calibration

QuantNova also supports model calibration using AD-powered optimization.

### SABR Calibration

```julia
# Market quotes
quotes = [
    OptionQuote(95.0, 0.5, 0.0, :call, 0.22),
    OptionQuote(100.0, 0.5, 0.0, :call, 0.20),
    OptionQuote(105.0, 0.5, 0.0, :call, 0.21),
]
smile = SmileData(0.5, 100.0, 0.05, quotes)

result = calibrate_sabr(smile; beta=1.0)
result.params.alpha  # ATM vol
result.params.rho    # Skew
result.params.nu     # Smile curvature
result.rmse          # Fit quality
```

### Heston Calibration

```julia
# Multiple expiries
surface = VolSurface([smile_3m, smile_6m, smile_1y])
result = calibrate_heston(surface)

result.params.v0     # Initial variance
result.params.kappa  # Mean reversion
result.params.theta  # Long-term variance
result.params.sigma  # Vol of vol
result.params.rho    # Correlation
```

## Visualization

QuantNova provides built-in visualization for optimization results using Makie.jl.

### Efficient Frontier

<img src="../assets/viz-frontier-light.png" class="only-light" alt="Efficient Frontier">
<img src="../assets/viz-frontier-dark.png" class="only-dark" alt="Efficient Frontier">

```julia
using CairoMakie  # or GLMakie for interactive plots

# Visualize the efficient frontier with individual assets
spec = visualize(result, :frontier;
    title="Risk-Return Tradeoff",
    μ=expected_returns,
    Σ=covariance_matrix,
    assets=[:AAPL, :MSFT, :GOOGL, :AMZN, :META]
)
fig = render(spec)
```

### Portfolio Weights

<img src="../assets/viz-weights-light.png" class="only-light" alt="Portfolio Weights">
<img src="../assets/viz-weights-dark.png" class="only-dark" alt="Portfolio Weights">

```julia
spec = visualize(result, :weights;
    title="Portfolio Allocation",
    assets=[:AAPL, :MSFT, :GOOGL, :AMZN, :META]
)
fig = render(spec)
```

For a complete optimization workflow with visualization, see the [Optimization Demo](../examples/optimization-demo.md).
