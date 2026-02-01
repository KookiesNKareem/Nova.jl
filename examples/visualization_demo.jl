# Visualization Module Demo
# Run with: julia --project examples/visualization_demo.jl

using QuantNova
using Dates
using Random
using Statistics: mean, std

Random.seed!(2024)

println("=== QuantNova Visualization Demo ===\n")

# Generate sample backtest data
n_days = 252 * 2  # 2 years
timestamps = [DateTime(2023, 1, 1) + Day(i) for i in 1:n_days]

# Simulate equity curve with realistic returns
# Daily drift of 0.0005 (~12% annual) with 1% daily vol (~16% annual)
returns = 0.0005 .+ 0.01 .* randn(n_days)
equity = zeros(n_days)
equity[1] = 10000.0
for i in 2:n_days
    equity[i] = equity[i-1] * (1 + returns[i])
end

# Compute metrics
sharpe = mean(returns) / std(returns) * sqrt(252)
max_drawdown = maximum(accumulate(max, equity) .- equity) / maximum(equity)
cagr = (equity[end] / equity[1])^(252/n_days) - 1
volatility = std(returns) * sqrt(252)

# Create BacktestResult
result = BacktestResult(
    10000.0,
    equity[end],
    equity,
    returns[2:end],  # returns has n-1 elements
    timestamps,
    Fill[],
    [Dict{Symbol,Float64}() for _ in 1:n_days],
    Dict{Symbol,Float64}(
        :sharpe_ratio => sharpe,
        :max_drawdown => -max_drawdown,
        :annualized_return => cagr,
        :volatility => volatility,
        :total_return => (equity[end] - equity[1]) / equity[1]
    )
)

println("BacktestResult created:")
println("  Initial: \$$(result.initial_value)")
println("  Final: \$$(round(result.final_value, digits=2))")
println("  Sharpe: $(round(result.metrics[:sharpe_ratio], digits=2))")
println("  Max DD: $(round(result.metrics[:max_drawdown] * 100, digits=1))%")
println()

# Demonstrate API
println("=== API Demo ===\n")

# 1. Basic visualization
println("1. Default visualization:")
spec = visualize(result)
println("   visualize(result) -> $(spec.view) view")

# 2. Specific view
println("\n2. Specific view:")
spec = visualize(result, :drawdown)
println("   visualize(result, :drawdown) -> $(spec.view) view")

# 3. Multiple views
println("\n3. Multiple views:")
specs = visualize(result, [:equity, :drawdown, :returns])
println("   visualize(result, [:equity, :drawdown, :returns]) -> $(length(specs)) specs")

# 4. With options
println("\n4. With options:")
spec = visualize(result, :equity; theme=:dark, title="My Portfolio")
println("   visualize(result, :equity; theme=:dark, title=\"My Portfolio\")")
println("   -> theme: $(spec.options[:theme][:backgroundcolor])")

# 5. Theme management
println("\n5. Theme management:")
set_theme!(:dark)
println("   set_theme!(:dark) -> bg: $(get_theme()[:backgroundcolor])")
set_theme!(:light)
println("   set_theme!(:light) -> bg: $(get_theme()[:backgroundcolor])")

# 6. Available views
println("\n6. Available views for BacktestResult:")
for view in available_views(result)
    println("   - :$view")
end

# 7. Dashboard construction
println("\n7. Dashboard construction:")
dashboard = Dashboard(
    title = "Strategy Monitor",
    theme = :dark,
    layout = [
        Row(visualize(result, :equity), weight=2),
        Row(visualize(result, :drawdown), visualize(result, :returns)),
    ]
)
println("   Dashboard with $(length(dashboard.layout)) rows")

println("\n=== Demo Complete ===")
println("\nTo render visualizations, load a Makie backend:")
println("  using GLMakie   # For desktop")
println("  using WGLMakie  # For notebooks/web")
println("  using CairoMakie # For static export")
println("\nThen call display(spec) or save(\"output.png\", spec)")
