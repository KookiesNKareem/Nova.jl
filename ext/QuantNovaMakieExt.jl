module QuantNovaMakieExt

using QuantNova
using QuantNova.Visualization: VisualizationSpec, get_theme, COLORS
using QuantNova.Backtesting: BacktestResult
using QuantNova.Optimization: OptimizationResult, EfficientFrontier

using Makie
using Dates
using Statistics: mean, std

# ============================================================================
# LTTB Downsampling Algorithm
# ============================================================================
# Largest Triangle Three Buckets algorithm for downsampling time series
# while preserving visual shape. Critical for plotting long time series.

"""
    lttb_downsample(x, y, n_out)

Downsample data using Largest Triangle Three Buckets algorithm.
Preserves visual shape of the data while reducing point count.

# Arguments
- `x`: x-coordinates (e.g., timestamps converted to numbers)
- `y`: y-coordinates (e.g., equity values)
- `n_out`: target number of output points

# Returns
- Tuple of (x_out, y_out) downsampled arrays
"""
function lttb_downsample(x::AbstractVector, y::AbstractVector, n_out::Int)
    n = length(x)

    # No downsampling needed
    if n <= n_out
        return (collect(x), collect(y))
    end

    # Always keep first and last points
    x_out = Vector{eltype(x)}(undef, n_out)
    y_out = Vector{eltype(y)}(undef, n_out)

    x_out[1] = x[1]
    y_out[1] = y[1]
    x_out[n_out] = x[n]
    y_out[n_out] = y[n]

    # Bucket size
    bucket_size = (n - 2) / (n_out - 2)

    a = 1  # Index of previously selected point

    for i in 2:(n_out - 1)
        # Calculate average point in next bucket (for area calculation)
        avg_start = floor(Int, (i - 1) * bucket_size) + 2
        avg_end = floor(Int, i * bucket_size) + 2
        avg_end = min(avg_end, n)

        avg_x = mean(x[avg_start:avg_end])
        avg_y = mean(y[avg_start:avg_end])

        # Range for current bucket
        range_start = floor(Int, (i - 2) * bucket_size) + 2
        range_end = floor(Int, (i - 1) * bucket_size) + 2
        range_end = min(range_end, n)

        # Find point in current bucket with largest triangle area
        max_area = -1.0
        max_idx = range_start

        for j in range_start:range_end
            # Calculate triangle area (Shoelace formula)
            area = abs((x[a] - avg_x) * (y[j] - y[a]) -
                      (x[a] - x[j]) * (avg_y - y[a])) * 0.5
            if area > max_area
                max_area = area
                max_idx = j
            end
        end

        x_out[i] = x[max_idx]
        y_out[i] = y[max_idx]
        a = max_idx
    end

    return (x_out, y_out)
end

"""
    downsample_for_display(timestamps, values; max_points=2000)

Prepare data for display, downsampling if necessary.

# Arguments
- `timestamps`: Vector of DateTime
- `values`: Vector of values
- `max_points`: Maximum points to display (default 2000)

# Returns
- Tuple of (timestamps, values) possibly downsampled
"""
function downsample_for_display(timestamps::Vector{DateTime}, values::AbstractVector;
                                 max_points::Int=2000)
    if length(timestamps) <= max_points
        return (timestamps, values)
    end

    # Convert timestamps to numeric for LTTB
    t0 = timestamps[1]
    x_numeric = [Dates.value(t - t0) / 1000.0 for t in timestamps]  # seconds

    x_down, y_down = lttb_downsample(x_numeric, values, max_points)

    # Convert back to DateTime
    timestamps_down = [t0 + Dates.Millisecond(round(Int, x * 1000)) for x in x_down]

    return (timestamps_down, y_down)
end

"""
    timestamps_to_numeric(timestamps)

Convert DateTime vector to numeric values for plotting.
Returns (numeric_values, tick_formatter) tuple.
"""
function timestamps_to_numeric(timestamps::Vector{DateTime})
    t0 = timestamps[1]
    # Days since start
    numeric = [Dates.value(t - t0) / (24 * 60 * 60 * 1000.0) for t in timestamps]

    # Create tick formatter
    function format_ticks(vals)
        formatted = String[]
        for v in vals
            t = t0 + Dates.Day(round(Int, v))
            push!(formatted, Dates.format(t, "yyyy-mm-dd"))
        end
        return formatted
    end

    return (numeric, format_ticks, t0)
end

# ============================================================================
# Theme Application
# ============================================================================

"""
    to_color(c)

Convert a color specification to a Makie-compatible color.
"""
function to_color(c)
    if c isa Symbol
        return Makie.to_color(c)
    elseif c isa String
        return Makie.parse(Makie.Colorant, c)
    else
        return c
    end
end

"""
    apply_theme!(ax, theme)

Apply theme settings to a Makie axis.
"""
function apply_theme!(ax, theme::Dict{Symbol,Any})
    ax.backgroundcolor[] = to_color(theme[:backgroundcolor])
    ax.xgridcolor[] = to_color(theme[:gridcolor])
    ax.ygridcolor[] = to_color(theme[:gridcolor])
    ax.xlabelcolor[] = to_color(theme[:textcolor])
    ax.ylabelcolor[] = to_color(theme[:textcolor])
    ax.xticklabelcolor[] = to_color(theme[:textcolor])
    ax.yticklabelcolor[] = to_color(theme[:textcolor])
    ax.titlecolor[] = to_color(theme[:textcolor])
end

"""
    get_color(theme, idx)

Get a color from the theme palette by index (cycling).
"""
function get_color(theme::Dict{Symbol,Any}, idx::Int)
    palette = theme[:palette]
    return palette[mod1(idx, length(palette))]
end

# ============================================================================
# Drawdown Calculation
# ============================================================================

"""
    compute_drawdown(equity_curve)

Compute drawdown series from equity curve.
"""
function compute_drawdown(equity_curve::Vector{Float64})
    n = length(equity_curve)
    drawdown = zeros(n)
    peak = equity_curve[1]

    for i in 1:n
        peak = max(peak, equity_curve[i])
        drawdown[i] = (equity_curve[i] - peak) / peak
    end

    return drawdown
end

# ============================================================================
# BacktestResult Render Functions
# ============================================================================

"""
    render(spec::VisualizationSpec{BacktestResult})

Render a backtest visualization specification.
"""
function QuantNova.Visualization.render(spec::VisualizationSpec{BacktestResult})
    view = spec.view

    if view == :equity
        return render_equity(spec)
    elseif view == :drawdown
        return render_drawdown(spec)
    elseif view == :returns
        return render_returns(spec)
    elseif view == :rolling
        return render_rolling(spec)
    elseif view == :monthly
        return render_monthly(spec)
    elseif view == :dashboard
        return render_backtest_dashboard(spec)
    else
        error("Unknown view for BacktestResult: $view. Available: :equity, :drawdown, :returns, :rolling, :monthly, :dashboard")
    end
end

"""
    render_equity(spec::VisualizationSpec{BacktestResult})

Render equity curve plot.
"""
function render_equity(spec::VisualizationSpec{BacktestResult})
    result = spec.data
    opts = spec.options
    theme = get(opts, :theme, get_theme())

    timestamps, equity = downsample_for_display(result.timestamps, result.equity_curve)
    x_numeric, format_ticks, _ = timestamps_to_numeric(timestamps)

    fig = Figure(backgroundcolor=to_color(theme[:backgroundcolor]), size=(800, 500))
    ax = Axis(fig[1, 1],
        title="Equity Curve",
        xlabel="Date",
        ylabel="Portfolio Value",
        titlesize=theme[:titlesize],
        xlabelsize=theme[:fontsize],
        ylabelsize=theme[:fontsize],
        xtickformat=format_ticks
    )
    apply_theme!(ax, theme)

    # Plot equity curve
    lines!(ax, x_numeric, equity, color=get_color(theme, 1), linewidth=2)

    # Add initial value reference line
    hlines!(ax, [result.initial_value], color=to_color(theme[:gridcolor]), linestyle=:dash, linewidth=1)

    return fig
end

"""
    render_drawdown(spec::VisualizationSpec{BacktestResult})

Render drawdown plot.
"""
function render_drawdown(spec::VisualizationSpec{BacktestResult})
    result = spec.data
    opts = spec.options
    theme = get(opts, :theme, get_theme())

    drawdown = compute_drawdown(result.equity_curve)
    timestamps, dd = downsample_for_display(result.timestamps, drawdown)
    x_numeric, format_ticks, _ = timestamps_to_numeric(timestamps)

    fig = Figure(backgroundcolor=to_color(theme[:backgroundcolor]), size=(800, 400))
    ax = Axis(fig[1, 1],
        title="Drawdown",
        xlabel="Date",
        ylabel="Drawdown (%)",
        titlesize=theme[:titlesize],
        xlabelsize=theme[:fontsize],
        ylabelsize=theme[:fontsize],
        xtickformat=format_ticks
    )
    apply_theme!(ax, theme)

    # Fill area under drawdown curve
    band!(ax, x_numeric, dd, zeros(length(dd)), color=(to_color(COLORS[:loss]), 0.3))
    lines!(ax, x_numeric, dd, color=to_color(COLORS[:loss]), linewidth=1.5)

    # Format y-axis as percentage
    ax.ytickformat = ys -> ["$(round(y*100, digits=1))%" for y in ys]

    return fig
end

"""
    render_returns(spec::VisualizationSpec{BacktestResult})

Render returns distribution histogram.
"""
function render_returns(spec::VisualizationSpec{BacktestResult})
    result = spec.data
    opts = spec.options
    theme = get(opts, :theme, get_theme())

    returns = result.returns

    fig = Figure(backgroundcolor=to_color(theme[:backgroundcolor]), size=(800, 400))
    ax = Axis(fig[1, 1],
        title="Returns Distribution",
        xlabel="Return",
        ylabel="Frequency",
        titlesize=theme[:titlesize],
        xlabelsize=theme[:fontsize],
        ylabelsize=theme[:fontsize]
    )
    apply_theme!(ax, theme)

    # Histogram with color based on sign
    hist!(ax, returns, bins=50, color=get_color(theme, 1), strokewidth=1, strokecolor=to_color(theme[:gridcolor]))

    # Add vertical line at zero
    vlines!(ax, [0.0], color=to_color(theme[:gridcolor]), linestyle=:dash)

    # Add mean line
    mean_ret = mean(returns)
    vlines!(ax, [mean_ret], color=to_color(COLORS[:highlight]), linewidth=2, label="Mean: $(round(mean_ret*100, digits=2))%")

    axislegend(ax, position=:rt, backgroundcolor=(to_color(theme[:backgroundcolor]), 0.8))

    return fig
end

"""
    render_rolling(spec::VisualizationSpec{BacktestResult})

Render rolling metrics (Sharpe ratio and volatility).
"""
function render_rolling(spec::VisualizationSpec{BacktestResult})
    result = spec.data
    opts = spec.options
    theme = get(opts, :theme, get_theme())
    window = get(opts, :window, 63)  # Default ~3 months for daily data

    returns = result.returns
    timestamps = result.timestamps
    n = length(returns)

    # Compute rolling metrics
    if n < window
        error("Not enough data points for rolling window of $window")
    end

    rolling_sharpe = Vector{Float64}(undef, n - window + 1)
    rolling_vol = Vector{Float64}(undef, n - window + 1)

    for i in 1:(n - window + 1)
        window_returns = returns[i:(i + window - 1)]
        μ = mean(window_returns)
        σ = std(window_returns)
        rolling_sharpe[i] = σ > 0 ? (μ / σ) * sqrt(252) : 0.0
        rolling_vol[i] = σ * sqrt(252)
    end

    rolling_timestamps = timestamps[window:end]
    ts_sharpe, sharpe_down = downsample_for_display(rolling_timestamps, rolling_sharpe)
    ts_vol, vol_down = downsample_for_display(rolling_timestamps, rolling_vol)

    x_sharpe, format_ticks_sharpe, _ = timestamps_to_numeric(ts_sharpe)
    x_vol, format_ticks_vol, _ = timestamps_to_numeric(ts_vol)

    fig = Figure(backgroundcolor=to_color(theme[:backgroundcolor]), size=(800, 600))

    # Rolling Sharpe
    ax1 = Axis(fig[1, 1],
        title="Rolling Sharpe Ratio ($(window)-day)",
        xlabel="",
        ylabel="Sharpe Ratio",
        titlesize=theme[:titlesize],
        xlabelsize=theme[:fontsize],
        ylabelsize=theme[:fontsize],
        xtickformat=format_ticks_sharpe
    )
    apply_theme!(ax1, theme)
    lines!(ax1, x_sharpe, sharpe_down, color=get_color(theme, 1), linewidth=1.5)
    hlines!(ax1, [0.0], color=to_color(theme[:gridcolor]), linestyle=:dash)

    # Rolling Volatility
    ax2 = Axis(fig[2, 1],
        title="Rolling Volatility ($(window)-day)",
        xlabel="Date",
        ylabel="Annualized Vol",
        titlesize=theme[:titlesize],
        xlabelsize=theme[:fontsize],
        ylabelsize=theme[:fontsize],
        xtickformat=format_ticks_vol
    )
    apply_theme!(ax2, theme)
    lines!(ax2, x_vol, vol_down, color=get_color(theme, 2), linewidth=1.5)
    ax2.ytickformat = ys -> ["$(round(y*100, digits=1))%" for y in ys]

    return fig
end

"""
    render_monthly(spec::VisualizationSpec{BacktestResult})

Render monthly returns heatmap.
"""
function render_monthly(spec::VisualizationSpec{BacktestResult})
    result = spec.data
    opts = spec.options
    theme = get(opts, :theme, get_theme())

    returns = result.returns
    timestamps = result.timestamps

    # Aggregate returns by month
    monthly_data = Dict{Tuple{Int,Int}, Float64}()

    for (i, t) in enumerate(timestamps)
        if i > length(returns)
            break
        end
        key = (year(t), month(t))
        if haskey(monthly_data, key)
            # Compound returns
            monthly_data[key] = (1 + monthly_data[key]) * (1 + returns[i]) - 1
        else
            monthly_data[key] = returns[i]
        end
    end

    if isempty(monthly_data)
        error("No monthly data to plot")
    end

    # Extract years and months
    years = sort(unique([k[1] for k in keys(monthly_data)]))
    months = 1:12

    # Create matrix for heatmap
    n_years = length(years)
    monthly_matrix = fill(NaN, 12, n_years)

    for ((y, m), ret) in monthly_data
        year_idx = findfirst(==(y), years)
        if year_idx !== nothing
            monthly_matrix[m, year_idx] = ret * 100  # Convert to percentage
        end
    end

    fig = Figure(backgroundcolor=to_color(theme[:backgroundcolor]), size=(max(400, n_years * 60), 500))
    ax = Axis(fig[1, 1],
        title="Monthly Returns (%)",
        xlabel="Year",
        ylabel="Month",
        titlesize=theme[:titlesize],
        xlabelsize=theme[:fontsize],
        ylabelsize=theme[:fontsize],
        xticks=(1:n_years, string.(years)),
        yticks=(1:12, ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]),
        yreversed=true
    )
    apply_theme!(ax, theme)

    # Heatmap with diverging colormap
    valid_vals = filter(!isnan, monthly_matrix)
    if !isempty(valid_vals)
        maxabs = max(abs(minimum(valid_vals)), abs(maximum(valid_vals)), 5.0)
        hm = heatmap!(ax, 1:n_years, 1:12, monthly_matrix',
                      colormap=:RdYlGn, colorrange=(-maxabs, maxabs))
        Colorbar(fig[1, 2], hm, label="Return (%)")
    end

    return fig
end

"""
    render_backtest_dashboard(spec::VisualizationSpec{BacktestResult})

Render comprehensive backtest dashboard with multiple views.
"""
function render_backtest_dashboard(spec::VisualizationSpec{BacktestResult})
    result = spec.data
    opts = spec.options
    theme = get(opts, :theme, get_theme())

    # Prepare data
    timestamps, equity = downsample_for_display(result.timestamps, result.equity_curve)
    drawdown = compute_drawdown(result.equity_curve)
    ts_dd, dd = downsample_for_display(result.timestamps, drawdown)
    returns = result.returns

    # Convert timestamps to numeric
    x_equity, format_ticks, _ = timestamps_to_numeric(timestamps)
    x_dd, format_ticks_dd, _ = timestamps_to_numeric(ts_dd)

    fig = Figure(backgroundcolor=to_color(theme[:backgroundcolor]), size=(1200, 800))

    # Title with key metrics
    metrics = result.metrics
    title_text = "Backtest Dashboard"
    if haskey(metrics, :sharpe)
        title_text *= " | Sharpe: $(round(metrics[:sharpe], digits=2))"
    end
    if haskey(metrics, :max_drawdown)
        title_text *= " | Max DD: $(round(metrics[:max_drawdown]*100, digits=1))%"
    end

    Label(fig[0, :], title_text, fontsize=theme[:titlesize] + 2,
          color=to_color(theme[:textcolor]), halign=:left)

    # Equity curve (top-left)
    ax1 = Axis(fig[1, 1],
        title="Equity Curve",
        xlabel="",
        ylabel="Value",
        titlesize=theme[:titlesize],
        xlabelsize=theme[:fontsize],
        ylabelsize=theme[:fontsize],
        xtickformat=format_ticks
    )
    apply_theme!(ax1, theme)
    lines!(ax1, x_equity, equity, color=get_color(theme, 1), linewidth=2)
    hlines!(ax1, [result.initial_value], color=to_color(theme[:gridcolor]), linestyle=:dash, linewidth=1)

    # Drawdown (top-right)
    ax2 = Axis(fig[1, 2],
        title="Drawdown",
        xlabel="",
        ylabel="Drawdown",
        titlesize=theme[:titlesize],
        xlabelsize=theme[:fontsize],
        ylabelsize=theme[:fontsize],
        xtickformat=format_ticks_dd
    )
    apply_theme!(ax2, theme)
    band!(ax2, x_dd, dd, zeros(length(dd)), color=(to_color(COLORS[:loss]), 0.3))
    lines!(ax2, x_dd, dd, color=to_color(COLORS[:loss]), linewidth=1.5)
    ax2.ytickformat = ys -> ["$(round(y*100, digits=1))%" for y in ys]

    # Returns distribution (bottom-left)
    ax3 = Axis(fig[2, 1],
        title="Returns Distribution",
        xlabel="Return",
        ylabel="Frequency",
        titlesize=theme[:titlesize],
        xlabelsize=theme[:fontsize],
        ylabelsize=theme[:fontsize]
    )
    apply_theme!(ax3, theme)
    hist!(ax3, returns, bins=30, color=get_color(theme, 1), strokewidth=1, strokecolor=to_color(theme[:gridcolor]))
    vlines!(ax3, [0.0], color=to_color(theme[:gridcolor]), linestyle=:dash)
    vlines!(ax3, [mean(returns)], color=to_color(COLORS[:highlight]), linewidth=2)

    # Metrics table (bottom-right)
    ax4 = Axis(fig[2, 2],
        title="Performance Metrics",
        titlesize=theme[:titlesize]
    )
    apply_theme!(ax4, theme)
    hidedecorations!(ax4)
    hidespines!(ax4)

    # Format metrics as text
    total_return = (result.final_value - result.initial_value) / result.initial_value
    metric_lines = [
        "Total Return: $(round(total_return * 100, digits=2))%",
        "Initial: \$$(round(result.initial_value, digits=2))",
        "Final: \$$(round(result.final_value, digits=2))",
    ]

    for (key, val) in metrics
        push!(metric_lines, "$(key): $(round(val, digits=4))")
    end

    for (i, line) in enumerate(metric_lines[1:min(8, length(metric_lines))])
        text!(ax4, 0.1, 1.0 - i * 0.1, text=line,
              fontsize=theme[:fontsize], color=to_color(theme[:textcolor]),
              align=(:left, :top))
    end

    return fig
end

# ============================================================================
# OptimizationResult Render Functions
# ============================================================================

"""
    render(spec::VisualizationSpec{OptimizationResult})

Render an optimization visualization specification.
"""
function QuantNova.Visualization.render(spec::VisualizationSpec{OptimizationResult})
    view = spec.view

    if view == :weights
        return render_weights(spec)
    elseif view == :frontier
        return render_frontier_single(spec)
    elseif view == :risk
        return render_risk(spec)
    else
        error("Unknown view for OptimizationResult: $view. Available: :weights, :frontier, :risk")
    end
end

"""
    render_weights(spec::VisualizationSpec{OptimizationResult})

Render portfolio weights bar chart.
"""
function render_weights(spec::VisualizationSpec{OptimizationResult})
    result = spec.data
    opts = spec.options
    theme = get(opts, :theme, get_theme())

    weights = result.weights
    n = length(weights)
    labels = get(opts, :labels, ["Asset $i" for i in 1:n])

    fig = Figure(backgroundcolor=to_color(theme[:backgroundcolor]), size=(800, 500))
    ax = Axis(fig[1, 1],
        title="Portfolio Weights",
        xlabel="Asset",
        ylabel="Weight",
        titlesize=theme[:titlesize],
        xlabelsize=theme[:fontsize],
        ylabelsize=theme[:fontsize],
        xticks=(1:n, labels),
        xticklabelrotation=pi/4
    )
    apply_theme!(ax, theme)

    # Color positive and negative weights differently
    colors = [w >= 0 ? to_color(COLORS[:profit]) : to_color(COLORS[:loss]) for w in weights]
    barplot!(ax, 1:n, weights, color=colors, strokewidth=1, strokecolor=to_color(theme[:gridcolor]))

    # Reference line at zero
    hlines!(ax, [0.0], color=to_color(theme[:gridcolor]), linestyle=:dash)

    ax.ytickformat = ys -> ["$(round(y*100, digits=1))%" for y in ys]

    return fig
end

"""
    render_frontier_single(spec::VisualizationSpec{OptimizationResult})

Render a single portfolio point on risk-return space.
"""
function render_frontier_single(spec::VisualizationSpec{OptimizationResult})
    result = spec.data
    opts = spec.options
    theme = get(opts, :theme, get_theme())

    # For a single optimization result, we just show the point
    # We need expected return and volatility from options or compute from weights
    expected_return = get(opts, :expected_return, result.objective)
    volatility = get(opts, :volatility, 0.0)

    fig = Figure(backgroundcolor=to_color(theme[:backgroundcolor]), size=(600, 500))
    ax = Axis(fig[1, 1],
        title="Optimal Portfolio",
        xlabel="Volatility (Annualized)",
        ylabel="Expected Return (Annualized)",
        titlesize=theme[:titlesize],
        xlabelsize=theme[:fontsize],
        ylabelsize=theme[:fontsize]
    )
    apply_theme!(ax, theme)

    scatter!(ax, [volatility], [expected_return],
             color=to_color(COLORS[:highlight]), markersize=15,
             marker=:star5, label="Optimal")

    axislegend(ax, position=:rb, backgroundcolor=(to_color(theme[:backgroundcolor]), 0.8))

    return fig
end

"""
    render_risk(spec::VisualizationSpec{OptimizationResult})

Render risk contribution breakdown.
"""
function render_risk(spec::VisualizationSpec{OptimizationResult})
    result = spec.data
    opts = spec.options
    theme = get(opts, :theme, get_theme())

    weights = result.weights
    n = length(weights)
    labels = get(opts, :labels, ["Asset $i" for i in 1:n])

    # Risk contributions need covariance matrix
    risk_contributions = get(opts, :risk_contributions, abs.(weights) / sum(abs.(weights)))

    fig = Figure(backgroundcolor=to_color(theme[:backgroundcolor]), size=(800, 500))
    ax = Axis(fig[1, 1],
        title="Risk Contribution",
        xlabel="Asset",
        ylabel="Risk Contribution",
        titlesize=theme[:titlesize],
        xlabelsize=theme[:fontsize],
        ylabelsize=theme[:fontsize],
        xticks=(1:n, labels),
        xticklabelrotation=pi/4
    )
    apply_theme!(ax, theme)

    colors = [get_color(theme, i) for i in 1:n]
    barplot!(ax, 1:n, risk_contributions, color=colors, strokewidth=1, strokecolor=to_color(theme[:gridcolor]))

    ax.ytickformat = ys -> ["$(round(y*100, digits=1))%" for y in ys]

    return fig
end

# ============================================================================
# EfficientFrontier Render Functions
# ============================================================================

"""
    render(spec::VisualizationSpec{EfficientFrontier})

Render an efficient frontier visualization.
"""
function QuantNova.Visualization.render(spec::VisualizationSpec{EfficientFrontier})
    view = spec.view

    if view == :frontier
        return render_frontier(spec)
    elseif view == :weights
        return render_frontier_weights(spec)
    else
        error("Unknown view for EfficientFrontier: $view. Available: :frontier, :weights")
    end
end

"""
    render_frontier(spec::VisualizationSpec{EfficientFrontier})

Render efficient frontier curve.
"""
function render_frontier(spec::VisualizationSpec{EfficientFrontier})
    frontier = spec.data
    opts = spec.options
    theme = get(opts, :theme, get_theme())

    fig = Figure(backgroundcolor=to_color(theme[:backgroundcolor]), size=(800, 600))
    ax = Axis(fig[1, 1],
        title="Efficient Frontier",
        xlabel="Volatility (Annualized)",
        ylabel="Expected Return (Annualized)",
        titlesize=theme[:titlesize],
        xlabelsize=theme[:fontsize],
        ylabelsize=theme[:fontsize]
    )
    apply_theme!(ax, theme)

    # Plot frontier curve
    lines!(ax, frontier.volatilities, frontier.returns,
           color=get_color(theme, 1), linewidth=2, label="Efficient Frontier")

    # Mark special portfolios
    scatter!(ax, [frontier.volatilities[frontier.min_variance_idx]],
             [frontier.returns[frontier.min_variance_idx]],
             color=to_color(COLORS[:profit]), markersize=12, marker=:circle,
             label="Min Variance")

    scatter!(ax, [frontier.volatilities[frontier.max_sharpe_idx]],
             [frontier.returns[frontier.max_sharpe_idx]],
             color=to_color(COLORS[:highlight]), markersize=15, marker=:star5,
             label="Max Sharpe")

    # Add individual assets if provided
    asset_returns = get(opts, :asset_returns, nothing)
    asset_vols = get(opts, :asset_volatilities, nothing)
    if asset_returns !== nothing && asset_vols !== nothing
        scatter!(ax, asset_vols, asset_returns,
                 color=to_color(theme[:gridcolor]), markersize=8, marker=:diamond,
                 label="Individual Assets")
    end

    axislegend(ax, position=:rb, backgroundcolor=(to_color(theme[:backgroundcolor]), 0.8))

    # Format axes as percentages
    ax.xtickformat = xs -> ["$(round(x*100, digits=1))%" for x in xs]
    ax.ytickformat = ys -> ["$(round(y*100, digits=1))%" for y in ys]

    return fig
end

"""
    render_frontier_weights(spec::VisualizationSpec{EfficientFrontier})

Render stacked area chart of portfolio weights along the frontier.
"""
function render_frontier_weights(spec::VisualizationSpec{EfficientFrontier})
    frontier = spec.data
    opts = spec.options
    theme = get(opts, :theme, get_theme())

    n_assets = frontier.n_assets
    labels = get(opts, :labels, ["Asset $i" for i in 1:n_assets])

    fig = Figure(backgroundcolor=to_color(theme[:backgroundcolor]), size=(900, 500))
    ax = Axis(fig[1, 1],
        title="Portfolio Weights Along Frontier",
        xlabel="Expected Return",
        ylabel="Weight",
        titlesize=theme[:titlesize],
        xlabelsize=theme[:fontsize],
        ylabelsize=theme[:fontsize]
    )
    apply_theme!(ax, theme)

    # Stacked area plot
    n_points = length(frontier.returns)
    weights_matrix = frontier.weights  # n_points x n_assets

    # Build cumulative weights for stacking
    lower = zeros(n_points)
    for i in 1:n_assets
        upper = lower .+ weights_matrix[:, i]
        band!(ax, frontier.returns, lower, upper,
              color=(get_color(theme, i), 0.7), label=labels[i])
        lower = upper
    end

    axislegend(ax, position=:rt, backgroundcolor=(to_color(theme[:backgroundcolor]), 0.8))

    ax.xtickformat = xs -> ["$(round(x*100, digits=1))%" for x in xs]

    return fig
end

# ============================================================================
# Base.display Extension
# ============================================================================

"""
Override display for VisualizationSpec to automatically render.
"""
function Base.display(d::AbstractDisplay, spec::VisualizationSpec)
    fig = QuantNova.Visualization.render(spec)
    display(d, fig)
end

function Base.display(spec::VisualizationSpec)
    fig = QuantNova.Visualization.render(spec)
    display(fig)
end

# ============================================================================
# Module Initialization
# ============================================================================

function __init__()
    @info "QuantNovaMakieExt loaded: Visualization rendering enabled"
end

end # module
