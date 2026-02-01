module Visualization

using Dates
using Statistics: mean, std
using ..Backtesting: BacktestResult
using ..Optimization: OptimizationResult

# Core types
abstract type AbstractVisualization end

"""
    VisualizationSpec

Lazy container for visualization configuration. Not rendered until displayed.
"""
struct VisualizationSpec{T}
    data::T
    view::Symbol
    options::Dict{Symbol,Any}
end

# Theme definitions
const LIGHT_THEME = Dict{Symbol,Any}(
    :backgroundcolor => :white,
    :textcolor => "#1a1a1a",
    :gridcolor => "#e0e0e0",
    :palette => [:steelblue, :coral, :seagreen, :mediumpurple, :goldenrod],
    :fontsize => 14,
    :titlesize => 18,
)

const DARK_THEME = Dict{Symbol,Any}(
    :backgroundcolor => "#0d1117",
    :textcolor => "#e6edf3",
    :gridcolor => "#30363d",
    :palette => ["#58a6ff", "#f97583", "#56d364", "#d2a8ff", "#e3b341"],
    :fontsize => 14,
    :titlesize => 18,
)

# Semantic colors
const COLORS = Dict{Symbol,String}(
    :profit => "#56d364",
    :loss => "#f97583",
    :benchmark => "#8b949e",
    :highlight => "#58a6ff",
)

# Global state
const CURRENT_THEME = Ref{Dict{Symbol,Any}}(LIGHT_THEME)

"""
    set_theme!(theme::Symbol)

Set the global visualization theme. Options: `:light`, `:dark`.
"""
function set_theme!(theme::Symbol)
    if theme == :light
        CURRENT_THEME[] = LIGHT_THEME
    elseif theme == :dark
        CURRENT_THEME[] = DARK_THEME
    else
        error("Unknown theme: $theme. Use :light or :dark.")
    end
end

"""
    get_theme()

Get the current theme dictionary.
"""
get_theme() = CURRENT_THEME[]

"""
    visualize(data; kwargs...)
    visualize(data, view::Symbol; kwargs...)
    visualize(data, views::Vector{Symbol}; kwargs...)

Create a visualization specification for the given data.

# Arguments
- `data`: Data to visualize (BacktestResult, OptimizationResult, etc.)
- `view`: Specific view to render (e.g., `:equity`, `:drawdown`, `:frontier`)
- `views`: Multiple views to render as linked panels
- `theme`: Override theme (`:light`, `:dark`, or custom Dict)
- `backend`: Override backend (`:gl`, `:wgl`, `:cairo`)

# Examples
```julia
result = backtest(strategy, data)
visualize(result)                    # Default view
visualize(result, :drawdown)         # Specific view
visualize(result, [:equity, :drawdown])  # Multiple linked views
```
"""
function visualize end

# Default view dispatch
visualize(data; kwargs...) = visualize(data, default_view(data); kwargs...)

# Single view
function visualize(data, view::Symbol; theme=nothing, backend=nothing, kwargs...)
    opts = Dict{Symbol,Any}(kwargs...)
    if !isnothing(theme)
        opts[:theme] = theme isa Symbol ? (theme == :dark ? DARK_THEME : LIGHT_THEME) : theme
    end
    if !isnothing(backend)
        opts[:backend] = backend
    end
    VisualizationSpec(data, view, opts)
end

# Multiple views
function visualize(data, views::Vector{Symbol}; kwargs...)
    [visualize(data, v; kwargs...) for v in views]
end

# Default views for each type
default_view(::BacktestResult) = :dashboard
default_view(::OptimizationResult) = :frontier
default_view(::Any) = :auto

# Available views registry
const AVAILABLE_VIEWS = Dict{Type,Vector{Symbol}}(
    BacktestResult => [:dashboard, :equity, :drawdown, :returns, :rolling, :trades, :monthly, :tearsheet],
    OptimizationResult => [:frontier, :weights, :risk, :correlation],
)

"""
    available_views(data)

Return the list of available visualization views for the given data type.
"""
available_views(data) = get(AVAILABLE_VIEWS, typeof(data), Symbol[])

# ============================================================================
# Render function - implemented by Makie extension
# ============================================================================

"""
    render(spec::VisualizationSpec)

Render a visualization specification to produce a plot.

This function is implemented by the Makie extension (QuantNovaMakieExt).
To use it, load Makie or one of its backends (CairoMakie, GLMakie, WGLMakie).

# Examples
```julia
using QuantNova
using CairoMakie  # or GLMakie, WGLMakie

result = backtest(strategy, data)
spec = visualize(result, :equity)
fig = render(spec)
```
"""
function render end

export AbstractVisualization, VisualizationSpec
export visualize, set_theme!, get_theme, available_views
export LIGHT_THEME, DARK_THEME, COLORS
export render

end # module
