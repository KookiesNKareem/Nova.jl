module MarketData

using ..Core
using Dates
using DelimitedFiles
using YFinance

# ============================================================================
# Abstract Types
# ============================================================================

"""
    AbstractMarketData

Base type for all market data containers.
"""
abstract type AbstractMarketData end

"""
    AbstractPriceHistory <: AbstractMarketData

Base type for historical price data (OHLCV).
"""
abstract type AbstractPriceHistory <: AbstractMarketData end

"""
    AbstractDataAdapter

Base type for data loading adapters.
"""
abstract type AbstractDataAdapter end

# ============================================================================
# OHLCV Price History
# ============================================================================

"""
    PriceHistory <: AbstractPriceHistory

Container for OHLCV (Open, High, Low, Close, Volume) price data.

# Fields
- `symbol::String` - Ticker symbol
- `timestamps::Vector{DateTime}` - Timestamps for each bar
- `open::Vector{Float64}` - Opening prices
- `high::Vector{Float64}` - High prices
- `low::Vector{Float64}` - Low prices
- `close::Vector{Float64}` - Closing prices
- `volume::Vector{Float64}` - Trading volume
"""
struct PriceHistory <: AbstractPriceHistory
    symbol::String
    timestamps::Vector{DateTime}
    open::Vector{Float64}
    high::Vector{Float64}
    low::Vector{Float64}
    close::Vector{Float64}
    volume::Vector{Float64}

    function PriceHistory(symbol, timestamps, open, high, low, close, volume)
        n = length(timestamps)
        length(open) == n || error("open must have same length as timestamps")
        length(high) == n || error("high must have same length as timestamps")
        length(low) == n || error("low must have same length as timestamps")
        length(close) == n || error("close must have same length as timestamps")
        length(volume) == n || error("volume must have same length as timestamps")
        new(symbol, timestamps, open, high, low, close, volume)
    end
end

# Convenience constructor for close-only data
function PriceHistory(symbol::String, timestamps::Vector{DateTime}, close::Vector{Float64})
    n = length(timestamps)
    PriceHistory(symbol, timestamps, close, close, close, close, zeros(n))
end

Base.length(ph::PriceHistory) = length(ph.timestamps)
Base.getindex(ph::PriceHistory, i::Int) = (
    timestamp=ph.timestamps[i],
    open=ph.open[i],
    high=ph.high[i],
    low=ph.low[i],
    close=ph.close[i],
    volume=ph.volume[i]
)
Base.getindex(ph::PriceHistory, r::UnitRange) = PriceHistory(
    ph.symbol,
    ph.timestamps[r],
    ph.open[r],
    ph.high[r],
    ph.low[r],
    ph.close[r],
    ph.volume[r]
)

"""
    returns(ph::PriceHistory; type=:simple)

Compute returns from price history.

# Arguments
- `type` - :simple for (P_t - P_{t-1})/P_{t-1}, :log for log(P_t/P_{t-1})
"""
function returns(ph::PriceHistory; type::Symbol=:simple)
    prices = ph.close
    n = length(prices)
    n < 2 && error("Need at least 2 prices to compute returns")

    if type == :simple
        return [(prices[i] - prices[i-1]) / prices[i-1] for i in 2:n]
    elseif type == :log
        return [log(prices[i] / prices[i-1]) for i in 2:n]
    else
        error("type must be :simple or :log")
    end
end

# ============================================================================
# CSV Adapter
# ============================================================================

"""
    CSVAdapter <: AbstractDataAdapter

Adapter for loading market data from CSV files.

Supports flexible column mapping for different CSV formats.
"""
struct CSVAdapter <: AbstractDataAdapter
    date_format::String
    date_col::Int
    open_col::Int
    high_col::Int
    low_col::Int
    close_col::Int
    volume_col::Int
    has_header::Bool

    function CSVAdapter(;
        date_format::String="yyyy-mm-dd",
        date_col::Int=1,
        open_col::Int=2,
        high_col::Int=3,
        low_col::Int=4,
        close_col::Int=5,
        volume_col::Int=6,
        has_header::Bool=true
    )
        new(date_format, date_col, open_col, high_col, low_col, close_col, volume_col, has_header)
    end
end

# Default adapter for Yahoo Finance CSV format
const YAHOO_ADAPTER = CSVAdapter(
    date_format="yyyy-mm-dd",
    date_col=1, open_col=2, high_col=3, low_col=4, close_col=5, volume_col=7,
    has_header=true
)

"""
    load(adapter::CSVAdapter, filepath::String, symbol::String) -> PriceHistory

Load price history from a CSV file.
"""
function load(adapter::CSVAdapter, filepath::String, symbol::String)
    isfile(filepath) || error("File not found: $filepath")

    data, header = readdlm(filepath, ',', Any, '\n'; header=adapter.has_header)

    n = size(data, 1)
    timestamps = Vector{DateTime}(undef, n)
    open_prices = Vector{Float64}(undef, n)
    high_prices = Vector{Float64}(undef, n)
    low_prices = Vector{Float64}(undef, n)
    close_prices = Vector{Float64}(undef, n)
    volumes = Vector{Float64}(undef, n)

    for i in 1:n
        timestamps[i] = DateTime(string(data[i, adapter.date_col]), adapter.date_format)
        open_prices[i] = Float64(data[i, adapter.open_col])
        high_prices[i] = Float64(data[i, adapter.high_col])
        low_prices[i] = Float64(data[i, adapter.low_col])
        close_prices[i] = Float64(data[i, adapter.close_col])
        volumes[i] = Float64(data[i, adapter.volume_col])
    end

    # Sort by timestamp (ascending)
    perm = sortperm(timestamps)
    PriceHistory(
        symbol,
        timestamps[perm],
        open_prices[perm],
        high_prices[perm],
        low_prices[perm],
        close_prices[perm],
        volumes[perm]
    )
end

"""
    save(adapter::CSVAdapter, ph::PriceHistory, filepath::String)

Save price history to a CSV file.
"""
function save(adapter::CSVAdapter, ph::PriceHistory, filepath::String)
    n = length(ph)
    data = Matrix{Any}(undef, n + 1, 6)

    # Header
    data[1, :] = ["Date", "Open", "High", "Low", "Close", "Volume"]

    # Data
    for i in 1:n
        data[i+1, 1] = Dates.format(ph.timestamps[i], adapter.date_format)
        data[i+1, 2] = ph.open[i]
        data[i+1, 3] = ph.high[i]
        data[i+1, 4] = ph.low[i]
        data[i+1, 5] = ph.close[i]
        data[i+1, 6] = ph.volume[i]
    end

    writedlm(filepath, data, ',')
end

# ============================================================================
# YFinance Integration
# ============================================================================

"""
    fetch_prices(symbol::String; range="1y", interval="1d",
                 startdt=nothing, enddt=nothing, autoadjust=true) -> PriceHistory

Fetch historical price data from Yahoo Finance.

# Arguments
- `symbol` - Ticker symbol (e.g., "AAPL", "MSFT", "SPY")
- `range` - Time range: "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"
- `interval` - Data interval: "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"
- `startdt` - Start date (overrides range if provided), format: "YYYY-MM-DD" or Date
- `enddt` - End date, format: "YYYY-MM-DD" or Date
- `autoadjust` - Whether to use adjusted prices (default: true)

# Examples
```julia
# Get 1 year of daily AAPL data
prices = fetch_prices("AAPL")

# Get 5 years of weekly data
prices = fetch_prices("AAPL", range="5y", interval="1wk")

# Get data for a specific date range
prices = fetch_prices("AAPL", startdt="2020-01-01", enddt="2024-01-01")
```
"""
function fetch_prices(symbol::String;
                      range::String="1y",
                      interval::String="1d",
                      startdt::Union{Nothing, String, Date}=nothing,
                      enddt::Union{Nothing, String, Date}=nothing,
                      autoadjust::Bool=true)

    # Build kwargs for YFinance
    kwargs = Dict{Symbol, Any}()
    kwargs[:interval] = interval
    kwargs[:autoadjust] = autoadjust

    if startdt !== nothing
        kwargs[:startdt] = string(startdt)
        if enddt !== nothing
            kwargs[:enddt] = string(enddt)
        end
    else
        kwargs[:range] = range
    end

    # Fetch from Yahoo Finance
    data = get_prices(symbol; kwargs...)

    # Convert to PriceHistory
    _yfinance_to_pricehistory(symbol, data)
end

"""
    fetch_multiple(symbols::Vector{String}; align_dates=true, kwargs...) -> Vector{PriceHistory}

Fetch historical price data for multiple symbols.

# Arguments
- `symbols` - Vector of ticker symbols
- `align_dates` - Whether to align all histories to common dates (default: true)
- `kwargs...` - Arguments passed to `fetch_prices`

# Examples
```julia
# Fetch and align multiple stocks
histories = fetch_multiple(["AAPL", "MSFT", "GOOGL"])

# Access individual histories
aapl, msft, googl = histories
```
"""
function fetch_multiple(symbols::Vector{String}; align_dates::Bool=true, kwargs...)
    histories = [fetch_prices(s; kwargs...) for s in symbols]

    if align_dates && length(histories) > 1
        return align(histories)
    end

    histories
end

"""
    fetch_returns(symbol::String; type=:simple, kwargs...) -> Vector{Float64}

Convenience function to fetch prices and compute returns directly.

# Arguments
- `symbol` - Ticker symbol
- `type` - :simple or :log returns
- `kwargs...` - Arguments passed to `fetch_prices`
"""
function fetch_returns(symbol::String; type::Symbol=:simple, kwargs...)
    ph = fetch_prices(symbol; kwargs...)
    returns(ph; type=type)
end

"""
    fetch_return_matrix(symbols::Vector{String}; type=:simple, kwargs...) -> Matrix{Float64}

Fetch aligned returns for multiple symbols as a matrix.

Returns an (n_periods x n_assets) matrix suitable for portfolio optimization.

# Examples
```julia
# Get return matrix for portfolio optimization
R = fetch_return_matrix(["AAPL", "MSFT", "GOOGL", "AMZN"], range="2y")
# R is (n_days-1) x 4 matrix
```
"""
function fetch_return_matrix(symbols::Vector{String}; type::Symbol=:simple, kwargs...)
    histories = fetch_multiple(symbols; align_dates=true, kwargs...)

    n_periods = length(histories[1]) - 1
    n_assets = length(symbols)

    R = Matrix{Float64}(undef, n_periods, n_assets)
    for (j, ph) in enumerate(histories)
        R[:, j] = returns(ph; type=type)
    end

    R
end

"""
    to_backtest_format(histories::Vector{PriceHistory}) -> (timestamps, price_series)

Convert aligned PriceHistory objects to backtest-compatible format.

Returns a tuple of:
- `timestamps::Vector{DateTime}` - Common timestamps
- `price_series::Dict{Symbol,Vector{Float64}}` - Close prices keyed by symbol

# Example
```julia
histories = fetch_multiple(["AAPL", "MSFT", "GOOGL"], range="1y")
timestamps, prices = to_backtest_format(histories)
result = backtest(strategy, timestamps, prices)
```
"""
function to_backtest_format(histories::Vector{PriceHistory})
    isempty(histories) && error("No price histories provided")

    timestamps = histories[1].timestamps
    price_series = Dict{Symbol,Vector{Float64}}()

    for ph in histories
        price_series[Symbol(ph.symbol)] = ph.close
    end

    return timestamps, price_series
end

# Internal: Convert YFinance output to PriceHistory
function _yfinance_to_pricehistory(symbol::String, data)
    # YFinance returns a dictionary with vectors
    # Keys: timestamp, open, high, low, close, vol (and adj_close if requested)

    timestamps = DateTime.(data["timestamp"])
    open_prices = Float64.(data["open"])
    high_prices = Float64.(data["high"])
    low_prices = Float64.(data["low"])
    close_prices = Float64.(data["close"])
    volumes = Float64.(data["vol"])

    # Sort by timestamp (YFinance usually returns sorted, but be safe)
    perm = sortperm(timestamps)

    PriceHistory(
        symbol,
        timestamps[perm],
        open_prices[perm],
        high_prices[perm],
        low_prices[perm],
        close_prices[perm],
        volumes[perm]
    )
end

# ============================================================================
# Parquet Adapter (Stub - requires Arrow.jl)
# ============================================================================

"""
    ParquetAdapter <: AbstractDataAdapter

Adapter for loading market data from Parquet files.

Note: Requires Arrow.jl to be loaded for full functionality.
"""
struct ParquetAdapter <: AbstractDataAdapter
    date_col::String
    open_col::String
    high_col::String
    low_col::String
    close_col::String
    volume_col::String

    function ParquetAdapter(;
        date_col::String="date",
        open_col::String="open",
        high_col::String="high",
        low_col::String="low",
        close_col::String="close",
        volume_col::String="volume"
    )
        new(date_col, open_col, high_col, low_col, close_col, volume_col)
    end
end

function load(adapter::ParquetAdapter, filepath::String, symbol::String)
    error("""
    ParquetAdapter requires Arrow.jl. Install and load it first:
        using Pkg; Pkg.add("Arrow")
        using Arrow

    Then use Arrow.Table to read the file and convert to PriceHistory.
    """)
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    resample(ph::PriceHistory, frequency::Symbol) -> PriceHistory

Resample price history to a different frequency.

# Arguments
- `frequency` - :daily, :weekly, :monthly
"""
function resample(ph::PriceHistory, frequency::Symbol)
    frequency in (:daily, :weekly, :monthly) || error("frequency must be :daily, :weekly, or :monthly")

    # Group by period
    if frequency == :daily
        return ph  # Already daily
    elseif frequency == :weekly
        period_fn = t -> (year(t), week(t))
    else  # monthly
        period_fn = t -> (year(t), month(t))
    end

    # Group indices by period
    periods = Dict{Tuple, Vector{Int}}()
    for (i, t) in enumerate(ph.timestamps)
        p = period_fn(t)
        if !haskey(periods, p)
            periods[p] = Int[]
        end
        push!(periods[p], i)
    end

    # Aggregate each period
    sorted_periods = sort(collect(keys(periods)))
    n = length(sorted_periods)

    timestamps = Vector{DateTime}(undef, n)
    open_prices = Vector{Float64}(undef, n)
    high_prices = Vector{Float64}(undef, n)
    low_prices = Vector{Float64}(undef, n)
    close_prices = Vector{Float64}(undef, n)
    volumes = Vector{Float64}(undef, n)

    for (j, p) in enumerate(sorted_periods)
        idxs = periods[p]
        timestamps[j] = ph.timestamps[idxs[end]]  # Last timestamp in period
        open_prices[j] = ph.open[idxs[1]]         # First open
        high_prices[j] = maximum(ph.high[idxs])   # Highest high
        low_prices[j] = minimum(ph.low[idxs])     # Lowest low
        close_prices[j] = ph.close[idxs[end]]     # Last close
        volumes[j] = sum(ph.volume[idxs])         # Total volume
    end

    PriceHistory(ph.symbol, timestamps, open_prices, high_prices, low_prices, close_prices, volumes)
end

"""
    align(histories::Vector{PriceHistory}) -> Vector{PriceHistory}

Align multiple price histories to common timestamps (inner join).
"""
function align(histories::Vector{PriceHistory})
    isempty(histories) && return histories

    # Find common timestamps
    common = Set(histories[1].timestamps)
    for ph in histories[2:end]
        intersect!(common, Set(ph.timestamps))
    end

    if isempty(common)
        error("No common timestamps found across price histories")
    end

    common_sorted = sort(collect(common))

    # Filter each history to common timestamps
    aligned = PriceHistory[]
    for ph in histories
        mask = [t in common for t in ph.timestamps]
        idxs = findall(mask)
        # Re-sort by common_sorted order
        idx_map = Dict(t => i for (i, t) in enumerate(ph.timestamps))
        final_idxs = [idx_map[t] for t in common_sorted]
        push!(aligned, PriceHistory(
            ph.symbol,
            common_sorted,
            ph.open[final_idxs],
            ph.high[final_idxs],
            ph.low[final_idxs],
            ph.close[final_idxs],
            ph.volume[final_idxs]
        ))
    end

    aligned
end

# ============================================================================
# Exports
# ============================================================================

export AbstractMarketData, AbstractPriceHistory, AbstractDataAdapter
export PriceHistory, returns, resample, align
export CSVAdapter, ParquetAdapter, YAHOO_ADAPTER
export load, save
export fetch_prices, fetch_multiple, fetch_returns, fetch_return_matrix
export to_backtest_format

end
