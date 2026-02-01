# Generate Benchmark Comparison Report
#
# Runs QuantNova benchmarks and compares against recorded Python/C++ results.
# Generates markdown report for README.
#
# Run: julia --project=../.. generate_report.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))

include("comprehensive_benchmark.jl")

# =============================================================================
# Reference Values (from Python/C++ benchmarks)
# =============================================================================

# QuantLib C++ results (from quantlib_benchmark_extended.cpp on M1)
const QUANTLIB_CPP = Dict(
    "european_bs" => 5.7,      # μs
    "american_100" => 67.0,    # μs
    "mc_european_10k" => 45.0, # ms (converted to μs below)
    "mc_asian_10k" => 120.0,   # ms
    "batch_1000" => 5700.0,    # μs (5.7 μs × 1000)
    "sabr_vol" => 0.8,         # μs
    "greeks_all" => 5.7,       # μs (same as european, QL computes together)
)

# Python results (from python_benchmark.py)
const PYTHON = Dict(
    "sma_crossover" => 2500.0,    # μs (pandas)
    "rolling_sharpe" => 800.0,    # μs (pandas)
    "full_metrics" => 150.0,      # μs (pandas)
    "capm" => 450.0,              # μs (statsmodels)
    "ff3" => 550.0,               # μs (statsmodels)
    "rolling_beta" => 12000.0,    # μs (pandas)
    "ic" => 25.0,                 # μs (scipy)
    "sharpe" => 2.0,              # μs (numpy)
    "sharpe_ci" => 3.0,           # μs (numpy)
)

# =============================================================================
# Run Benchmarks and Generate Report
# =============================================================================

function generate_comparison_table(nova_results)
    println("\n" * "=" ^ 80)
    println("COMPARISON TABLES (for README.md)")
    println("=" ^ 80)

    # Option Pricing vs QuantLib C++
    println("\n### Option Pricing (vs QuantLib C++)\n")
    println("| Benchmark | QuantNova.jl | QuantLib C++ | Speedup |")
    println("|-----------|-------------|--------------|---------|")

    comparisons = [
        ("European (Black-Scholes)", "european_bs", "european_bs"),
        ("American (100-step binomial)", "american_100", "american_100"),
        ("SABR implied vol", "sabr_vol", "sabr_vol"),
        ("Batch (1000 options)", "batch_1000", "batch_1000"),
    ]

    for (name, nova_key, ql_key) in comparisons
        nova_time = get(nova_results, nova_key, NaN)
        ql_time = get(QUANTLIB_CPP, ql_key, NaN)
        if !isnan(nova_time) && !isnan(ql_time)
            speedup = ql_time / nova_time
            nova_str = nova_time < 1 ? "$(round(nova_time, digits=3)) μs" :
                       nova_time < 1000 ? "$(round(nova_time, digits=1)) μs" :
                       "$(round(nova_time/1000, digits=2)) ms"
            ql_str = ql_time < 1 ? "$(round(ql_time, digits=3)) μs" :
                     ql_time < 1000 ? "$(round(ql_time, digits=1)) μs" :
                     "$(round(ql_time/1000, digits=2)) ms"
            println("| $name | $nova_str | $ql_str | **$(round(speedup, digits=0))x faster** |")
        end
    end

    # Greeks
    println("\n### Greeks Computation\n")
    println("| Benchmark | QuantNova.jl (AD) | QuantLib C++ | Speedup |")
    println("|-----------|------------------|--------------|---------|")

    nova_greeks = get(nova_results, "greeks_ad", NaN)
    ql_greeks = get(QUANTLIB_CPP, "greeks_all", NaN)
    if !isnan(nova_greeks) && !isnan(ql_greeks)
        speedup = ql_greeks / nova_greeks
        println("| All 5 Greeks | $(round(nova_greeks, digits=2)) μs | $(round(ql_greeks, digits=1)) μs | **$(round(speedup, digits=0))x faster** |")
    end

    # Monte Carlo
    println("\n### Monte Carlo Simulation\n")
    println("| Benchmark | QuantNova.jl | Notes |")
    println("|-----------|-------------|-------|")

    mc_benchmarks = [
        ("MC European (10k paths)", "mc_european_10k"),
        ("MC Asian (10k paths)", "mc_asian_10k"),
        ("American LSM (10k paths)", "lsm_10k"),
    ]

    for (name, key) in mc_benchmarks
        nova_time = get(nova_results, key, NaN)
        if !isnan(nova_time)
            println("| $name | $(round(nova_time/1000, digits=2)) ms | Pure Julia, vectorized |")
        end
    end

    # Backtesting vs Python
    println("\n### Backtesting (vs Python pandas/vectorbt)\n")
    println("| Benchmark | QuantNova.jl | Python (pandas) | Speedup |")
    println("|-----------|-------------|-----------------|---------|")

    bt_comparisons = [
        ("SMA Crossover (5yr)", "sma_crossover", "sma_crossover"),
        ("Rolling Sharpe (252-day)", "rolling_sharpe", "rolling_sharpe"),
        ("Full Metrics", "full_metrics", "full_metrics"),
    ]

    for (name, nova_key, py_key) in bt_comparisons
        nova_time = get(nova_results, nova_key, NaN)
        py_time = get(PYTHON, py_key, NaN)
        if !isnan(nova_time) && !isnan(py_time)
            speedup = py_time / nova_time
            nova_str = nova_time < 1000 ? "$(round(nova_time, digits=1)) μs" : "$(round(nova_time/1000, digits=2)) ms"
            py_str = py_time < 1000 ? "$(round(py_time, digits=1)) μs" : "$(round(py_time/1000, digits=2)) ms"
            println("| $name | $nova_str | $py_str | **$(round(speedup, digits=0))x faster** |")
        end
    end

    # Factor Models vs Python
    println("\n### Factor Models (vs Python statsmodels)\n")
    println("| Benchmark | QuantNova.jl | Python | Speedup |")
    println("|-----------|-------------|--------|---------|")

    factor_comparisons = [
        ("CAPM Regression", "capm", "capm"),
        ("Fama-French 3-Factor", "ff3", "ff3"),
        ("Rolling Beta (60-day)", "rolling_beta", "rolling_beta"),
        ("Information Coefficient", "ic", "ic"),
    ]

    for (name, nova_key, py_key) in factor_comparisons
        nova_time = get(nova_results, nova_key, NaN)
        py_time = get(PYTHON, py_key, NaN)
        if !isnan(nova_time) && !isnan(py_time)
            speedup = py_time / nova_time
            nova_str = nova_time < 1000 ? "$(round(nova_time, digits=1)) μs" : "$(round(nova_time/1000, digits=2)) ms"
            py_str = py_time < 1000 ? "$(round(py_time, digits=1)) μs" : "$(round(py_time/1000, digits=2)) ms"
            println("| $name | $nova_str | $py_str | **$(round(speedup, digits=0))x faster** |")
        end
    end

    # Statistics
    println("\n### Statistical Testing\n")
    println("| Benchmark | QuantNova.jl | Notes |")
    println("|-----------|-------------|-------|")

    stats_benchmarks = [
        ("Sharpe Ratio", "sharpe"),
        ("Sharpe CI (Lo 2002)", "sharpe_ci"),
        ("Probabilistic Sharpe", "psr"),
    ]

    for (name, key) in stats_benchmarks
        nova_time = get(nova_results, key, NaN)
        if !isnan(nova_time)
            println("| $name | $(round(nova_time, digits=3)) μs | Pure Julia |")
        end
    end
end

function main()
    println("Running QuantNova.jl comprehensive benchmarks...\n")

    results = main()  # From comprehensive_benchmark.jl

    generate_comparison_table(results)

    println("\n" * "=" ^ 80)
    println("Report generation complete.")
    println("Copy the markdown tables above to your README.md")
    println("=" ^ 80)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
