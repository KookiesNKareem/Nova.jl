# Pre-compiled GPU Calibration Benchmark
#
# Key finding: GPU is faster PER-CALIBRATION once compiled.
#
# Apple Silicon benchmark results (30 strikes, 200 iterations):
#   CPU (ForwardDiff):     ~62 ms per calibration
#   GPU (pre-compiled):    ~22 ms per calibration (2.9x faster)
#   Compilation overhead:  ~18 seconds (one-time)
#   Break-even:            ~440 calibrations
#
# When GPU wins:
# - Production systems with many daily calibrations
# - Batch processing (calibrate N underlyings)
# - Re-running same calibration structure with different data
#
# When CPU wins:
# - Interactive/ad-hoc calibration
# - Single calibration runs
# - Very small problems (< 10 strikes)

using Quasar
using Printf
using Statistics: median

"""
    run_precompiled_benchmark(; n_strikes=20, n_iterations=100)

Compare pre-compiled vs standard calibration performance.
"""
function run_precompiled_benchmark(; n_strikes::Int=20, n_iterations::Int=100, verbose::Bool=true)
    # Test parameters
    F, T, r = 100.0, 1.0, 0.05
    β = 0.5

    # Generate synthetic market data
    moneyness = range(0.8, 1.2, length=n_strikes)
    strikes = collect(F .* moneyness)

    # True params for synthetic data
    α_true, ρ_true, ν_true = 0.25, -0.3, 0.4
    market_vols = [Quasar.BatchPricing._sabr_vol_gpu(F, K, T, α_true, β, ρ_true, ν_true) for K in strikes]

    # Starting point
    x0 = [0.2, 0.0, log(0.3)]

    if verbose
        println()
        println("Pre-compiled Calibration Benchmark")
        println("=" ^ 60)
        println("Configuration: $n_strikes strikes, $n_iterations iterations")
        println("=" ^ 60)
    end

    results = Dict{String, Any}()

    # CPU baseline (ForwardDiff)
    if verbose
        println()
        println("CPU (ForwardDiff)")
        println("-" ^ 60)
    end

    smile = SmileData(T, F, r, [
        OptionQuote(strikes[i], T, 0.0, :call, market_vols[i])
        for i in eachindex(strikes)
    ])

    # Warmup
    calibrate_sabr(smile; backend=ForwardDiffBackend(), max_iter=10)

    # Time CPU
    cpu_times = Float64[]
    for _ in 1:5
        t0 = time_ns()
        calibrate_sabr(smile; backend=ForwardDiffBackend(), max_iter=n_iterations)
        push!(cpu_times, (time_ns() - t0) / 1e6)
    end
    cpu_median = median(cpu_times)
    results["cpu_ms"] = cpu_median

    if verbose
        @printf("  Median time: %.2f ms\n", cpu_median)
    end

    # Check if Reactant is available
    reactant_available = false
    try
        test_backend = ReactantBackend()
        test_grad = gradient(x -> sum(x.^2), [1.0, 2.0]; backend=test_backend)
        reactant_available = test_grad ≈ [2.0, 4.0]
    catch
    end

    if !reactant_available
        if verbose
            println()
            println("Reactant not available. To enable GPU benchmarks:")
            println("  using Reactant")
            println("  using Quasar")
        end
        return results
    end

    # Pre-compiled GPU calibration
    if verbose
        println()
        println("GPU (Reactant) - Pre-compiled")
        println("-" ^ 60)
    end

    cal = PrecompiledSABRCalibrator(F, T, β, strikes, market_vols)

    # Time compilation (one-time cost)
    t_compile = time_ns()
    compile_gpu!(cal)
    compile_time = (time_ns() - t_compile) / 1e6
    results["compile_ms"] = compile_time

    if verbose
        @printf("  Compilation time: %.2f ms (one-time)\n", compile_time)
    end

    # Time calibration runs
    gpu_times = Float64[]
    for _ in 1:5
        t0 = time_ns()
        calibrate!(cal, x0; max_iter=n_iterations)
        push!(gpu_times, (time_ns() - t0) / 1e6)
    end
    gpu_median = median(gpu_times)
    results["gpu_ms"] = gpu_median

    if verbose
        @printf("  Median calibration: %.2f ms\n", gpu_median)
    end

    # Summary
    if verbose
        println()
        println("Summary")
        println("-" ^ 60)
        @printf("CPU (ForwardDiff):     %.2f ms\n", cpu_median)
        @printf("GPU (pre-compiled):    %.2f ms (+ %.2f ms compile)\n", gpu_median, compile_time)

        if gpu_median < cpu_median
            speedup = cpu_median / gpu_median
            @printf("Speedup:               %.2fx (excluding compilation)\n", speedup)
            breakeven = compile_time / (cpu_median - gpu_median)
            @printf("Break-even:            %.0f calibrations to amortize compile cost\n", breakeven)
        else
            @printf("Note: CPU faster for this problem size\n")
        end
    end

    results["speedup"] = cpu_median / gpu_median
    return results
end

"""
    run_scaling_benchmark(; sizes=[10, 20, 50, 100])

Test how pre-compiled GPU scales with problem size.
"""
function run_scaling_benchmark(; sizes::Vector{Int}=[10, 20, 50, 100])
    println()
    println("Scaling Benchmark: CPU vs Pre-compiled GPU")
    println("=" ^ 70)
    @printf("%-12s %-15s %-15s %-15s\n", "Strikes", "CPU (ms)", "GPU (ms)", "Speedup")
    println("-" ^ 70)

    for n in sizes
        results = run_precompiled_benchmark(n_strikes=n, n_iterations=200, verbose=false)
        cpu = get(results, "cpu_ms", NaN)
        gpu = get(results, "gpu_ms", NaN)
        speedup = get(results, "speedup", NaN)
        @printf("%-12d %-15.2f %-15.2f %-15.2fx\n", n, cpu, gpu, speedup)
    end
    println()
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_precompiled_benchmark()
end
