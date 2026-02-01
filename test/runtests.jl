using Test
using QuantNova

# NOTE: The following test categories have been implemented:
# ✓ Edge case tests (T→0, S→0, K→∞, σ→0) - see instruments.jl
# ✓ Error case tests (invalid inputs) - see core.jl, instruments.jl
# ✓ Gradient accuracy tests (AD vs analytical vs finite diff) - see ad.jl
# ✓ ImmutableDict immutability tests - see core.jl
# ✓ Bond pricing consistency tests - see interest_rates.jl
# ✓ Curve interpolation accuracy tests - see interest_rates.jl
# ✓ Nelson-Siegel/Svensson curve fitting tests - see interest_rates.jl
# ✓ Day count convention tests - see interest_rates.jl

# Remaining TODOs for future work:
# TODO: Enable doctests: @testset "Doctests" begin doctest(QuantNova) end
# TODO: Add integration tests (end-to-end workflows)
# TODO: Add performance regression tests
# TODO: Add GPU backend tests (Reactant, Enzyme) - requires GPU hardware
# TODO: Add extension loading tests
# TODO: Add MCResult confidence intervals correctness tests
# TODO: Add calibration robustness tests (noisy data, edge cases)

@testset "QuantNova.jl" begin
    include("core.jl")
    include("ad.jl")
    include("instruments.jl")
    include("portfolio.jl")
    include("risk.jl")
    include("optimization.jl")
    include("calibration.jl")
    include("montecarlo.jl")
    include("full_pipeline.jl")
    include("backend_parity.jl")
    include("interest_rates.jl")
    include("simulation.jl")
    include("backtesting.jl")
    include("scenario_analysis.jl")
    include("research_toolkit.jl")

    # Live data tests (requires network, skipped by default)
    # Run with: QUANTNOVA_TEST_LIVE_DATA=1 julia --project=. -e 'using Pkg; Pkg.test()'
    include("live_data.jl")
end
