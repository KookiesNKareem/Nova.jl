using Test
using Quasar

@testset "Quasar.jl" begin
    include("Core/test_core.jl")
    include("AD/test_ad.jl")
    include("Instruments/test_instruments.jl")
    include("Portfolio/test_portfolio.jl")
    include("Risk/test_risk.jl")
    include("Optimization/test_optimization.jl")
end
