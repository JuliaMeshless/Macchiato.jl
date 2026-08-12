using Macchiato
using Test

testfiles = [
    "aqua.jl",
    "core/test_domain.jl",
    "core/test_utils.jl",
    "core/test_api_surface.jl",
    "operators/test_upwinding.jl",
    "io/test_io.jl",
    "end_2_end/2d_Laplacian_MoMS.jl",
    "end_2_end/2d_complex_steady.jl",
    "simulation/test_set.jl",
    "simulation/test_simulation.jl",
    "simulation/test_ghost.jl",
    "models/test_linear_elasticity.jl",
]

@testset "Macchiato.jl" begin
    for testfile in testfiles
        println("testing $testfile...")
        include(testfile)
    end
end
