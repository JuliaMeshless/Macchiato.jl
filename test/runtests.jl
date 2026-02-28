using Macchiato
using TestItemRunner
using Test

@run_package_tests
testfiles = [
    "end_2_end/2d_Laplacian_MoMS.jl",
    "simulation/test_callbacks.jl",
    "simulation/test_set.jl",
    "simulation/test_output_writers.jl",
    "simulation/test_simulation.jl",
    "models/test_linear_elasticity.jl"
]

@testset "Macchiato.jl" begin
    for testfile in testfiles
        println("testing $testfile...")
        include(testfile)
    end
end
