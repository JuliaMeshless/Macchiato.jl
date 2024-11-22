using MeshlessMultiphysics
using TestItemRunner

@run_package_tests
#=

testfiles = [
    "points.jl"
    "surfaces.jl"
    "normals.jl"
    #"node_generation.jl"
]

@testset "MeshlessMultiphysics.jl" begin
    for testfile in testfiles
        println("testing $testfile...")
        include(testfile)
    end
end
=#
