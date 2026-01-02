using MeshlessMultiphysics
using TestItemRunner
using Test

@run_package_tests
testfiles = [
    "end_2_end/2d_Laplacian_MoMS.jl"
]
#=

testfiles = [
    "points.jl"
    "surfaces.jl"
    "normals.jl"
    #"node_generation.jl"
]

=#
@testset "MeshlessMultiphysics.jl" begin
    for testfile in testfiles
        println("testing $testfile...")
        include(testfile)
    end
end
