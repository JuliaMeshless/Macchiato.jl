using Test
using Macchiato
import Macchiato as MM
using WhatsThePoint
import WhatsThePoint as WTP
using SparseArrays
using StaticArrays
using Unitful: m, ustrip

include(joinpath(@__DIR__, "..", "end_2_end", "2d_square.jl"))

function create_utils_test_cloud()
    dx = 1 / 9 * m
    part = create_2d_square_domain(dx)
    return WTP.discretize(part, ConstantSpacing(dx), alg = VanDerSandeFornberg())
end

@testset "utils" begin
    @testset "findmin_turbo matches findmin" begin
        x = [3.0, 1.0, 2.0, -5.0, 7.0]
        @test MM.findmin_turbo(x) == (-5.0, 4)
        @test MM.findmin_turbo(x) == findmin(x)

        y = 100 .* sin.(1:101)
        @test MM.findmin_turbo(y) == findmin(y)

        # @turbo lane reduction does not preserve Base.findmin's first-occurrence
        # tie-breaking (returns index 2 here), so only assert a valid minimizer
        ties = [1.0, 1.0]
        tie_val, tie_idx = MM.findmin_turbo(ties)
        @test tie_val == 1.0
        @test tie_idx in eachindex(ties)
        @test ties[tie_idx] == tie_val
    end

    @testset "findmin_turbo empty input sentinel" begin
        @test MM.findmin_turbo(Float64[]) == (Inf, 0)
    end

    @testset "findmin_turbo over id subset returns the global id" begin
        x = [5.0, 1.0, 4.0, 0.5, 2.0]
        ids = [1, 3, 5]
        @test MM.findmin_turbo(x, ids) == (2.0, 5)
    end

    @testset "zero_rows! zeros stored entries only in given rows" begin
        rows = [1, 2, 3, 4, 1, 2, 3, 4]
        cols = [1, 2, 3, 4, 3, 4, 1, 2]
        vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        A = sparse(rows, cols, vals, 4, 4)
        B = copy(A)

        MM.zero_rows!(A, Set([2, 4]))
        @test all(iszero, A[2, :])
        @test all(iszero, A[4, :])
        @test A[1, :] == B[1, :]
        @test A[3, :] == B[3, :]
        @test nnz(A) == nnz(B)

        C = copy(B)
        MM.zero_rows!(C, Set{Int}())
        @test C == B
    end

    @testset "_coords extracts coordinate SVectors" begin
        pts2d = [WTP.Point(0.1m, 0.2m), WTP.Point(0.3m, 0.4m)]
        cs2d = MM._coords(pts2d)
        @test cs2d == [SVector(0.1m, 0.2m), SVector(0.3m, 0.4m)]

        pts3d = [WTP.Point(0.1m, 0.2m, 0.3m), WTP.Point(0.4m, 0.5m, 0.6m)]
        cs3d = MM._coords(pts3d)
        @test cs3d == [SVector(0.1m, 0.2m, 0.3m), SVector(0.4m, 0.5m, 0.6m)]

        cloud = create_utils_test_cloud()
        cs_cloud = MM._coords(cloud)
        @test length(cs_cloud) == length(WTP.points(cloud))
        c1 = WTP.coords(WTP.points(cloud)[1])
        @test cs_cloud[1] == SVector(c1.x, c1.y)
    end

    @testset "get_node_coords strips units" begin
        cloud = create_utils_test_cloud()
        g = MM.get_node_coords(cloud, 1)
        @test g isa SVector{2, Float64}
        c1 = WTP.coords(WTP.points(cloud)[1])
        @test g == ustrip.(SVector(c1.x, c1.y))
    end
end
