using Test
using Macchiato
using StaticArrays

unit_square_grid(n = 11) = [
    SVector(x, y) for x in range(0.0, 1.0; length = n) for y in range(0.0, 1.0; length = n)
]

@testset "upwind operators" begin
    data = unit_square_grid()
    N = length(data)
    ϕ_lin = [2 * p[1] + 3 * p[2] for p in data]
    dux = upwind(data, 1)

    @testset "advective derivative of linear field is exact" begin
        for θ in (0.0, 0.5, 1.0)
            @test dux(ϕ_lin, ones(N), θ) ≈ fill(2.0, N) atol = 1e-6
        end
    end

    @testset "operator scales by velocity" begin
        @test dux(ϕ_lin, fill(2.0, N), 1.0) ≈ fill(4.0, N) atol = 1e-6
    end

    @testset "negative velocity switches stencil" begin
        @test dux(ϕ_lin, -ones(N), 1.0) ≈ fill(-2.0, N) atol = 1e-6
    end

    @testset "zero velocity yields zero advection" begin
        @test dux(ϕ_lin, zeros(N), 1.0) == zeros(N)
    end

    @testset "θ blends one-sided and centered stencils" begin
        ϕ_sin = [sin(2π * p[1]) for p in data]
        full = dux(ϕ_sin, ones(N), 1.0)
        centered = dux(ϕ_sin, ones(N), 0.0)
        half = dux(ϕ_sin, ones(N), 0.5)

        @test !(full ≈ centered)
        @test half ≈ (full .+ centered) ./ 2
    end

    @testset "two-argument form evaluates at data points" begin
        dux_explicit = upwind(data, data, 1)
        @test dux_explicit(ϕ_lin, ones(N), 1.0) ≈ dux(ϕ_lin, ones(N), 1.0)
    end

    @testset "custom eval points" begin
        du = upwind(data, [SVector(0.5, 0.5)], 1)
        result = du(ϕ_lin, [1.0], 1.0)

        @test length(result) == 1
        @test result[1] ≈ 2.0 atol = 1e-6
    end

    @testset "keyword overrides accepted" begin
        dux_k = upwind(data, 1; k = 15)
        @test dux_k(ϕ_lin, ones(N), 1.0) ≈ fill(2.0, N) atol = 1e-6

        dux_Δ = upwind(data, 1; Δ = 0.05)
        @test dux_Δ(ϕ_lin, ones(N), 1.0) ≈ fill(2.0, N) atol = 1e-6
    end
end
