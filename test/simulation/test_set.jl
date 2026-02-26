using Test
using Macchiato
import Macchiato as MM
using WhatsThePoint
import WhatsThePoint as WTP
using Unitful: m, °

include(joinpath(@__DIR__, "..", "end_2_end", "2d_square.jl"))

function create_test_domain()
    dx = 1 / 17 * m
    part = create_2d_square_domain(dx)
    cloud = WTP.discretize(part, ConstantSpacing(dx), alg=VanDerSandeFornberg())

    k, ρ, cₚ = 1.0, 1.0, 1.0
    bcs = Dict(
        :surface1 => MM.Temperature(0.0),
        :surface2 => MM.Temperature(0.0),
        :surface3 => MM.Temperature(0.0),
        :surface4 => MM.Temperature(0.0)
    )
    model = MM.SolidEnergy(k=k, ρ=ρ, cₚ=cₚ)
    return MM.Domain(cloud, bcs, model)
end

@testset "set! and field accessors" begin
    @testset "set! with uniform value" begin
        domain = create_test_domain()
        sim = Simulation(domain)

        set!(sim, T=300.0)

        @test sim.u0 !== nothing
        @test all(sim.u0 .== 300.0)
    end

    @testset "set! with function" begin
        domain = create_test_domain()
        sim = Simulation(domain)

        set!(sim, T=x -> x[1] + x[2])

        @test sim.u0 !== nothing
        coords = MM._coords(domain.cloud)
        for (i, pt) in enumerate(coords)
            expected = MM.ustrip(pt.x) + MM.ustrip(pt.y)
            @test sim.u0[i] ≈ expected
        end
    end

    @testset "set! with vector" begin
        domain = create_test_domain()
        sim = Simulation(domain)

        n = length(domain.cloud)
        values = collect(1.0:n)
        set!(sim, T=values)

        @test sim.u0 == values
    end

    @testset "set! vector dimension mismatch" begin
        domain = create_test_domain()
        sim = Simulation(domain)

        @test_throws DimensionMismatch set!(sim, T=[1.0, 2.0, 3.0])
    end

    @testset "set! unknown field" begin
        domain = create_test_domain()
        sim = Simulation(domain)

        @test_throws ArgumentError set!(sim, unknown_field=1.0)
    end

    @testset "temperature accessor" begin
        domain = create_test_domain()
        sim = Simulation(domain)
        set!(sim, T=100.0)

        T = temperature(sim)
        @test all(T .== 100.0)
        @test length(T) == length(domain.cloud)
    end

    @testset "_field_indices" begin
        domain = create_test_domain()
        sim = Simulation(domain)
        set!(sim, T=0.0)

        indices = MM._field_indices(sim, :T)
        @test length(indices) == length(domain.cloud)
        @test first(indices) == 1
        @test last(indices) == length(domain.cloud)
    end

    @testset "_has_field" begin
        domain = create_test_domain()
        sim = Simulation(domain)

        @test MM._has_field(sim, :T)
        @test MM._has_field(sim, :temperature)
        @test !MM._has_field(sim, :u)
        @test !MM._has_field(sim, :p)
    end
end
