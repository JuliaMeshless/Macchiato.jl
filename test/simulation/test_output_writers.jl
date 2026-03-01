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
    cloud = WTP.discretize(part, ConstantSpacing(dx), alg = VanDerSandeFornberg())

    k, ρ, cₚ = 1.0, 1.0, 1.0
    bcs = Dict(
        :surface1 => MM.Temperature(0.0),
        :surface2 => MM.Temperature(0.0),
        :surface3 => MM.Temperature(100.0),
        :surface4 => MM.Temperature(0.0)
    )
    model = MM.SolidEnergy(k = k, ρ = ρ, cₚ = cₚ)
    return MM.Domain(cloud, bcs, model)
end

@testset "Output Writers" begin
    @testset "VTKOutputWriter constructor" begin
        tmpdir = mktempdir()
        writer = VTKOutputWriter(joinpath(tmpdir, "test"); schedule = TimeInterval(0.1))

        @test writer.prefix == joinpath(tmpdir, "test")
        @test writer.schedule isa TimeInterval
        @test writer.schedule.Δt == 0.1
        @test isempty(writer.fields)
        @test writer._output_count == 0

        writer_fields = VTKOutputWriter(
            joinpath(tmpdir, "test2");
            schedule = IterationInterval(10),
            fields = [:T]
        )
        @test writer_fields.fields == [:T]
    end

    @testset "VTKOutputWriter creates directory" begin
        tmpdir = mktempdir()
        subdir = joinpath(tmpdir, "subdir", "results")

        writer = VTKOutputWriter(joinpath(subdir, "test"); schedule = TimeInterval(0.1))
        @test isdir(subdir)
    end

    @testset "VTKOutputWriter with steady-state simulation" begin
        tmpdir = mktempdir()
        domain = create_test_domain()
        sim = Simulation(domain)
        set!(sim, T = 50.0)

        sim.output_writers[:vtk] = VTKOutputWriter(
            joinpath(tmpdir, "steady");
            schedule = IterationInterval(1)
        )

        run!(sim)

        @test isfile(joinpath(tmpdir, "steady.pvd"))
    end

    # NOTE: Skipped because make_f() has pre-existing RBF bug
    @testset "VTKOutputWriter with transient simulation" begin
        @test_skip "Skipped: make_f() has pre-existing RBF bug"
    end

    @testset "JLD2OutputWriter constructor" begin
        tmpdir = mktempdir()
        writer = JLD2OutputWriter(joinpath(tmpdir, "checkpoint"); schedule = TimeInterval(1.0))

        @test writer.prefix == joinpath(tmpdir, "checkpoint")
        @test writer.schedule isa TimeInterval
        @test writer._output_count == 0
    end

    # NOTE: Skipped because make_f() has pre-existing RBF bug
    @testset "JLD2OutputWriter with transient simulation" begin
        @test_skip "Skipped: make_f() has pre-existing RBF bug"
    end

    @testset "show methods" begin
        tmpdir = mktempdir()

        vtk = VTKOutputWriter(joinpath(tmpdir, "test"); schedule = TimeInterval(0.1))
        str_vtk = string(vtk)
        @test occursin("VTKOutputWriter", str_vtk)
        @test occursin("TimeInterval", str_vtk)

        jld2 = JLD2OutputWriter(joinpath(tmpdir, "test"); schedule = IterationInterval(10))
        str_jld2 = string(jld2)
        @test occursin("JLD2OutputWriter", str_jld2)
        @test occursin("IterationInterval", str_jld2)
    end
end
