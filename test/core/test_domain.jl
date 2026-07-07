using Test
using Macchiato
import Macchiato as MM
using WhatsThePoint
import WhatsThePoint as WTP
using Unitful: m

include(joinpath(@__DIR__, "..", "end_2_end", "2d_square.jl"))

function domain_test_cloud(dx = 1 / 17 * m)
    part = create_2d_square_domain(dx)
    return WTP.discretize(part, ConstantSpacing(dx), alg = VanDerSandeFornberg())
end

function domain_test_bcs()
    return Dict(s => MM.Temperature(0.0) for s in (:surface1, :surface2, :surface3, :surface4))
end

@testset "Domain" begin
    cloud = domain_test_cloud()
    surfs = (:surface1, :surface2, :surface3, :surface4)

    @testset "constructor assigns contiguous boundary ranges" begin
        model = SolidEnergy(k = 1.0, ρ = 1.0, cₚ = 1.0)
        domain = Domain(cloud, domain_test_bcs(), model)

        @test Set(keys(domain.boundaries)) == Set(surfs)
        for surf in surfs
            ids, _ = domain.boundaries[surf]
            @test length(ids) == length(cloud[surf])
        end

        all_ids = reduce(vcat, [collect(domain.boundaries[surf][1]) for surf in surfs])
        total = sum(length(cloud[surf]) for surf in surfs)
        @test allunique(all_ids)
        @test sort(all_ids) == 1:total

        @test domain.models == [model]
        @test domain.name == :domain1
    end

    @testset "constructor rejects BC for unknown surface" begin
        bcs = domain_test_bcs()
        bcs[:bogus] = MM.Temperature(0.0)
        @test_throws ArgumentError Domain(cloud, bcs, SolidEnergy(k = 1.0, ρ = 1.0, cₚ = 1.0))
    end

    @testset "constructor requires BC for every surface" begin
        bcs = domain_test_bcs()
        pop!(bcs, :surface4)
        @test_throws KeyError Domain(cloud, bcs, SolidEnergy(k = 1.0, ρ = 1.0, cₚ = 1.0))
    end

    @testset "add! appends a model" begin
        domain = Domain(cloud, domain_test_bcs(), SolidEnergy(k = 1.0, ρ = 1.0, cₚ = 1.0))
        add!(domain, SolidEnergy(k = 2.0, ρ = 1.0, cₚ = 1.0))

        @test length(domain.models) == 2
        @test last(domain.models).k == 2.0
    end

    @testset "add! replaces BC preserving index range" begin
        domain = Domain(cloud, domain_test_bcs(), SolidEnergy(k = 1.0, ρ = 1.0, cₚ = 1.0))
        ids_before = domain.boundaries[:surface1][1]

        ret = add!(domain, MM.HeatFlux(-10.0), :surface1)
        @test ret === domain

        ids_after, bc = domain.boundaries[:surface1]
        @test ids_after == ids_before
        @test bc isa MM.PrescribedFlux
        @test bc.name == :HeatFlux
    end

    @testset "add! rejects unknown surface" begin
        domain = Domain(cloud, domain_test_bcs(), SolidEnergy(k = 1.0, ρ = 1.0, cₚ = 1.0))
        @test_throws ArgumentError add!(domain, MM.Temperature(1.0), :missing_surface)
    end

    @testset "delete! removes matching model" begin
        model1 = SolidEnergy(k = 1.0, ρ = 1.0, cₚ = 1.0)
        model2 = SolidEnergy(k = 2.0, ρ = 1.0, cₚ = 1.0)
        domain = Domain(cloud, domain_test_bcs(), model1)
        add!(domain, model2)

        ret = delete!(domain, model2)
        @test ret === domain
        @test length(domain.models) == 1
        @test only(domain.models).k == 1.0

        delete!(domain, model1)
        @test isempty(domain.models)

        @test MM.delete! === Base.delete!
    end
end
