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

@testset "Simulation" begin
    @testset "Constructor - default is Steady" begin
        domain = create_test_domain()

        sim = Simulation(domain)
        @test sim.mode isa Steady
        @test sim.u0 === nothing
        @test sim.time == 0.0
        @test sim.running == false
        @test sim._solution === nothing
    end

    @testset "Constructor - explicit Steady" begin
        domain = create_test_domain()

        sim = Simulation(domain, Steady())
        @test sim.mode isa Steady
    end

    @testset "Constructor - Transient" begin
        domain = create_test_domain()

        sim = Simulation(domain, Transient(Δt = 0.001, stop_time = 1.0))
        @test sim.mode isa Transient
        @test sim.mode.Δt == 0.001
        @test sim.mode.stop_time == 1.0
        @test sim.mode.solver isa MM.OrdinaryDiffEq.FBDF
    end

    @testset "Constructor - Transient with custom solver" begin
        domain = create_test_domain()

        sim = Simulation(domain, Transient(Δt = 0.001, stop_time = 1.0, solver = MM.OrdinaryDiffEq.Tsit5()))
        @test sim.mode.solver isa MM.OrdinaryDiffEq.Tsit5
    end

    @testset "show methods" begin
        domain = create_test_domain()

        sim_steady = Simulation(domain)
        str_steady = string(sim_steady)
        @test occursin("steady-state", str_steady)

        sim_transient = Simulation(domain, Transient(Δt = 0.001, stop_time = 1.0))
        str_transient = string(sim_transient)
        @test occursin("transient", str_transient)
        @test occursin("Δt=0.001", str_transient)
    end

    @testset "Steady-state run!" begin
        domain = create_test_domain()
        sim = Simulation(domain)
        set!(sim, T = 0.0)

        run!(sim)

        @test sim._solution !== nothing
        @test !sim.running

        T = temperature(sim)
        @test length(T) == length(domain.cloud)
    end

    @testset "Transient run!" begin
        dx = 1 / 17 * m
        part = create_2d_square_domain(dx)
        cloud = WTP.discretize(part, ConstantSpacing(dx), alg = VanDerSandeFornberg())
        model = MM.SolidEnergy(k = 1.0, ρ = 1.0, cₚ = 1.0)

        dirichlet = Dict(
            :surface1 => MM.Temperature(0.0), :surface2 => MM.Temperature(0.0),
            :surface3 => MM.Temperature(100.0), :surface4 => MM.Temperature(0.0)
        )

        # Transient Dirichlet must converge to the steady-state solution.
        steady = Simulation(MM.Domain(cloud, dirichlet, model))
        set!(steady, T = 0.0)
        run!(steady)
        Ts = temperature(steady)

        sim = Simulation(MM.Domain(cloud, dirichlet, model), Transient(Δt = 1.0e-2, stop_time = 10.0))
        set!(sim, T = 0.0)
        run!(sim)
        Tt = temperature(sim)
        @test !sim.running
        @test all(isfinite, Tt)
        @test maximum(abs, Tt .- Ts) < 0.5

        # Flux/Robin conditions are folded into the diffusion operator via ghost nodes;
        # each must remain bounded (the earlier penalty form blew up to ~1e5 for Adiabatic).
        for bc in (MM.Adiabatic(), MM.HeatFlux(-10.0), MM.Convection(5.0, 1.0, 20.0))
            bcs = Dict(
                :surface1 => bc, :surface2 => MM.Temperature(0.0),
                :surface3 => MM.Temperature(100.0), :surface4 => MM.Temperature(0.0)
            )
            flux_sim = Simulation(MM.Domain(cloud, bcs, model), Transient(Δt = 1.0e-2, stop_time = 10.0))
            set!(flux_sim, T = 0.0)
            run!(flux_sim)
            T = temperature(flux_sim)
            @test all(isfinite, T)
            @test all(-10 .< T .< 110)
        end
    end

    @testset "run! returns simulation" begin
        domain = create_test_domain()
        sim = Simulation(domain)
        set!(sim, T = 0.0)

        result = run!(sim)
        @test result === sim
    end
end
