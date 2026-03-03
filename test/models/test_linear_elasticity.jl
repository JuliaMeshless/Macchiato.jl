using Test
using Macchiato
import Macchiato as MM
using WhatsThePoint
import WhatsThePoint as WTP
using Unitful: m, °, ustrip
using LinearAlgebra
using LinearSolve

include(joinpath(@__DIR__, "..", "end_2_end", "2d_square.jl"))

# ============================================================================
# Unit Tests
# ============================================================================

@testset "LinearElasticity unit tests" begin
    @testset "Lamé parameter computation" begin
        # Steel-like: E=200e3, ν=0.3
        model = LinearElasticity(E = 200.0e3, ν = 0.3)
        μ, λstar = lame_parameters(model)

        E, ν = 200.0e3, 0.3
        μ_expected = E / (2 * (1 + ν))
        λ_full = E * ν / ((1 + ν) * (1 - 2ν))
        λstar_expected = 2μ_expected * λ_full / (λ_full + 2μ_expected)

        @test μ ≈ μ_expected rtol = 1.0e-14
        @test λstar ≈ λstar_expected rtol = 1.0e-14

        println("  μ = $μ (expected $μ_expected)")
        println("  λ* = $λstar (expected $λstar_expected)")
    end

    @testset "Lamé parameters for incompressible limit" begin
        # ν → 0.5 makes λ → ∞ but λ* stays finite for plane stress
        model = LinearElasticity(E = 1.0, ν = 0.499)
        μ, λstar = lame_parameters(model)
        @test isfinite(μ)
        @test isfinite(λstar)
        # For ν → 0.5 (plane stress): λ* → 2μ = 2E/3
        @test λstar ≈ 2.0 / 3 rtol = 0.01
    end

    @testset "_num_vars" begin
        model = LinearElasticity(E = 200.0e3, ν = 0.3)
        @test MM._num_vars(model, 2) == 2
        @test MM._num_vars(model, 3) == 3
    end

    @testset "show method" begin
        model = LinearElasticity(E = 200.0e3, ν = 0.3)
        str = string(model)
        @test occursin("LinearElasticity", str)
        @test occursin("200000", str)
        @test occursin("0.3", str)
    end
end

# ============================================================================
# Boundary Condition Tests
# ============================================================================

@testset "Mechanics BCs" begin
    @testset "Displacement construction" begin
        bc1 = Displacement(0.0, 0.0)
        @test bc1([0.0, 0.0], 0.0) == (0.0, 0.0)

        bc2 = Displacement((x, t) -> (x[1], -x[2]))
        @test bc2([1.0, 2.0], 0.0) == (1.0, -2.0)

        bc3 = Displacement(x -> x[1], x -> -x[2])
        @test bc3([3.0, 4.0], 0.0) == (3.0, -4.0)
    end

    @testset "Traction construction" begin
        bc1 = Traction(0.0, -1000.0)
        @test bc1([0.0, 0.0], 0.0) == (0.0, -1000.0)

        bc2 = Traction((x, t) -> (0.0, -x[2]))
        @test bc2([0.0, 5.0], 0.0) == (0.0, -5.0)
    end

    @testset "TractionFree" begin
        bc = TractionFree()
        @test bc([0.0, 0.0], 0.0) == (0.0, 0.0)
    end

end

# ============================================================================
# Field Index and Accessor Tests
# ============================================================================

@testset "Displacement field indices and accessors" begin
    dx = 1 / 17 * m
    part = create_2d_square_domain(dx)
    cloud = WTP.discretize(part, ConstantSpacing(dx), alg = VanDerSandeFornberg())

    bcs = Dict(
        :surface1 => Displacement(0.0, 0.0),
        :surface2 => Displacement(0.0, 0.0),
        :surface3 => Displacement(0.0, 0.0),
        :surface4 => Displacement(0.0, 0.0),
    )
    model = LinearElasticity(E = 200.0e3, ν = 0.3)
    domain = MM.Domain(cloud, bcs, model)
    sim = Simulation(domain)

    @testset "set! with uniform displacement" begin
        set!(sim, ux = 1.0, uy = -2.0)
        @test sim.u0 !== nothing
        N = length(domain.cloud)
        @test length(sim.u0) == 2N
        @test all(sim.u0[1:N] .== 1.0)
        @test all(sim.u0[(N + 1):(2N)] .== -2.0)
    end

    @testset "set! with function" begin
        set!(sim, ux = x -> x[1], uy = x -> -x[2])
        N = length(domain.cloud)
        coords = MM._coords(domain.cloud)
        for (i, pt) in enumerate(coords)
            @test sim.u0[i] ≈ MM.ustrip(pt.x)
            @test sim.u0[i + N] ≈ -MM.ustrip(pt.y)
        end
    end

    @testset "_has_field" begin
        @test MM._has_field(sim, :ux)
        @test MM._has_field(sim, :uy)
        @test !MM._has_field(sim, :T)
        @test !MM._has_field(sim, :p)
    end

    @testset "_field_indices" begin
        N = length(domain.cloud)
        idx_ux = MM._field_indices(sim, :ux)
        idx_uy = MM._field_indices(sim, :uy)
        @test first(idx_ux) == 1
        @test last(idx_ux) == N
        @test first(idx_uy) == N + 1
        @test last(idx_uy) == 2N
    end

    @testset "displacement accessor" begin
        set!(sim, ux = 5.0, uy = -3.0)
        ux, uy = displacement(sim)
        N = length(domain.cloud)
        @test length(ux) == N
        @test length(uy) == N
        @test all(ux .== 5.0)
        @test all(uy .== -3.0)
    end
end

# ============================================================================
# Patch Test for 2D Linear Elasticity
# ============================================================================

@testset "2D Linear Elasticity Patch Test" begin
    # Linear displacement: u = a + bx + cy, v = d + ex + fy
    # Constant strain everywhere => body force = 0
    # Must be reproduced to near machine precision
    u_exact(x, y) = 1.0e-3 + 2.0e-3 * x + 3.0e-3 * y
    v_exact(x, y) = 4.0e-3 + 5.0e-3 * x + 6.0e-3 * y

    bc_func(x, t) = (u_exact(x[1], x[2]), v_exact(x[1], x[2]))

    dx = 1 / 17 * m
    part = create_2d_square_domain(dx)
    cloud = WTP.discretize(part, ConstantSpacing(dx), alg = VanDerSandeFornberg())

    bcs = Dict(
        :surface1 => Displacement(bc_func),
        :surface2 => Displacement(bc_func),
        :surface3 => Displacement(bc_func),
        :surface4 => Displacement(bc_func),
    )
    model = LinearElasticity(E = 1.0, ν = 0.3)
    domain = MM.Domain(cloud, bcs, model)
    prob = MM.LinearProblem(domain)
    sol = solve(prob)

    N = length(cloud)
    u_num = sol.u[1:N]
    v_num = sol.u[(N + 1):(2N)]

    coords = MM._coords(cloud)
    u_ana = [u_exact(ustrip(pt.x), ustrip(pt.y)) for pt in coords]
    v_ana = [v_exact(ustrip(pt.x), ustrip(pt.y)) for pt in coords]

    @test norm(u_num .- u_ana, Inf) < 1.0e-10
    @test norm(v_num .- v_ana, Inf) < 1.0e-10
end

# ============================================================================
# MMS Convergence Rate Test for 2D Linear Elasticity
# ============================================================================

@testset "2D Linear Elasticity MMS Convergence" begin
    # Manufactured solution: u = x²y, v = -xy²
    # Body forces: fx = -2μy, fy = 2μx

    E_val = 1.0
    ν_val = 0.3
    μ_val = E_val / (2 * (1 + ν_val))

    u_exact(x, y) = x^2 * y
    v_exact(x, y) = -x * y^2
    body_force(x) = (-2μ_val * x[2], 2μ_val * x[1])
    bc_func(x, t) = (u_exact(x[1], x[2]), v_exact(x[1], x[2]))

    resolutions = [1 / 15, 1 / 21, 1 / 29]
    L2_errors = Float64[]

    for res in resolutions
        dx = res * m
        part = create_2d_square_domain(dx)
        cloud = WTP.discretize(part, ConstantSpacing(dx), alg = VanDerSandeFornberg())

        bcs = Dict(
            :surface1 => Displacement(bc_func),
            :surface2 => Displacement(bc_func),
            :surface3 => Displacement(bc_func),
            :surface4 => Displacement(bc_func),
        )
        model = LinearElasticity(E = E_val, ν = ν_val, body_force = body_force)
        domain = MM.Domain(cloud, bcs, model)
        prob = MM.LinearProblem(domain)
        sol = solve(prob)

        N = length(cloud)
        u_num = sol.u[1:N]
        v_num = sol.u[(N + 1):(2N)]
        coords = MM._coords(cloud)
        u_ana = [u_exact(ustrip(pt.x), ustrip(pt.y)) for pt in coords]
        v_ana = [v_exact(ustrip(pt.x), ustrip(pt.y)) for pt in coords]

        L2 = sqrt(norm(u_num .- u_ana)^2 + norm(v_num .- v_ana)^2) / sqrt(2N)
        push!(L2_errors, L2)
    end

    # Compute convergence rates between successive refinements
    h = Float64.(resolutions)
    rates = [
        log(L2_errors[i] / L2_errors[i + 1]) / log(h[i] / h[i + 1])
            for i in 1:(length(h) - 1)
    ]

    println("\nMMS Convergence:")
    for (i, res) in enumerate(resolutions)
        println("  h=$(round(res, digits = 4)): L2=$(L2_errors[i])")
    end
    println("  Rates: ", rates)

    # RBF with polyharmonic spline basis expects ~2nd order convergence
    # Threshold of 1.5 accounts for meshless method variability
    for rate in rates
        @test rate > 1.5
    end
end

# ============================================================================
# Simulation API Integration Test
# ============================================================================

@testset "LinearElasticity Simulation API" begin
    dx = 1 / 17 * m
    part = create_2d_square_domain(dx)
    cloud = WTP.discretize(part, ConstantSpacing(dx), alg = VanDerSandeFornberg())

    bcs = Dict(
        :surface1 => Displacement(0.0, 0.0),
        :surface2 => Displacement(0.0, 0.0),
        :surface3 => Displacement(0.0, 0.0),
        :surface4 => Displacement(0.0, 0.0),
    )
    model = LinearElasticity(E = 200.0e3, ν = 0.3)
    domain = MM.Domain(cloud, bcs, model)

    sim = Simulation(domain)
    set!(sim, ux = 0.0, uy = 0.0)

    @test sim.mode isa Steady

    run!(sim)

    @test sim._solution !== nothing

    ux, uy = displacement(sim)
    N = length(domain.cloud)
    @test length(ux) == N
    @test length(uy) == N

    # All-zero Dirichlet with no body force → zero displacement
    @test norm(ux) < 1.0e-10
    @test norm(uy) < 1.0e-10

    println("  Simulation API integration test passed")
end
