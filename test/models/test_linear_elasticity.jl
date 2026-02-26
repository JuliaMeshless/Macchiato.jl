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
        model = LinearElasticity(E = 200e3, ν = 0.3)
        μ, λstar = lame_parameters(model)

        E, ν = 200e3, 0.3
        μ_expected = E / (2 * (1 + ν))
        λ_full = E * ν / ((1 + ν) * (1 - 2ν))
        λstar_expected = 2μ_expected * λ_full / (λ_full + 2μ_expected)

        @test μ ≈ μ_expected rtol = 1e-14
        @test λstar ≈ λstar_expected rtol = 1e-14

        println("  μ = $μ (expected $μ_expected)")
        println("  λ* = $λstar (expected $λstar_expected)")
    end

    @testset "Lamé parameters for incompressible limit" begin
        # ν → 0.5 makes λ → ∞ but λ* stays finite for plane stress
        model = LinearElasticity(E = 1.0, ν = 0.499)
        μ, λstar = lame_parameters(model)
        @test isfinite(μ)
        @test isfinite(λstar)
        # For ν → 0.5 (plane stress): λ* → E/3
        @test λstar ≈ 1.0 / 3 rtol = 0.01
    end

    @testset "_num_vars" begin
        model = LinearElasticity(E = 200e3, ν = 0.3)
        @test MM._num_vars(model, 2) == 2
        @test MM._num_vars(model, 3) == 3
    end

    @testset "physics_domain" begin
        @test MM.physics_domain(LinearElasticity{Float64, Float64, Nothing, Nothing}) isa MechanicsPhysics
    end

    @testset "show method" begin
        model = LinearElasticity(E = 200e3, ν = 0.3)
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

    @testset "Displacement physics domain" begin
        @test MM.physics_domain(typeof(Displacement(0.0, 0.0))) isa MechanicsPhysics
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

    @testset "BC compatibility with LinearElasticity" begin
        @test MM.is_compatible(MechanicsPhysics(), MechanicsPhysics())
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
    model = LinearElasticity(E = 200e3, ν = 0.3)
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
# Method of Manufactured Solutions Test for 2D Linear Elasticity
# ============================================================================

@testset "2D Linear Elasticity MMS (Dirichlet only)" begin
    # Manufactured displacement field (polynomial):
    # u(x, y) = x²y
    # v(x, y) = -xy²
    #
    # Partial derivatives:
    # ∂u/∂x = 2xy,     ∂u/∂y = x²
    # ∂²u/∂x² = 2y,    ∂²u/∂y² = 0
    # ∂²u/∂x∂y = 2x
    #
    # ∂v/∂x = -y²,     ∂v/∂y = -2xy
    # ∂²v/∂x² = 0,     ∂²v/∂y² = -2x
    # ∂²v/∂x∂y = -2y
    #
    # Navier-Cauchy (plane stress):
    # fₓ = -[(λ*+2μ)∂²u/∂x² + μ∂²u/∂y² + (λ*+μ)∂²v/∂x∂y]
    # fᵧ = -[(λ*+μ)∂²u/∂x∂y + μ∂²v/∂x² + (λ*+2μ)∂²v/∂y²]
    #
    # fₓ = -[(λ*+2μ)(2y) + μ(0) + (λ*+μ)(-2y)]
    #     = -[2y(λ*+2μ) - 2y(λ*+μ)]
    #     = -[2y·μ]
    #     = -2μy
    #
    # fᵧ = -[(λ*+μ)(2x) + μ(0) + (λ*+2μ)(-2x)]
    #     = -[2x(λ*+μ) - 2x(λ*+2μ)]
    #     = -[-2x·μ]
    #     = 2μx

    E_val = 1.0
    ν_val = 0.3
    μ_val = E_val / (2 * (1 + ν_val))
    λ_full = E_val * ν_val / ((1 + ν_val) * (1 - 2ν_val))
    λstar_val = 2μ_val * λ_full / (λ_full + 2μ_val)

    u_exact(x, y) = x^2 * y
    v_exact(x, y) = -x * y^2

    fx_exact(x, y) = -2μ_val * y
    fy_exact(x, y) = 2μ_val * x

    body_force(x) = (fx_exact(x[1], x[2]), fy_exact(x[1], x[2]))

    bc_func(x, t) = (u_exact(x[1], x[2]), v_exact(x[1], x[2]))

    dx = 1 / 25 * m
    part = create_2d_square_domain(dx)
    cloud = WTP.discretize(part, ConstantSpacing(dx), alg = VanDerSandeFornberg())
    cloud, _ = repel(cloud, ConstantSpacing(dx); α = dx / 20, max_iters = 500)

    bcs = Dict(
        :surface1 => Displacement(bc_func),
        :surface2 => Displacement(bc_func),
        :surface3 => Displacement(bc_func),
        :surface4 => Displacement(bc_func),
    )

    model = LinearElasticity(E = E_val, ν = ν_val, body_force = body_force)
    domain = MM.Domain(cloud, bcs, model)

    println("\nMMS Model: ", model)
    println("μ = $μ_val, λ* = $λstar_val")

    prob = MM.LinearProblem(domain)
    sol = solve(prob)

    N = length(cloud)
    u_num = sol.u[1:N]
    v_num = sol.u[(N + 1):(2N)]

    coords = MM._coords(cloud)
    u_ana = [u_exact(ustrip(pt.x), ustrip(pt.y)) for pt in coords]
    v_ana = [v_exact(ustrip(pt.x), ustrip(pt.y)) for pt in coords]

    err_u = u_num .- u_ana
    err_v = v_num .- v_ana

    L2_u = norm(err_u) / sqrt(N)
    L2_v = norm(err_v) / sqrt(N)
    Linf_u = norm(err_u, Inf)
    Linf_v = norm(err_v, Inf)

    println("\nMMS Error Analysis:")
    println("  u-displacement: L2=$L2_u, L∞=$Linf_u")
    println("  v-displacement: L2=$L2_v, L∞=$Linf_v")

    @test L2_u < 5e-2
    @test L2_v < 5e-2
    @test Linf_u < 1e-1
    @test Linf_v < 1e-1

    println("\n  2D Linear Elasticity MMS validated")
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
    model = LinearElasticity(E = 200e3, ν = 0.3)
    domain = MM.Domain(cloud, bcs, model)

    sim = Simulation(domain)
    set!(sim, ux = 0.0, uy = 0.0)

    @test sim.mode == MM.SteadyState

    run!(sim)

    @test sim._solution !== nothing
    @test sim.iteration == 1

    ux, uy = displacement(sim)
    N = length(domain.cloud)
    @test length(ux) == N
    @test length(uy) == N

    # All-zero Dirichlet with no body force → zero displacement
    @test norm(ux) < 1e-10
    @test norm(uy) < 1e-10

    println("  Simulation API integration test passed")
end
