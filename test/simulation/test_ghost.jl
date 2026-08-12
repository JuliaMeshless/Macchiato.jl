using Test
using Macchiato
import Macchiato as MM
using WhatsThePoint
import WhatsThePoint as WTP
using Unitful: m, °
using StaticArrays: SVector
using LinearAlgebra: mul!
using SparseArrays: nnz

include(joinpath(@__DIR__, "..", "end_2_end", "2d_square.jl"))

function ghost_test_domain(bcs, model; dx = 1 / 17 * m)
    part = create_2d_square_domain(dx)
    cloud = WTP.discretize(part, ConstantSpacing(dx))
    return MM.Domain(cloud, bcs, model)
end

# Custom transient model consuming the anisotropic operator path end-to-end —
# the same way an example-script model (e.g. the Niederer monodomain) does.
struct AnisoDiffusion{V} <: Macchiato.AbstractModel
    D::V
end

Macchiato.num_vars(::AnisoDiffusion, _) = 1

function Macchiato.make_f(model::AnisoDiffusion, domain; kwargs...)
    D = model.D
    W, G, source_appliers = MM.build_neumann_diffusion(
        domain; k = 30,
        operator = (data; eval_points, k) -> custom(
            data, @operator(∇ ⋅ (D * ∇));
            eval_points = eval_points, k = k, basis = PHS(3; poly_deg = 2),
        ),
    )
    n_ghost = size(G, 2)
    cbuf = zeros(n_ghost)
    return function f(du, u, p, t)
        mul!(du, W, u)
        if n_ghost > 0
            for apply! in source_appliers
                apply!(cbuf, t)
            end
            mul!(du, G, cbuf, 1, 1)
        end
        return nothing
    end
end

@testset "build_neumann_diffusion operator kwarg" begin
    # surface1 = bottom, surface2 = right, surface3 = top, surface4 = left
    bcs = Dict(
        :surface1 => ZeroFlux(), :surface3 => ZeroFlux(),
        :surface2 => PrescribedValue(100.0), :surface4 => PrescribedValue(0.0),
    )
    # build_neumann_diffusion only reads the cloud and BCs; the model is a placeholder
    domain = ghost_test_domain(bcs, MM.SolidEnergy(k = 1.0, ρ = 1.0, cₚ = 1.0))
    N = length(domain.cloud)

    @testset "explicit laplacian closure reproduces the default" begin
        W_default, G_default, _ = MM.build_neumann_diffusion(domain)
        W_expl, G_expl, _ = MM.build_neumann_diffusion(
            domain;
            operator = (data; eval_points, k) -> laplacian(data; eval_points = eval_points, k = k),
        )
        println("  default W: size = $(size(W_default)), nnz = $(nnz(W_default))")
        @test size(W_expl) == size(W_default)
        @test nnz(W_expl) == nnz(W_default)
        @test W_expl ≈ W_default rtol = 1.0e-12
        @test G_expl ≈ G_default rtol = 1.0e-12
    end

    @testset "isotropic custom operator matches laplacian" begin
        Dones = SVector(1.0, 1.0)
        basis = PHS(3; poly_deg = 2)
        W_lap, _, _ = MM.build_neumann_diffusion(
            domain;
            operator = (data; eval_points, k) ->
            laplacian(data; eval_points = eval_points, k = k, basis = basis),
        )
        W_cus, _, _ = MM.build_neumann_diffusion(
            domain;
            operator = (data; eval_points, k) -> custom(
                data, @operator(∇ ⋅ (Dones * ∇));
                eval_points = eval_points, k = k, basis = basis,
            ),
        )
        # the two assembly paths (composed ∂²x + ∂²y vs fused ∇²) agree only up to
        # roundoff on O(1/dx²) ghost-amplified weights, so compare relative to scale
        reldiff = maximum(abs, W_cus - W_lap) / maximum(abs, W_lap)
        println("  |W_custom(D = I) - W_laplacian|_max / |W_laplacian|_max = $reldiff")
        @test reldiff < 1.0e-11
    end

    @testset "anisotropic operator: finite weights, zero row-sum" begin
        D = SVector(2.0, 0.5)
        W, G, appliers = MM.build_neumann_diffusion(
            domain; k = 30,
            operator = (data; eval_points, k) -> custom(
                data, @operator(∇ ⋅ (D * ∇));
                eval_points = eval_points, k = k, basis = PHS(3; poly_deg = 2),
            ),
        )
        row_sum = maximum(abs, vec(sum(W; dims = 2)))
        println(
            "  anisotropic W: nnz = $(nnz(W)), max|row-sum| = $row_sum, " *
                "max|W| = $(maximum(abs, W))"
        )
        @test size(W) == (N, N)
        @test all(isfinite, W.nzval)
        # ∇·(D∇) annihilates constants; the ghost→shadow fold must preserve that
        @test row_sum < 1.0e-8
        n_flux = sum(
            length(ids) for (name, (ids, bc)) in domain.boundaries if bc isa ZeroFlux
        )
        println("  ghost count: $(size(G, 2)) (ZeroFlux boundary nodes: $n_flux)")
        @test size(G, 2) == n_flux
        @test length(appliers) == 2  # one per ZeroFlux surface
    end

    @testset "end-to-end transient anisotropic diffusion" begin
        # ∇·(D∇T) = 0 with T(left) = 0, T(right) = 100 and zero-flux top/bottom has
        # the anisotropy-independent steady limit T = 100x, exact for the operator
        # up to discretization error.
        domain2 = ghost_test_domain(bcs, AnisoDiffusion(SVector(2.0, 0.5)))
        sim = Simulation(domain2, Transient(Δt = 1.0e-2, stop_time = 10.0))
        # custom models have no set! field map — assign u0 directly
        sim.u0 = zeros(length(domain2.cloud))
        run!(sim)
        T = solution(sim)
        x = [p[1] for p in node_coordinates(domain2)]
        err = maximum(abs, T .- 100 .* x)
        println("  transient anisotropic steady-limit L∞ error: $err")
        @test all(isfinite, T)
        @test err < 1.0
    end
end
