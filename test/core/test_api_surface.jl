using Test
using Macchiato
import Macchiato as MM
import RadialBasisFunctions
using WhatsThePoint
import WhatsThePoint as WTP
using StaticArrays
using Unitful: m, ustrip

include(joinpath(@__DIR__, "..", "end_2_end", "2d_square.jl"))

function create_api_test_cloud()
    dx = 1 / 9 * m
    return create_2d_square_cloud(dx)
end

struct APITestModel <: AbstractModel end

@testset "API surface" begin
    @testset "RadialBasisFunctions operator surface is re-exported" begin
        for name in (
                :laplacian, :partial, :mixed_partial, :gradient, :custom, :weights,
                Symbol("@operator"), :Laplacian, :Partial, :MixedPartial, :Identity,
                :PHS, :PHS1, :PHS3, :PHS5, :PHS7, :IMQ, :Gaussian,
                :find_neighbors, :autoselect_k,
            )
            @test Base.isexported(Macchiato, name)
        end
        # BC vocabulary is Macchiato's own
        @test Base.isexported(Macchiato, :Dirichlet)  # Macchiato's abstract type
    end

    @testset "RadialBasisFunctions owns no boundary-condition vocabulary" begin
        # Layer invariant: RBF is an operator/stencil library. Boundary conditions live
        # here, geometry lives in WhatsThePoint. If any of these reappear upstream, the
        # packages have re-coupled — see the 0.9.0 Hermite removal.
        for name in (
                :BoundaryCondition, :Dirichlet, :Neumann, :Robin, :Internal,
                :is_dirichlet, :is_neumann, :is_robin, :is_internal,
                :HermiteStencilData, :classify_stencil,
            )
            @test !isdefined(RadialBasisFunctions, name)
        end
    end

    @testset "num_vars rename keeps _num_vars back-compat alias" begin
        @test MM._num_vars === num_vars
        # methods defined through the old name land on the new function
        MM._num_vars(::APITestModel, _) = 7
        @test num_vars(APITestModel(), 2) == 7
    end

    @testset "node_coordinates" begin
        cloud = create_api_test_cloud()
        x = node_coordinates(cloud)
        @test x isa AbstractVector{<:SVector{2, Float64}}
        @test length(x) == length(points(cloud))
        @test x == MM._ustrip(MM._coords(cloud))
        # units kept on request
        xu = node_coordinates(cloud; strip_units = false)
        @test ustrip.(first(xu)) ≈ first(x)
        println("  node_coordinates: $(length(x)) nodes, eltype $(eltype(x))")
    end

    @testset "LinearElasticity assembles via mixed_partial + operator algebra" begin
        # Regression for the migration off the hand-rolled _ℒ_mixed_partial:
        # make_system must equal the manually assembled block matrix.
        cloud = create_api_test_cloud()
        model = LinearElasticity(E = 200.0e3, ν = 0.3)
        # assembly only touches domain.cloud, so a NamedTuple stands in for Domain
        A, b = MM.make_system(model, (; cloud = cloud))

        x = node_coordinates(cloud)
        k = MM.DEFAULT_STENCIL_SIZE
        adjl = find_neighbors(x, k)
        W∂²x = weights(partial(x, 2, 1; k = k, adjl = adjl))
        W∂²y = weights(partial(x, 2, 2; k = k, adjl = adjl))
        W∂²xy = weights(mixed_partial(x, 1, 2; k = k, adjl = adjl))

        μ, λstar = lame_parameters(model)
        A₁₁ = (λstar + 2μ) * W∂²x + μ * W∂²y
        A₁₂ = (λstar + μ) * W∂²xy
        A₂₂ = μ * W∂²x + (λstar + 2μ) * W∂²y
        A_ref = [A₁₁ A₁₂; A₁₂ A₂₂]

        @test size(A) == size(A_ref)
        @test A ≈ A_ref rtol = 1.0e-12
        @test all(iszero, b)
        println("  elasticity assembly: $(size(A, 1))×$(size(A, 2)), matches reference")
    end
end
