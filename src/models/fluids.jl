abstract type AbstractViscosity end
struct NewtonianViscosity{M <: Real} <: AbstractViscosity
    μ::M
end
(μ::NewtonianViscosity)(_) = μ.μ

function Base.show(io::IO, e::NewtonianViscosity)
    print(io, "Newtonian Viscosity: $(e.μ)")
end

@kwdef struct CarreauYasudaViscosity{MI, M0, N, L, A} <: AbstractViscosity
    μ_inf::MI
    μ_0::M0
    n::N = 0.333
    λ::L = 0.31
    a::A = 2
end
function (μ::CarreauYasudaViscosity)(γ::T) where {T}
    (; μ_inf, μ_0, n, λ, a) = μ
    μ_inf + (μ_0 - μ_inf) * (one(T) + (λ * γ)^a)^((n - one(T)) / a)
end

function Base.show(io::IO, e::CarreauYasudaViscosity)
    print(io,
        "Carreau-Yasuda Viscosity: (μ_inf = $(e.μ_inf), μ_0 = $(e.μ_0), n = $(e.n), λ = $(e.λ))")
end

@kwdef struct IncompressibleNavierStokes{M <: AbstractViscosity, P} <: Fluid
    μ::M
    ρ::P
end

function IncompressibleNavierStokes(μ::M, ρ) where {M <: Real}
    IncompressibleNavierStokes(NewtonianViscosity(μ), ρ)
end

_num_vars(::IncompressibleNavierStokes, dim::Int) = dim + 1

# Physics domain trait - imported from boundary_conditions/physics_domains.jl
physics_domain(::Type{<:IncompressibleNavierStokes}) = FluidPhysics()

function make_f(
        model::IncompressibleNavierStokes, domain::Domain{Dim}; kwargs...) where {Dim}
    (; μ, ρ) = model
    vol = _coords(domain.cloud.volume)
    all_points = _coords(domain.cloud)
    ∇² = laplacian(all_points, vol; k = 40)
    update_weights!(∇²)
    α = k / (cₚ * ρ)
    w = α * ∇².weights
    vol_ids = only(domain.cloud.volume.points.indices)

    function f(du, u, p, t)
        # calculate intermediate velocity
        # calculate pressure correction
        # correct velocity and correct pressure at boundaries
        # correct velocities and pressure on interior points
        mul!(view(du, vol_ids), w, u)
        return nothing
    end

    return f
end

function Base.show(io::IO, e::IncompressibleNavierStokes)
    print(io, "Incompressible Navier Stokes: (μ = $(e.μ), ρ = $(e.ρ))")
end
