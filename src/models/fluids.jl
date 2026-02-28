"""
    AbstractViscosity

Abstract supertype for viscosity models used with [`IncompressibleNavierStokes`](@ref).
Concrete subtypes must be callable as `μ(γ̇)` returning the dynamic viscosity at shear rate `γ̇`.
"""
abstract type AbstractViscosity end

"""
    NewtonianViscosity(μ)

Constant (Newtonian) viscosity model. Returns the same viscosity `μ` regardless of shear rate.

# Example
```julia
visc = NewtonianViscosity(0.001)  # water-like viscosity
visc(100.0)  # => 0.001
```
"""
struct NewtonianViscosity{M <: Real} <: AbstractViscosity
    μ::M
end
(μ::NewtonianViscosity)(_) = μ.μ

function Base.show(io::IO, e::NewtonianViscosity)
    print(io, "Newtonian Viscosity: $(e.μ)")
end

"""
    CarreauYasudaViscosity(; μ_inf, μ_0, n=0.333, λ=0.31, a=2)

Generalized Newtonian viscosity model for shear-thinning (or shear-thickening) fluids.

The Carreau-Yasuda model:
```
μ(γ̇) = μ_inf + (μ_0 - μ_inf) * (1 + (λ γ̇)^a)^((n - 1) / a)
```

# Fields
- `μ_inf`: Infinite-shear-rate viscosity
- `μ_0`: Zero-shear-rate viscosity
- `n`: Power-law index (< 1 for shear-thinning)
- `λ`: Relaxation time
- `a`: Yasuda parameter (a = 2 recovers the Carreau model)

# Example
```julia
visc = CarreauYasudaViscosity(μ_inf=0.0035, μ_0=0.056, n=0.333, λ=0.31)
```
"""
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

"""
    IncompressibleNavierStokes(; μ, ρ)
    IncompressibleNavierStokes(μ::Real, ρ)

Incompressible Navier-Stokes fluid model.

When `μ` is a plain number, it is automatically wrapped in [`NewtonianViscosity`](@ref).
Pass an [`AbstractViscosity`](@ref) subtype (e.g., [`CarreauYasudaViscosity`](@ref)) for
non-Newtonian behavior.

!!! warning
    The fluid solver is under active development and not yet fully functional.

# Fields
- `μ::AbstractViscosity`: Dynamic viscosity model
- `ρ`: Fluid density

# Example
```julia
model = IncompressibleNavierStokes(μ=0.001, ρ=1000.0)
model = IncompressibleNavierStokes(μ=CarreauYasudaViscosity(μ_inf=0.0035, μ_0=0.056), ρ=1060.0)
```
"""
@kwdef struct IncompressibleNavierStokes{M <: AbstractViscosity, P} <: Fluid
    μ::M
    ρ::P
end

function IncompressibleNavierStokes(μ::M, ρ) where {M <: Real}
    IncompressibleNavierStokes(NewtonianViscosity(μ), ρ)
end

_num_vars(::IncompressibleNavierStokes, dim::Int) = dim + 1

function make_f(
        model::IncompressibleNavierStokes, domain::Domain{Dim}; kwargs...) where {Dim}
    (; μ, ρ) = model
    vol = _coords(domain.cloud.volume)
    all_points = _coords(domain.cloud)
    ∇² = laplacian(all_points, vol; k = 40)
    α = k / (cₚ * ρ)
    w = α * ∇².weights
    n_boundary = length(boundary(domain.cloud))
    vol_ids = (n_boundary + 1):(n_boundary + length(domain.cloud.volume))

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
