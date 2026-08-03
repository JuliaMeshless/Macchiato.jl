"""
    LinearElasticity{E, Nu, Rho, F} <: Solid

Linear isotropic elasticity model for solid mechanics (Navier-Cauchy equations).

Supports 2D plane stress formulation. The governing equations in displacement form:
```
(λ*+2μ) ∂²u/∂x² + μ ∂²u/∂y² + (λ*+μ) ∂²v/∂x∂y + fₓ = 0
(λ*+μ) ∂²u/∂x∂y + μ ∂²v/∂x² + (λ*+2μ) ∂²v/∂y² + fᵧ = 0
```

# Fields
- `E`: Young's modulus
- `ν`: Poisson's ratio
- `ρ`: Density (optional, for body forces / dynamics)
- `body_force`: Body force function `f(x) -> (fx, fy)` (optional)

# Example
```julia
model = LinearElasticity(E=200e3, ν=0.3)
model = LinearElasticity(E=200e3, ν=0.3, body_force=x -> (0.0, -9.81))
```
"""
@kwdef struct LinearElasticity{TE, TNu, TRho, TF} <: Solid
    E::TE
    ν::TNu
    ρ::TRho = nothing
    body_force::TF = nothing
end

num_vars(::LinearElasticity, dim::Int) = dim

"""
    lame_parameters(model::LinearElasticity)

Compute Lamé parameters for plane stress:
- μ = E / (2(1+ν))
- λ = Eν / ((1+ν)(1-2ν))
- λ* = 2μλ / (λ+2μ)  (plane stress modification)

Returns (μ, λstar).
"""
function lame_parameters(model::LinearElasticity)
    (; E, ν) = model
    μ = E / (2 * (1 + ν))
    λ = E * ν / ((1 + ν) * (1 - 2ν))
    λstar = 2μ * λ / (λ + 2μ)
    return μ, λstar
end

"""
    make_system(model::LinearElasticity, domain; kwargs...)

Assemble the 2N×2N linear system for steady-state linear elasticity.

System structure:
```
[A₁₁  A₁₂] [u]   [bₓ]
[A₂₁  A₂₂] [v] = [bᵧ]
```
Where:
- A₁₁ = (λ*+2μ) ∂²/∂x² + μ ∂²/∂y²
- A₁₂ = (λ*+μ) ∂²/∂x∂y
- A₂₁ = (λ*+μ) ∂²/∂x∂y
- A₂₂ = μ ∂²/∂x² + (λ*+2μ) ∂²/∂y²
"""
function make_system(model::LinearElasticity, domain; kwargs...)
    μ, λstar = lame_parameters(model)
    coords = _ustrip(_coords(domain.cloud))
    N = length(coords)

    # Pre-compute shared KNN adjacency list (same coords and k for all 3 operators)
    k = get(kwargs, :k, DEFAULT_STENCIL_SIZE)
    adjl = find_neighbors(coords, k)

    # Build RBF operators (KernelAbstractions parallelizes internally)
    ∂²x = partial(coords, 2, 1; k = k, adjl = adjl, kwargs...)
    ∂²y = partial(coords, 2, 2; k = k, adjl = adjl, kwargs...)
    ∂²xy = mixed_partial(coords, 1, 2; k = k, adjl = adjl, kwargs...)

    # Assemble 2N×2N system from blocks via operator algebra
    A₁₁ = weights((λstar + 2μ) * ∂²x + μ * ∂²y)
    A₁₂ = weights((λstar + μ) * ∂²xy)
    A₂₂ = weights(μ * ∂²x + (λstar + 2μ) * ∂²y)

    # Combine into 2N×2N sparse matrix
    A = [
        A₁₁ A₁₂;
        A₁₂ A₂₂
    ]

    # Build RHS from body force
    # PDE: L[u] + f = 0  =>  A*u = -f
    b = zeros(eltype(A), 2N)
    if model.body_force !== nothing
        for (i, pt) in enumerate(coords)
            fx, fy = model.body_force(pt)
            b[i] = -fx
            b[i + N] = -fy
        end
    end

    return A, b
end

function Base.show(io::IO, m::LinearElasticity)
    bf_str = m.body_force === nothing ? "" : ", body_force"
    ρ_str = m.ρ === nothing ? "" : ", ρ = $(m.ρ)"
    return print(io, "LinearElasticity: (E = $(m.E), ν = $(m.ν)$(ρ_str)$(bf_str))")
end
