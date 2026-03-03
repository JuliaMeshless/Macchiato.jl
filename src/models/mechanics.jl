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

_num_vars(::LinearElasticity, dim::Int) = dim

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
    _ℒ_mixed_partial(basis::AbstractRadialBasis)

Mixed partial derivative ∂²/∂x∂y for radial basis functions.
Uses the D² (directional second derivative) functor with orthogonal unit vectors.
"""
function _ℒ_mixed_partial(basis::RadialBasisFunctions.AbstractRadialBasis)
    return RadialBasisFunctions.D²(basis, SVector(1.0, 0.0), SVector(0.0, 1.0))
end

"""
    _ℒ_mixed_partial(basis::MonomialBasis{2, Deg})

Mixed partial derivative ∂²/∂x∂y for 2D monomial basis.

Hand-coded to match the MonomialBasis evaluation ordering. The generic
`∂exponents`/`monomial_recursive_list`/`build_monomial_basis` pipeline
produces results in `multiexponents` ordering which differs from the
`MonomialBasis` evaluator ordering used in the RBF-FD collocation matrix.
"""
function _ℒ_mixed_partial(::RadialBasisFunctions.MonomialBasis{2, 0})
    function basis!(b, x)
        b .= zero(eltype(x))
        return nothing
    end
    return RadialBasisFunctions.ℒMonomialBasis(2, 0, basis!)
end

function _ℒ_mixed_partial(::RadialBasisFunctions.MonomialBasis{2, 1})
    function basis!(b, x)
        b .= zero(eltype(x))
        return nothing
    end
    return RadialBasisFunctions.ℒMonomialBasis(2, 1, basis!)
end

function _ℒ_mixed_partial(::RadialBasisFunctions.MonomialBasis{2, 2})
    # Monomial ordering: [1, x, y, xy, x², y²]
    # ∂²/∂x∂y:          [0, 0, 0,  1,  0,  0]
    function basis!(b, x)
        T = eltype(x)
        b .= zero(T)
        b[4] = one(T)
        return nothing
    end
    return RadialBasisFunctions.ℒMonomialBasis(2, 2, basis!)
end

function _ℒ_mixed_partial(mon::RadialBasisFunctions.MonomialBasis{2, Deg}) where {Deg}
    n = binomial(2 + Deg, 2)

    # Probe the monomial evaluator to determine exponents in its native ordering
    b_x = zeros(n)
    b_y = zeros(n)
    mon.f(b_x, [2.0, 1.0])  # b_x[i] = 2^(x_exponent_i)
    mon.f(b_y, [1.0, 2.0])  # b_y[i] = 2^(y_exponent_i)

    ax = round.(Int, log2.(b_x))
    ay = round.(Int, log2.(b_y))

    # Precompute active indices: ∂²/∂x∂y is non-zero only when both exponents ≥ 1
    active = [(i, ax[i], ay[i]) for i in 1:n if ax[i] >= 1 && ay[i] >= 1]

    function basis!(b, x)
        T = eltype(x)
        b .= zero(T)
        for (i, a, c) in active
            b[i] = T(a * c) * x[1]^(a - 1) * x[2]^(c - 1)
        end
        return nothing
    end
    return RadialBasisFunctions.ℒMonomialBasis(2, Deg, basis!)
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
    k = get(kwargs, :k, 40)
    adjl = find_neighbors(coords, k)

    # Build RBF operators (KernelAbstractions parallelizes internally)
    ∂²x = partial(coords, 2, 1; k = k, adjl = adjl, kwargs...)
    ∂²y = partial(coords, 2, 2; k = k, adjl = adjl, kwargs...)
    ∂²xy = custom(coords, _ℒ_mixed_partial; k = k, adjl = adjl, kwargs...)

    # Assemble 2N×2N system from blocks
    W_∂²x = ∂²x.weights
    W_∂²y = ∂²y.weights
    W_∂²xy = ∂²xy.weights

    # A₁₁ = (λ*+2μ) ∂²/∂x² + μ ∂²/∂y²
    A₁₁ = (λstar + 2μ) * W_∂²x + μ * W_∂²y

    # A₁₂ = A₂₁ = (λ*+μ) ∂²/∂x∂y
    A₁₂ = (λstar + μ) * W_∂²xy

    # A₂₂ = μ ∂²/∂x² + (λ*+2μ) ∂²/∂y²
    A₂₂ = μ * W_∂²x + (λstar + 2μ) * W_∂²y

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
