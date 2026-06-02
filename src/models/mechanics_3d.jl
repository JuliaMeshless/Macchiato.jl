# ============================================================================
# mechanics_3d.jl — 3D linear isotropic elasticity (Navier-Cauchy equations).
#
# The 2D LinearElasticity in mechanics.jl is plane-stress only. This file adds
# the full 3D version for the 3D shape-optimization pipeline.
# ============================================================================

"""
    LinearElasticity3D{E, Nu, Rho, F} <: Solid

3D linear isotropic elasticity model.

The governing equations in displacement form (Navier-Cauchy):
```
μ∇²u + (λ+μ)∇(∇·u) + f = 0
```

In 3D component form:
```
(λ+2μ)∂²u/∂x² + μ(∂²u/∂y² + ∂²u/∂z²) + (λ+μ)(∂²v/∂x∂y + ∂²w/∂x∂z) + fₓ = 0
(λ+2μ)∂²v/∂y² + μ(∂²v/∂x² + ∂²v/∂z²) + (λ+μ)(∂²u/∂x∂y + ∂²w/∂y∂z) + fᵧ = 0
(λ+2μ)∂²w/∂z² + μ(∂²w/∂x² + ∂²w/∂y²) + (λ+μ)(∂²u/∂x∂z + ∂²v/∂y∂z) + f_z = 0
```

Fields:
- `E`: Young's modulus
- `ν`: Poisson's ratio
- `ρ`: Density (optional)
- `body_force`: Body force function `f(x) -> (fx, fy, fz)` (optional)

Unlike the 2D plane-stress case, 3D uses the unmodified Lamé parameters
``λ = Eν/((1+ν)(1-2ν))``, ``μ = E/(2(1+ν))``.
"""
@kwdef struct LinearElasticity3D{TE, TNu, TRho, TF} <: Solid
    E::TE
    ν::TNu
    ρ::TRho = nothing
    body_force::TF = nothing
end

_num_vars(::LinearElasticity3D, ::Int) = 3  # (u, v, w)

"""
    lame_parameters_3d(model::LinearElasticity3D) -> (μ, λ)

Compute 3D Lamé parameters (no plane-stress modification):
- μ = E / (2(1+ν))
- λ = Eν / ((1+ν)(1-2ν))

Returns `(μ, λ)`.
"""
function lame_parameters_3d(model::LinearElasticity3D)
    (; E, ν) = model
    μ = E / (2 * (1 + ν))
    λ = E * ν / ((1 + ν) * (1 - 2ν))
    return μ, λ
end

function Base.show(io::IO, m::LinearElasticity3D)
    bf_str = m.body_force === nothing ? "" : ", body_force"
    ρ_str = m.ρ === nothing ? "" : ", ρ = $(m.ρ)"
    return print(io, "LinearElasticity3D: (E = $(m.E), ν = $(m.ν)$(ρ_str)$(bf_str))")
end

# ============================================================================
# 3D block assembly — analog of assemble_elasticity_from_weights for 3D.
# ============================================================================

"""
    assemble_elasticity_3d_from_weights(
        W_d2x, W_d2y, W_d2z, W_d2xy, W_d2xz, W_d2yz,
        N, λ, μ,
    ) -> SparseMatrixCSC

Build the 3N×3N Navier-Cauchy system from pre-computed 3D second-derivative
weight matrices.  Block structure:

    A[i,    j   ] = (λ+2μ)W_xx[i,j] + μ·W_yy[i,j] + μ·W_zz[i,j]
    A[i,    j+N ] = (λ+μ)W_xy[i,j]
    A[i,    j+2N] = (λ+μ)W_xz[i,j]
    A[i+N,  j   ] = (λ+μ)W_xy[i,j]
    A[i+N,  j+N ] = μ·W_xx[i,j] + (λ+2μ)W_yy[i,j] + μ·W_zz[i,j]
    A[i+N,  j+2N] = (λ+μ)W_yz[i,j]
    A[i+2N, j   ] = (λ+μ)W_xz[i,j]
    A[i+2N, j+N ] = (λ+μ)W_yz[i,j]
    A[i+2N, j+2N] = μ·W_xx[i,j] + μ·W_yy[i,j] + (λ+2μ)W_zz[i,j]

Returns a 3N×3N sparse matrix `[A11 A12 A13; A21 A22 A23; A31 A32 A33]`.
"""
function assemble_elasticity_3d_from_weights(
    W_d2x::SparseMatrixCSC{<:Number, <:Integer},
    W_d2y::SparseMatrixCSC{<:Number, <:Integer},
    W_d2z::SparseMatrixCSC{<:Number, <:Integer},
    W_d2xy::SparseMatrixCSC{<:Number, <:Integer},
    W_d2xz::SparseMatrixCSC{<:Number, <:Integer},
    W_d2yz::SparseMatrixCSC{<:Number, <:Integer},
    N::Int,
    λ::Real,
    μ::Real,
)
    c1 = λ + 2μ       # coefficient on "aligned" second derivatives
    c2 = μ             # coefficient on "transverse" second derivatives
    c3 = λ + μ         # coefficient on mixed partials

    A11 = c1 * W_d2x + c2 * W_d2y + c2 * W_d2z
    A12 = c3 * W_d2xy
    A13 = c3 * W_d2xz
    A22 = c2 * W_d2x + c1 * W_d2y + c2 * W_d2z
    A23 = c3 * W_d2yz
    A33 = c2 * W_d2x + c2 * W_d2y + c1 * W_d2z

    return [A11 A12 A13; A12 A22 A23; A13 A23 A33]
end

# ============================================================================
# 3D make_system (for use with the Simulation API — optional for shape opt).
# ============================================================================

function make_system(model::LinearElasticity3D, domain; kwargs...)
    μ, λ = lame_parameters_3d(model)
    coords = _ustrip(_coords(domain.cloud))
    N = length(coords)

    k = get(kwargs, :k, 40)
    adjl = find_neighbors(coords, k)

    ∂²x  = partial(coords, 2, 1; k = k, adjl = adjl, kwargs...)
    ∂²y  = partial(coords, 2, 2; k = k, adjl = adjl, kwargs...)
    ∂²z  = partial(coords, 2, 3; k = k, adjl = adjl, kwargs...)
    ∂²xy = custom(coords, _ℒ_mixed_partial_3d(1, 2); k = k, adjl = adjl, kwargs...)
    ∂²xz = custom(coords, _ℒ_mixed_partial_3d(1, 3); k = k, adjl = adjl, kwargs...)
    ∂²yz = custom(coords, _ℒ_mixed_partial_3d(2, 3); k = k, adjl = adjl, kwargs...)

    A = assemble_elasticity_3d_from_weights(
        ∂²x.weights, ∂²y.weights, ∂²z.weights,
        ∂²xy.weights, ∂²xz.weights, ∂²yz.weights,
        N, λ, μ,
    )

    b = zeros(eltype(A), 3N)
    if model.body_force !== nothing
        for (i, pt) in enumerate(coords)
            fx, fy, fz = model.body_force(pt)
            b[i]       = -fx
            b[i + N]   = -fy
            b[i + 2N]  = -fz
        end
    end

    return A, b
end

"""
    _ℒ_mixed_partial_3d(d1::Int, d2::Int)

Return a mixed-partial operator `∂²/∂x_{d1} ∂x_{d2}` for 3D RBF-FD.
Uses `D²` (directional second derivative) with orthogonal unit vectors.
"""
function _ℒ_mixed_partial_3d(d1::Int, d2::Int)
    e1 = ntuple(i -> Float64(i == d1), 3)
    e2 = ntuple(i -> Float64(i == d2), 3)
    return RadialBasisFunctions.D²(SVector{3,Float64}(e1), SVector{3,Float64}(e2))
end
