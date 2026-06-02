# ============================================================================
# morph_extension.jl — Interior mesh extension (morph) for shape optimization.
#
# When boundary nodes move under a design step, interior nodes must follow to
# maintain cloud quality. An extension (morph) is a smooth, differentiable map
# from boundary displacement to interior displacement. The adjoint flows back
# through its transpose.
#
# FRONT 1 in the two-front architecture: the morph keeps the cloud
# differentiable within a remesh interval. The discrete adjoint flows through
# it exactly (transpose correction).
#
# Concrete extensions shipped here:
#   - LaplaceExtension  (harmonic interpolation: ∇²(Δx) = 0 with Dirichlet BCs)
#
# The Laplace operator is dimension-agnostic; the same struct works for 2D and
# 3D — only the Laplacian assembly changes (2 terms in 2D, 3 in 3D).
# ============================================================================

# (uses RadialBasisFunctions._build_weights, Partial — already in scope from Macchiato.jl)
using SparseArrays, LinearAlgebra, StaticArrays

"""
    AbstractExtension

Abstract type for interior mesh extension (morph) operators.

Interface (documented; dispatch is duck-typed):
- `morph(ext, boundary_new) -> full_pts` — extend interior to follow the new
  boundary positions; returns the complete point cloud.
- `morph_transpose(ext, g_all, hole_idx) -> g_hole` — carry the full nodal
  gradient back to the hole boundary via the morph transpose:
  ``ĝ_bnd = g_bnd + Mᵀ g_int`` where ``M = -L_int⁻¹ L_int_bnd``.
- `build_laplace_extension(...)` — convenience constructor for LaplaceExtension.
"""
abstract type AbstractExtension end

# ============================================================================
# LaplaceExtension — harmonic morph (∇²(Δx) = 0 with Dirichlet BCs on boundary).
# ============================================================================

"""
    LaplaceExtension <: AbstractExtension

Laplace-equation interior extension. The interior displacement is slaved to
the boundary by solving

    Δx_int = -L_int⁻¹ · L_int_bnd · Δx_bnd

where ``L`` is the RBF-FD Laplace matrix and the blocks are
`L[interior, interior]` and `L[interior, boundary]`. The outer boundary
displacement is always zero (the outer frame is fixed); only the hole
boundary moves.

The adjoint flows through the transpose:

    ĝ_bnd = g_bnd + Mᵀ g_int,   M = -L_int⁻¹ L_int_bnd

so each hole node's gradient receives a correction from every interior
node whose displacement it influences.

Construction via `build_laplace_extension(pts, adjl, basis, interior_idx, boundary_idx, ref_interior, ref_hole, n_outer, nθ)`.

Fields:
- `morph_fact` — `lu(L_int)`, factorized once per anchor.
- `L_int_bnd` — `L[interior_idx, boundary_idx]`, sparse.
- `interior_idx` — global indices of interior nodes.
- `boundary_idx` — global indices of boundary nodes (ordered: [outer; hole]).
- `n_outer` — number of outer (fixed) boundary nodes.
- `nθ` — number of hole (design) boundary nodes.
- `ref_pts` — full point cloud at anchor (the morph reference).
- `ref_interior` — interior positions at anchor.
- `ref_hole` — hole boundary positions at anchor.
"""
struct LaplaceExtension{T_FACT, T_MAT, T_PTS, T_VEC} <: AbstractExtension
    morph_fact::T_FACT       # lu(L_int)
    L_int_bnd::T_MAT         # Laplacian[interior, boundary]
    interior_idx::Vector{Int}
    boundary_idx::Vector{Int}  # [outer; hole] ordering
    n_outer::Int
    nθ::Int
    ref_pts::T_PTS           # full point cloud at anchor
    ref_interior::T_VEC      # interior positions at anchor
    ref_hole::T_VEC          # hole boundary positions at anchor
end

"""
    build_laplace_extension(pts, adjl, basis, interior_idx, boundary_idx, ref_interior, ref_hole, n_outer, nθ) -> LaplaceExtension

Build the Laplace morph operator from a reference point cloud.
Factorizes `L[interior, interior]` once (via LU); the factorization
is reused for every morph and transpose call within the interval.

Arguments:
- `pts` — full point cloud at anchor (Vector{SVector{2}}).
- `adjl` — adjacency list.
- `basis` — RBF basis (e.g. `PHS(3; poly_deg=3)`).
- `interior_idx` — global indices of interior nodes.
- `boundary_idx` — global indices of boundary nodes, ordered as `[outer; hole]`.
- `ref_interior` — interior point positions at anchor.
- `ref_hole` — hole boundary point positions at anchor.
- `n_outer` — number of outer (fixed) boundary nodes.
- `nθ` — number of hole (design) boundary nodes.
"""
function build_laplace_extension(
    pts::AbstractVector{<:AbstractVector{Float64}},
    adjl::AbstractVector,
    basis,
    interior_idx::AbstractVector{Int},
    boundary_idx::AbstractVector{Int},
    ref_interior::AbstractVector{<:AbstractVector{Float64}},
    ref_hole::AbstractVector{<:AbstractVector{Float64}},
    n_outer::Int,
    nθ::Int,
)
    Wxx = _build_weights(Partial(2, 1), pts, pts, adjl, basis)
    Wyy = _build_weights(Partial(2, 2), pts, pts, adjl, basis)
    Wlap = Wxx + Wyy
    L_int     = Wlap[interior_idx, interior_idx] + 1e-8 * I
    L_int_bnd = Wlap[interior_idx, boundary_idx]

    return LaplaceExtension(
        lu(L_int), L_int_bnd,
        collect(Int, interior_idx), collect(Int, boundary_idx),
        n_outer, nθ,
        copy(pts), copy(ref_interior), copy(ref_hole),
    )
end

# Outer constructor — delegates to build_laplace_extension for discoverability.
LaplaceExtension(args...; kwargs...) = build_laplace_extension(args...; kwargs...)

"""
    morph(ext::LaplaceExtension, hole_new) -> Vector{SVector{2,Float64}}

Extend the interior to follow the new hole boundary positions.
Returns the complete point cloud (interior morphed, outer at reference,
hole at `hole_new`).

The outer boundary nodes stay at their reference positions (fixed frame).
"""
function morph(
    ext::LaplaceExtension,
    hole_new::AbstractVector{<:AbstractVector{Float64}},
)
    # Boundary displacement: outer = 0, hole = hole_new - ref_hole
    Δhx = vcat(zeros(ext.n_outer),
               [hole_new[j][1] - ext.ref_hole[j][1] for j in 1:ext.nθ])
    Δhy = vcat(zeros(ext.n_outer),
               [hole_new[j][2] - ext.ref_hole[j][2] for j in 1:ext.nθ])

    Δix = ext.morph_fact \ (-ext.L_int_bnd * Δhx)
    Δiy = ext.morph_fact \ (-ext.L_int_bnd * Δhy)

    pts = copy(ext.ref_pts)
    @inbounds for (kk, i) in enumerate(ext.interior_idx)
        pts[i] = ext.ref_interior[kk] + SVector{2,Float64}(Δix[kk], Δiy[kk])
    end
    @inbounds for (kk, i) in enumerate(ext.boundary_idx[ext.n_outer + 1:end])
        pts[i] = hole_new[kk]
    end
    return pts
end

"""
    morph_transpose(ext::LaplaceExtension, g_all, hole_idx) -> Vector{SVector{2,Float64}}

Carry the full nodal gradient back to the hole boundary via the morph transpose:

    ĝ_hole = g_hole + Mᵀ g_int,    M = -L_int⁻¹ L_int_bnd

Arguments:
- `ext::LaplaceExtension` — the morph operator.
- `g_all` — full nodal gradient (one SVector{2} per point).
- `hole_idx` — global indices of the hole boundary nodes.

Returns the morph-corrected hole boundary gradient (length `ext.nθ`).
"""
function morph_transpose(
    ext::LaplaceExtension,
    g_all::AbstractVector{<:AbstractVector{Float64}},
    hole_idx::AbstractVector{Int},
)
    gix = [g_all[i][1] for i in ext.interior_idx]
    giy = [g_all[i][2] for i in ext.interior_idx]

    # cb = -L_int_bndᵀ · L_int⁻ᵀ · g_int   (length n_boundary = n_outer + nθ)
    cbx = -(ext.L_int_bnd' * (ext.morph_fact' \ gix))
    cby = -(ext.L_int_bnd' * (ext.morph_fact' \ giy))

    # Hole boundary = last nθ entries of the boundary ordering
    return [SVector{2,Float64}(
        g_all[hole_idx[j]][1] + cbx[ext.n_outer + j],
        g_all[hole_idx[j]][2] + cby[ext.n_outer + j],
    ) for j in 1:ext.nθ]
end
