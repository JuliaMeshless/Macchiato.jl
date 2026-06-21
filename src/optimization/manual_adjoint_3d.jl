# ============================================================================
# manual_adjoint_3d.jl — 3D elasticity adjoint: Step 3 extraction + shape_gradient_3d.
#
# Mirrors the 2D pipeline in manual_adjoint.jl and MacchiatoMooncakeExt.jl
# but for the 3D Navier-Cauchy system (3 displacement components, 6 second-
# derivative operators for interior, 3 first-derivative operators for Neumann).
#
# For the first pass we implement interior-only (Dirichlet BCs); Neumann
# (traction) BCs are added in a follow-up.
# ============================================================================

import RadialBasisFunctions: _build_weights_and_cache, Partial, MixedPartial, _pullback_weights!

# ============================================================================
# Step 3: 3D coefficient extraction
# ============================================================================

"""
    extract_weight_sensitivities_elasticity_3d!(
        ΔW_d2x, ΔW_d2y, ΔW_d2z, ΔW_d2xy, ΔW_d2xz, ΔW_d2yz,
        W_template, η, u, active, λ, μ;
        interior_rows = nothing,
    )

Step 3 of the manual adjoint for 3D linear elasticity.

Given the global adjoint η = Aᵀ \\ ∂L/∂u (length 3N) and forward solution u
(length 3N), populate the gradients of the six second-derivative weight
matrices.  The 3N×3N system has nine 3×3 block entries:

    t_ab = active[row_a] ? -η[row_a] · u[col_b] : 0.0

The pullback onto the six operators is:

    ΔW_xx  = c1·t₁₁ + c2·t₂₂ + c2·t₃₃
    ΔW_yy  = c2·t₁₁ + c1·t₂₂ + c2·t₃₃
    ΔW_zz  = c2·t₁₁ + c2·t₂₂ + c1·t₃₃
    ΔW_xy  = c3·(t₁₂ + t₂₁)
    ΔW_xz  = c3·(t₁₃ + t₃₁)
    ΔW_yz  = c3·(t₂₃ + t₃₂)

where c1 = λ + 2μ, c2 = μ, c3 = λ + μ.

`W_template` provides the shared sparsity pattern via `nzrange`.  All six ΔW
outputs must be pre-allocated with that exact pattern.
"""
function extract_weight_sensitivities_elasticity_3d!(
    ΔW_d2x::SparseMatrixCSC{Float64, Int},
    ΔW_d2y::SparseMatrixCSC{Float64, Int},
    ΔW_d2z::SparseMatrixCSC{Float64, Int},
    ΔW_d2xy::SparseMatrixCSC{Float64, Int},
    ΔW_d2xz::SparseMatrixCSC{Float64, Int},
    ΔW_d2yz::SparseMatrixCSC{Float64, Int},
    W_template::SparseMatrixCSC{Float64, Int},
    η::AbstractVector{Float64},
    u::AbstractVector{Float64},
    active::AbstractVector{Bool},
    λ::Real,
    μ::Real;
    interior_rows::Union{Nothing, BitVector} = nothing,
)
    N = size(W_template, 1)
    @assert length(η) == 3N
    @assert length(u) == 3N
    @assert length(active) == 3N

    c1 = λ + 2μ
    c2 = μ
    c3 = λ + μ

    fill!(ΔW_d2x.nzval,  0.0)
    fill!(ΔW_d2y.nzval,  0.0)
    fill!(ΔW_d2z.nzval,  0.0)
    fill!(ΔW_d2xy.nzval, 0.0)
    fill!(ΔW_d2xz.nzval, 0.0)
    fill!(ΔW_d2yz.nzval, 0.0)

    @inbounds for j in 1:N
        for idx in nzrange(W_template, j)
            i = W_template.rowval[idx]

            if interior_rows !== nothing && !interior_rows[i]
                continue
            end

            # Nine block entries t_ab = -η[row_a] · u[col_b]
            t11 = active[i]        ? -η[i]        * u[j]        : 0.0
            t22 = active[i + N]    ? -η[i + N]    * u[j + N]    : 0.0
            t33 = active[i + 2N]   ? -η[i + 2N]   * u[j + 2N]   : 0.0
            t12 = active[i]        ? -η[i]        * u[j + N]    : 0.0
            t21 = active[i + N]    ? -η[i + N]    * u[j]        : 0.0
            t13 = active[i]        ? -η[i]        * u[j + 2N]   : 0.0
            t31 = active[i + 2N]   ? -η[i + 2N]   * u[j]        : 0.0
            t23 = active[i + N]    ? -η[i + N]    * u[j + 2N]   : 0.0
            t32 = active[i + 2N]   ? -η[i + 2N]   * u[j + N]    : 0.0

            ΔW_d2x.nzval[idx]  = c1 * t11 + c2 * t22 + c2 * t33
            ΔW_d2y.nzval[idx]  = c2 * t11 + c1 * t22 + c2 * t33
            ΔW_d2z.nzval[idx]  = c2 * t11 + c2 * t22 + c1 * t33
            ΔW_d2xy.nzval[idx] = c3 * (t12 + t21)
            ΔW_d2xz.nzval[idx] = c3 * (t13 + t31)
            ΔW_d2yz.nzval[idx] = c3 * (t23 + t32)
        end
    end

    return nothing
end

# ============================================================================
# 3D weight-gradient propagation
# ============================================================================
# `_propagate_weight_gradient!` (manual_adjoint.jl) infers the spatial dimension
# from `length(first(pts))`, so it handles 2D and 3D with no separate code path.
# 3D `shape_gradient_3d` calls it directly.

# ============================================================================
# 3D Neumann (traction) BCs for linear elasticity — frozen-normals (Phase-B
# analogue).  The 3×3 traction-row family: σ·n = t with
#
#   σ_xx=(λ+2μ)u_x+λv_y+λw_z,  σ_yy=λu_x+(λ+2μ)v_y+λw_z,  σ_zz=λu_x+λv_y+(λ+2μ)w_z,
#   σ_xy=μ(u_y+v_x),  σ_xz=μ(u_z+w_x),  σ_yz=μ(v_z+w_y),
#   t = (σ_xx n_x + σ_xy n_y + σ_xz n_z,  σ_xy n_x + σ_yy n_y + σ_yz n_z,
#        σ_xz n_x + σ_yz n_y + σ_zz n_z).
#
# Grouping each traction component by displacement component (u,v,w) and weight
# operator (W_dx,W_dy,W_dz) gives the per-(equation, column-block) coefficient
# triples encoded in `build_traction_layout_3d`.
# ============================================================================

"""
    TractionLayout3D

3D analogue of `TractionLayout` (M = 3 weight matrices: W_dx, W_dy, W_dz).
Each Neumann point contributes 3 equations (x,y,z) × 3 column-blocks (u,v,w) ×
its stencil.  Interleaved row ordering: `rows[3i-2]=g`, `rows[3i-1]=g+N`,
`rows[3i]=g+2N` for global node `g = neumann_ids[i]`.

Fields mirror `TractionLayout`:
- `rows` — global A row indices, length `3·n_neumann`.
- `weight_rows` — row in W_dx/W_dy/W_dz for each equation (= local Neumann index).
- `col_ptr` — offsets (length `3·n_neumann + 1`) into the entry arrays.
- `a_cols` — column of A for each entry.
- `w_cols` — column of W for each entry (≤ N).
- `coeffs` — flat `M·entries` array; entry `p`'s triple at `(p-1)*3 .+ (1:3)`,
  ordered `(c_dx, c_dy, c_dz)`.
- `b_vals` — RHS (traction components in row order).
- `M::Int` — 3.
"""
struct TractionLayout3D
    rows::Vector{Int}
    weight_rows::Vector{Int}
    col_ptr::Vector{Int}
    a_cols::Vector{Int}
    w_cols::Vector{Int}
    coeffs::Vector{Float64}
    b_vals::Vector{Float64}
    M::Int
end

# Coefficient triples (c_dx, c_dy, c_dz) for (equation eq ∈ 1:3, col-block cb ∈ 1:3).
# eq: 1=x, 2=y, 3=z.  cb: 1=u, 2=v, 3=w.  See the header derivation.
@inline function _traction_coeff_3d(eq::Int, cb::Int, nx, ny, nz, λ, μ)
    c1 = λ + 2μ
    if eq == 1                      # t_x
        cb == 1 && return (nx * c1, ny * μ,  nz * μ)     # u
        cb == 2 && return (ny * μ,  nx * λ,  0.0)        # v
        return            (nz * μ,  0.0,     nx * λ)     # w
    elseif eq == 2                  # t_y
        cb == 1 && return (ny * λ,  nx * μ,  0.0)        # u
        cb == 2 && return (nx * μ,  ny * c1, nz * μ)     # v
        return            (0.0,     nz * μ,  ny * λ)     # w
    else                            # t_z
        cb == 1 && return (nz * λ,  0.0,     nx * μ)     # u
        cb == 2 && return (0.0,     nz * λ,  ny * μ)     # v
        return            (nx * μ,  ny * μ,  nz * c1)    # w
    end
end

"""
    build_traction_layout_3d(neumann_ids, neumann_adjl, normals, tractions, λ, μ, N)

Build a `TractionLayout3D`.  `normals`/`tractions` are `Vector{SVector{3,Float64}}`
of length `n_neumann`.  Frozen-normals assumption: the normals are taken at the
reference configuration and treated as constants under design perturbations.
"""
function build_traction_layout_3d(
    neumann_ids::Vector{Int},
    neumann_adjl::Vector{Vector{Int}},
    normals::Vector{<:AbstractVector{Float64}},
    tractions::Vector{<:AbstractVector{Float64}},
    λ::Real,
    μ::Real,
    N::Int,
)
    n_neu = length(neumann_ids)
    @assert length(neumann_adjl) == n_neu
    @assert length(normals) == n_neu
    @assert length(tractions) == n_neu

    M = 3
    n_eq = 3 * n_neu
    total_entries = 0
    for s in neumann_adjl
        total_entries += 3 * 3 * length(s)   # 3 eqs × 3 col-blocks × stencil
    end

    rows        = Vector{Int}(undef, n_eq)
    weight_rows = Vector{Int}(undef, n_eq)
    col_ptr     = Vector{Int}(undef, n_eq + 1)
    a_cols      = Vector{Int}(undef, total_entries)
    w_cols      = Vector{Int}(undef, total_entries)
    coeffs      = Vector{Float64}(undef, M * total_entries)
    b_vals      = Vector{Float64}(undef, n_eq)

    ptr = 1
    for i_local in 1:n_neu
        g  = neumann_ids[i_local]
        nx, ny, nz = normals[i_local][1], normals[i_local][2], normals[i_local][3]
        s  = neumann_adjl[i_local]
        for eq in 1:3
            r = 3 * (i_local - 1) + eq
            rows[r]        = g + (eq - 1) * N
            weight_rows[r] = i_local
            b_vals[r]      = tractions[i_local][eq]
            col_ptr[r]     = ptr
            for j in eachindex(s)
                sj = s[j]
                for cb in 1:3
                    a_cols[ptr] = sj + (cb - 1) * N
                    w_cols[ptr] = sj
                    c_dx, c_dy, c_dz = _traction_coeff_3d(eq, cb, nx, ny, nz, λ, μ)
                    base = (ptr - 1) * M
                    coeffs[base + 1] = c_dx
                    coeffs[base + 2] = c_dy
                    coeffs[base + 3] = c_dz
                    ptr += 1
                end
            end
        end
    end
    col_ptr[n_eq + 1] = ptr

    return TractionLayout3D(rows, weight_rows, col_ptr, a_cols, w_cols, coeffs, b_vals, M)
end

"""
    apply_traction_3d!(A, b, layout, W_dx, W_dy, W_dz)

Overwrite the Neumann rows of `A` with `Σ_m coeff_m · W_m[weight_row, w_col]`
and set `b` to `layout.b_vals`.  Non-traced forward pass.
"""
function apply_traction_3d!(
    A::SparseMatrixCSC{<:Number, <:Integer},
    b::AbstractVector{<:Number},
    layout::TractionLayout3D,
    W_dx::SparseMatrixCSC{<:Number, <:Integer},
    W_dy::SparseMatrixCSC{<:Number, <:Integer},
    W_dz::SparseMatrixCSC{<:Number, <:Integer},
)
    M = layout.M
    @assert M == 3

    row_set = Set{Int}(layout.rows)
    for j in 1:size(A, 2)
        for idx in nzrange(A, j)
            A.rowval[idx] in row_set && (A.nzval[idx] = 0.0)
        end
    end

    for k in eachindex(layout.rows)
        g  = layout.rows[k]
        wr = layout.weight_rows[k]
        for p in layout.col_ptr[k]:(layout.col_ptr[k + 1] - 1)
            a_col = layout.a_cols[p]
            w_col = layout.w_cols[p]
            base = (p - 1) * M
            v = 0.0
            ix = _find_nzval(W_dx, wr, w_col); v += ix > 0 ? layout.coeffs[base + 1] * W_dx.nzval[ix] : 0.0
            iy = _find_nzval(W_dy, wr, w_col); v += iy > 0 ? layout.coeffs[base + 2] * W_dy.nzval[iy] : 0.0
            iz = _find_nzval(W_dz, wr, w_col); v += iz > 0 ? layout.coeffs[base + 3] * W_dz.nzval[iz] : 0.0
            A[g, a_col] = v
        end
        b[g] = layout.b_vals[k]
    end
    return nothing
end

"""
    extract_neumann_sensitivities_3d!(ΔW_dx, ΔW_dy, ΔW_dz, layout, η, u, active)

Step 3 — 3D Neumann contribution.  Distributes `ΔA[g, a_col] = -η[g]·u[a_col]`
(when `active[g]`) back into ΔW_dx/ΔW_dy/ΔW_dz via the coefficient transpose.
Accumulates (does not zero the ΔW first).
"""
function extract_neumann_sensitivities_3d!(
    ΔW_dx::SparseMatrixCSC{Float64, Int},
    ΔW_dy::SparseMatrixCSC{Float64, Int},
    ΔW_dz::SparseMatrixCSC{Float64, Int},
    layout::TractionLayout3D,
    η::AbstractVector{Float64},
    u::AbstractVector{Float64},
    active::AbstractVector{Bool},
)
    M = layout.M
    @assert M == 3
    for k in eachindex(layout.rows)
        g = layout.rows[k]
        active[g] || continue
        wr = layout.weight_rows[k]
        for p in layout.col_ptr[k]:(layout.col_ptr[k + 1] - 1)
            a_col = layout.a_cols[p]
            w_col = layout.w_cols[p]
            Δa = -η[g] * u[a_col]
            base = (p - 1) * M
            ix = _find_nzval(ΔW_dx, wr, w_col); ix > 0 && (ΔW_dx.nzval[ix] += layout.coeffs[base + 1] * Δa)
            iy = _find_nzval(ΔW_dy, wr, w_col); iy > 0 && (ΔW_dy.nzval[iy] += layout.coeffs[base + 2] * Δa)
            iz = _find_nzval(ΔW_dz, wr, w_col); iz > 0 && (ΔW_dz.nzval[iz] += layout.coeffs[base + 3] * Δa)
        end
    end
    return nothing
end

# ============================================================================
# Phase D L2 (3D): differentiable triangle-based vertex normals
# ============================================================================
# The 3D analogue of 2D's `polyline_normals`/`NormalJacobian`.  A boundary
# vertex normal is the area-weighted average of the normals of its incident
# triangles (the 1-ring of the surface mesh):
#
#     g_i = Σ_{f ∋ i} (p_b - p_a) × (p_c - p_a),   N_i = g_i / ‖g_i‖
#
# Each triangle's un-normalized cross product has magnitude 2·area, so the sum
# is exactly the area weighting.  The face connectivity is FIXED across design
# perturbations (the discrete-adjoint contract); only the vertex coordinates
# move, so `∂N_i/∂pts` is sparse on the 1-ring vertices.

"""
    NormalJacobian3D

Sparse Jacobian of an area-weighted vertex normal `N_i` at a Neumann vertex
w.r.t. the vertices of the triangles incident to it (its 1-ring, including
itself).  See the section header for the normal definition.  Mirrors the 2D
`NormalJacobian`, but the column set is variable-length (the 1-ring) instead
of a fixed `(prev, next)` pair.

Fields:
- `cols::Vector{Int}` — global indices (1..N) of the 1-ring vertices.
- `blocks::Vector{SMatrix{3,3,Float64,9}}` — `blocks[k] = ∂N_i/∂p_{cols[k]}`.
"""
struct NormalJacobian3D
    cols::Vector{Int}
    blocks::Vector{SMatrix{3,3,Float64,9}}
end

# skew(v)·w = v × w  (column-major SMatrix literal)
@inline _skew3(v) = SMatrix{3,3,Float64,9}(
    0.0,   v[3], -v[2],
   -v[3],  0.0,   v[1],
    v[2], -v[1],  0.0,
)

"""
    triangle_normals(pts_flat, faces, neumann_ids, vertex_faces)
        -> (normals::Vector{SVector{3,Float64}},
            jacobians::Vector{NormalJacobian3D})

3D analogue of `polyline_normals`.  Compute area-weighted vertex normals and
their sparse Jacobians at the Neumann vertices of a fixed triangle surface mesh.

Arguments:
- `pts_flat` — flat coordinate vector `[x1,y1,z1, x2,y2,z2, …]` (length `3N`).
- `faces` — triangle connectivity, each a `(a,b,c)` triple of global vertex
  indices (1..N) wound CCW as seen from OUTSIDE (so `(p_b-p_a)×(p_c-p_a)` points
  outward).
- `neumann_ids` — global indices of the Neumann vertices (length `n_neu`).
- `vertex_faces[i_local]` — indices into `faces` of the triangles incident to
  Neumann vertex `neumann_ids[i_local]`.

Returns `(normals, jacobians)`, each of length `n_neu`.

Per-triangle derivatives (`m = (p_b-p_a)×(p_c-p_a)`):

    ∂m/∂p_a = skew(p_c - p_b),  ∂m/∂p_b = skew(p_a - p_c),  ∂m/∂p_c = skew(p_b - p_a)

(`skew(v)·w = v×w`; their sum is zero ⇒ translation invariance).  Then
`g = Σ_f m_f`, `N = g/‖g‖`, and the normalization pullback
`∂N/∂g = (I − N Nᵀ)/‖g‖` gives `∂N/∂p_j = (∂N/∂g)·(∂g/∂p_j)`.
"""
function triangle_normals(
    pts_flat::AbstractVector{Float64},
    faces::AbstractVector{<:NTuple{3,Int}},
    neumann_ids::AbstractVector{Int},
    vertex_faces::AbstractVector{<:AbstractVector{Int}},
)
    n_neu = length(neumann_ids)
    @assert length(vertex_faces) == n_neu
    P(i) = SVector{3,Float64}(pts_flat[3i - 2], pts_flat[3i - 1], pts_flat[3i])

    normals   = Vector{SVector{3,Float64}}(undef, n_neu)
    jacobians = Vector{NormalJacobian3D}(undef, n_neu)

    for i_local in 1:n_neu
        g  = zero(SVector{3,Float64})
        dg = Dict{Int, MMatrix{3,3,Float64,9}}()
        acc!(j, B) = (haskey(dg, j) ? (dg[j] .+= B) :
                      (dg[j] = MMatrix{3,3,Float64,9}(B)))
        for fidx in vertex_faces[i_local]
            a, b, c = faces[fidx]
            pa, pb, pc = P(a), P(b), P(c)
            g += cross(pb - pa, pc - pa)
            acc!(a, _skew3(pc - pb))
            acc!(b, _skew3(pa - pc))
            acc!(c, _skew3(pb - pa))
        end
        gn = norm(g)
        gn > 0.0 || error("triangle_normals: degenerate normal at Neumann $i_local")
        N  = g / gn
        Pn = (SMatrix{3,3,Float64,9}(I) - N * N') / gn      # ∂N/∂g
        cols   = collect(keys(dg))
        blocks = Vector{SMatrix{3,3,Float64,9}}(undef, length(cols))
        for (k, j) in enumerate(cols)
            blocks[k] = Pn * SMatrix{3,3,Float64,9}(dg[j])
        end
        normals[i_local]   = N
        jacobians[i_local] = NormalJacobian3D(cols, blocks)
    end
    return normals, jacobians
end

"""
    update_traction_coeffs_3d!(layout::TractionLayout3D, normals, λ, μ)

Rebuild `layout.coeffs` in place from a fresh set of per-Neumann-point normals.
Mirrors the coefficient assignment in `build_traction_layout_3d` exactly (same
`(eq, stencil, cb)` entry ordering) but touches no other field.  Use each
iteration when normals are live (Phase D Level 2, 3D).  Analogue of the 2D
`update_traction_coeffs!`.
"""
function update_traction_coeffs_3d!(
    layout::TractionLayout3D,
    normals::AbstractVector{<:AbstractVector{Float64}},
    λ::Real,
    μ::Real,
)
    M = layout.M
    @assert M == 3
    n_neu = length(layout.rows) ÷ 3
    @assert length(normals) == n_neu

    @inbounds for i_local in 1:n_neu
        nx, ny, nz = normals[i_local][1], normals[i_local][2], normals[i_local][3]
        for eq in 1:3
            r    = 3 * (i_local - 1) + eq
            ptr0 = layout.col_ptr[r]
            ptr1 = layout.col_ptr[r + 1] - 1
            for p in ptr0:ptr1
                cb = ((p - ptr0) % 3) + 1
                c_dx, c_dy, c_dz = _traction_coeff_3d(eq, cb, nx, ny, nz, λ, μ)
                base = (p - 1) * M
                layout.coeffs[base + 1] = c_dx
                layout.coeffs[base + 2] = c_dy
                layout.coeffs[base + 3] = c_dz
            end
        end
    end
    return nothing
end

"""
    extract_normal_sensitivities_3d!(
        Δpts, layout, η, u, W_dx, W_dy, W_dz, normal_jacobians, λ, μ;
        active = nothing,
    )

Step 3 (n-side, 3D) — accumulate `-ηᵀ·(∂A/∂n · ∂n/∂pts)·u` into `Δpts`.
3D analogue of `extract_normal_sensitivities!`.

For each Neumann vertex, all three equation rows (x,y,z) share the same normal
`n_i`.  We form `Sₙ = ∂L/∂n_i` (a 3-vector) by contracting `Δa = -η[g]·u[a_col]`
against the analytic `∂coeff/∂n` of each Neumann-row entry (the traction
coefficients are linear in `n`, so `∂coeff/∂n_d = _traction_coeff_3d(eq, cb, e_d, …)`,
derived from the same routine the forward uses — never hardcoded), then push it
through the sparse Jacobian: `Δp_j += (∂N_i/∂p_j)ᵀ · Sₙ`.

`active`, if provided, is the 3N per-DOF mask (Dirichlet rows ⇒ false).
"""
function extract_normal_sensitivities_3d!(
    Δpts::AbstractVector{Float64},
    layout::TractionLayout3D,
    η::AbstractVector{Float64},
    u::AbstractVector{Float64},
    W_dx::SparseMatrixCSC{Float64, Int},
    W_dy::SparseMatrixCSC{Float64, Int},
    W_dz::SparseMatrixCSC{Float64, Int},
    normal_jacobians::AbstractVector{NormalJacobian3D},
    λ::Real,
    μ::Real;
    active::Union{Nothing, AbstractVector{Bool}} = nothing,
)
    M = layout.M
    @assert M == 3
    n_neu = length(normal_jacobians)
    @assert length(layout.rows) == 3 * n_neu

    @inbounds for i_local in 1:n_neu
        wr  = layout.weight_rows[3 * (i_local - 1) + 1]   # = i_local
        Sn1 = 0.0; Sn2 = 0.0; Sn3 = 0.0
        any_active = false

        for eq in 1:3
            r = 3 * (i_local - 1) + eq
            g = layout.rows[r]
            (active === nothing || active[g]) || continue
            any_active = true
            ηg = η[g]

            # ∂coeff/∂n_d for this (eq, cb) — coeff is linear in n, so the
            # derivative is the coeff evaluated at the unit normal e_d.
            ptr0 = layout.col_ptr[r]
            ptr1 = layout.col_ptr[r + 1] - 1
            # Cache the 3 (cb) × 3 (d) derivative triples once per row.
            for p in ptr0:ptr1
                cb    = ((p - ptr0) % 3) + 1
                a_col = layout.a_cols[p]
                w_col = layout.w_cols[p]
                ix = _find_nzval(W_dx, wr, w_col); wx = ix > 0 ? W_dx.nzval[ix] : 0.0
                iy = _find_nzval(W_dy, wr, w_col); wy = iy > 0 ? W_dy.nzval[iy] : 0.0
                iz = _find_nzval(W_dz, wr, w_col); wz = iz > 0 ? W_dz.nzval[iz] : 0.0
                up = u[a_col]
                base_contrib = -ηg * up

                cx1, cy1, cz1 = _traction_coeff_3d(eq, cb, 1.0, 0.0, 0.0, λ, μ)
                cx2, cy2, cz2 = _traction_coeff_3d(eq, cb, 0.0, 1.0, 0.0, λ, μ)
                cx3, cy3, cz3 = _traction_coeff_3d(eq, cb, 0.0, 0.0, 1.0, λ, μ)
                Sn1 += base_contrib * (cx1 * wx + cy1 * wy + cz1 * wz)
                Sn2 += base_contrib * (cx2 * wx + cy2 * wy + cz2 * wz)
                Sn3 += base_contrib * (cx3 * wx + cy3 * wy + cz3 * wz)
            end
        end
        any_active || continue

        nj = normal_jacobians[i_local]
        for k in eachindex(nj.cols)
            j = nj.cols[k]
            B = nj.blocks[k]                     # ∂N_i/∂p_j  (3×3)
            Δpts[3j - 2] += B[1, 1] * Sn1 + B[2, 1] * Sn2 + B[3, 1] * Sn3
            Δpts[3j - 1] += B[1, 2] * Sn1 + B[2, 2] * Sn2 + B[3, 2] * Sn3
            Δpts[3j]     += B[1, 3] * Sn1 + B[2, 3] * Sn2 + B[3, 3] * Sn3
        end
    end
    return nothing
end

# ============================================================================
# shape_gradient_3d — interior + optional Neumann (frozen normals).
# ============================================================================

"""
    shape_gradient_3d(
        pts_flat, model::LinearElasticity3D, N, adjl, basis, active,
        dirichlet_dofs, dirichlet_vals, ∂L_∂u;
        interior_rows = nothing,
    ) -> (u, Δpts)

Manual adjoint for 3D linear elasticity.  Dirichlet-only by default; mixed
Dirichlet + traction when the Neumann kwargs are supplied (frozen normals,
the Phase-B analogue).

Implements the same five-step pipeline as the 2D `shape_gradient`:

1. Forward solve: build the 6 second-derivative weight matrices (+ 3 first-
   derivative matrices on the Neumann nodes when traction is present), assemble
   the 3N×3N system, apply Dirichlet + traction BCs, LU-factorize, solve for u.
2. Adjoint solve: Aᵀ η = ∂L/∂u via the cached factorization.
3. Extract ΔW: per-operator sensitivities from ΔA = -η·uᵀ (interior 3D
   coefficient structure + Neumann contribution).
4. Backward through RBF stencils: `_pullback_weights!` for each operator.
5. (External b-side / normal contributions — added by caller; same as 2D.)

Arguments (same convention as the 2D `shape_gradient`):
- `pts_flat::Vector{Float64}` — flat coordinates [x1,y1,z1, x2,y2,z2, …] (length 3N).
- `model::LinearElasticity3D` — 3D elasticity model.
- `N::Int` — number of points.
- `adjl` — adjacency list.
- `basis` — RBF basis.
- `active::AbstractVector{Bool}` — per-DOF mask (length 3N).
- `dirichlet_dofs::Vector{Int}` — Dirichlet row indices (in 1..3N).
- `dirichlet_vals::Vector{Float64}` — prescribed values at Dirichlet rows.
- `∂L_∂u::Function` — `u -> ∂L/∂u` (length 3N).

Keywords:
- `interior_rows::BitVector` — length N, true where the interior assembly
  applies (not overwritten by BC rows).
- `traction_layout::TractionLayout3D`, `neumann_ids::Vector{Int}`,
  `neumann_adjl::Vector{Vector{Int}}` — supply all three for mixed BC.
- `rigid_modes::AbstractMatrix{Float64}` — `3N×m` rigid-body modes for a
  PURE-traction problem (no Dirichlet pins).  When supplied, the singular
  operator is regularized by a bordered Lagrange system `[A R; Rᵀ 0]` that
  fixes the gauge SYMMETRICALLY (`Rᵀu = 0`) instead of pinning asymmetric
  points — so no spurious point-reaction stress perturbs the field.  **R must
  be FROZEN at the anchor geometry** (built once per remesh, constant within a
  morph interval), exactly like the boundary connectivity/normals; then
  `∂R/∂pts = 0` and the adjoint extraction below is unchanged (the bordered
  multipliers drop out, `η = η̄[1:3N]`).  Use `active = trues(3N)`,
  `dirichlet_dofs = Int[]`.  Default `nothing` ⇒ the Dirichlet-pin path.

Returns `(u, Δpts, η)`.
"""
function shape_gradient_3d(
    pts_flat::Vector{Float64},
    model::LinearElasticity3D,
    N::Int,
    adjl,
    basis,
    active::AbstractVector{Bool},
    dirichlet_dofs::Vector{Int},
    dirichlet_vals::Vector{Float64},
    ∂L_∂u::Function;
    interior_rows::Union{Nothing, BitVector} = nothing,
    traction_layout::Union{Nothing, TractionLayout3D} = nothing,
    neumann_ids::Union{Nothing, Vector{Int}} = nothing,
    neumann_adjl::Union{Nothing, Vector{Vector{Int}}} = nothing,
    normal_jacobians::Union{Nothing, AbstractVector{NormalJacobian3D}} = nothing,
    rigid_modes::Union{Nothing, AbstractMatrix{Float64}} = nothing,
)
    has_neumann = traction_layout !== nothing
    μ, λ = lame_parameters_3d(model)

    # --- Step 1: Forward ------------------------------------------------------
    pts = _pts_from_flat_3d(pts_flat)

    W_d2x,  cache_d2x  = _build_weights_and_cache(Partial(2, 1),      pts, pts, adjl, basis)
    W_d2y,  cache_d2y  = _build_weights_and_cache(Partial(2, 2),      pts, pts, adjl, basis)
    W_d2z,  cache_d2z  = _build_weights_and_cache(Partial(2, 3),      pts, pts, adjl, basis)
    W_d2xy, cache_d2xy = _build_weights_and_cache(MixedPartial(1, 2), pts, pts, adjl, basis)
    W_d2xz, cache_d2xz = _build_weights_and_cache(MixedPartial(1, 3), pts, pts, adjl, basis)
    W_d2yz, cache_d2yz = _build_weights_and_cache(MixedPartial(2, 3), pts, pts, adjl, basis)

    if has_neumann
        neumann_pts = pts[neumann_ids]
        W_dx, cache_dx = _build_weights_and_cache(Partial(1, 1), pts, neumann_pts, neumann_adjl, basis)
        W_dy, cache_dy = _build_weights_and_cache(Partial(1, 2), pts, neumann_pts, neumann_adjl, basis)
        W_dz, cache_dz = _build_weights_and_cache(Partial(1, 3), pts, neumann_pts, neumann_adjl, basis)
    end

    A = assemble_elasticity_3d_from_weights(
        W_d2x, W_d2y, W_d2z, W_d2xy, W_d2xz, W_d2yz, N, λ, μ,
    )
    b = zeros(3N)
    apply_dirichlet!(A, b, dirichlet_dofs, dirichlet_vals)
    if has_neumann
        apply_traction_3d!(A, b, traction_layout, W_dx, W_dy, W_dz)
    end

    if rigid_modes === nothing
        F = lu(A)
        u = F \ b

        # --- Step 2: Adjoint -------------------------------------------------
        η = F' \ ∂L_∂u(u)
    else
        # Bordered Lagrange system [A R; Rᵀ 0] — symmetric gauge fix (Rᵀu = 0)
        # for the pure-traction operator.  R frozen ⇒ ∂M/∂pts = [∂A/∂pts 0; 0 0],
        # so the multiplier block drops out of the gradient and only η[1:3N] feeds
        # the unchanged extraction below.
        nr = size(rigid_modes, 2)
        Rs = sparse(rigid_modes)
        M  = [A Rs; permutedims(Rs) spzeros(nr, nr)]
        Fm = lu(M)
        ub = Fm \ vcat(b, zeros(nr))
        u  = ub[1:3N]
        ηb = Fm' \ vcat(∂L_∂u(u), zeros(nr))
        η  = ηb[1:3N]
    end

    # --- Step 3: Extract ΔW --------------------------------------------------
    ΔW_d2x, ΔW_d2y, ΔW_d2z, ΔW_d2xy, ΔW_d2xz, ΔW_d2yz =
        allocate_weight_gradients(W_d2x, W_d2x, W_d2x, W_d2x, W_d2x, W_d2x)
    extract_weight_sensitivities_elasticity_3d!(
        ΔW_d2x, ΔW_d2y, ΔW_d2z, ΔW_d2xy, ΔW_d2xz, ΔW_d2yz,
        W_d2x, η, u, active, λ, μ;
        interior_rows = interior_rows,
    )
    if has_neumann
        ΔW_dx, ΔW_dy, ΔW_dz = allocate_weight_gradients(W_dx, W_dy, W_dz)
        extract_neumann_sensitivities_3d!(ΔW_dx, ΔW_dy, ΔW_dz, traction_layout, η, u, active)
    end

    # --- Step 4: Backward through RBF stencils (dim inferred from pts) -------
    Δpts = zeros(Float64, 3N)
    _propagate_weight_gradient!(Δpts, ΔW_d2x,  W_d2x,  cache_d2x,  pts, pts, adjl, basis, Partial(2, 1))
    _propagate_weight_gradient!(Δpts, ΔW_d2y,  W_d2y,  cache_d2y,  pts, pts, adjl, basis, Partial(2, 2))
    _propagate_weight_gradient!(Δpts, ΔW_d2z,  W_d2z,  cache_d2z,  pts, pts, adjl, basis, Partial(2, 3))
    _propagate_weight_gradient!(Δpts, ΔW_d2xy, W_d2xy, cache_d2xy, pts, pts, adjl, basis, MixedPartial(1, 2))
    _propagate_weight_gradient!(Δpts, ΔW_d2xz, W_d2xz, cache_d2xz, pts, pts, adjl, basis, MixedPartial(1, 3))
    _propagate_weight_gradient!(Δpts, ΔW_d2yz, W_d2yz, cache_d2yz, pts, pts, adjl, basis, MixedPartial(2, 3))
    if has_neumann
        _propagate_weight_gradient!(Δpts, ΔW_dx, W_dx, cache_dx,
                                    pts, neumann_pts, neumann_adjl, basis, Partial(1, 1); eval_offset = neumann_ids)
        _propagate_weight_gradient!(Δpts, ΔW_dy, W_dy, cache_dy,
                                    pts, neumann_pts, neumann_adjl, basis, Partial(1, 2); eval_offset = neumann_ids)
        _propagate_weight_gradient!(Δpts, ΔW_dz, W_dz, cache_dz,
                                    pts, neumann_pts, neumann_adjl, basis, Partial(1, 3); eval_offset = neumann_ids)
    end

    # --- Step 4c: Phase D L2 (3D) — differentiable normals -------------------
    # -ηᵀ·(∂A/∂n · ∂n/∂pts)·u, distributed via the sparse NormalJacobian3Ds.
    # No-op unless the caller supplied normal Jacobians (and refreshed the
    # layout's coeffs to the live normals via `update_traction_coeffs_3d!`).
    if has_neumann && normal_jacobians !== nothing
        extract_normal_sensitivities_3d!(Δpts, traction_layout, η, u,
                                         W_dx, W_dy, W_dz, normal_jacobians, λ, μ;
                                         active = active)
    end

    return (u = u, Δpts = Δpts, η = η)
end

# ============================================================================
# 3D helpers
# ============================================================================

"""
    _pts_from_flat_3d(p::AbstractVector{T}) -> Vector{SVector{3,T}}

Reshape flat [x1,y1,z1, x2,y2,z2, …] → [SVector(x1,y1,z1), …].
"""
function _pts_from_flat_3d(p::AbstractVector{T}) where {T<:Real}
    N = length(p) ÷ 3
    return [SVector{3, T}(p[3i - 2], p[3i - 1], p[3i]) for i in 1:N]
end

"""
    rigid_body_modes_3d(pts) -> Matrix{Float64}  (3N × 6)

The six rigid-body modes (3 translations + 3 rotations) of a 3D body, in the
BLOCK DOF ordering `[x₁..x_N, y₁..y_N, z₁..z_N]` the assembled operator uses
(DOF of node `i`, component `d∈{0,1,2}` is `i + d·N`).  Rotations are taken
about the centroid and each column is unit-normalized for conditioning.

This is the `R` block of the bordered Lagrange system in `shape_gradient_3d`,
which fixes the pure-traction gauge symmetrically (`Rᵀu = 0`) in place of
asymmetric 3-2-1 point pins.  **Build ONCE per anchor and hold it FROZEN within
a morph interval** (same contract as boundary connectivity/normals): a frozen R
has `∂R/∂pts = 0`, keeping the discrete shape gradient exact.
"""
function rigid_body_modes_3d(pts::AbstractVector{<:AbstractVector{<:Real}})
    N = length(pts)
    R = zeros(Float64, 3N, 6)
    c = sum(pts) / N
    for i in 1:N
        x, y, z = pts[i][1] - c[1], pts[i][2] - c[2], pts[i][3] - c[3]
        R[i,      1] = 1.0                          # Tx
        R[i + N,  2] = 1.0                          # Ty
        R[i + 2N, 3] = 1.0                          # Tz
        R[i + N,  4] = -z;  R[i + 2N, 4] =  y       # Rx: (0,-z, y)
        R[i,      5] =  z;  R[i + 2N, 5] = -x       # Ry: (z, 0,-x)
        R[i,      6] = -y;  R[i + N,  6] =  x       # Rz: (-y,x, 0)
    end
    for k in 1:6
        R[:, k] ./= norm(@view R[:, k])
    end
    return R
end

# Weight-gradient propagation (Step 4) uses `_propagate_weight_gradient!` from
# manual_adjoint.jl, which infers the spatial dimension from the points — no 3D
# duplicate needed.
