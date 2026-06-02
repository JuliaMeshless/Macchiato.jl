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

    F = lu(A)
    u = F \ b

    # --- Step 2: Adjoint -----------------------------------------------------
    η = F' \ ∂L_∂u(u)

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

# Weight-gradient propagation (Step 4) uses `_propagate_weight_gradient!` from
# manual_adjoint.jl, which infers the spatial dimension from the points — no 3D
# duplicate needed.
