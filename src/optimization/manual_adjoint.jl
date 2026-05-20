import RadialBasisFunctions: _build_weights, Partial, MixedPartial,
    _build_weights_and_cache, _pullback_weights!

"""
    _pts_from_flat(p)

Reshape a flat coordinate vector `[x1, y1, x2, y2, …]` of length `2N` into a
`Vector{SVector{2, eltype(p)}}` of length `N`. Inverse of
`vcat([collect(pt) for pt in pts]...)`.

No longer a Mooncake primitive — the manual adjoint Step 4 now uses
`_pullback_weights!` from RadialBasisFunctions.jl directly, avoiding
`build_rrule` entirely.
"""
function _pts_from_flat(p::AbstractVector{T}) where {T<:Real}
    N = length(p) ÷ 2
    return [SVector{2, T}(p[2i - 1], p[2i]) for i in 1:N]
end

"""
    extract_weight_sensitivities_elasticity!(
        ΔW_d2x, ΔW_d2y, ΔW_d2xy,
        W_template, η, u, active, λstar, μ;
        interior_rows = nothing,
    )

Step 3 of the manual adjoint for 2D linear elasticity. Given the global adjoint
vector `η = Aᵀ \\ ∂L/∂u` and forward solution `u`, populate the gradients of
the three weight matrices that assemble A:

    A11 = c1·W_d2x + c2·W_d2y     (c1 = λstar + 2μ, c2 = μ)
    A12 = c3·W_d2xy               (c3 = λstar + μ)
    A22 = c2·W_d2x + c1·W_d2y

The IFT gives `ΔA = -η · uᵀ` at active rows. Per-DOF gating (not per-point AND
— see plan_manual_adjoint.md §Corrections §1):

    ΔA11[i,j] = active[i]   ? -η[i]   · u[j]   : 0
    ΔA22[i,j] = active[i+N] ? -η[i+N] · u[j+N] : 0
    ΔA12[i,j] = active[i]   ? -η[i]   · u[j+N] : 0
    ΔA21[i,j] = active[i+N] ? -η[i+N] · u[j]   : 0

`W_template` provides the shared sparsity pattern; the three ΔW outputs must
be pre-allocated with that exact pattern (see `allocate_weight_gradients`).

`interior_rows` is an optional `BitVector` of length N to exclude Neumann rows
from the main extraction loop (Phase B). Default `nothing` ⇒ visit all rows.
"""
function extract_weight_sensitivities_elasticity!(
    ΔW_d2x::SparseMatrixCSC{Float64, Int},
    ΔW_d2y::SparseMatrixCSC{Float64, Int},
    ΔW_d2xy::SparseMatrixCSC{Float64, Int},
    W_template::SparseMatrixCSC{Float64, Int},
    η::AbstractVector{Float64},
    u::AbstractVector{Float64},
    active::AbstractVector{Bool},
    λstar::Real,
    μ::Real;
    interior_rows::Union{Nothing, BitVector} = nothing,
)
    N = size(W_template, 1)
    @assert length(η) == 2N
    @assert length(u) == 2N
    @assert length(active) == 2N

    c1 = λstar + 2μ
    c2 = μ
    c3 = λstar + μ

    fill!(ΔW_d2x.nzval, 0.0)
    fill!(ΔW_d2y.nzval, 0.0)
    fill!(ΔW_d2xy.nzval, 0.0)

    @inbounds for j in 1:N
        for idx in nzrange(W_template, j)
            i = W_template.rowval[idx]

            if interior_rows !== nothing && !interior_rows[i]
                continue
            end

            t11 = active[i]      ? -η[i]     * u[j]     : 0.0
            t22 = active[i + N]  ? -η[i + N] * u[j + N] : 0.0
            t12 = active[i]      ? -η[i]     * u[j + N] : 0.0
            t21 = active[i + N]  ? -η[i + N] * u[j]     : 0.0

            ΔW_d2x.nzval[idx]  = c1 * t11 + c2 * t22
            ΔW_d2y.nzval[idx]  = c2 * t11 + c1 * t22
            ΔW_d2xy.nzval[idx] = c3 * (t12 + t21)
        end
    end

    return nothing
end

"""
    allocate_weight_gradients(templates...)

Allocate zero `SparseMatrixCSC` matrices sharing the sparsity pattern of each
template. Returns a tuple of the same length as the input.
"""
function allocate_weight_gradients(templates::SparseMatrixCSC{Float64,Int}...)
    return map(templates) do W
        SparseMatrixCSC(size(W, 1), size(W, 2),
                        copy(W.colptr), copy(W.rowval),
                        zeros(length(W.nzval)))
    end
end

"""
    assemble_elasticity_from_weights(W_d2x, W_d2y, W_d2xy, N, λstar, μ)

Build the 2N×2N plane-stress Navier system from pre-computed weight matrices.
Mirrors `make_system_differentiable` but takes the already-evaluated `W`
matrices, so the forward pass of the manual adjoint can reuse them in Step 3
without rebuilding. Outside Mooncake's trace ⇒ we can use sparse-sparse `+`
and `hvcat` freely.
"""
function assemble_elasticity_from_weights(
    W_d2x::SparseMatrixCSC{Float64, Int},
    W_d2y::SparseMatrixCSC{Float64, Int},
    W_d2xy::SparseMatrixCSC{Float64, Int},
    N::Int,
    λstar::Real,
    μ::Real,
)
    c1 = λstar + 2μ
    c2 = μ
    c3 = λstar + μ

    A11 = c1 * W_d2x + c2 * W_d2y
    A12 = c3 * W_d2xy
    A22 = c2 * W_d2x + c1 * W_d2y

    return [A11 A12; A12 A22]
end

# ============================================================================
# Phase B: Neumann (traction) BCs for 2D linear elasticity
# ============================================================================

"""
    TractionLayout

Pre-computed coefficient bundle for traction (Neumann) BC assembly and its
adjoint. All vectors use **interleaved row ordering** `[x_1, y_1, x_2, y_2, …,
x_n, y_n]`, so `col_ptr[k]` and `rows[k]` always refer to the same equation.

Fields:
- `rows::Vector{Int}` — global A row indices, length `2·n_neumann`,
  `rows[2i-1] = neumann_ids[i]`, `rows[2i] = neumann_ids[i] + N`.
- `weight_rows::Vector{Int}` — row in `W_dx`/`W_dy` for each equation,
  `weight_rows[2i-1] = weight_rows[2i] = i` (local Neumann index).
- `col_ptr::Vector{Int}` — offsets (length `2·n_neumann + 1`); entries
  `col_ptr[k]:(col_ptr[k+1]-1)` belong to equation `k`.
- `a_cols::Vector{Int}` — column of A for each entry.
- `w_cols::Vector{Int}` — column of W for each entry (≤ N).
- `coeffs::Vector{Float64}` — flat `M·E` array, interleaved per entry: for
  entry `p`, the `M` coefficients are at indices `(p-1)·M + (1:M)`.
- `b_vals::Vector{Float64}` — RHS value at each equation (= traction
  components in order).
- `M::Int` — number of weight matrices contributing (= 2 for plane stress).
"""
struct TractionLayout
    rows::Vector{Int}
    weight_rows::Vector{Int}
    col_ptr::Vector{Int}
    a_cols::Vector{Int}
    w_cols::Vector{Int}
    coeffs::Vector{Float64}
    b_vals::Vector{Float64}
    M::Int
end

"""
    build_traction_layout(neumann_ids, neumann_adjl, normals, tractions,
                          λstar, μ, N)

Build a `TractionLayout` for 2D linear elasticity traction BCs. `normals` and
`tractions` are `Vector{SVector{2, Float64}}` of length `n_neumann`.

For each Neumann point `i_local` (global `i_global = neumann_ids[i_local]`)
with outward normal `n = (nx, ny)`, traction `t = (tx, ty)`, and stencil
`s = neumann_adjl[i_local]`:

  Row x-eq (global `i_global`):
    For each stencil entry j ∈ s:
      A[i_global, j  ] = nx·(λ+2μ)·W_dx[i_local,j] + ny·μ·W_dy[i_local,j]
      A[i_global, j+N] = ny·μ·W_dx[i_local,j] + nx·λ·W_dy[i_local,j]
    b[i_global] = tx

  Row y-eq (global `i_global + N`):
    For each stencil entry j ∈ s:
      A[i_global+N, j  ] = ny·λ·W_dx[i_local,j] + nx·μ·W_dy[i_local,j]
      A[i_global+N, j+N] = nx·μ·W_dx[i_local,j] + ny·(λ+2μ)·W_dy[i_local,j]
    b[i_global+N] = ty

Frozen-normals assumption (plan Level 1): the normal vectors are taken at the
reference configuration and treated as constants under design perturbations.
"""
function build_traction_layout(
    neumann_ids::Vector{Int},
    neumann_adjl::Vector{Vector{Int}},
    normals::Vector{<:AbstractVector{Float64}},
    tractions::Vector{<:AbstractVector{Float64}},
    λstar::Real,
    μ::Real,
    N::Int,
)
    n_neu = length(neumann_ids)
    @assert length(neumann_adjl) == n_neu
    @assert length(normals) == n_neu
    @assert length(tractions) == n_neu

    M = 2  # W_dx, W_dy
    n_eq = 2 * n_neu
    total_entries = sum(2 * length(s) for s in neumann_adjl)  # 2 cols per stencil entry × 2 eqns / 2 = …

    # 2 equations × stencil-size × 2 column-blocks (u,v)
    total_entries = 0
    for s in neumann_adjl
        total_entries += 2 * 2 * length(s)   # 2 eqs (x,y) × 2 col-blocks × k
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
        nx = normals[i_local][1]
        ny = normals[i_local][2]
        tx = tractions[i_local][1]
        ty = tractions[i_local][2]
        s  = neumann_adjl[i_local]

        # x-equation (interleaved row 2i-1)
        rx = 2 * i_local - 1
        rows[rx]        = g
        weight_rows[rx] = i_local
        b_vals[rx]      = tx
        col_ptr[rx]     = ptr
        for j in eachindex(s)
            sj = s[j]
            # u column
            a_cols[ptr] = sj
            w_cols[ptr] = sj
            coeffs[(ptr - 1) * M + 1] = nx * (λstar + 2μ)
            coeffs[(ptr - 1) * M + 2] = ny * μ
            ptr += 1
            # v column
            a_cols[ptr] = sj + N
            w_cols[ptr] = sj
            coeffs[(ptr - 1) * M + 1] = ny * μ
            coeffs[(ptr - 1) * M + 2] = nx * λstar
            ptr += 1
        end

        # y-equation (interleaved row 2i)
        ry = 2 * i_local
        rows[ry]        = g + N
        weight_rows[ry] = i_local
        b_vals[ry]      = ty
        col_ptr[ry]     = ptr
        for j in eachindex(s)
            sj = s[j]
            a_cols[ptr] = sj
            w_cols[ptr] = sj
            coeffs[(ptr - 1) * M + 1] = ny * λstar
            coeffs[(ptr - 1) * M + 2] = nx * μ
            ptr += 1
            a_cols[ptr] = sj + N
            w_cols[ptr] = sj
            coeffs[(ptr - 1) * M + 1] = nx * μ
            coeffs[(ptr - 1) * M + 2] = ny * (λstar + 2μ)
            ptr += 1
        end
    end
    col_ptr[n_eq + 1] = ptr

    return TractionLayout(rows, weight_rows, col_ptr, a_cols, w_cols, coeffs,
                          b_vals, M)
end

"""
    apply_traction!(A, b, layout, W_dx, W_dy)

Apply traction BC assembly to the system. Overwrites rows `layout.rows` of `A`
with the linear combination `Σ_m coeff_m · W_m[weight_row, w_col]`, and sets
the corresponding entries of `b` to `layout.b_vals`. Non-traced — Phase B's
forward pass runs this outside any Mooncake context.
"""
function apply_traction!(
    A::SparseMatrixCSC{Float64, Int},
    b::Vector{Float64},
    layout::TractionLayout,
    W_dx::SparseMatrixCSC{Float64, Int},
    W_dy::SparseMatrixCSC{Float64, Int},
)
    M = layout.M
    @assert M == 2 "apply_traction! assumes (W_dx, W_dy) only"

    # Zero the rows we are about to overwrite
    row_set = Set{Int}(layout.rows)
    for j in 1:size(A, 2)
        for idx in nzrange(A, j)
            if A.rowval[idx] in row_set
                A.nzval[idx] = 0.0
            end
        end
    end

    for k in eachindex(layout.rows)
        g  = layout.rows[k]
        wr = layout.weight_rows[k]
        for p in layout.col_ptr[k]:(layout.col_ptr[k + 1] - 1)
            a_col = layout.a_cols[p]
            w_col = layout.w_cols[p]
            c1 = layout.coeffs[(p - 1) * M + 1]
            c2 = layout.coeffs[(p - 1) * M + 2]
            v = 0.0
            ix = _find_nzval(W_dx, wr, w_col); v += ix > 0 ? c1 * W_dx.nzval[ix] : 0.0
            iy = _find_nzval(W_dy, wr, w_col); v += iy > 0 ? c2 * W_dy.nzval[iy] : 0.0
            A[g, a_col] = v
        end
        b[g] = layout.b_vals[k]
    end
    return nothing
end

"""
    extract_neumann_sensitivities!(ΔW_dx, ΔW_dy, layout, η, u, active)

Step 3 — Neumann contribution. Given the global adjoint `η` and forward `u`,
distribute `ΔA[g, a_col] = -η[g] · u[a_col]` (when `active[g]`) back into
`ΔW_dx`, `ΔW_dy` at `(weight_row, w_col)` using the coefficient transpose
encoded in `layout`. Gated per row by `active[g]` so Dirichlet rows (should
not appear in Neumann row set anyway) contribute zero.

Accumulates into the provided ΔW matrices (does **not** zero them first —
caller is responsible).
"""
function extract_neumann_sensitivities!(
    ΔW_dx::SparseMatrixCSC{Float64, Int},
    ΔW_dy::SparseMatrixCSC{Float64, Int},
    layout::TractionLayout,
    η::AbstractVector{Float64},
    u::AbstractVector{Float64},
    active::AbstractVector{Bool},
)
    M = layout.M
    @assert M == 2

    for k in eachindex(layout.rows)
        g = layout.rows[k]
        active[g] || continue
        wr = layout.weight_rows[k]
        for p in layout.col_ptr[k]:(layout.col_ptr[k + 1] - 1)
            a_col = layout.a_cols[p]
            w_col = layout.w_cols[p]
            Δa = -η[g] * u[a_col]
            c1 = layout.coeffs[(p - 1) * M + 1]
            c2 = layout.coeffs[(p - 1) * M + 2]
            ix = _find_nzval(ΔW_dx, wr, w_col)
            if ix > 0
                ΔW_dx.nzval[ix] += c1 * Δa
            end
            iy = _find_nzval(ΔW_dy, wr, w_col)
            if iy > 0
                ΔW_dy.nzval[iy] += c2 * Δa
            end
        end
    end
    return nothing
end

"""
    extract_load_sensitivities!(Δpts, layout, η, neumann_ids, traction_jacobians)

Step 3 (b-side) — accumulate `ηᵀ · ∂b/∂pts` into `Δpts` for the dead-load
case where each Neumann point's traction depends on its own coordinates,
`t = t(p_i)`.

`traction_jacobians[i_local]` is a 2×2 matrix `J` with `J[a, b] = ∂t_a/∂p_b`
evaluated at the current location of Neumann point `i_local` (whose global
index is `neumann_ids[i_local]`). For the x-equation row `g = neumann_ids[i_local]`
and y-equation row `g + N`, the contribution to `Δpts` is

    Δpts[2g-1] += η[g] · J[1,1] + η[g+N] · J[2,1]
    Δpts[2g]   += η[g] · J[1,2] + η[g+N] · J[2,2]

(positive sign — recall `dL/dpts = +ηᵀ·∂b/∂pts − ηᵀ·∂A/∂pts·u`).
"""
function extract_load_sensitivities!(
    Δpts::AbstractVector{Float64},
    layout::TractionLayout,
    η::AbstractVector{Float64},
    neumann_ids::AbstractVector{Int},
    traction_jacobians::AbstractVector{<:AbstractMatrix{Float64}},
)
    N = length(η) ÷ 2
    @assert length(traction_jacobians) == length(neumann_ids)
    @inbounds for i_local in eachindex(neumann_ids)
        g  = neumann_ids[i_local]
        ηx = η[g]
        ηy = η[g + N]
        J  = traction_jacobians[i_local]
        Δpts[2g - 1] += ηx * J[1, 1] + ηy * J[2, 1]
        Δpts[2g]     += ηx * J[1, 2] + ηy * J[2, 2]
    end
    return nothing
end

# ============================================================================
# Direct gradient propagation (no Mooncake — calls RBF._pullback_weights!)
# ============================================================================

"""
    _propagate_weight_gradient!(Δpts_flat, ΔW, W, cache, pts, eval_pts, adjl, basis, op)

Propagate a weight-matrix gradient `ΔW` back to the flat coordinate gradient
`Δpts_flat`. Uses RBF's `_pullback_weights!` (pure Julia, sub-millisecond).

- `Δpts_flat`: `Vector{Float64}` (length 2N), accumulated in-place.
- `ΔW`: sparse matrix with gradient values in `nzval`.
- `W`, `cache`: from `_build_weights_and_cache(op, pts, eval_pts, adjl, basis)`.
- `pts`: all data points (length N).
- `eval_pts`: evaluation points (same as `pts` for full operators, subset for Neumann).
- `eval_offset`: if `eval_pts` is a subset of `pts`, the global indices of eval points.
"""
function _propagate_weight_gradient!(
    Δpts_flat::Vector{Float64},
    ΔW::SparseMatrixCSC{Float64,Int},
    W::SparseMatrixCSC{Float64,Int},
    cache,
    pts::AbstractVector{<:AbstractVector{Float64}},
    eval_pts::AbstractVector{<:AbstractVector{Float64}},
    adjl::AbstractVector,
    basis,
    op;
    eval_offset::Union{Nothing,Vector{Int}} = nothing,
)
    TD = Float64
    dim = 2
    N = length(pts)
    M = length(eval_pts)

    Δdata = [zeros(TD, dim) for _ in 1:N]
    Δeval  = [zeros(TD, dim) for _ in 1:M]

    _pullback_weights!(Δdata, Δeval, ΔW.nzval, W, cache, pts, eval_pts, adjl, basis, op)

    # Accumulate Δdata → Δpts_flat (all N data points)
    @inbounds for i in 1:N
        Δpts_flat[2i - 1] += Δdata[i][1]
        Δpts_flat[2i]     += Δdata[i][2]
    end

    # Accumulate Δeval → Δpts_flat
    if eval_offset !== nothing
        # Subset case: eval_pts correspond to specific global indices
        @inbounds for i_local in 1:M
            i_global = eval_offset[i_local]
            Δpts_flat[2i_global - 1] += Δeval[i_local][1]
            Δpts_flat[2i_global]     += Δeval[i_local][2]
        end
    else
        # Full case: eval_pts === pts, add both contributions
        @inbounds for i in 1:N
            Δpts_flat[2i - 1] += Δeval[i][1]
            Δpts_flat[2i]     += Δeval[i][2]
        end
    end

    return nothing
end

# ============================================================================
# Mooncake-extension stubs (real implementations in MacchiatoMooncakeExt)
# ============================================================================

function shape_gradient(args...; kwargs...)
    return error("Mooncake.jl must be loaded to use `shape_gradient`. " *
                 "Run `using Mooncake` first.")
end
