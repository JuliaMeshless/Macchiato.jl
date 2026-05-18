import RadialBasisFunctions: _build_weights, Partial, MixedPartial, find_neighbors

"""
    make_system_differentiable(::LinearElasticity, pts_flat, N, adjl, basis, λstar, μ)

Assemble the raw 2N×2N plane-stress Navier system from a flat coordinate vector.
Returns the system matrix without any BC modification so that Mooncake can trace
through the five `_build_weights` calls via their existing rrule!!s.

This operates on stripped scalars and plain arrays — no Unitful, no Domain — so
the AD chain is unobstructed.
"""
function make_system_differentiable(
    ::LinearElasticity,
    pts_flat::AbstractVector{<:Real},
    N::Int,
    adjl,
    basis,
    λstar::Real,
    μ::Real,
)
    pts = [SVector{2}(pts_flat[2i - 1], pts_flat[2i]) for i in 1:N]

    W_d2x  = _build_weights(Partial(2, 1),      pts, pts, adjl, basis)
    W_d2y  = _build_weights(Partial(2, 2),      pts, pts, adjl, basis)
    W_d2xy = _build_weights(MixedPartial(1, 2), pts, pts, adjl, basis)

    c1 = λstar + 2μ
    c2 = μ
    c3 = λstar + μ

    # All 3 share the same sparsity pattern — build CSC format directly to avoid
    # sparse-sparse addition and hvcat, which Mooncake traces at scalar level.
    cp = W_d2x.colptr
    rv = W_d2x.rowval
    nz = length(W_d2x.nzval)

    # Block nzvals: A11 = c1*W_d2x + c2*W_d2y,  A12 = c3*W_d2xy,
    #               A21 = A12 (symmetry),         A22 = c2*W_d2x + c1*W_d2y
    a11 = c1 .* W_d2x.nzval .+ c2 .* W_d2y.nzval
    a12 = c3 .* W_d2xy.nzval
    a22 = c2 .* W_d2x.nzval .+ c1 .* W_d2y.nzval

    total_nz = 4 * nz
    colptr_out = Vector{Int}(undef, 2N + 1)
    rowval_out = Vector{Int}(undef, total_nz)
    nzval_out   = Vector{Float64}(undef, total_nz)

    colptr_out[1] = 1
    ptr = 1
    # First N columns: A11 (rows 1:N) then A21 (rows N+1:2N)
    for j in 1:N
        for idx in cp[j]:(cp[j+1] - 1)
            rowval_out[ptr] = rv[idx]
            nzval_out[ptr] = a11[idx]
            ptr += 1
        end
        for idx in cp[j]:(cp[j+1] - 1)
            rowval_out[ptr] = rv[idx] + N
            nzval_out[ptr] = a12[idx]
            ptr += 1
        end
        colptr_out[j+1] = ptr
    end
    # Second N columns: A12 (rows 1:N) then A22 (rows N+1:2N)
    for j in 1:N
        for idx in cp[j]:(cp[j+1] - 1)
            rowval_out[ptr] = rv[idx]
            nzval_out[ptr] = a12[idx]
            ptr += 1
        end
        for idx in cp[j]:(cp[j+1] - 1)
            rowval_out[ptr] = rv[idx] + N
            nzval_out[ptr] = a22[idx]
            ptr += 1
        end
        colptr_out[N+j+1] = ptr
    end

    return SparseMatrixCSC(2N, 2N, colptr_out, rowval_out, nzval_out)
end

"""
    PDESolveIFT

Mooncake-differentiable linear solver for assembled PDE systems. Wraps `A\\b`
with a custom rrule!! that implements the adjoint-based IFT:

    ∂L/∂A[i,j] = -η[i] * u[j]   for active_dofs[i] == true
    ∂L/∂b      = η               where η = Aᵀ \\ (∂L/∂u)

`active_dofs[i] == true` marks rows that are genuine PDE equations. Identity
rows inserted by `apply_dirichlet!` are `false` — their gradient is zero.
"""
struct PDESolveIFT
    active_dofs::BitVector
end

function (s::PDESolveIFT)(A::SparseMatrixCSC{Float64, Int}, b::Vector{Float64})
    return lu(A) \ b
end

"""
    make_active_dofs_elasticity(interior_idx, N)

Build the `active_dofs` BitVector for a 2D elasticity system of size 2N.
`interior_idx` lists the row indices (1-based, in the u_x block) that hold
genuine PDE equations. The u_y block is offset by N.
"""
function make_active_dofs_elasticity(interior_idx::AbstractVector{Int}, N::Int)
    active = falses(2N)
    active[interior_idx]       .= true
    active[interior_idx .+ N]  .= true
    return active
end

"""
    apply_dirichlet!(A, b, dirichlet_dofs, vals)

Overwrite Dirichlet rows of `A` with identity and set the corresponding
entries of `b`. This step is intentionally outside the differentiable path —
Dirichlet rows do not depend on `pts`.
"""
function apply_dirichlet!(
    A::AbstractMatrix,
    b::AbstractVector,
    dirichlet_dofs::AbstractVector{Int},
    vals::AbstractVector{<:Real},
)
    for (k, i) in enumerate(dirichlet_dofs)
        A[i, :] .= 0.0
        A[i, i]  = 1.0
        b[i]     = vals[k]
    end
    return nothing
end

"""
    compute_von_mises(u_flat, N, pts_flat, adjl, basis, λstar, μ)

Compute the von Mises stress field (plane stress) from a displacement solution
`u_flat = [u_x; u_y]` and point coordinates `pts_flat`.

This is fully differentiable via the `_build_weights` rrule!! — no custom rrule
needed here.
"""
function compute_von_mises(
    u_flat::AbstractVector{<:Real},
    N::Int,
    pts_flat::AbstractVector{<:Real},
    adjl,
    basis,
    λstar::Real,
    μ::Real,
)
    pts = [SVector{2}(pts_flat[2i - 1], pts_flat[2i]) for i in 1:N]
    u_x = u_flat[1:N]
    u_y = u_flat[(N + 1):(2N)]

    W_dx = _build_weights(Partial(1, 1), pts, pts, adjl, basis)
    W_dy = _build_weights(Partial(1, 2), pts, pts, adjl, basis)

    dux_dx = W_dx * u_x
    dux_dy = W_dy * u_x
    duy_dx = W_dx * u_y
    duy_dy = W_dy * u_y

    σ_xx = (λstar + 2μ) .* dux_dx .+ λstar .* duy_dy
    σ_yy = λstar .* dux_dx .+ (λstar + 2μ) .* duy_dy
    σ_xy = μ .* (dux_dy .+ duy_dx)

    return sqrt.(σ_xx .^ 2 .- σ_xx .* σ_yy .+ σ_yy .^ 2 .+ 3 .* σ_xy .^ 2)
end

# ============================================================================
# Phase 2: Domain-aware AD setup
# ============================================================================

"""
    active_dofs(domain::Domain)

Build the `active_dofs` BitVector from the domain's boundary conditions.

A DOF is active (`true`) if its row in the assembled system contains a genuine
PDE equation. Dirichlet BC rows (identity rows) are inactive (`false`).
Neumann/Robin boundaries and interior points are active.

This replaces the Phase 1 manual `make_active_dofs_elasticity(interior_idx, N)`.
"""
function active_dofs(domain::Domain)
    model = only(domain.models)
    dim = length(first(_coords(domain.cloud)))
    n_vars = _num_vars(model, dim)
    N = length(points(domain.cloud))
    active = trues(n_vars * N)

    for (_surf_name, (ids, bc)) in domain.boundaries
        bc_family(typeof(bc)) == Dirichlet || continue
        for v in 0:(n_vars - 1)
            active[ids .+ v * N] .= false
        end
    end

    return active
end

"""
    build_dirichlet_info(domain::Domain, t=0.0)

Extract Dirichlet DOF indices and prescribed values from a domain.

Returns `(dirichlet_dofs::Vector{Int}, dirichlet_vals::Vector{Float64})` where
`dirichlet_dofs[k]` is the global row index and `dirichlet_vals[k]` is the
prescribed value. For vector-valued models (e.g. elasticity with n_vars=2),
each boundary point contributes `n_vars` DOFs.
"""
function build_dirichlet_info(domain::Domain, t::Real = 0.0)
    model = only(domain.models)
    dim = length(first(_coords(domain.cloud)))
    n_vars = _num_vars(model, dim)
    N = length(points(domain.cloud))

    dirichlet_dofs = Int[]
    dirichlet_vals = Float64[]

    for (surf_name, (ids, bc)) in domain.boundaries
        bc_family(typeof(bc)) == Dirichlet || continue
        surf = domain.cloud[surf_name]
        for (local_i, global_i) in enumerate(ids)
            x = get_node_coords(surf, local_i)
            vals = bc(x, t)
            if n_vars == 1
                push!(dirichlet_dofs, global_i)
                push!(dirichlet_vals, vals)
            else
                for v in 0:(n_vars - 1)
                    push!(dirichlet_dofs, global_i + v * N)
                    push!(dirichlet_vals, vals[v + 1])
                end
            end
        end
    end

    return dirichlet_dofs, dirichlet_vals
end

"""
    make_system_differentiable(model::LinearElasticity, domain; k=35, basis=PHS(3; poly_deg=3))

Domain-aware overload of `make_system_differentiable`. Extracts coordinates
from the domain, strips units, builds the adjacency list, and delegates to
the low-level `_build_weights`-based assembly.

Returns `(A, b)` where both are traceable through Mooncake.
"""
function make_system_differentiable(
    model::LinearElasticity,
    domain::Domain;
    k::Int = 35,
    basis = PHS(3; poly_deg = 3),
)
    μ, λstar = lame_parameters(model)
    coords = _ustrip(_coords(domain.cloud))
    N = length(coords)
    adjl = find_neighbors(coords, k)

    pts_flat = vcat([collect(p) for p in coords]...)

    A = make_system_differentiable(model, pts_flat, N, adjl, basis, λstar, μ)

    b = zeros(2N)
    if model.body_force !== nothing
        for (i, pt) in enumerate(coords)
            fx, fy = model.body_force(pt)
            b[i] = -fx
            b[i + N] = -fy
        end
    end

    return A, b
end

"""
    gradient(sim, loss_function; wrt=:pts, ...)

Compute the gradient of `loss_function` with respect to point coordinates.

Requires Mooncake.jl to be loaded (`using Mooncake`).
The loss function signature is `loss(pts_flat, setup)` where `setup` is a
named tuple containing `(; sim, adjl, active, dirichlet_dofs, dirichlet_vals,
basis, solver)`.

Returns `Vector{SVector{2, Float64}}` (one gradient vector per point).
"""
function gradient(args...; kwargs...)
    return error("Mooncake.jl must be loaded to use `gradient`. Run `using Mooncake` first.")
end

# ============================================================================
# Neumann BC differentiability — generic sparse row assembly
# ============================================================================

"""
    _find_nzval(W::SparseMatrixCSC, row::Int, col::Int)

Return the nzval index of entry `(row, col)` in sparse matrix `W`, or `0` if
the entry is a structural zero.
"""
function _find_nzval(W::SparseMatrixCSC, row::Int, col::Int)
    @inbounds for idx in nzrange(W, col)
        if W.rowval[idx] == row
            return idx
        end
    end
    return 0
end

"""
    batch_overwrite_sparse_rows!(A, b, rows, col_ptr, a_cols, w_cols, weights,
                                 coeffs, weight_rows, b_vals)

Generic primitive for overwriting rows of a sparse matrix with values computed
as linear combinations of weight-matrix entries. Model-agnostic — the physics
lives in the `coeffs` computation, which the caller performs before invoking
this function.

# Arguments
- `A, b`: system matrix and RHS (mutated in-place)
- `rows::Vector{Int}`: global row indices to overwrite (length `R`)
- `col_ptr::Vector{Int}`: offset array (length `R+1`). Row `k` uses entries
  `col_ptr[k]:col_ptr[k+1]-1`.
- `a_cols::Vector{Int}`: column indices in `A` for each entry (length `E`).
- `w_cols::Vector{Int}`: column indices in `weights` for each entry (length `E`).
  Separate from `a_cols` because vector-valued problems have different column
  spaces (e.g. elasticity v-block columns are offset by N).
- `weights::Vector{SparseMatrixCSC{Float64,Int}}`: `M` sparse weight matrices
  from `_build_weights`, each with the same sparsity pattern.
- `coeffs::Vector{Float64}`: flat coefficient array (length `M * E`,
  interleaved: all `M` coefficients for entry 1, then entry 2, ...).
- `weight_rows::Vector{Int}`: row in each weight matrix for global `rows[k]`
  (length `R`).
- `b_vals::Vector{Float64}`: RHS value for each row (length `R`).

# Forward pass
For each row `k` (global index `rows[k]`):
  A[rows[k], a_cols[e]] = Σ_m coeffs[e*M + m] * weights[m][weight_rows[k], w_cols[e]]
  b[rows[k]] = b_vals[k]

# Backward pass (handled by Mooncake rrule!!)
Redirects `ΔA` at the overwritten entries to `Δweights` using the transpose of
`coeffs`, then zeros `ΔA`/`Δb` at those rows.
"""
function batch_overwrite_sparse_rows!(
    A::SparseMatrixCSC{Float64, Int},
    b::Vector{Float64},
    rows::Vector{Int},
    col_ptr::Vector{Int},
    a_cols::Vector{Int},
    w_cols::Vector{Int},
    weights::Vector{SparseMatrixCSC{Float64, Int}},
    coeffs::Vector{Float64},
    weight_rows::Vector{Int},
    b_vals::Vector{Float64},
)
    M = length(weights)
    row_set = Set{Int}(rows)
    zero_rows!(A, row_set)

    for k in 1:length(rows)
        global_row = rows[k]
        wr = weight_rows[k]
        for p in col_ptr[k]:(col_ptr[k + 1] - 1)
            a_col = a_cols[p]
            w_col = w_cols[p]
            val = 0.0
            for m in 1:M
                idx_w = _find_nzval(weights[m], wr, w_col)
                if idx_w > 0
                    val += coeffs[(p - 1) * M + m] * weights[m].nzval[idx_w]
                end
            end
            A[global_row, a_col] = val
        end
        b[global_row] = b_vals[k]
    end

    return nothing
end
