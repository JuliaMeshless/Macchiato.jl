# Boundary Condition System
# Includes: core traits, BC hierarchy, generic types, and numerical methods

# Core infrastructure
include("core/bc_hierarchy.jl")
include("core/generic_types.jl")

# ============================================================================
# ODE-style BC Application (for transient problems)
# ============================================================================

"""
    make_bc(bc, surf, domain, ids; kwargs...)

Create boundary condition function for ODE integration.
Returns a function `f(du, u, p, t)` that applies the BC to du.
"""
function make_bc(bc::T, surf, domain, ids; kwargs...) where {T}
    return make_bc(bc_family(T), bc, surf, domain, ids; kwargs...)
end

function make_bc(::Type{Dirichlet}, bc, surf, domain, ids; kwargs...)
    function bc_func(du, u, p, t)
        for (local_i, global_i) in enumerate(ids)
            x = get_node_coords(surf, local_i)
            du[global_i] = bc(x, t) - u[global_i]
        end
        return nothing
    end
    return bc_func
end

# Transient Neumann/Robin conditions are not applied here: they are folded into the
# diffusion operator via ghost nodes (see `build_neumann_diffusion`), so the boundary
# nodes stay dynamic. Only Dirichlet surfaces get an ODE-style BC closure.
function make_bc(::Type{DerivativeBoundaryCondition}, bc, surf, domain, ids; kwargs...)
    throw(
        ArgumentError(
            "Transient Neumann/Robin conditions are folded into the diffusion operator " *
                "(see `build_neumann_diffusion`), not applied through `make_bc`."
        )
    )
end

# ============================================================================
# Matrix-style BC Application Dispatchers (for steady-state problems)
# ============================================================================

function make_bc!(A, b, boundary::T, surf, domain, ids; kwargs...) where {T}
    return make_bc!(bc_family(T), A, b, boundary, surf, domain, ids; kwargs...)
end

function make_bc!(::Type{Dirichlet}, A, b, boundary, surf, domain, ids; kwargs...)
    return write_bc_dirichlet!(A, b, ids, boundary, surf)
end

function make_bc!(
        ::Type{DerivativeBoundaryCondition}, A, b, boundary, surf, domain, ids;
        scheme = nothing, kwargs...
    )
    return write_bc_derivative!(
        bc_type(typeof(boundary)), A, b, ids, boundary, surf, domain, scheme; kwargs...
    )
end

function write_bc_derivative!(
        ::Type{Neumann}, A, b, ids, boundary, surf, domain, scheme; kwargs...
    )
    return write_bc_neumann!(A, b, ids, boundary, surf, domain, scheme; kwargs...)
end

function write_bc_derivative!(
        ::Type{Robin}, A, b, ids, boundary, surf, domain, scheme; kwargs...
    )
    return write_bc_robin!(A, b, ids, boundary, surf, domain, scheme; kwargs...)
end

# ============================================================================
# BC Implementation Functions
# ============================================================================

function write_bc_dirichlet!(
        A::AbstractMatrix{TA}, b::AbstractVector{TB},
        ids, bc, surf, n_vars::Int = 1, t = 0.0
    ) where {TA, TB}
    N = div(size(A, 1), n_vars)

    # Batch zero all BC rows in a single O(nnz) pass
    row_set = Set{Int}()
    for global_i in ids
        for v in 0:(n_vars - 1)
            push!(row_set, global_i + v * N)
        end
    end
    zero_rows!(A, row_set)

    # Set diagonals and RHS
    for (local_i, global_i) in enumerate(ids)
        x = get_node_coords(surf, local_i)
        vals = bc(x, t)
        if n_vars == 1
            A[global_i, global_i] = one(TA)
            b[global_i] = convert(TB, vals)
        else
            for v in 1:n_vars
                row = global_i + (v - 1) * N
                A[row, row] = one(TA)
                b[row] = convert(TB, vals[v])
            end
        end
    end
    return
end

"""
    scatter_rows!(A, rows, R)

Overwrite `A[rows[r], :]` with `R[r, :]`, in place.

Each row of `R` must be supported on columns already stored in the
corresponding row of `A`; throws otherwise.
"""
function scatter_rows!(A::SparseMatrixCSC, rows, R::SparseMatrixCSC)
    zero_rows!(A, Set(rows))
    rowv = collect(rows)
    Ir, Jr, Vr = findnz(R)
    @inbounds for t in eachindex(Ir)
        i, j = rowv[Ir[t]], Jr[t]
        rng = A.colptr[j]:(A.colptr[j + 1] - 1)
        p = searchsortedfirst(view(A.rowval, rng), i)
        (p <= length(rng) && A.rowval[rng[p]] == i) || throw(
            ArgumentError(
                "boundary row $i has support at column $j that the assembled operator " *
                    "does not: the boundary and interior stencils disagree. Pass the same " *
                    "`adjl` to both, or omit it from both."
            )
        )
        A.nzval[rng[p]] = Vr[t]
    end
    return A
end

# Writes `β ∂u/∂n + α u = g` on `ids`. Neumann is α = 0, β = 1; Robin takes α
# and β from the boundary condition.
function write_bc_derivative_block!(
        A::AbstractMatrix{TA}, b::AbstractVector{TB},
        ids, boundary, surf, domain, scheme, α_val, β_val, t = 0.0; kwargs...
    ) where {TA, TB}
    n = length(ids)
    R = convert(TA, β_val) .*
        derivative_rows(surf, domain, scheme, ids, normal(surf); A = A, kwargs...)
    if !iszero(α_val)
        # α on the diagonal, added as a block rather than element by element.
        R += sparse(1:n, collect(ids), fill(convert(TA, α_val), n), n, size(A, 2))
    end
    for (local_i, global_i) in enumerate(ids)
        b[global_i] = convert(TB, boundary(get_node_coords(surf, local_i), t))
    end
    scatter_rows!(A, ids, R)
    return
end

function write_bc_neumann!(A, b, ids, boundary, surf, domain, scheme, t = 0.0; kwargs...)
    return write_bc_derivative_block!(
        A, b, ids, boundary, surf, domain, scheme, 0.0, 1.0, t; kwargs...
    )
end

function write_bc_robin!(A, b, ids, boundary, surf, domain, scheme, t = 0.0; kwargs...)
    return write_bc_derivative_block!(
        A, b, ids, boundary, surf, domain, scheme, α(boundary), β(boundary), t; kwargs...
    )
end
