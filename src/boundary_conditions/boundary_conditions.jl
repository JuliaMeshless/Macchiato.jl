# Boundary Condition System
# Includes: core traits, BC hierarchy, generic types, and numerical methods

# Core infrastructure
include("core/physics_traits.jl")
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

function make_bc(::Type{DerivativeBoundaryCondition}, bc, surf, domain, ids; kwargs...)
    function bc_func(du, u, p, t)
        for (local_i, global_i) in enumerate(ids)
            du[global_i] = zero(eltype(du))
        end
        return nothing
    end
    return bc_func
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

function make_bc!(::Type{DerivativeBoundaryCondition}, A, b, boundary, surf, domain, ids;
        scheme = nothing, kwargs...)
    return write_bc_derivative!(
        bc_type(typeof(boundary)), A, b, ids, boundary, surf, domain, scheme; kwargs...)
end

function write_bc_derivative!(
        ::Type{Neumann}, A, b, ids, boundary, surf, domain, scheme; kwargs...)
    return write_bc_neumann!(A, b, ids, boundary, surf, domain, scheme; kwargs...)
end

function write_bc_derivative!(
        ::Type{Robin}, A, b, ids, boundary, surf, domain, scheme; kwargs...)
    return write_bc_robin!(A, b, ids, boundary, surf, domain, scheme; kwargs...)
end

# ============================================================================
# BC Implementation Functions
# ============================================================================

function write_bc_dirichlet!(A::AbstractMatrix{TA}, b::AbstractVector{TB},
        ids, bc, surf, n_vars::Int = 1, t = 0.0) where {TA, TB}
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
end

function write_bc_neumann!(A::AbstractMatrix{TA}, b::AbstractVector{TB},
        ids, boundary, surf, domain, scheme, t = 0.0; kwargs...) where {TA, TB}
    normals = normal(surf)

    for (local_i, global_i) in enumerate(ids)
        x = get_node_coords(surf, local_i)
        nbs, weights = compute_local_derivative_weights(
            surf, domain, scheme, A, global_i, local_i, normals; kwargs...)

        sv = SparseVector(size(A, 2), nbs, weights)
        A[global_i, :] = sv
        b[global_i] = convert(TB, boundary(x, t))
    end
end

function write_bc_robin!(A::AbstractMatrix{TA}, b::AbstractVector{TB},
        ids, boundary, surf, domain, scheme, t = 0.0; kwargs...) where {TA, TB}
    α_val = convert(TA, α(boundary))
    β_val = convert(TA, β(boundary))
    normals = normal(surf)

    for (local_i, global_i) in enumerate(ids)
        x = get_node_coords(surf, local_i)
        nbs, weights = compute_local_derivative_weights(
            surf, domain, scheme, A, global_i, local_i, normals; kwargs...)

        robin_weights = convert(TA, β_val) .* weights
        diag_idx = searchsortedfirst(nbs, global_i)
        robin_weights[diag_idx] += convert(TA, α_val)

        sv = SparseVector(size(A, 2), nbs, robin_weights)
        A[global_i, :] = sv
        b[global_i] = convert(TB, boundary(x, t))
    end
end
