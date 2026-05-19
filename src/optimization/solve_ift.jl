# ============================================================================
# Shared helpers used by both manual adjoint and monolithic trace
# ============================================================================

"""
    apply_dirichlet!(A, b, dirichlet_dofs, vals)

Overwrite Dirichlet rows of `A` with identity and set the corresponding
entries of `b`. Used by `shape_gradient` in the manual adjoint.
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

# ============================================================================
# Domain-aware BC helpers (useful for Phase 3 optimization loop)
# ============================================================================

"""
    active_dofs(domain::Domain)

Build the `active_dofs` BitVector from the domain's boundary conditions.

A DOF is active (`true`) if its row in the assembled system contains a genuine
PDE equation. Dirichlet BC rows (identity rows) are inactive (`false`).
Neumann/Robin boundaries and interior points are active.
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

# ============================================================================
# Sparse matrix utilities
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
