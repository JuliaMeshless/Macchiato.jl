"""
    boundary_conditions.jl

Generic boundary condition framework for MeshlessMultiphysics.jl.

Imports the boundary condition system from RadialBasisFunctions.jl:
    Robin BC: α*u + β*∂ₙu = g

Special cases:
- Dirichlet: α=1, β=0  →  u = g
- Neumann:   α=0, β=1  →  ∂ₙu = g
- Robin:     α≠0, β≠0  →  α*u + β*∂ₙu = g
"""

using RadialBasisFunctions: BoundaryCondition, Dirichlet, Neumann, Robin, Internal
using RadialBasisFunctions: α, β, is_dirichlet, is_neumann, is_robin, is_internal

export BoundaryCondition, Dirichlet, Neumann, Robin, Internal
export α, β, is_dirichlet, is_neumann, is_robin, is_internal

# ============================================================================
# Default BC Implementation
# ============================================================================

"""
    make_bc!(A, b, boundary, surf, domain, ids; kwargs...)

Apply boundary condition to linear system. Dispatches based on BC type.

Concrete BC types should define:
- bc_type(::MyBC) → BoundaryCondition (Dirichlet/Neumann/Robin)
- bc_value(bc::MyBC) → prescribed value

Special cases can override for custom behavior.
"""
function make_bc!(A, b, boundary, surf, domain, ids; kwargs...)
    bc = bc_type(boundary)
    value = bc_value(boundary)

    if is_dirichlet(bc)
        make_bc_dirichlet!(A, b, ids, value)
    elseif is_neumann(bc)
        make_bc_neumann!(A, b, surf, domain, ids, value; kwargs...)
    else  # Robin
        make_bc_robin!(A, b, α(bc), β(bc), surf, domain, ids, value; kwargs...)
    end

    return A
end

# ============================================================================
# BC Type Implementations
# ============================================================================

function make_bc_dirichlet!(A::AbstractMatrix{TA}, b::AbstractVector{TB},
        ids, value) where {TA, TB}
    for i in ids
        A[i, :] .= zero(TA)
        A[i, i] = one(TA)
        b[i] = value isa AbstractVector ? value[i - first(ids) + 1] : convert(TB, value)
    end
end

function make_bc_neumann!(A::AbstractMatrix{TA}, b::AbstractVector{TB},
        surf, domain, ids, flux_value; kwargs...) where {TA, TB}
    d = directional(coordinates(domain.cloud), coordinates(surf), normals(surf);
        k = get(kwargs, :k, 40))
    update_weights!(d)

    for i in ids
        A[i, :] .= d.weights[i, :]
        b[i] = convert(TB, flux_value)
    end
end

function make_bc_robin!(A::AbstractMatrix{TA}, b::AbstractVector{TB},
        α_val, β_val, surf, domain, ids, value; kwargs...) where {TA, TB}
    d = directional(coordinates(domain.cloud), coordinates(surf), normals(surf);
        k = get(kwargs, :k, 40))
    update_weights!(d)

    for i in ids
        A[i, :] .= convert(TA, β_val) .* d.weights[i, :]
        A[i, i] += convert(TA, α_val)
        b[i] = convert(TB, value)
    end
end
