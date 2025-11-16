"""
    abstract_boundary_conditions.jl

Generic boundary condition framework for MeshlessMultiphysics.jl.

This module imports the boundary condition system from RadialBasisFunctions.jl,
which provides a unified, Robin-based boundary condition framework:
    Robin BC: α*u + β*∂ₙu = g

Special cases:
- Dirichlet: α=1, β=0  →  u = g
- Neumann:   α=0, β=1  →  ∂ₙu = g
- Robin:     α≠0, β≠0  →  α*u + β*∂ₙu = g

The BoundaryCondition struct from RBF.jl stores (α, β) coefficients.
Domain-specific boundary types (Temperature, HeatFlux, Wall, etc.) store 
the prescribed value g directly in their own fields.
"""

using RadialBasisFunctions: BoundaryCondition, Dirichlet, Neumann, Robin, Internal
using RadialBasisFunctions: α, β, is_dirichlet, is_neumann, is_robin, is_internal

# Re-export for convenience
export BoundaryCondition, Dirichlet, Neumann, Robin, Internal
export α, β, is_dirichlet, is_neumann, is_robin, is_internal

# ============================================================================
# Generic Boundary Condition Implementation
# ============================================================================

"""
    apply_bc!(A::AbstractMatrix, b::AbstractVector, bc_type::BoundaryCondition,
              surf, domain, ids, value; kwargs...)

Unified boundary condition application that dispatches based on BC type (Dirichlet/Neumann/Robin).
Automatically selects the appropriate application method based on α and β coefficients.

# Arguments
- `A`: System matrix
- `b`: Right-hand side vector
- `bc_type`: BoundaryCondition with α, β coefficients (from bc_type() helper)
- `surf`: Surface where BC is applied
- `domain`: Problem domain
- `ids`: Node indices for the boundary
- `value`: Prescribed value (or function)
- `kwargs...`: Additional arguments (e.g., k for neighbor count)
"""
function apply_bc!(A::AbstractMatrix, b::AbstractVector,
        bc::BoundaryCondition, surf, domain, ids, value; kwargs...)
    # Dispatch based on boundary condition type
    if is_dirichlet(bc)
        return apply_dirichlet!(A, b, ids, value)
    elseif is_neumann(bc)
        return apply_neumann!(A, b, surf, domain, ids, value; kwargs...)
    else  # Robin
        return apply_robin!(A, b, α(bc), β(bc), surf, domain, ids, value; kwargs...)
    end
end

# ============================================================================
# Dispatch to specific cases: Dirichlet, Neumann or Robin
# ============================================================================

"""
    apply_dirichlet!(A::AbstractMatrix, b::AbstractVector, ids, value)

Apply Dirichlet boundary condition to linear system.
Sets A[i, :] to zero except A[i, i] = 1, and b[i] = value.

For sparse matrices, this efficiently modifies only the relevant entries.
"""
function apply_dirichlet!(A::AbstractMatrix{TA}, b::AbstractVector{TB},
        ids, value::Union{TB, AbstractVector{TB}}) where {TA, TB}
    for i in ids
        A[i, :] .= zero(TA)
        A[i, i] = one(TA)
        b[i] = value isa AbstractVector ? value[i - first(ids) + 1] : convert(TB, value)
    end
    return A, b
end

"""
    apply_neumann!(A::AbstractMatrix, b::AbstractVector, surf, domain, ids, flux_value; kwargs...)

Apply Neumann boundary condition to linear system using directional derivatives.
Computes the directional derivative operator in the normal direction and applies it.

For Neumann BC: ∂ₙu = g, the matrix row becomes the directional derivative weights.
"""
function apply_neumann!(A::AbstractMatrix{TA}, b::AbstractVector{TB},
        surf, domain, ids, flux_value; kwargs...) where {TA, TB}
    # Compute directional derivative operator
    d = directional(coordinates(domain.cloud), coordinates(surf), normals(surf);
        k = get(kwargs, :k, 40))
    update_weights!(d)

    offset = first(ids) - 1
    for i in ids
        # Directly assign the weights (no need to zero first)
        A[i, :] .= d.weights[i - offset, :]
        b[i] = convert(TB, flux_value)
    end

    return A, b
end

"""
    apply_robin!(A::AbstractMatrix, b::AbstractVector, α, β, 
                 surf, domain, ids, specified_value; kwargs...)

Apply Robin boundary condition: α*u + β*∂ₙu = specified_value to linear system.

The matrix row becomes: A[i, :] = α*I[i, :] + β*∂ₙ[i, :]
where I is identity and ∂ₙ is the directional derivative operator.
"""
function apply_robin!(A::AbstractMatrix{TA}, b::AbstractVector{TB},
        α_val, β_val, surf, domain, ids, specified_value;
        kwargs...) where {TA, TB}
    # For Robin: α*u + β*∂ₙu = g

    d = directional(coordinates(domain.cloud), coordinates(surf), normals(surf);
        k = get(kwargs, :k, 40))
    update_weights!(d)

    offset = first(ids) - 1
    for i in ids
        # Build Robin row: α*I + β*D
        A[i, :] .= convert(TA, β_val) .* d.weights[i - offset, :]
        A[i, i] += convert(TA, α_val)  # Add identity term
        b[i] = convert(TB, specified_value)
    end

    return A, b
end
