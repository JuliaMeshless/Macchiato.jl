# ============================================================================
# Boundary Condition System
# ============================================================================
#
# This file orchestrates the boundary condition system, which consists of:
#
# 1. Core Infrastructure (core/):
#    - physics_traits.jl: Physics domain trait system for BC-model validation
#    - bc_hierarchy.jl: Mathematical BC type hierarchy (Dirichlet, Neumann, Robin)
#    - generic_types.jl: Reusable generic BC types (FixedValue, Flux, ZeroGradient)
#
# 2. Numerical Methods (numerical/):
#    - derivatives.jl: Derivative computation for Neumann and Robin BCs
#
# 3. Physics-Specific BCs (at package level):
#    - energy.jl: Energy/thermal boundary conditions
#    - fluids.jl: Fluid dynamics boundary conditions
#    - walls.jl: Wall boundary conditions
#
# ============================================================================

# Core infrastructure
include("core/physics_traits.jl")
include("core/bc_hierarchy.jl")
include("core/generic_types.jl")

function make_bc!(A, b, boundary::T, surf, domain, ids; kwargs...) where {T}
    return make_bc!(bc_family(T), A, b, boundary, surf, domain, ids; kwargs...)
end

function make_bc!(::Type{Dirichlet}, A, b, boundary, surf, domain, ids; kwargs...)
    return write_bc_dirichlet!(A, b, ids, boundary, surf)
end

function make_bc!(::Type{DerivativeBoundaryCondition}, A, b, boundary, surf, domain, ids;
        scheme = nothing, kwargs...)
    # scheme comes from kwargs, passed down from LinearProblem
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

function get_bc_value_at_index(value::Number, surf, ids, i)
    # Scalar value - return as-is
    return value
end

function get_bc_value_at_index(value::AbstractVector, surf, ids, i)
    # Vector of values - index into it
    local_idx = i - first(ids) + 1
    return value[local_idx]
end

function get_bc_value_at_index(value::Function, surf, ids, i)
    # Function - evaluate at surface point
    local_idx = i - first(ids) + 1
    point = coordinates(surf)[local_idx]
    return value(point)
end

function write_bc_dirichlet!(A::AbstractMatrix{TA}, b::AbstractVector{TB},
        ids, boundary, surf) where {TA, TB}
    bc_value = boundary()  # Can be scalar, vector, or function

    for i in ids
        A[i, :] .= zero(TA)
        A[i, i] = one(TA)
        b[i] = convert(TB, get_bc_value_at_index(bc_value, surf, ids, i))
    end
end

function write_bc_neumann!(A::AbstractMatrix{TA}, b::AbstractVector{TB},
        ids, boundary, surf, domain, scheme; kwargs...) where {TA, TB}
    bc_value = boundary()  # Can be scalar, vector, or function
    normals = normal(surf)

    for (local_i, global_i) in enumerate(ids)
        nbs, weights = compute_local_derivative_weights(
            surf, domain, scheme, A, global_i, local_i, normals; kwargs...)

        sv = SparseVector(size(A, 2), nbs, weights)
        A[global_i, :] = sv

        b[global_i] = convert(TB, get_bc_value_at_index(bc_value, surf, ids, global_i))
    end
end

function write_bc_robin!(A::AbstractMatrix{TA}, b::AbstractVector{TB},
        ids, boundary, surf, domain, scheme; kwargs...) where {TA, TB}
    α_val = convert(TA, α(boundary))
    β_val = convert(TA, β(boundary))
    bc_value = boundary()  # Can be scalar, vector, or function
    normals = normal(surf)

    for (local_i, global_i) in enumerate(ids)
        nbs, weights = compute_local_derivative_weights(
            surf, domain, scheme, A, global_i, local_i, normals; kwargs...)

        # Robin BC: β * ∂u/∂n + α * u = g
        robin_weights = convert(TA, β_val) .* weights
        diag_idx = searchsortedfirst(nbs, global_i)
        robin_weights[diag_idx] += convert(TA, α_val)

        sv = SparseVector(size(A, 2), nbs, robin_weights)
        A[global_i, :] = sv

        b[global_i] = convert(TB, get_bc_value_at_index(bc_value, surf, ids, global_i))
    end
end