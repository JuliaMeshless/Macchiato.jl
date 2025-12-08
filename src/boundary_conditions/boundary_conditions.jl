# Core infrastructure
include("core/physics_traits.jl")
include("core/time_traits.jl")
include("core/bc_hierarchy.jl")
include("core/generic_types.jl")

# Main dispatch functions for applying BCs to system matrices/vectors

#case of LinearProblem (probably steady state)
function make_bc!(A, b, boundary::T, surf, domain, ids; kwargs...) where {T}
    return make_bc!(bc_family(T), A, b, boundary, surf, domain, ids; kwargs...)
end

#case of ODEProblem (time-dependent)
function make_bc(
        boundary::T, surf, domain, ids; kwargs...) where {T <: AbstractBoundaryCondition}
    return make_bc(bc_family(T), boundary, surf, domain, ids; kwargs...)
end

# Dispatch implementations of Dirichlet BC

# case of LinearProblem (steady state)
function make_bc!(::Type{Dirichlet}, A, b, boundary, surf, domain, ids; kwargs...)
    return write_bc_dirichlet!(A, b, ids, boundary, surf)
end

# case of ODEProblem (time-dependent)
function make_bc(::Type{Dirichlet}, boundary, surf, domain, ids; kwargs...)
    # Dispatch based on the time dependence trait of the boundary value
    val = boundary()
    return make_bc_dirichlet(time_dependence(val), val, surf, domain, ids; kwargs...)
end

# 1. Steady: Constant in time -> du/dt = 0
function make_bc_dirichlet(::SteadyTime, val, surf, domain, ids; kwargs...)
    return function (du, u, p, t)
        du[ids] .= 0.0
        return nothing
    end
end

# 2. Transient: Function of (x, t) -> Compute time derivative
function make_bc_dirichlet(::Transient, func, surf, domain, ids; kwargs...)
    # Get the time derivative function (handles FD logic)
    df_dt = time_derivative(func)

    return function (du, u, p, t)
        for i in ids
            # Evaluate derivative at (x, t)
            # get_bc_value_at_index handles the coordinate lookup and (x,t) dispatch
            du[i] = get_bc_value_at_index(df_dt, surf, ids, i, t)
        end
        return nothing
    end
end

# Dispatch implementations of Neumann and Robin BCs
# case of LinearProblem (steady state)
function make_bc!(::Type{DerivativeBoundaryCondition}, A, b, boundary, surf, domain, ids;
        scheme = nothing, kwargs...)
    # scheme comes from kwargs, passed down from LinearProblem
    return write_bc_derivative!(
        bc_type(typeof(boundary)), A, b, ids, boundary, surf, domain, scheme; kwargs...)
end

# case of ODEProblem (time-dependent) (not implemented yet)
function make_bc(
        ::Type{<:DerivativeBoundaryCondition}, boundary, surf, domain, ids; kwargs...)
    return (du, u, p, t) -> nothing
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
    point = _coords(surf)[local_idx]
    return value(point)
end

function get_bc_value_at_index(val, surf, ids, i, t)
    get_bc_value_at_index(time_dependence(val), val, surf, ids, i, t)
end

# 1. SteadyTime: Ignore time 't', just use spatial index
function get_bc_value_at_index(::SteadyTime, val, surf, ids, i, t)
    return get_bc_value_at_index(val, surf, ids, i) # Call existing spatial-only version
end

# 2. Transient: Must be a function f(x, t)
function get_bc_value_at_index(::Transient, func, surf, ids, i, t)
    local_idx = i - first(ids) + 1
    point = _coords(surf)[local_idx]
    return func(point, t)
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