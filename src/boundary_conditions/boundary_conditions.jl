abstract type AbstractBoundaryCondition end
abstract type DerivativeBoundaryCondition <: AbstractBoundaryCondition end

abstract type Dirichlet <: AbstractBoundaryCondition end
abstract type Neumann <: DerivativeBoundaryCondition end
abstract type Robin <: DerivativeBoundaryCondition end

bc_family(::Type{<:Dirichlet}) = Dirichlet
bc_family(::Type{<:DerivativeBoundaryCondition}) = DerivativeBoundaryCondition

bc_type(::Type{<:Dirichlet}) = Dirichlet
bc_type(::Type{<:Neumann}) = Neumann
bc_type(::Type{<:Robin}) = Robin

function make_bc!(A, b, boundary::T, surf, domain, ids; kwargs...) where {T}
    return make_bc!(bc_family(T), A, b, boundary, surf, domain, ids; kwargs...)
end

function make_bc!(::Type{Dirichlet}, A, b, boundary, surf, domain, ids; kwargs...)
    return write_bc_dirichlet!(A, b, ids, boundary, surf)
end

function make_bc!(::Type{DerivativeBoundaryCondition}, A, b, boundary, surf, domain, ids;
        scheme = nothing, kwargs...)
    # scheme comes from kwargs, passed down from LinearProblem
    weights = compute_derivative_weights(surf, domain, scheme; kwargs...)
    return write_bc_derivative!(
        bc_type(typeof(boundary)), A, b, ids, weights, boundary, surf; kwargs...)
end

function write_bc_derivative!(
        ::Type{Neumann}, A, b, ids, weights, boundary, surf; kwargs...)
    return write_bc_neumann!(A, b, ids, weights, boundary, surf; kwargs...)
end

function write_bc_derivative!(::Type{Robin}, A, b, ids, weights, boundary, surf; kwargs...)
    return write_bc_robin!(A, b, ids, weights, boundary, surf; kwargs...)
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
        ids, weights, boundary, surf; kwargs...) where {TA, TB}
    bc_value = boundary()  # Can be scalar, vector, or function

    for i in ids
        local_idx = i - first(ids) + 1
        A[i, :] .= weights[local_idx, :]
        b[i] = convert(TB, get_bc_value_at_index(bc_value, surf, ids, i))
    end
end

function write_bc_robin!(A::AbstractMatrix{TA}, b::AbstractVector{TB},
        ids, weights, boundary, surf; kwargs...) where {TA, TB}
    α_val = convert(TA, α(boundary))
    β_val = convert(TA, β(boundary))
    bc_value = boundary()  # Can be scalar, vector, or function

    for i in ids
        local_idx = i - first(ids) + 1
        A[i, :] .= convert(TA, β_val) .* weights[local_idx, :]
        A[i, i] += convert(TA, α_val)
        b[i] = convert(TB, get_bc_value_at_index(bc_value, surf, ids, i))
    end
end