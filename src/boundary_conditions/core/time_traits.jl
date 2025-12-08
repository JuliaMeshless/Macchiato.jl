# ============================================================================
# Time Dependence Traits
# ============================================================================

abstract type AbstractTimeDependence end

struct SteadyTime <: AbstractTimeDependence end

struct Transient <: AbstractTimeDependence end

time_dependence(x) = time_dependence(typeof(x))

time_dependence(::Type{T}) where {T} = SteadyTime() # Default fallback

time_dependence(::Type{<:Number}) = SteadyTime()

time_dependence(::Type{<:AbstractArray}) = SteadyTime()

function time_dependence(f::Function)
    if hasmethod(f, Tuple{Any, Number})
        return Transient()
    elseif hasmethod(f, Tuple{Any})
        return SteadyTime()
    else
        error("Boundary function must accept either f(x) or f(x, t)")
    end
end

time_derivative(val) = time_derivative(time_dependence(val), val)

time_derivative(::SteadyTime, val) = 0.0

function time_derivative(::Transient, func)
    dt_eps = 1e-6
    inv_2dt = 1.0 / (2 * dt_eps)

    return function (x, t)
        # Central difference: (f(x, t+dt) - f(x, t-dt)) / 2dt
        v_plus = func(x, t + dt_eps)
        v_minus = func(x, t - dt_eps)
        return (v_plus - v_minus) * inv_2dt
    end
end
