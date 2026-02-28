"""
    Time <: AbstractModel

Abstract type for time-mode models that indicate whether a simulation is steady-state
or transient. See [`Steady`](@ref) and [`Unsteady`](@ref).
"""
abstract type Time <:AbstractModel end

"""
    Steady(max_time)

Marker for steady-state simulations. `max_time` may be used as a convergence limit.
"""
struct Steady{T} <: Time
    max_time::T
end

"""
    Unsteady(max_time)

Marker for transient (time-dependent) simulations. `max_time` sets the end time.
"""
struct Unsteady{T} <: Time
    max_time::T
end
