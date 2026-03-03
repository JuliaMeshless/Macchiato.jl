"""
    AbstractSimulationMode

Abstract supertype for simulation modes. See [`Steady`](@ref) and [`Transient`](@ref).
"""
abstract type AbstractSimulationMode end

"""
    Steady()

Mode for steady-state simulations solved via `LinearSolve`.
"""
struct Steady <: AbstractSimulationMode end

"""
    Transient(; Δt, stop_time, solver=Tsit5())

Mode for transient (time-dependent) simulations solved via `OrdinaryDiffEq.solve`.

# Arguments
- `Δt`: Time step size
- `stop_time`: End time for the simulation
- `solver`: ODE solver (default: `Tsit5()`)

# Examples
```julia
Transient(Δt=0.001, stop_time=1.0)
Transient(Δt=0.001, stop_time=1.0, solver=RK4())
```
"""
struct Transient{S} <: AbstractSimulationMode
    Δt::Float64
    stop_time::Float64
    solver::S
end

function Transient(; Δt, stop_time, solver=Tsit5())
    return Transient(Float64(Δt), Float64(stop_time), solver)
end
