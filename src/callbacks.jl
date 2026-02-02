"""
    AbstractSchedule

Abstract type for callback schedules. Concrete implementations determine when callbacks fire.
"""
abstract type AbstractSchedule end

"""
    IterationInterval(N::Int)

Schedule that triggers every `N` iterations.

# Example
```julia
callback = Callback(print_progress, IterationInterval(100))  # Every 100 iterations
```
"""
struct IterationInterval <: AbstractSchedule
    N::Int
    function IterationInterval(N::Int)
        N > 0 || throw(ArgumentError("IterationInterval must be positive, got $N"))
        new(N)
    end
end

"""
    TimeInterval(Δt::Float64)

Schedule that triggers every `Δt` simulation time units.

# Example
```julia
writer = VTKOutputWriter("results/", schedule=TimeInterval(0.1))  # Every 0.1 time units
```
"""
struct TimeInterval <: AbstractSchedule
    Δt::Float64
    function TimeInterval(Δt::Float64)
        Δt > 0 || throw(ArgumentError("TimeInterval must be positive, got $Δt"))
        new(Δt)
    end
end
TimeInterval(Δt::Real) = TimeInterval(Float64(Δt))

"""
    WallTimeInterval(seconds::Float64)

Schedule that triggers every `seconds` wall-clock seconds.

# Example
```julia
callback = Callback(checkpoint, WallTimeInterval(300.0))  # Every 5 minutes
```
"""
struct WallTimeInterval <: AbstractSchedule
    seconds::Float64
    function WallTimeInterval(seconds::Float64)
        seconds > 0 || throw(ArgumentError("WallTimeInterval must be positive, got $seconds"))
        new(seconds)
    end
end
WallTimeInterval(seconds::Real) = WallTimeInterval(Float64(seconds))

"""
    SpecifiedTimes(times::Vector{Float64})

Schedule that triggers at specific simulation times.

# Example
```julia
callback = Callback(snapshot, SpecifiedTimes([0.0, 0.5, 1.0, 2.0]))
```
"""
struct SpecifiedTimes <: AbstractSchedule
    times::Vector{Float64}
    function SpecifiedTimes(times::Vector{Float64})
        isempty(times) && throw(ArgumentError("SpecifiedTimes cannot be empty"))
        issorted(times) || throw(ArgumentError("SpecifiedTimes must be sorted"))
        new(times)
    end
end
SpecifiedTimes(times::AbstractVector{<:Real}) = SpecifiedTimes(Float64.(collect(times)))
SpecifiedTimes(times::Real...) = SpecifiedTimes(collect(Float64.(times)))

"""
    ScheduleState

Mutable state for tracking when a schedule was last triggered.
"""
mutable struct ScheduleState
    last_iteration::Int
    last_time::Float64
    last_wall_time::Float64
    next_specified_idx::Int
end
ScheduleState() = ScheduleState(0, 0.0, time(), 1)

"""
    should_execute(schedule, state, iteration, sim_time) -> Bool

Check if a callback should execute given the current state.
"""
function should_execute(schedule::IterationInterval, state::ScheduleState, iteration::Int, sim_time::Float64)
    if iteration - state.last_iteration >= schedule.N
        state.last_iteration = iteration
        return true
    end
    return false
end

function should_execute(schedule::TimeInterval, state::ScheduleState, iteration::Int, sim_time::Float64)
    if sim_time - state.last_time >= schedule.Δt
        state.last_time = sim_time
        return true
    end
    return false
end

function should_execute(schedule::WallTimeInterval, state::ScheduleState, iteration::Int, sim_time::Float64)
    current_wall = time()
    if current_wall - state.last_wall_time >= schedule.seconds
        state.last_wall_time = current_wall
        return true
    end
    return false
end

function should_execute(schedule::SpecifiedTimes, state::ScheduleState, iteration::Int, sim_time::Float64)
    if state.next_specified_idx > length(schedule.times)
        return false
    end
    if sim_time >= schedule.times[state.next_specified_idx]
        state.next_specified_idx += 1
        return true
    end
    return false
end

"""
    Callback{F, S<:AbstractSchedule, P}

A callback that executes a function according to a schedule.

# Fields
- `func::F`: Function to call. Signature: `func(sim)` or `func(sim, parameters)`
- `schedule::S`: When to execute the callback
- `parameters::P`: Optional parameters passed to the function
- `_state::ScheduleState`: Internal state for schedule tracking

# Example
```julia
function print_progress(sim)
    println("Iteration \$(sim.iteration), t = \$(sim.time)")
end
callback = Callback(print_progress, IterationInterval(100))
```
"""
struct Callback{F, S<:AbstractSchedule, P}
    func::F
    schedule::S
    parameters::P
    _state::ScheduleState
end

function Callback(func::F, schedule::S; parameters::P=nothing) where {F, S<:AbstractSchedule, P}
    Callback{F, S, P}(func, schedule, parameters, ScheduleState())
end

function Callback(func::F, schedule::S, parameters::P) where {F, S<:AbstractSchedule, P}
    Callback{F, S, P}(func, schedule, parameters, ScheduleState())
end

"""
    execute!(callback, sim)

Execute the callback if its schedule condition is met.
Returns true if the callback was executed, false otherwise.
"""
function execute!(callback::Callback, sim)
    if should_execute(callback.schedule, callback._state, sim.iteration, sim.time)
        if callback.parameters === nothing
            callback.func(sim)
        else
            callback.func(sim, callback.parameters)
        end
        return true
    end
    return false
end

"""
    reset!(callback)

Reset callback state for a new simulation run.
"""
function reset!(callback::Callback)
    callback._state.last_iteration = 0
    callback._state.last_time = 0.0
    callback._state.last_wall_time = time()
    callback._state.next_specified_idx = 1
    return nothing
end

function Base.show(io::IO, s::IterationInterval)
    print(io, "IterationInterval($(s.N))")
end

function Base.show(io::IO, s::TimeInterval)
    print(io, "TimeInterval($(s.Δt))")
end

function Base.show(io::IO, s::WallTimeInterval)
    print(io, "WallTimeInterval($(s.seconds)s)")
end

function Base.show(io::IO, s::SpecifiedTimes)
    n = length(s.times)
    if n <= 4
        print(io, "SpecifiedTimes($(s.times))")
    else
        print(io, "SpecifiedTimes([$(s.times[1]), ..., $(s.times[end])] ($n times))")
    end
end

function Base.show(io::IO, c::Callback)
    print(io, "Callback($(c.schedule))")
end
