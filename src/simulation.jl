"""
    SimulationMode

Enum indicating whether simulation is transient (time-stepping) or steady-state.
"""
@enum SimulationMode Transient SteadyState

"""
    Simulation{D<:Domain}

High-level simulation container that manages time stepping, callbacks, and output.

# Fields
- `domain::D`: The computational domain with models and boundary conditions
- `mode::SimulationMode`: Whether this is a transient or steady-state simulation
- `Δt::Union{Nothing, Float64}`: Time step (transient only)
- `stop_time::Union{Nothing, Float64}`: End time (transient only)
- `stop_iteration::Int`: Maximum iterations (0 = unlimited)
- `solver::Symbol`: ODE solver to use (transient only)
- `solver_options::NamedTuple`: Additional solver options
- `callbacks::Dict{Symbol, Callback}`: Named callbacks
- `output_writers::Dict{Symbol, AbstractOutputWriter}`: Named output writers
- `u0::Union{Nothing, Vector{Float64}}`: Initial condition vector
- `iteration::Int`: Current iteration count
- `time::Float64`: Current simulation time
- `running::Bool`: Whether simulation is currently running
- `_integrator`: ODE integrator (transient only)
- `_solution`: Solution vector (steady-state or final transient)

# Constructors

## Transient simulation
```julia
sim = Simulation(domain; Δt=0.001, stop_time=1.0)
sim = Simulation(domain; Δt=0.001, stop_iteration=1000)
```

## SteadyState-state simulation
```julia
sim = Simulation(domain)  # No Δt → steady-state mode
```

# Mode Detection
- If `Δt` and/or `stop_time` provided → transient mode
- If neither provided → steady-state mode
"""
mutable struct Simulation{M, C}
    domain::Domain{M, C}
    mode::SimulationMode
    Δt::Union{Nothing, Float64}
    stop_time::Union{Nothing, Float64}
    stop_iteration::Int
    solver::Symbol
    solver_options::NamedTuple
    callbacks::Dict{Symbol, Callback}
    output_writers::Dict{Symbol, AbstractOutputWriter}
    u0::Union{Nothing, Vector{Float64}}
    iteration::Int
    time::Float64
    running::Bool
    _integrator::Any
    _solution::Union{Nothing, Vector{Float64}}
end

function Simulation(domain::Domain{M, C};
    Δt::Union{Nothing, Real}=nothing,
    stop_time::Union{Nothing, Real}=nothing,
    stop_iteration::Int=0,
    solver::Symbol=:Tsit5,
    solver_options::NamedTuple=NamedTuple()
) where {M, C}

    mode = (Δt === nothing && stop_time === nothing) ? SteadyState : Transient

    if mode == Transient
        if Δt === nothing
            throw(ArgumentError("Transient simulation requires Δt"))
        end
        if stop_time === nothing && stop_iteration == 0
            throw(ArgumentError("Transient simulation requires stop_time or stop_iteration"))
        end
    end

    Simulation{M, C}(
        domain,
        mode,
        Δt === nothing ? nothing : Float64(Δt),
        stop_time === nothing ? nothing : Float64(stop_time),
        stop_iteration,
        solver,
        solver_options,
        Dict{Symbol, Callback}(),
        Dict{Symbol, AbstractOutputWriter}(),
        nothing,
        0,
        0.0,
        false,
        nothing,
        nothing
    )
end

"""
    run!(sim::Simulation)

Execute the simulation. Dispatches to `_run_transient!` or `_run_steady!` based on mode.

Returns the simulation object for chaining.
"""
function run!(sim::Simulation; kwargs...)
    sim.running = true

    _initialize_callbacks!(sim)
    _initialize_output_writers!(sim)

    try
        if sim.mode == Transient
            _run_transient!(sim)
        else
            _run_steady!(sim; kwargs...)
        end
    finally
        sim.running = false
        _finalize_output_writers!(sim)
    end

    return sim
end

function _run_transient!(sim::Simulation)
    _ensure_u0_initialized!(sim)

    tspan = (0.0, sim.stop_time === nothing ? Inf : sim.stop_time)

    prob = _create_ode_problem(sim.domain, sim.u0, tspan)

    ode_solver = _get_ode_solver(sim.solver)

    save_opts = (; save_everystep=false, save_end=true)
    merged_opts = merge(save_opts, sim.solver_options)

    sim._integrator = OrdinaryDiffEq.init(prob, ode_solver; dt=sim.Δt, merged_opts...)

    _execute_callbacks!(sim)
    _write_outputs!(sim)

    while !_should_stop(sim)
        OrdinaryDiffEq.step!(sim._integrator)

        sim.iteration += 1
        sim.time = sim._integrator.t

        _execute_callbacks!(sim)
        _write_outputs!(sim)
    end

    sim._solution = copy(sim._integrator.u)
    return nothing
end

function _run_steady!(sim::Simulation; kwargs...)
    prob = LinearSolve.LinearProblem(sim.domain; kwargs...)
    sol = LinearSolve.solve(prob)

    sim._solution = sol.u
    sim.iteration = 1

    _execute_callbacks!(sim)
    _write_outputs!(sim)

    return nothing
end

function _create_ode_problem(domain::Domain, u0::AbstractVector, tspan)
    boundary_funcs = [
        make_bc(bc, domain.cloud[surf_name], domain, ids)
        for (surf_name, (ids, bc)) in domain.boundaries
    ]

    model_funcs = [make_f(m, domain) for m in domain.models]

    function f(du, u, p, t)
        for model in model_funcs
            model(du, u, p, t)
        end
        for bc in boundary_funcs
            bc(du, u, p, t)
        end
        return nothing
    end

    return ODEProblem(f, u0, tspan)
end

function _should_stop(sim::Simulation)
    if sim.stop_time !== nothing && sim.time >= sim.stop_time
        return true
    end
    if sim.stop_iteration > 0 && sim.iteration >= sim.stop_iteration
        return true
    end
    return false
end

function _get_ode_solver(solver::Symbol)
    if solver === :Euler
        return Euler()
    elseif solver === :RK4
        return RK4()
    elseif solver === :Tsit5
        return Tsit5()
    elseif solver === :DP5
        return DP5()
    elseif solver === :SSPRK33
        return SSPRK33()
    elseif solver === :SSPRK43
        return SSPRK43()
    else
        throw(ArgumentError("Unknown solver: $solver. Available: :Euler, :RK4, :Tsit5, :DP5, :SSPRK33, :SSPRK43"))
    end
end

function _initialize_callbacks!(sim::Simulation)
    for (_, callback) in sim.callbacks
        reset!(callback)
    end
    return nothing
end

function _execute_callbacks!(sim::Simulation)
    for (_, callback) in sim.callbacks
        execute!(callback, sim)
    end
    return nothing
end

function _initialize_output_writers!(sim::Simulation)
    for (_, writer) in sim.output_writers
        initialize!(writer, sim)
    end
    return nothing
end

function _write_outputs!(sim::Simulation)
    for (_, writer) in sim.output_writers
        write_output!(writer, sim)
    end
    return nothing
end

function _finalize_output_writers!(sim::Simulation)
    for (_, writer) in sim.output_writers
        finalize!(writer, sim)
    end
    return nothing
end

function Base.show(io::IO, sim::Simulation)
    print(io, "Simulation(")
    print(io, sim.mode == Transient ? "transient" : "steady-state")
    if sim.mode == Transient
        print(io, ", Δt=$(sim.Δt)")
        if sim.stop_time !== nothing
            print(io, ", stop_time=$(sim.stop_time)")
        end
        if sim.stop_iteration > 0
            print(io, ", stop_iteration=$(sim.stop_iteration)")
        end
        print(io, ", solver=$(sim.solver)")
    end
    print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", sim::Simulation)
    println(io, "Simulation")
    println(io, "├── Mode: ", sim.mode == Transient ? "Transient" : "Steady-state")
    if sim.mode == Transient
        println(io, "├── Δt: ", sim.Δt)
        println(io, "├── Stop time: ", sim.stop_time === nothing ? "none" : sim.stop_time)
        println(io, "├── Stop iteration: ", sim.stop_iteration == 0 ? "none" : sim.stop_iteration)
        println(io, "├── Solver: ", sim.solver)
    end
    println(io, "├── Iteration: ", sim.iteration)
    println(io, "├── Time: ", sim.time)
    println(io, "├── Callbacks: ", length(sim.callbacks))
    for (name, cb) in sim.callbacks
        println(io, "│   └── ", name, ": ", cb)
    end
    println(io, "├── Output writers: ", length(sim.output_writers))
    for (name, ow) in sim.output_writers
        println(io, "│   └── ", name, ": ", ow)
    end
    print(io, "└── Running: ", sim.running)
end
