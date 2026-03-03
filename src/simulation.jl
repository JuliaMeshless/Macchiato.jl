"""
    Simulation{M, C, Mode}

High-level simulation container that manages solving and solution storage.

# Fields
- `domain::Domain{M, C}`: The computational domain with models and boundary conditions
- `mode::Mode`: Simulation mode ([`Steady`](@ref) or [`Transient`](@ref))
- `u0`: Initial condition vector (transient only)
- `time`: Current simulation time
- `running`: Whether simulation is currently running
- `_solution`: Solution vector

# Constructors

## Steady-state simulation
```julia
sim = Simulation(domain)
sim = Simulation(domain, Steady())
```

## Transient simulation
```julia
sim = Simulation(domain, Transient(Δt=0.001, stop_time=1.0))
```
"""
mutable struct Simulation{M, C, Mode <: AbstractSimulationMode}
    domain::Domain{M, C}
    mode::Mode
    u0::Union{Nothing, Vector{Float64}}
    time::Float64
    running::Bool
    _solution::Union{Nothing, Vector{Float64}}
end

function Simulation(domain::Domain{M, C}, mode::AbstractSimulationMode=Steady()) where {M, C}
    return Simulation{M, C, typeof(mode)}(domain, mode, nothing, 0.0, false, nothing)
end

"""
    run!(sim::Simulation; kwargs...)

Execute the simulation. Dispatches to transient or steady-state path based on mode.
Extra `kwargs` are forwarded to `OrdinaryDiffEq.solve` (transient) or `LinearSolve.LinearProblem` (steady).

Returns the simulation object.
"""
function run!(sim::Simulation; kwargs...)
    sim.running = true
    try
        _run!(sim, sim.mode; kwargs...)
    finally
        sim.running = false
    end
    return sim
end

function _run!(sim::Simulation, mode::Transient; kwargs...)
    _ensure_u0_initialized!(sim)
    tspan = (0.0, mode.stop_time)
    prob = _create_ode_problem(sim.domain, sim.u0, tspan)
    sol = OrdinaryDiffEq.solve(
        prob, mode.solver;
        dt=mode.Δt, save_everystep=false, save_end=true, kwargs...
    )
    sim._solution = copy(sol.u[end])
    sim.time = sol.t[end]
    return nothing
end

function _run!(sim::Simulation, ::Steady; kwargs...)
    prob = LinearSolve.LinearProblem(sim.domain; kwargs...)
    sol = LinearSolve.solve(prob)
    sim._solution = sol.u
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

function Base.show(io::IO, sim::Simulation{M, C, <:Steady}) where {M, C}
    return print(io, "Simulation(steady-state)")
end

function Base.show(io::IO, sim::Simulation{M, C, <:Transient}) where {M, C}
    return print(io, "Simulation(transient, Δt=$(sim.mode.Δt), stop_time=$(sim.mode.stop_time))")
end

function Base.show(io::IO, ::MIME"text/plain", sim::Simulation{M, C, <:Steady}) where {M, C}
    println(io, "Simulation")
    println(io, "├── Mode: Steady-state")
    println(io, "├── Time: ", sim.time)
    return print(io, "└── Running: ", sim.running)
end

function Base.show(io::IO, ::MIME"text/plain", sim::Simulation{M, C, <:Transient}) where {M, C}
    println(io, "Simulation")
    println(io, "├── Mode: Transient")
    println(io, "├── Δt: ", sim.mode.Δt)
    println(io, "├── Stop time: ", sim.mode.stop_time)
    println(io, "├── Solver: ", typeof(sim.mode.solver))
    println(io, "├── Time: ", sim.time)
    return print(io, "└── Running: ", sim.running)
end
