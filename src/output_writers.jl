"""
    AbstractOutputWriter

Abstract type for output writers. Implementations must define:
- `initialize!(writer, sim)` - called before simulation starts
- `write_output!(writer, sim)` - called according to schedule
- `finalize!(writer, sim)` - called after simulation ends
"""
abstract type AbstractOutputWriter end

"""
    initialize!(writer::AbstractOutputWriter, sim)

Initialize the output writer before simulation starts.
Default implementation does nothing.
"""
initialize!(::AbstractOutputWriter, sim) = nothing

"""
    write_output!(writer::AbstractOutputWriter, sim)

Write output for the current simulation state.
Must be implemented by concrete types.
"""
function write_output! end

"""
    finalize!(writer::AbstractOutputWriter, sim)

Finalize the output writer after simulation ends.
Default implementation does nothing.
"""
finalize!(::AbstractOutputWriter, sim) = nothing

"""
    VTKOutputWriter{S<:AbstractSchedule}

Writes simulation output to VTK format for visualization in ParaView.

# Fields
- `prefix::String`: Output file prefix (directory/filename without extension)
- `schedule::S`: When to write output
- `fields::Vector{Symbol}`: Which fields to output (empty = all fields)
- `_state::ScheduleState`: Internal state for schedule tracking
- `_pvd::Any`: ParaView collection file handle
- `_output_count::Int`: Number of outputs written

# Example
```julia
writer = VTKOutputWriter("results/sim", schedule=TimeInterval(0.1))
writer = VTKOutputWriter("results/sim", schedule=IterationInterval(100), fields=[:T])
```
"""
mutable struct VTKOutputWriter{S<:AbstractSchedule} <: AbstractOutputWriter
    prefix::String
    schedule::S
    fields::Vector{Symbol}
    _state::ScheduleState
    _pvd::Any
    _output_count::Int
end

function VTKOutputWriter(prefix::String; schedule::S=TimeInterval(1.0), fields::Vector{Symbol}=Symbol[]) where {S<:AbstractSchedule}
    dir = dirname(prefix)
    if !isempty(dir) && !isdir(dir)
        mkpath(dir)
    end
    VTKOutputWriter{S}(prefix, schedule, fields, ScheduleState(), nothing, 0)
end

function initialize!(writer::VTKOutputWriter, sim)
    writer._pvd = createpvd(writer.prefix; append=false)
    writer._output_count = 0
    writer._state.last_iteration = 0
    writer._state.last_time = 0.0
    writer._state.last_wall_time = time()
    writer._state.next_specified_idx = 1
    return nothing
end

function write_output!(writer::VTKOutputWriter, sim)
    if !should_execute(writer.schedule, writer._state, sim.iteration, sim.time)
        return false
    end

    cloud = sim.domain.cloud
    coords = reduce(hcat, _ustrip(_coords(cloud)))
    cells = createvtkcells(coords)

    filename = "$(writer.prefix)_$(lpad(writer._output_count, 6, '0'))"
    vtkfile = createvtkfile(filename, coords, cells)

    field_data = _get_field_data(sim, writer.fields)
    for (name, data) in field_data
        addfieldvtk!(vtkfile, String(name), data)
    end

    pvdappend!(writer._pvd, sim.time, vtkfile)
    savevtk!(vtkfile)

    writer._output_count += 1
    return true
end

function finalize!(writer::VTKOutputWriter, sim)
    if writer._pvd !== nothing
        vtk_save(writer._pvd)
        writer._pvd = nothing
    end
    return nothing
end

function _get_field_data(sim, requested_fields::Vector{Symbol})
    result = Pair{Symbol, Vector{Float64}}[]

    if isempty(requested_fields)
        for model in sim.domain.models
            append!(result, _model_fields(model, sim))
        end
    else
        for field in requested_fields
            data = _get_single_field(sim, field)
            if data !== nothing
                push!(result, field => data)
            end
        end
    end

    return result
end

function _model_fields(model::SolidEnergy, sim)
    T = temperature(sim)
    return [(:T => T)]
end

function _model_fields(model::IncompressibleNavierStokes, sim)
    result = Pair{Symbol, Vector{Float64}}[]
    u, v = velocity(sim)
    p = pressure(sim)
    push!(result, :u => u)
    push!(result, :v => v)
    push!(result, :p => p)
    return result
end

function _get_single_field(sim, field::Symbol)
    if field === :T || field === :temperature
        return temperature(sim)
    elseif field === :u
        u, _ = velocity(sim)
        return u
    elseif field === :v
        _, v = velocity(sim)
        return v
    elseif field === :p || field === :pressure
        return pressure(sim)
    end
    return nothing
end

function Base.show(io::IO, w::VTKOutputWriter)
    fields_str = isempty(w.fields) ? "all" : join(w.fields, ", ")
    print(io, "VTKOutputWriter(\"$(w.prefix)\", $(w.schedule), fields=[$fields_str])")
end

"""
    JLD2OutputWriter{S<:AbstractSchedule}

Writes simulation checkpoints to JLD2 format for restart capability.

# Fields
- `prefix::String`: Output file prefix
- `schedule::S`: When to write checkpoints
- `_state::ScheduleState`: Internal state for schedule tracking
- `_output_count::Int`: Number of checkpoints written

# Example
```julia
writer = JLD2OutputWriter("checkpoints/sim", schedule=WallTimeInterval(300.0))
```
"""
mutable struct JLD2OutputWriter{S<:AbstractSchedule} <: AbstractOutputWriter
    prefix::String
    schedule::S
    _state::ScheduleState
    _output_count::Int
end

function JLD2OutputWriter(prefix::String; schedule::S=TimeInterval(1.0)) where {S<:AbstractSchedule}
    dir = dirname(prefix)
    if !isempty(dir) && !isdir(dir)
        mkpath(dir)
    end
    JLD2OutputWriter{S}(prefix, schedule, ScheduleState(), 0)
end

function initialize!(writer::JLD2OutputWriter, sim)
    writer._output_count = 0
    writer._state.last_iteration = 0
    writer._state.last_time = 0.0
    writer._state.last_wall_time = time()
    writer._state.next_specified_idx = 1
    return nothing
end

function write_output!(writer::JLD2OutputWriter, sim)
    if !should_execute(writer.schedule, writer._state, sim.iteration, sim.time)
        return false
    end

    filename = "$(writer.prefix)_$(lpad(writer._output_count, 6, '0')).jld2"

    JLD2.jldsave(filename;
        u=_get_solution_vector(sim),
        iteration=sim.iteration,
        time=sim.time,
        Δt=sim.Δt,
        stop_time=sim.stop_time
    )

    writer._output_count += 1
    return true
end

function finalize!(writer::JLD2OutputWriter, sim)
    return nothing
end

function Base.show(io::IO, w::JLD2OutputWriter)
    print(io, "JLD2OutputWriter(\"$(w.prefix)\", $(w.schedule))")
end
