"""
    set!(sim::Simulation; kwargs...)

Set initial conditions for simulation fields.

# Arguments
- `sim`: Simulation to set initial conditions for
- `kwargs`: Field name/value pairs

# Supported value types
- `Number`: Uniform value for entire field
- `Function`: Called as `f(x)` where `x` is coordinate vector [x, y] or [x, y, z]
- `Vector`: Direct assignment (must match field length)

# Examples
```julia
set!(sim, T=300.0)                           # Uniform temperature
set!(sim, T=x -> 300 + 10*x[1])              # Temperature function of position
set!(sim, u=0.0, v=0.0, p=0.0)               # Multiple fields
```
"""
function set!(sim; kwargs...)
    _ensure_u0_initialized!(sim)

    for (field, value) in kwargs
        _set_field!(sim, field, value)
    end
    return sim
end

function _ensure_u0_initialized!(sim)
    if sim.u0 === nothing
        n_points = length(sim.domain.cloud)
        dim = _get_dimension(sim.domain)
        n_vars = _num_vars(sim.domain.models, dim)
        sim.u0 = zeros(n_points * n_vars)
    end
    return nothing
end

function _get_dimension(domain::Domain)
    pt = first(_coords(domain.cloud))
    return length(pt)
end

"""
    _set_field!(sim, field::Symbol, value)

Set a specific field to the given value.
"""
function _set_field!(sim, field::Symbol, value::Number)
    indices = _field_indices(sim, field)
    sim.u0[indices] .= value
    return nothing
end

function _set_field!(sim, field::Symbol, value::Function)
    indices = _field_indices(sim, field)
    coords = _coords(sim.domain.cloud)

    for (i, idx) in enumerate(indices)
        pt = coords[i]
        x = _point_to_vector(pt)
        sim.u0[idx] = value(x)
    end
    return nothing
end

function _set_field!(sim, field::Symbol, value::AbstractVector)
    indices = _field_indices(sim, field)
    length(value) == length(indices) || throw(DimensionMismatch(
        "Vector length $(length(value)) does not match field size $(length(indices))"))
    sim.u0[indices] .= value
    return nothing
end

function _point_to_vector(pt::SVector{N}) where {N}
    return Float64[ustrip(pt[i]) for i in 1:N]
end

"""
    _field_indices(sim, field::Symbol) -> UnitRange{Int}

Return the indices in the solution vector corresponding to the given field.

Field layout in solution vector:
- SolidEnergy (1 var): [T₁, T₂, ..., Tₙ]
- NavierStokes 2D (3 vars): [u₁..uₙ, v₁..vₙ, p₁..pₙ]
- NavierStokes 3D (4 vars): [u₁..uₙ, v₁..vₙ, w₁..wₙ, p₁..pₙ]
"""
function _field_indices(sim, field::Symbol)
    n_points = length(sim.domain.cloud)
    dim = _get_dimension(sim.domain)

    var_offset = 0
    for model in sim.domain.models
        field_idx = _field_index_in_model(model, field, dim)
        if field_idx !== nothing
            start = var_offset * n_points + 1
            stop = start + n_points - 1
            return (start + (field_idx - 1) * n_points):(start + field_idx * n_points - 1)
        end
        var_offset += _num_vars(model, dim)
    end

    throw(ArgumentError("Unknown field: $field"))
end

function _field_index_in_model(model::SolidEnergy, field::Symbol, dim)
    if field === :T || field === :temperature
        return 1
    end
    return nothing
end

function _field_index_in_model(model::IncompressibleNavierStokes, field::Symbol, dim)
    if field === :u
        return 1
    elseif field === :v
        return 2
    elseif field === :w && dim == 3
        return 3
    elseif field === :p || field === :pressure
        return dim + 1
    end
    return nothing
end

"""
    temperature(sim) -> Vector{Float64}

Extract temperature field from simulation.
"""
function temperature(sim)
    _has_field(sim, :T) || throw(ArgumentError("Simulation does not have temperature field"))
    indices = _field_indices(sim, :T)
    return _get_solution_vector(sim)[indices]
end

"""
    velocity(sim) -> Tuple{Vector{Float64}, ...}

Extract velocity components from simulation.
Returns (u, v) for 2D or (u, v, w) for 3D.
"""
function velocity(sim)
    _has_field(sim, :u) || throw(ArgumentError("Simulation does not have velocity field"))
    dim = _get_dimension(sim.domain)
    u = _get_solution_vector(sim)[_field_indices(sim, :u)]
    v = _get_solution_vector(sim)[_field_indices(sim, :v)]
    if dim == 3
        w = _get_solution_vector(sim)[_field_indices(sim, :w)]
        return (u, v, w)
    end
    return (u, v)
end

"""
    pressure(sim) -> Vector{Float64}

Extract pressure field from simulation.
"""
function pressure(sim)
    _has_field(sim, :p) || throw(ArgumentError("Simulation does not have pressure field"))
    indices = _field_indices(sim, :p)
    return _get_solution_vector(sim)[indices]
end

function _has_field(sim, field::Symbol)
    dim = _get_dimension(sim.domain)
    for model in sim.domain.models
        if _field_index_in_model(model, field, dim) !== nothing
            return true
        end
    end
    return false
end

function _get_solution_vector(sim)
    if sim._solution !== nothing
        return sim._solution
    elseif sim._integrator !== nothing
        return sim._integrator.u
    elseif sim.u0 !== nothing
        return sim.u0
    else
        throw(ErrorException("No solution available"))
    end
end
