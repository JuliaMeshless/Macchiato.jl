"""
    EnergyBoundaryCondition <: AbstractBoundaryCondition

Abstract type for energy boundary conditions.
"""
abstract type EnergyBoundaryCondition <: AbstractBoundaryCondition end

# ============================================================================
# Temperature (Dirichlet for energy equation)
# ============================================================================

"""
    Temperature{T} <: EnergyBoundaryCondition

Temperature boundary condition - Dirichlet type (α=1, β=0).
Prescribes the temperature value at the boundary.
"""
struct Temperature{T} <: EnergyBoundaryCondition
    temperature::T
end

# Accessor
(bc::Temperature)() = bc.temperature
(bc::Temperature{<:Function})(x, t) = bc.temperature(x, t)

# Helpers
bc_type(::Temperature) = Dirichlet()
bc_value(bc::Temperature) = bc.temperature
bc_value(bc::Temperature{<:Function}, x, t) = bc.temperature(x, t)

# Time evolution
function make_bc(boundary::Temperature, surf, domain, ids; kwargs...)
    bc(du, u, p, t) = (u[ids] .= bc_value(boundary, nothing, t); nothing)
end

# Linear system
function make_bc!(A, b, boundary::Temperature, surf, domain, ids; kwargs...)
    (apply_bc!(A, b, bc_type(boundary), surf, domain, ids, bc_value(boundary)); A)
end

Base.show(io::IO, bc::Temperature) = print(io, "Temperature: $(bc.temperature)")

# ============================================================================
# HeatFlux (Neumann for energy equation)
# ============================================================================

"""
    HeatFlux{T} <: EnergyBoundaryCondition

Heat flux boundary condition - Neumann type (α=0, β=1).
Prescribes the heat flux (normal derivative of temperature) at the boundary.
"""
struct HeatFlux{T} <: EnergyBoundaryCondition
    heat_flux::T
end

# Helpers
bc_type(::HeatFlux) = Neumann()
bc_value(bc::HeatFlux) = bc.heat_flux

# Linear system
function make_bc!(A, b, boundary::HeatFlux, surf, domain, ids; kwargs...)
    (apply_bc!(A, b, bc_type(boundary), surf, domain, ids, bc_value(boundary); kwargs...); A)
end

Base.show(io::IO, bc::HeatFlux) = print(io, "HeatFlux: $(bc.heat_flux)")

# ============================================================================
# Convection (Robin for energy equation)
# ============================================================================

"""
    Convection{C, T} <: EnergyBoundaryCondition

Convection boundary condition - Robin type.
Represents heat transfer to surrounding fluid: q = h*(T - T∞)
This is a Robin condition: h*T + k*∂ₙT = h*T∞
"""
struct Convection{C, T} <: EnergyBoundaryCondition
    coefficient::T  # h (heat transfer coefficient)
    T∞::T  # Ambient temperature

    function Convection(coefficient::C, T∞::T) where {C, T}
        coefficient < 0 && throw(ArgumentError("The coefficient must be non-negative."))
        return new{C, T}(coefficient, T∞)
    end
end

# Helpers
bc_type(bc::Convection) = Robin(bc.coefficient, 1.0)
bc_value(bc::Convection) = bc.coefficient * bc.T∞

# Linear system
function make_bc!(A, b, boundary::Convection, surf, domain, ids; kwargs...)
    (apply_bc!(A, b, bc_type(boundary), surf, domain, ids, bc_value(boundary); kwargs...); A)
end

function Base.show(io::IO, bc::Convection)
    print(io, "Convection: coeff=$(bc.coefficient), T∞=$(bc.T∞)")
end

# ============================================================================
# Adiabatic (Neumann with zero flux)
# ============================================================================

"""
    Adiabatic <: EnergyBoundaryCondition

Adiabatic boundary condition - Neumann type with zero heat flux (α=0, β=1, g=0).
Represents a thermally insulated boundary where ∂ₙT = 0.

# Fields
- `op::T`: Operator type to use when calculating the normal gradient (e.g., ShadowPoints)
"""
struct Adiabatic{T} <: EnergyBoundaryCondition
    op::T
end

# Constructors
Adiabatic() = Adiabatic(nothing)
Adiabatic(Δ::Number) = Adiabatic(ShadowPoints(Δ, 1))
Adiabatic(Δ::Number, order::T) where {T <: Int} = Adiabatic(ShadowPoints(Δ, order))

# Helpers
bc_type(::Adiabatic) = Neumann()
bc_value(::Adiabatic) = 0.0  # Zero flux

# Time evolution - specialized implementations
function make_bc(boundary::Adiabatic{<:ShadowPoints}, surf, domain, ids; kwargs...)
    shadow_points = generate_shadows(surf, boundary.op)
    coords = _coords(domain.cloud)
    method = KNearestSearch(domain.cloud, 40)
    adjl = search.(shadow_points, Ref(method))
    d = regrid(_ustrip(coords), _ustrip(_coords(shadow_points)); adjl = adjl)
    update_weights!(d)

    function bc(du, u, p, t)
        u[ids] .= d(u)
        return nothing
    end
    return bc
end
function make_bc(boundary::Adiabatic, surf, domain, ids; kwargs...)
    println("creating Adiabatic BC")
    d = directional(_coords(domain.cloud), _coords(surf), normals(surf); kwargs...)
    update_weights!(d)
    w = d.weights
    wi = diag(w)
    w[diagind(w)] .= 0
    dropzeros!(w)

    function bc(du, u, p, t)
        # TODO is this correct?
        #du[surf_ids] .= d(u) .- u[surf_ids]
        u[ids] .= (w * u) ./ wi
        return nothing
    end
    return bc
end

function cone(cloud, surf, k)
    all_points = _coords(cloud)
    surf_points = _coords(surf)
    normal = normals(surf)
    offset = first(only(surf.points.indices))

    tree = KDTree(all_points)
    adjl, _ = knn(tree, surf_points, k, true)

    for (i, neighbors) in enumerate(adjl)
        O = all_points[first(neighbors)]
        n = -normal[first(neighbors) - offset + 1]
        L = 0
        new_k = k
        local new_neighbors
        while L < k
            a, _ = knn(tree, O, new_k, true)
            new_neighbors = filter(a) do i
                v = all_points[i] - O
                abs(∠(v, n)) < (56 * π / 180)
            end
            L = length(new_neighbors)
            new_k += 10
        end
        adjl[i] = new_neighbors
    end
    return adjl
end

function make_bc!(
        A::AbstractMatrix{TA}, b::AbstractVector{TB}, boundary::Adiabatic,
        surf, domain, ids; kwargs...) where {TA, TB}
    apply_bc!(A, b, boundary, surf, domain, ids, bc_value(boundary))
end

function columnwise_div(A::SparseMatrixCSC, B::AbstractVector)
    I, J, V = findnz(A)
    for idx in eachindex(V)
        V[idx] /= B[I[idx]]
    end
    return sparse(I, J, V)
end
columnwise_div(A::SparseMatrixCSC, B::Number) = A ./ B

function replace_rows(A, weights, ids, offset)
    I, J, V = findnz(A)
    I2, J2, V2 = findnz(weights)
    i = findall(i -> i ∈ ids, I)
    i2 = findall(i -> i ∈ (ids .- offset), I2)
    deleteat!(I, i)
    deleteat!(J, i)
    deleteat!(V, i)
    append!(I, I2[i2] .+ offset)
    append!(J, J2[i2])
    append!(V, V2[i2])

    return sparse(I, J, V)
end

function make_bc!(
        A::AbstractMatrix{TA}, b::AbstractVector{TB}, boundary::Adiabatic{<:ShadowPoints},
        surf, domain, ids; kwargs...) where {TA, TB}
    shadow_points = generate_shadows(surf, boundary.op)
    coords = _ustrip(_coords(domain.cloud))
    method = KNearestSearch(domain.cloud, 40)
    adjl = search.(shadow_points, Ref(method))

    println("building surf")
    @time surf = regrid(coords, _ustrip(_coords(surf)); adjl = adjl, kwargs...)
    @time update_weights!(surf)

    println("building shadow")
    @time shadow = regrid(coords, _ustrip(_coords(shadow_points)); kwargs...)
    @time update_weights!(shadow)
    weights = columnwise_div(surf.weights .- shadow.weights, ustrip(boundary.op.Δ(1)))

    offset = first(ids) - 1
    println("zeroing")
    @time b[ids] .= zero(TB)

    println("placing weights")
    @time A = replace_rows(A, weights, ids, offset)
    return A
end

function make_bc!(
        A::AbstractMatrix{TA}, b::AbstractVector{TB}, boundary::Adiabatic{<:ShadowPoints{2}},
        surf, domain, ids; kwargs...) where {TA, TB}
    coords = _coords(domain.cloud)
    shadow_points1 = generate_shadows(surf, boundary.op)
    shadow_points2 = generate_shadows(
        surf, ShadowPoints(ConstantSpacing(boundary.op.Δ.Δx * 2)))

    println("building surf")
    surf = regrid(coords, _coords(surf); kwargs...)
    @time update_weights!(surf)

    println("building shadow 1")
    shadow1 = regrid(coords, shadow_points1; kwargs...)
    @time update_weights!(shadow1)

    println("building shadow 2")
    shadow2 = regrid(coords, shadow_points2; kwargs...)
    @time update_weights!(shadow2)

    num = 3 * surf.weights .- 4 * shadow1.weights .+ shadow2.weights
    weights = columnwise_div(num, 2 * boundary.op.Δ.Δx)

    offset = first(ids) - 1
    println("zeroing")
    @time b[ids] .= zero(TB)

    println("placing weights")
    @time A = replace_rows(A, weights, ids, offset)
    return A
end

function add_shadows!(cloud, surf, shadow)
    s = cloud[surf]
    delete!(cloud.surfaces, surf)
    cloud[surf] = s(shadow)

    shadow_points = generate_shadows(cloud[surf], shadow)

    # add shadow points to the cloud
    append!(cloud.points, shadow_points)
    cloud_length = length(cloud.points)
    shadow_ids = (cloud_length - length(shadow_points) + 1):cloud_length
    vol_ids = only(cloud.volume.points.indices)
    new_ids = first(vol_ids):(last(vol_ids) + length(shadow_points))
    cloud.volume = PointVolume(view(cloud.points, new_ids))

    return shadow_ids
end

Base.show(io::IO, ::Adiabatic) = print(io, "Adiabatic")
