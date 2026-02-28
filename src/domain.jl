"""
    Domain{M, C}

Central container that ties together a point cloud, boundary conditions, and physics models.

# Fields
- `cloud::PointCloud{M, C}`: The discretized geometry (boundary + interior points)
- `boundaries::Dict{Symbol, Tuple{UnitRange, AbstractBoundaryCondition}}`: Mapping from surface
  names to `(index_range, bc)` pairs, where `index_range` gives the global indices of that
  surface's points in the assembled system
- `models::AbstractVector{<:AbstractModel}`: Physics models (e.g., `SolidEnergy`, `LinearElasticity`)
- `name::Symbol`: Domain identifier (defaults to `:domain1`)

# Constructors
```julia
Domain(cloud, boundaries, model)    # cloud + BCs + model(s)
Domain(cloud, model)                # cloud + model(s), no BCs
```

At construction, the `Domain` validates that:
1. Every boundary condition key matches a surface name in the point cloud
"""
struct Domain{M <: Manifold, C <: CRS}
    cloud::PointCloud{M, C}
    boundaries::Dict{Symbol, <:Tuple{UnitRange, AbstractBoundaryCondition}}
    models::AbstractVector{<:AbstractModel}
    name::Symbol
end

"""
    Domain(cloud::PointCloud, boundaries::Dict{Symbol, <:AbstractBoundaryCondition}, models)

Construct a `Domain` from a point cloud, a dictionary mapping surface names to boundary
conditions, and one or more physics models. Validates BC–surface and BC–model compatibility.
"""
function Domain(cloud::PointCloud, boundaries, models)
    for bc_surf_name in keys(boundaries)
        if !hassurface(cloud.boundary, bc_surf_name)
            throw(ArgumentError("The boundary condition $bc_surf_name is not associated with a `PointCloud` boundary."))
        end
    end
    # make sure it is a vector so we can push! to it later
    models = models isa Vector ? models : [models]

    ids_bcs = Dict{Symbol, Tuple{UnitRange, AbstractBoundaryCondition}}()
    offset = 0
    for surf_name in names(cloud.boundary)
        N = length(cloud[surf_name])
        ids = offset .+ (1:N)
        offset += N
        ids_bcs[surf_name] = (ids, boundaries[surf_name])
    end

    return Domain(cloud, Dict(ids_bcs), models, :domain1)
end

"""
    Domain(cloud::PointCloud, models)

Construct a `Domain` without boundary conditions — useful for interior-only problems
or when BCs will be added later via [`add!`](@ref).
"""
function Domain(cloud::PointCloud{M, C}, models) where {M <: Manifold, C <: CRS}
    boundaries = Dict{Symbol, AbstractBoundaryCondition}()
    models = models isa Vector ? models : [models]
    return Domain(cloud, boundaries, models, :domain1)
end

"""
    add!(domain::Domain, model::AbstractModel)

Append a physics model to the domain's model list.
"""
add!(domain::Domain, model::AbstractModel) = push!(domain.models, model)

"""
    add!(domain::Domain, boundary::AbstractBoundaryCondition, name::Symbol)

Attach a boundary condition to the named surface on the domain.
"""
function add!(
        domain::Domain, boundary::AbstractBoundaryCondition, name::Symbol)
    domain.boundaries[name] = boundary
end

"""
    delete!(domain::Domain, model::AbstractModel)

Remove a physics model from the domain.
"""
function delete!(domain::Domain, model::AbstractModel)
    deleteat!(domain.models, findall(x -> x == model, domain.models))
end

function Base.show(io::IO, domain::Domain)
    print(io, "$(domain.name): Domain")
    println()
    show(io, domain.models)
end
