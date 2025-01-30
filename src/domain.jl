struct Domain{M <: Manifold, C <: CRS}
    cloud::PointCloud{M, C}
    boundaries::Dict{Symbol, <:Tuple{UnitRange, AbstractBoundaryCondition}}
    models::AbstractVector{<:AbstractModel}
    name::Symbol
end

function Domain(cloud::PointCloud, boundaries, models)
    for bc_surf_name in keys(boundaries)
        if !hassurface(cloud.boundary, bc_surf_name)
            throw(ArgumentError("The boundary condition $bc_name is not associated with a `PointCloud` boundary."))
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

    display(ids_bcs)

    return Domain(cloud, Dict(ids_bcs), models, :domain1)
end

function Domain(cloud::PointCloud{M, C}, models) where {M <: Manifold, C <: CRS}
    boundaries = Dict{Symbol, AbstractBoundaryCondition}()
    models = models isa Vector ? models : [models]
    return Domain(cloud, boundaries, models, :domain1)
end

add!(domain::Domain, model::AbstractModel) = push!(domain.models, model)
function add!(
        domain::Domain, boundary::AbstractBoundaryCondition, name::Symbol)
    domain.boundaries[name] = boundary
end

function delete!(domain::Domain, model::AbstractModel)
    deleteat!(domain.models, findall(x -> x == model, domain.models))
end

function Base.show(io::IO, domain::Domain)
    print(io, "$(domain.name): Domain")
    println()
    show(io, domain.models)
end
