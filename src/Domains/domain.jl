struct Domain{Dim, T}
    cloud::PointCloud{Dim, T}
    boundaries::Dict{Symbol, <:AbstractBoundary}
    models::AbstractVector{<:AbstractModel}
    name::Symbol
end

function Domain(cloud::PointCloud, boundaries, models)
    for bc_name in keys(boundaries)
        if !haskey(cloud.surfaces, bc_name)
            throw(ArgumentError("The boundary condition $bc_name is not associated with a `PointCloud` surface."))
        end
    end
    #_build_ops!(cloud, boundaries)
    models = models isa Vector ? models : [models]
    return Domain(cloud, boundaries, models, :domain1)
end

function Domain(cloud::PointCloud{Dim, T}, models) where {Dim, T}
    boundaries = Dict{Symbol, AbstractBoundary}()
    models = models isa Vector ? models : [models]
    return Domain(cloud, boundaries, models, :domain1)
end

add_model!(domain::Domain, model::AbstractModel) = push!(domain.models, model)
function add_boundary_condition!(domain::Domain, boundary::AbstractBoundary, name::Symbol)
    domain.boundaries[name] = boundary
end

function rm_model!(domain::Domain, model::AbstractModel)
    deleteat!(domain.models, findall(x -> x == model, domain.models))
end

#function _build_ops!(cloud::PointCloud, boundaries)
#    for (name, boundary) in boundaries
#    end
#end

function Base.show(io::IO, domain::Domain)
    print(io, "$(domain.name): Domain")
    println()
    show(io, domain.models)
end
