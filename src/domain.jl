struct Domain{M <: Manifold, C <: CRS}
    cloud::PointCloud{M, C}
    boundaries::Dict{Symbol, <:Tuple{UnitRange, AbstractBoundaryCondition}}
    models::AbstractVector{<:AbstractModel}
    name::Symbol
end

function Domain(cloud::PointCloud, boundaries, models)
    for bc_name in keys(boundaries)
        if !haskey(boundary(cloud).surfaces, bc_name)
            throw(ArgumentError("The boundary condition $bc_name is not associated with a `PointCloud` boundary."))
        end
    end
    # make sure it is a vector so we can push! to it later
    models = models isa Vector ? models : [models]

    # assign ids to each boundary condition
    a, b = Iterators.peel(zip(keys(boundaries), values(boundaries)))
    init = first(a) => (1:length(cloud[first(a)]), a[2])
    rest = accumulate(b; init = init) do x, y
        ids = x.second[1][end] .+ (1:length(cloud[y[1]]))
        y[1] => (ids, y[2])
    end
    ids_bcs = vcat([init], rest)

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
