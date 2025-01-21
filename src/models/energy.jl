@kwdef struct SolidEnergy{K, P, C} <: AbstractModel
    k::K
    ρ::P
    cₚ::C
end

_num_vars(::SolidEnergy, _) = 1

function make_f(model::SolidEnergy, domain; neighbors = 40, kwargs...)
    (; k, ρ, cₚ) = model
    #vol = _coords(domain.cloud.volume)
    all_points = _coords(domain.cloud)

    #method = KNearestSearch(domain.cloud, neighbors)
    #adjl = search.(domain.cloud.volume.points, Ref(method))

    #∇² = laplacian(_ustrip(all_points), _ustrip(vol); k = neighbors, adjl = adjl)
    ∇² = laplacian(_ustrip(all_points); k = neighbors)
    update_weights!(∇²)
    α = k / (cₚ * ρ)
    w = α * ∇².weights

    #=
    start = maximum(domain.boundaries) do b
        b[2][1][end]
    end
    vol_ids = (start + 1):(start + length(vol))
    =#

    function f(du, u, p, t)
        mul!(du, w, u)
        return nothing
    end

    return f
end

function make_system(model::SolidEnergy, domain; kwargs...)
    (; k, ρ, cₚ) = model
    coords = _coords(domain.cloud)
    ∇² = laplacian(_ustrip(coords); k = 40, kwargs...)
    update_weights!(∇²)
    α = k / (cₚ * ρ)
    A = α * ∇².weights
    b = zeros(eltype(A), length(coords))
    return A, b
end

function Base.show(io::IO, e::SolidEnergy)
    print(io, "Energy: (k = $(e.k), ρ = $(e.ρ), cₚ = $(e.cₚ))")
end
