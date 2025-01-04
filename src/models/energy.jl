@kwdef struct SolidEnergy{K, P, C} <: AbstractModel
    k::K
    ρ::P
    cₚ::C
end

_num_vars(::SolidEnergy, _) = 1

function make_f(model::SolidEnergy, domain; kwargs...)
    (; k, ρ, cₚ) = model
    vol = _coords(domain.cloud.volume)
    all_points = _coords(domain.cloud)

    method = KNearestSearch(domain.cloud, 40)
    adjl = search.(domain.cloud.volume.points, Ref(method))

    ∇² = laplacian(_ustrip(all_points), _ustrip(vol); k = 40, adjl = adjl)
    update_weights!(∇²)
    α = k / (cₚ * ρ)
    w = α * ∇².weights
    start = maximum(domain.boundaries) do b
        b[2][1][end]
    end
    vol_ids = (start + 1):(start + length(vol))
    @show vol_ids

    function f(du, u, p, t)
        mul!(view(du, vol_ids), w, u)
        return nothing
    end

    return f
end

function make_system(model::SolidEnergy, domain; kwargs...)
    (; k, ρ, cₚ) = model
    coords = _coords(domain.cloud)
    ∇² = laplacian(coords; kwargs...)
    update_weights!(∇²)
    α = k / (cₚ * ρ)
    A = α * ∇².weights
    b = zeros(eltype(A), length(coords))
    return A, b
end

function Base.show(io::IO, e::SolidEnergy)
    print(io, "Energy: (k = $(e.k), ρ = $(e.ρ), cₚ = $(e.cₚ))")
end
