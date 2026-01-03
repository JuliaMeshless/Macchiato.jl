@kwdef struct SolidEnergy{K, P, C, S} <: AbstractModel
    k::K
    ρ::P
    cₚ::C
    source::S = nothing  # Optional source term: f(x, t) -> value
end

_num_vars(::SolidEnergy, _) = 1

# Physics domain trait - imported from boundary_conditions/physics_domains.jl
physics_domain(::Type{<:SolidEnergy}) = EnergyPhysics()

function make_f(model::SolidEnergy, domain; neighbors = 40, kwargs...)
    (; k, ρ, cₚ, source) = model
    vol = _coords(domain.cloud.volume)
    all_points = _coords(domain.cloud)

    method = KNearestSearch(domain.cloud, neighbors)
    adjl = search.(points(domain.cloud.volume), Ref(method))

    ∇² = laplacian(_ustrip(all_points), _ustrip(vol); k = neighbors, adjl = adjl)
    update_weights!(∇²)
    α = k / (cₚ * ρ)
    w = α * ∇².weights

    start = maximum(domain.boundaries) do b
        b[2][1][end]
    end
    vol_ids = (start + 1):(start + length(vol))

    # Transient heat equation: ∂T/∂t = α∇²T + f/(ρcₚ)
    # Create single function that handles both cases to avoid method overwriting
    function f(du, u, p, t)
        mul!(view(du, vol_ids), w, u)

        # Add source term contribution if present
        if source !== nothing
            for (i, pt) in enumerate(vol)
                x = [ustrip(pt.x), ustrip(pt.y), ustrip(pt.z)]
                du[vol_ids[i]] += source(x, t) / (ρ * cₚ)
            end
        end
        return nothing
    end

    return f
end

function make_system(model::SolidEnergy, domain; kwargs...)
    (; k, ρ, cₚ, source) = model
    coords = _coords(domain.cloud)
    ∇² = laplacian(_ustrip(coords); k = 40, kwargs...)
    update_weights!(∇²)
    α = k / (cₚ * ρ)
    A = α * ∇².weights

    # Compute RHS from source term
    # Steady-state heat equation: ∇²T = f/α → (α∇²)T = f
    if source === nothing
        b = zeros(eltype(A), length(coords))
    else
        # Evaluate source at each point
        b = map(coords) do pt
            # x = [ustrip(pt.x), ustrip(pt.y), ustrip(pt.z)]
            source(ustrip.(pt), 0.0)  # Steady-state: t=0
        end
    end

    return A, b
end

function Base.show(io::IO, e::SolidEnergy)
    source_str = e.source === nothing ? "" : ", source"
    print(io, "Energy: (k = $(e.k), ρ = $(e.ρ), cₚ = $(e.cₚ)$(source_str))")
end
