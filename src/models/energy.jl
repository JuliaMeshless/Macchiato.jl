"""
    SolidEnergy(; k, ρ, cₚ, source=nothing)

Solid-body energy (heat) transport model.

Solves the heat equation in a solid medium:
- **Steady-state**: `k ∇²T = -f` (Poisson equation)
- **Transient**: `ρ cₚ ∂T/∂t = k ∇²T + f`

# Fields
- `k`: Thermal conductivity
- `ρ`: Density
- `cₚ`: Specific heat capacity
- `source`: Optional volumetric source term `f(x, t) -> value` (default: `nothing`)

# Example
```julia
model = SolidEnergy(k=50.0, ρ=7800.0, cₚ=500.0)
model = SolidEnergy(k=1.0, ρ=1.0, cₚ=1.0, source=(x, t) -> -4.0)
```
"""
@kwdef struct SolidEnergy{K, P, C, S} <: AbstractModel
    k::K
    ρ::P
    cₚ::C
    source::S = nothing  # Optional source term: f(x, t) -> value
end

num_vars(::SolidEnergy, _) = 1

function make_f(model::SolidEnergy, domain; neighbors = DEFAULT_STENCIL_SIZE, kwargs...)
    (; k, ρ, cₚ, source) = model
    vol = _coords(domain.cloud.volume)
    α = k / (cₚ * ρ)

    W, G, source_appliers = build_neumann_diffusion(domain; k = neighbors)
    Wα = α .* W
    Gα = α .* G
    n_ghost = size(G, 2)
    cbuf = zeros(n_ghost)

    start = maximum(domain.boundaries) do b
        b[2][1][end]
    end
    vol_ids = (start + 1):(start + length(vol))

    # Transient heat equation: ∂T/∂t = α∇²T + f/(ρcₚ).
    # Neumann/Robin flux is folded into Wα/Gα via ghost nodes so those boundary
    # nodes stay dynamic; Dirichlet nodes are pinned afterwards by their BC closure.
    function f(du, u, p, t)
        mul!(du, Wα, u)
        if n_ghost > 0
            for apply! in source_appliers
                apply!(cbuf, t)
            end
            mul!(du, Gα, cbuf, 1, 1)
        end

        if source !== nothing
            for (i, pt) in enumerate(vol)
                du[vol_ids[i]] += source(ustrip.(pt), t) / (ρ * cₚ)
            end
        end
        return nothing
    end

    return f
end

function make_system(model::SolidEnergy, domain; kwargs...)
    (; k, ρ, cₚ, source) = model
    coords = _coords(domain.cloud)
    ∇² = laplacian(_ustrip(coords); k = DEFAULT_STENCIL_SIZE, kwargs...)
    α = k / (cₚ * ρ)
    A = weights(α * ∇²)

    # Compute RHS from source term
    # Steady-state heat equation: ∇²T = f/α → (α∇²)T = f
    if source === nothing
        b = zeros(eltype(A), length(coords))
    else
        # Evaluate source at each point
        b = map(coords) do pt
            source(ustrip.(pt), 0.0)  # Steady-state: t=0
        end
    end

    return A, b
end

function Base.show(io::IO, e::SolidEnergy)
    source_str = e.source === nothing ? "" : ", source"
    return print(io, "Energy: (k = $(e.k), ρ = $(e.ρ), cₚ = $(e.cₚ)$(source_str))")
end
