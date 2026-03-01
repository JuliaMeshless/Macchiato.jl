"""
    upwind(data, eval_points, dim[, basis]; Δ=nothing, k=autoselect_k(data, basis))
    upwind(data, dim[, basis]; Δ=nothing, k=autoselect_k(data, basis))

Build an upwind finite-difference-style operator using RBF interpolation.

Computes backward, forward, and centered partial derivatives with respect to dimension `dim`,
then returns a function `(ϕ, v, θ)` that blends them based on flow direction `v` and
upwind parameter `θ ∈ [0, 1]` (1 = full upwind, 0 = centered).

The single-argument form `upwind(data, dim)` evaluates at the data points themselves.

# Arguments
- `data`: Stencil points for RBF approximation
- `eval_points`: Points where the derivative is evaluated
- `dim`: Spatial dimension (1 = x, 2 = y, …)
- `basis`: Radial basis function (default: `PHS(3; poly_deg=2)`)
- `Δ`: Virtual node offset distance (auto-detected if `nothing`)
- `k`: Number of nearest neighbors for stencil
"""
function upwind(
        data::AbstractVector,
        eval_points::AbstractVector,
        dim,
        basis::B = PHS(3; poly_deg = 2);
        Δ = nothing,
        k::T = autoselect_k(data, basis)
    ) where {T <: Int, B <: AbstractRadialBasis}
    Δ === nothing && (Δ = _find_smallest_dist(data, k))
    backward = ∂virtual(data, eval_points, dim, basis; Δ = Δ, backward = true, k = k)
    forward = ∂virtual(data, eval_points, dim, basis; Δ = Δ, backward = false, k = k)
    center = partial(data, eval_points, 1, dim, basis; k = k)

    du = let backward = backward, forward = forward, center = center
        (ϕ, v, θ) -> begin
            wl = max.(v, Ref(0)) .* (θ .* backward(ϕ) .+ (1 .- θ) .* center(ϕ))
            wr = min.(v, Ref(0)) .* (θ .* forward(ϕ) .+ (1 .- θ) .* center(ϕ))
            wl .+ wr
        end
    end

    return du
end

function upwind(
        data::AbstractVector,
        dim,
        basis::B = PHS(3; poly_deg = 2);
        Δ = nothing,
        k::T = autoselect_k(data, basis)
    ) where {T <: Int, B <: AbstractRadialBasis}
    return upwind(data, data, dim, basis; Δ = Δ, k = k)
end
