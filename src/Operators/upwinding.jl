struct Upwind{L <: Function, T <: Int} <: AbstractOperator
    ℒ::L
    dim::T
end

# convienience constructors
"""
    function upwind(data, dim, basis; k=autoselect_k(data, basis))

Builds a `RadialBasisOperator` where the operator is the partial derivative, `Partial`, of `order` with respect to `dim`.
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
