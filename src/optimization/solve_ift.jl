import RadialBasisFunctions: _build_weights, Partial, MixedPartial

"""
    make_system_differentiable(::LinearElasticity, pts_flat, N, adjl, basis, λstar, μ)

Assemble the raw 2N×2N plane-stress Navier system from a flat coordinate vector.
Returns the system matrix without any BC modification so that Mooncake can trace
through the five `_build_weights` calls via their existing rrule!!s.

This operates on stripped scalars and plain arrays — no Unitful, no Domain — so
the AD chain is unobstructed.
"""
function make_system_differentiable(
    ::LinearElasticity,
    pts_flat::AbstractVector{<:Real},
    N::Int,
    adjl,
    basis,
    λstar::Real,
    μ::Real,
)
    pts = [SVector{2}(pts_flat[2i - 1], pts_flat[2i]) for i in 1:N]

    W_d2x  = _build_weights(Partial(2, 1),      pts, pts, adjl, basis)
    W_d2y  = _build_weights(Partial(2, 2),      pts, pts, adjl, basis)
    W_d2xy = _build_weights(MixedPartial(1, 2), pts, pts, adjl, basis)

    A11 = (λstar + 2μ) * W_d2x + μ * W_d2y
    A12 = (λstar + μ)  * W_d2xy
    A22 = μ * W_d2x + (λstar + 2μ) * W_d2y

    return [A11 A12; A12 A22]
end

"""
    PDESolveIFT

Mooncake-differentiable linear solver for assembled PDE systems. Wraps `A\\b`
with a custom rrule!! that implements the adjoint-based IFT:

    ∂L/∂A[i,j] = -η[i] * u[j]   for active_dofs[i] == true
    ∂L/∂b      = η               where η = Aᵀ \\ (∂L/∂u)

`active_dofs[i] == true` marks rows that are genuine PDE equations. Identity
rows inserted by `apply_dirichlet!` are `false` — their gradient is zero.
"""
struct PDESolveIFT
    active_dofs::BitVector
end

function (s::PDESolveIFT)(A::SparseMatrixCSC{Float64, Int}, b::Vector{Float64})
    return lu(A) \ b
end

"""
    make_active_dofs_elasticity(interior_idx, N)

Build the `active_dofs` BitVector for a 2D elasticity system of size 2N.
`interior_idx` lists the row indices (1-based, in the u_x block) that hold
genuine PDE equations. The u_y block is offset by N.
"""
function make_active_dofs_elasticity(interior_idx::AbstractVector{Int}, N::Int)
    active = falses(2N)
    active[interior_idx]       .= true
    active[interior_idx .+ N]  .= true
    return active
end

"""
    apply_dirichlet!(A, b, dirichlet_dofs, vals)

Overwrite Dirichlet rows of `A` with identity and set the corresponding
entries of `b`. This step is intentionally outside the differentiable path —
Dirichlet rows do not depend on `pts`.
"""
function apply_dirichlet!(
    A::AbstractMatrix,
    b::AbstractVector,
    dirichlet_dofs::AbstractVector{Int},
    vals::AbstractVector{<:Real},
)
    for (k, i) in enumerate(dirichlet_dofs)
        A[i, :] .= 0.0
        A[i, i]  = 1.0
        b[i]     = vals[k]
    end
    return nothing
end

"""
    compute_von_mises(u_flat, N, pts_flat, adjl, basis, λstar, μ)

Compute the von Mises stress field (plane stress) from a displacement solution
`u_flat = [u_x; u_y]` and point coordinates `pts_flat`.

This is fully differentiable via the `_build_weights` rrule!! — no custom rrule
needed here.
"""
function compute_von_mises(
    u_flat::AbstractVector{<:Real},
    N::Int,
    pts_flat::AbstractVector{<:Real},
    adjl,
    basis,
    λstar::Real,
    μ::Real,
)
    pts = [SVector{2}(pts_flat[2i - 1], pts_flat[2i]) for i in 1:N]
    u_x = u_flat[1:N]
    u_y = u_flat[(N + 1):(2N)]

    W_dx = _build_weights(Partial(1, 1), pts, pts, adjl, basis)
    W_dy = _build_weights(Partial(1, 2), pts, pts, adjl, basis)

    dux_dx = W_dx * u_x
    dux_dy = W_dy * u_x
    duy_dx = W_dx * u_y
    duy_dy = W_dy * u_y

    σ_xx = (λstar + 2μ) .* dux_dx .+ λstar .* duy_dy
    σ_yy = λstar .* dux_dx .+ (λstar + 2μ) .* duy_dy
    σ_xy = μ .* (dux_dy .+ duy_dx)

    return sqrt.(σ_xx .^ 2 .- σ_xx .* σ_yy .+ σ_yy .^ 2 .+ 3 .* σ_xy .^ 2)
end
