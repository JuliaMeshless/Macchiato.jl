# ============================================================================
# Displacement (Dirichlet) - prescribes displacement components
# ============================================================================

"""
    Displacement{F<:Function} <: Dirichlet

Prescribed displacement BC for solid mechanics. The function returns a tuple
of displacement components: `f(x, t) -> (ux, uy)` for 2D.

# Constructors
```julia
Displacement((x, t) -> (0.0, 0.0))          # Function returning tuple
Displacement(0.0, 0.0)                       # Constant displacement
Displacement(ux::Function, uy::Function)     # Per-component functions
```
"""
struct Displacement{F <: Function} <: Dirichlet
    f::F
end

Displacement(ux::Number, uy::Number) = Displacement((x, t) -> (ux, uy))
Displacement(ux::Function, uy::Function) = Displacement((x, t) -> (ux(x), uy(x)))

(bc::Displacement)(x, t) = bc.f(x, t)

Base.show(io::IO, ::Displacement) = print(io, "Displacement")

# ============================================================================
# Traction (Neumann) - prescribes traction vector σ·n at boundary
# ============================================================================

"""
    Traction{F<:Function} <: Neumann

Prescribed traction BC for solid mechanics. The function returns a tuple
of traction components: `f(x, t) -> (tx, ty)` for 2D.

In terms of stress: t = σ·n where n is the outward normal.

# Constructors
```julia
Traction((x, t) -> (0.0, -1000.0))          # Function returning tuple
Traction(tx::Number, ty::Number)             # Constant traction
Traction(tx::Function, ty::Function)         # Per-component functions
```
"""
struct Traction{F <: Function} <: Neumann
    f::F
end

Traction(tx::Number, ty::Number) = Traction((x, t) -> (tx, ty))
Traction(tx::Function, ty::Function) = Traction((x, t) -> (tx(x, t), ty(x, t)))

(bc::Traction)(x, t) = bc.f(x, t)

Base.show(io::IO, ::Traction) = print(io, "Traction")

# ============================================================================
# TractionFree - zero traction (free surface)
# ============================================================================

"""
    TractionFree()

Zero-traction (free surface) BC: σ·n = 0. Convenience for `Traction(0.0, 0.0)`.
"""
TractionFree() = Traction(0.0, 0.0)

# ============================================================================
# Custom make_bc! for vector-valued mechanics BCs
# ============================================================================

"""
    make_bc!(A, b, bc::Displacement, surf, domain, ids; kwargs...)

Apply displacement (Dirichlet) BC to the 2N×2N mechanics system.
Dispatches to the generalized `write_bc_dirichlet!` with `n_vars=2`.
"""
function make_bc!(
        A::AbstractMatrix, b::AbstractVector,
        bc::Displacement, surf, domain, ids; kwargs...
    )
    return write_bc_dirichlet!(A, b, ids, bc, surf, 2)
end

"""
    make_bc!(A, b, bc::Traction, surf, domain, ids; kwargs...)

Apply traction (Neumann) BC to the 2N×2N mechanics system.
For each boundary node i with outward normal n = (nx, ny):
  - Row i:   encode σ₁₁·nx + σ₁₂·ny = tx using displacement derivative weights
  - Row i+N: encode σ₂₁·nx + σ₂₂·ny = ty using displacement derivative weights

The stress-displacement relation (plane stress):
  σ₁₁ = (λ*+2μ)∂u/∂x + λ*∂v/∂y
  σ₂₂ = λ*∂u/∂x + (λ*+2μ)∂v/∂y
  σ₁₂ = μ(∂u/∂y + ∂v/∂x)

So the traction rows become:
  Row i:   nx[(λ*+2μ)∂/∂x]u + ny[μ∂/∂y]u + nx[λ*∂/∂y]v + ny[μ∂/∂x]v = tx
  Row i+N: nx[μ∂/∂y]u + ny[λ*∂/∂x]u + nx[μ∂/∂x]v + ny[(λ*+2μ)∂/∂y]v = ty
"""
function make_bc!(
        A::AbstractMatrix{TA}, b::AbstractVector{TB},
        bc::Traction, surf, domain, ids;
        λstar::Real, μ_lame::Real,
        scheme = nothing,
        kwargs...
    ) where {TA, TB}
    N = div(size(A, 1), 2)
    normals = normal(surf)
    coords_all = _ustrip(_coords(domain.cloud))

    # Build first-derivative operators with shared KNN adjacency list
    eval_pts = [get_node_coords(surf, i) for i in 1:length(ids)]
    k = get(kwargs, :k, 40)
    adjl = find_neighbors(coords_all, eval_pts, k)

    # Build operators (KernelAbstractions parallelizes internally)
    ∂x_op = partial(coords_all, eval_pts, 1, 1; k = k, adjl = adjl, kwargs...)
    ∂y_op = partial(coords_all, eval_pts, 1, 2; k = k, adjl = adjl, kwargs...)

    # Zero all traction BC rows in a single O(nnz) pass
    row_set = Set{Int}()
    for global_i in ids
        push!(row_set, global_i)
        push!(row_set, global_i + N)
    end
    zero_rows!(A, row_set)

    # Pre-allocate weight buffers for the inner loop (sized to max stencil width)
    max_k = maximum(length, adjl)
    w_dx_buf = Vector{Float64}(undef, max_k)
    w_dy_buf = Vector{Float64}(undef, max_k)

    W_dx = ∂x_op.weights
    W_dy = ∂y_op.weights

    for (local_i, global_i) in enumerate(ids)
        x = get_node_coords(surf, local_i)
        tx, ty = bc(x, 0.0)
        n = ustrip(normals[local_i])
        n_x, n_y = n[1], n[2]

        # Collect unique nonzero column indices from both weight rows
        # Use the adjacency list directly — it gives us the stencil indices
        stencil = adjl[local_i]
        n_stencil = length(stencil)

        # Read weights for this row's stencil points
        @inbounds for j in 1:n_stencil
            col = stencil[j]
            w_dx_buf[j] = W_dx[local_i, col]
            w_dy_buf[j] = W_dy[local_i, col]
        end

        # Row i (x-traction): nx(λ*+2μ)∂u/∂x + nyμ∂u/∂y + nxλ*∂v/∂y + nyμ∂v/∂x = tx
        c_ux = n_x * (λstar + 2μ_lame)
        c_uy = n_y * μ_lame
        c_vx = n_y * μ_lame
        c_vy = n_x * λstar
        @inbounds for j in 1:n_stencil
            col = stencil[j]
            A[global_i, col] = convert(TA, c_ux * w_dx_buf[j] + c_uy * w_dy_buf[j])
            A[global_i, col + N] = convert(TA, c_vy * w_dy_buf[j] + c_vx * w_dx_buf[j])
        end
        b[global_i] = convert(TB, tx)

        # Row i+N (y-traction): nxμ∂u/∂y + nyλ*∂u/∂x + nxμ∂v/∂x + ny(λ*+2μ)∂v/∂y = ty
        c_ux2 = n_y * λstar
        c_uy2 = n_x * μ_lame
        c_vx2 = n_x * μ_lame
        c_vy2 = n_y * (λstar + 2μ_lame)
        @inbounds for j in 1:n_stencil
            col = stencil[j]
            A[global_i + N, col] = convert(TA, c_ux2 * w_dx_buf[j] + c_uy2 * w_dy_buf[j])
            A[global_i + N, col + N] = convert(TA, c_vx2 * w_dx_buf[j] + c_vy2 * w_dy_buf[j])
        end
        b[global_i + N] = convert(TB, ty)
    end
    return
end
