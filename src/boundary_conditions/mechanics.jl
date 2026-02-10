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
Displacement(ux::Function, uy::Function) = Displacement((x, t) -> (ux(x, t), uy(x, t)))

(bc::Displacement)(x, t) = bc.f(x, t)

physics_domain(::Type{<:Displacement}) = MechanicsPhysics()

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

physics_domain(::Type{<:Traction}) = MechanicsPhysics()

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
For each boundary node i:
  - Row i:   A[i, :] = 0, A[i, i] = 1, b[i] = ux
  - Row i+N: A[i+N, :] = 0, A[i+N, i+N] = 1, b[i+N] = uy
"""
function make_bc!(A::AbstractMatrix{TA}, b::AbstractVector{TB},
        bc::Displacement, surf, domain, ids;
        kwargs...) where {TA, TB}
    N = div(size(A, 1), 2)
    for (local_i, global_i) in enumerate(ids)
        x = get_node_coords(surf, local_i)
        ux, uy = bc(x, 0.0)

        # x-displacement row
        A[global_i, :] .= zero(TA)
        A[global_i, global_i] = one(TA)
        b[global_i] = convert(TB, ux)

        # y-displacement row
        A[global_i + N, :] .= zero(TA)
        A[global_i + N, global_i + N] = one(TA)
        b[global_i + N] = convert(TB, uy)
    end
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
  Row i+N: nx[μ∂/∂y]u + ny[μ∂/∂x]u + nx[λ*∂/∂x]v + ny[(λ*+2μ)∂/∂y]v = ty
"""
function make_bc!(A::AbstractMatrix{TA}, b::AbstractVector{TB},
        bc::Traction, surf, domain, ids;
        λstar::Real, μ_lame::Real,
        kwargs...) where {TA, TB}
    N = div(size(A, 1), 2)
    normals = normal(surf)
    coords_all = _ustrip(_coords(domain.cloud))

    # Build first-derivative operators evaluated at all boundary points in this surface
    eval_pts = [get_node_coords(surf, i) for i in 1:length(ids)]
    ∂x_op = partial(coords_all, eval_pts, 1, 1; k = 40)
    update_weights!(∂x_op)
    ∂y_op = partial(coords_all, eval_pts, 1, 2; k = 40)
    update_weights!(∂y_op)

    for (local_i, global_i) in enumerate(ids)
        x = get_node_coords(surf, local_i)
        tx, ty = bc(x, 0.0)
        n = ustrip(normals[local_i])
        n_x, n_y = n[1], n[2]

        # Extract sparse weight rows and find non-zero entries (stencil neighbors)
        w_dx_sv = ∂x_op.weights[local_i, :]
        w_dy_sv = ∂y_op.weights[local_i, :]

        nbs_dx_idx, _ = findnz(w_dx_sv)
        nbs_dy_idx, _ = findnz(w_dy_sv)
        all_nbs = sort(unique(vcat(nbs_dx_idx, nbs_dy_idx)))

        # Build weight vectors for all neighbors
        w_∂x = [w_dx_sv[nb] for nb in all_nbs]
        w_∂y = [w_dy_sv[nb] for nb in all_nbs]

        # Row i (x-traction): nx(λ*+2μ)∂u/∂x + nyμ∂u/∂y + nxλ*∂v/∂y + nyμ∂v/∂x = tx
        w_u = n_x * (λstar + 2μ_lame) .* w_∂x .+ n_y * μ_lame .* w_∂y
        w_v = n_x * λstar .* w_∂y .+ n_y * μ_lame .* w_∂x

        A[global_i, :] .= zero(TA)
        for (j, nb) in enumerate(all_nbs)
            A[global_i, nb] = convert(TA, w_u[j])
            A[global_i, nb + N] = convert(TA, w_v[j])
        end
        b[global_i] = convert(TB, tx)

        # Row i+N (y-traction): nxμ∂u/∂y + nyμ∂u/∂x + nxλ*∂v/∂x + ny(λ*+2μ)∂v/∂y = ty
        w_u2 = n_x * μ_lame .* w_∂y .+ n_y * μ_lame .* w_∂x
        w_v2 = n_x * λstar .* w_∂x .+ n_y * (λstar + 2μ_lame) .* w_∂y

        A[global_i + N, :] .= zero(TA)
        for (j, nb) in enumerate(all_nbs)
            A[global_i + N, nb] = convert(TA, w_u2[j])
            A[global_i + N, nb + N] = convert(TA, w_v2[j])
        end
        b[global_i + N] = convert(TB, ty)
    end
end
