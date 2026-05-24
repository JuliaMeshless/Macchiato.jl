# ============================================================================
# Phase B: Manual-adjoint validation with mixed Dirichlet + Neumann BCs
# ============================================================================
# Setup (cantilever, frozen normals — plan Level 1):
#   - 9×5 regular grid on [0, L] × [-D, D]
#   - Left edge (x=0):   Dirichlet, clamped (u = v = 0)
#   - Right edge (x=L):  Traction (parabolic shear: t = (0, P·(D²-y²)/(2I)))
#   - Top   (y=+D):      Traction-free (t = 0, n = (0, +1))
#   - Bottom(y=-D):      Traction-free (t = 0, n = (0, -1))
#   - Corners (L, ±D) follow the right-edge traction.
#
# Loss: L = sum(u.^2)  (state-only ⇒ ∂L/∂u = 2u, no direct ∂L/∂pts term)
#
# Validates Phase B's manual adjoint vs central FD on full Δpts vector.
# ============================================================================
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Macchiato
using Mooncake
using RadialBasisFunctions
import RadialBasisFunctions: _build_weights, Partial, MixedPartial, find_neighbors
using StaticArrays
using SparseArrays
using LinearAlgebra
using FiniteDifferences
using Printf

# ============================================================================
# Parameters
# ============================================================================

L     = 8.0
D     = 1.0
P     = 1000.0
E_val = 1.0e7
ν_val = 0.3
I_val = 2D^3 / 3

model    = LinearElasticity(E = E_val, ν = ν_val)
μ, λstar = lame_parameters(model)

# ============================================================================
# Grid + BC classification
# ============================================================================

nx, ny = 9, 5
xs = range(0.0, L; length = nx)
ys = range(-D, D; length = ny)
points = [SVector{2}(x, y) for x in xs for y in ys]
N = length(points)

is_left(p)   = p[1] ≈ 0.0
is_right(p)  = p[1] ≈ L
is_top(p)    = p[2] ≈ D  && !is_left(p)
is_bottom(p) = p[2] ≈ -D && !is_left(p)
is_dirichlet(p) = is_left(p)
is_neumann(p)   = !is_dirichlet(p) && (is_right(p) || is_top(p) || is_bottom(p))
is_interior(p)  = !is_dirichlet(p) && !is_neumann(p)

dirichlet_idx = findall(is_dirichlet, points)
neumann_idx   = findall(is_neumann,   points)
interior_idx  = findall(is_interior,  points)

println("Points: $N  (Dirichlet: $(length(dirichlet_idx)), " *
        "Neumann: $(length(neumann_idx)), interior: $(length(interior_idx)))")

# DOF masks (2N)
active_v = trues(2N)
for i in dirichlet_idx
    active_v[i]     = false
    active_v[i + N] = false
end
active = BitVector(active_v)

interior_rows = falses(N)
for i in interior_idx
    interior_rows[i] = true
end

# ============================================================================
# Dirichlet info: clamped (u = v = 0) on left edge
# ============================================================================

dirichlet_dofs = vcat(dirichlet_idx, dirichlet_idx .+ N)
dirichlet_vals = zeros(2 * length(dirichlet_idx))

# ============================================================================
# Neumann info: normal + traction per Neumann point
# ============================================================================
# Right edge (n = (1, 0)):    t = (0, P·(D²-y²)/(2I))
# Top edge  (n = (0, 1)):    t = (0, 0)
# Bottom    (n = (0, -1)):   t = (0, 0)

function normal_of(p)
    if is_right(p)
        return SVector{2}(1.0, 0.0)
    elseif is_top(p)
        return SVector{2}(0.0, 1.0)
    elseif is_bottom(p)
        return SVector{2}(0.0, -1.0)
    else
        error("not a Neumann point: $p")
    end
end

function traction_of(p)
    if is_right(p)
        return SVector{2}(0.0, P * (D^2 - p[2]^2) / (2 * I_val))
    else
        return SVector{2}(0.0, 0.0)
    end
end

neumann_normals   = [normal_of(points[i])   for i in neumann_idx]
neumann_tractions = [traction_of(points[i]) for i in neumann_idx]

# ============================================================================
# RBF setup
# ============================================================================

basis = PHS(3; poly_deg = 3)
k     = 35
adjl  = find_neighbors(points, k)

# Neumann adjacency: stencils for the neumann_idx eval points, indices into full pts
neumann_adjl = adjl[neumann_idx]

pts_flat = vcat([collect(p) for p in points]...)

# ============================================================================
# Traction layout (one-shot)
# ============================================================================

traction_layout = build_traction_layout(
    neumann_idx, neumann_adjl, neumann_normals, neumann_tractions,
    λstar, μ, N,
)

# ============================================================================
# Sanity check: forward solve runs and produces non-zero displacement
# ============================================================================

pts = [SVector{2}(pts_flat[2i - 1], pts_flat[2i]) for i in 1:N]
W_d2x  = _build_weights(Partial(2, 1),      pts, pts, adjl, basis)
W_d2y  = _build_weights(Partial(2, 2),      pts, pts, adjl, basis)
W_d2xy = _build_weights(MixedPartial(1, 2), pts, pts, adjl, basis)

neumann_pts = pts[neumann_idx]
W_dx = _build_weights(Partial(1, 1), pts, neumann_pts, neumann_adjl, basis)
W_dy = _build_weights(Partial(1, 2), pts, neumann_pts, neumann_adjl, basis)

A_ref = assemble_elasticity_from_weights(W_d2x, W_d2y, W_d2xy, N, λstar, μ)
b_ref = zeros(2N)
apply_dirichlet!(A_ref, b_ref, dirichlet_dofs, dirichlet_vals)
apply_traction!(A_ref, b_ref, traction_layout, W_dx, W_dy)

u_ref = lu(A_ref) \ b_ref
println(@sprintf("Forward solve:  ‖u‖ = %.4e   max|u| = %.4e",
                  norm(u_ref), maximum(abs, u_ref)))
@assert all(isfinite, u_ref) "Forward solve produced non-finite values"

# ============================================================================
# Loss closure (for FD) and ∂L/∂u
# ============================================================================

loss_from_u(u) = sum(abs2, u)
∂loss_∂u(u)    = 2 .* u

function loss_pts(p::Vector{Float64})
    pts_v = [SVector{2}(p[2i - 1], p[2i]) for i in 1:N]
    W_d2x_l  = _build_weights(Partial(2, 1),      pts_v, pts_v, adjl, basis)
    W_d2y_l  = _build_weights(Partial(2, 2),      pts_v, pts_v, adjl, basis)
    W_d2xy_l = _build_weights(MixedPartial(1, 2), pts_v, pts_v, adjl, basis)
    neu_pts_l = pts_v[neumann_idx]
    W_dx_l = _build_weights(Partial(1, 1), pts_v, neu_pts_l, neumann_adjl, basis)
    W_dy_l = _build_weights(Partial(1, 2), pts_v, neu_pts_l, neumann_adjl, basis)
    A_l = assemble_elasticity_from_weights(W_d2x_l, W_d2y_l, W_d2xy_l, N, λstar, μ)
    b_l = zeros(2N)
    apply_dirichlet!(A_l, b_l, dirichlet_dofs, dirichlet_vals)
    apply_traction!(A_l, b_l, traction_layout, W_dx_l, W_dy_l)
    u_l = lu(A_l) \ b_l
    return loss_from_u(u_l)
end

L0 = loss_pts(pts_flat)
println(@sprintf("Loss at reference: %.6e", L0))

# ============================================================================
# Manual-adjoint gradient (no rule-building — direct RBF backward)
# ============================================================================

println("\nManual-adjoint gradient (cold)...")
t_ad_cold = @elapsed begin
    result = shape_gradient(
        pts_flat, model, N, adjl, basis,
        active,
        dirichlet_dofs, dirichlet_vals,
        ∂loss_∂u;
        interior_rows = interior_rows,
        traction_layout = traction_layout,
        neumann_ids = neumann_idx,
        neumann_adjl = neumann_adjl,
    )
end
grad_ad = result.Δpts
println(@sprintf("  cold: %.3f s   ‖Δpts‖ = %.4e", t_ad_cold, norm(grad_ad)))

# Warm call
t_ad_warm = @elapsed shape_gradient(
    pts_flat, model, N, adjl, basis,
    active,
    dirichlet_dofs, dirichlet_vals,
    ∂loss_∂u;
    interior_rows = interior_rows,
    traction_layout = traction_layout,
    neumann_ids = neumann_idx,
    neumann_adjl = neumann_adjl,
)
println(@sprintf("  warm: %.4f s", t_ad_warm))

L_ad = loss_from_u(result.u)
@assert isapprox(L_ad, L0; rtol = 1e-10) "forward solve drift between _build_weights and _build_weights_and_cache"

# ============================================================================
# FD reference
# ============================================================================

println("\nComputing FD gradient (central, order-5)...")
t_fd = @elapsed begin
    grad_fd = FiniteDifferences.grad(FiniteDifferences.central_fdm(5, 1),
                                      loss_pts, pts_flat)[1]
end
println(@sprintf("  %.2f s   ‖grad_fd‖ = %.4e", t_fd, norm(grad_fd)))

# ============================================================================
# Validation
# ============================================================================

rel_err   = norm(grad_ad .- grad_fd) / norm(grad_fd)
max_abs   = maximum(abs, grad_ad .- grad_fd)
ratio     = norm(grad_ad) / norm(grad_fd)

dir_dofs = vcat(dirichlet_idx, dirichlet_idx .+ N)
neu_dofs = vcat(neumann_idx,   neumann_idx   .+ N)
int_dofs = vcat(interior_idx,  interior_idx  .+ N)

dir_relerr = norm(grad_ad[dir_dofs] .- grad_fd[dir_dofs]) /
              max(norm(grad_fd[dir_dofs]), eps())
neu_relerr = norm(grad_ad[neu_dofs] .- grad_fd[neu_dofs]) / norm(grad_fd[neu_dofs])
int_relerr = norm(grad_ad[int_dofs] .- grad_fd[int_dofs]) / norm(grad_fd[int_dofs])

println("\n========================================")
println("Phase B: Manual-Adjoint Validation (mixed BC)")
println("========================================")
println(@sprintf("N=%d  k=%d  poly_deg=3", N, k))
println(@sprintf("Dirichlet=%d  Neumann=%d  interior=%d",
                  length(dirichlet_idx), length(neumann_idx), length(interior_idx)))
println(@sprintf("‖grad_AD‖ / ‖grad_FD‖    = %.6f", ratio))
println(@sprintf("AD/FD relative error     = %.4e", rel_err))
println(@sprintf("  Dirichlet DOFs only    = %.4e   ‖grad_fd[d]‖ = %.4e",
                  dir_relerr, norm(grad_fd[dir_dofs])))
println(@sprintf("  Neumann DOFs only      = %.4e", neu_relerr))
println(@sprintf("  Interior DOFs only     = %.4e", int_relerr))
println(@sprintf("max abs componentwise    = %.4e", max_abs))
println()
if rel_err < 1e-3
    println("PASS (rtol < 1e-3)")
else
    @warn "FAIL — rel_err = $rel_err  (target < 1e-3)"
    diff = grad_ad .- grad_fd
    perm = sortperm(abs.(diff); rev = true)
    println("\nTop 10 component mismatches:")
    for k in 1:min(10, length(perm))
        i = perm[k]
        which_dof = i > N ? "v" : "u"
        pt = i > N ? i - N : i
        println(@sprintf("  dof %3d (%s_%d):  AD = %+10.3e   FD = %+10.3e   Δ = %+10.3e",
                          i, which_dof, pt, grad_ad[i], grad_fd[i], diff[i]))
    end
end
