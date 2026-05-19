# ============================================================================
# Phase A: Manual-adjoint validation (Dirichlet-only)
# ============================================================================
# Validates plan_manual_adjoint.md Phase A:
#   - Steps 1-5 of the manual adjoint on the Phase 1 cantilever setup
#   - State-only loss L = sum(u.^2)  ⇒  ∂L/∂u = 2u, no direct ∂L/∂pts
#   - Compare Δpts (boundary + interior) against central FD
#
# Setup matches examples/shape_optimization_ad_phase1.jl:
#   9×5 regular grid on [0,L]×[-D,D], all 4 sides Dirichlet with Timoshenko
#   exact displacement, PHS(3; poly_deg=3), k=35.
#
# Targets (from plan §Phase A):
#   - AD/FD relative error < 1e-3  (Phase 1 baseline was ~1e-10)
#   - Rule-build wall time         (cold; warm should be ms after first call)
# ============================================================================
using Pkg
Pkg.activate(@__DIR__)

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
# Parameters (mirror Phase 1)
# ============================================================================

L     = 8.0
D     = 1.0
P     = 1000.0
E_val = 1.0e7
ν_val = 0.3
I_val = 2D^3 / 3

model    = LinearElasticity(E = E_val, ν = ν_val)
μ, λstar = lame_parameters(model)

u_exact(x, y) = -P / (6E_val * I_val) * y * ((6L - 3x) * x + (2 + ν_val) * (y^2 - D^2))
v_exact(x, y) =  P / (6E_val * I_val) * (3ν_val * y^2 * (L - x) + (4 + 5ν_val) * D^2 * x + (3L - x) * x^2)

# ============================================================================
# Regular grid: boundary Dirichlet, interior PDE
# ============================================================================

nx, ny = 9, 5
xs = range(0.0, L; length = nx)
ys = range(-D, D; length = ny)
points = [SVector{2}(x, y) for x in xs for y in ys]
N = length(points)

on_boundary(p) = p[1] ≈ 0.0 || p[1] ≈ L || p[2] ≈ -D || p[2] ≈ D
boundary_idx = findall(on_boundary, points)
interior_idx = findall(!on_boundary, points)

println("Points: $N  (boundary: $(length(boundary_idx)), interior: $(length(interior_idx)))")

pts_flat = vcat([collect(p) for p in points]...)

basis = PHS(3; poly_deg = 3)
k     = 35
adjl  = find_neighbors(points, k)

dirichlet_dofs = vcat(boundary_idx, boundary_idx .+ N)
dirichlet_vals = vcat(
    [u_exact(p[1], p[2]) for p in points[boundary_idx]],
    [v_exact(p[1], p[2]) for p in points[boundary_idx]],
)
active = falses(2N)
active[interior_idx]      .= true
active[interior_idx .+ N] .= true

# ============================================================================
# Sanity check: Timoshenko recovery at reference config
# ============================================================================

W0_d2x  = _build_weights(Partial(2, 1),      points, points, adjl, basis)
W0_d2y  = _build_weights(Partial(2, 2),      points, points, adjl, basis)
W0_d2xy = _build_weights(MixedPartial(1, 2), points, points, adjl, basis)
A0 = assemble_elasticity_from_weights(W0_d2x, W0_d2y, W0_d2xy, N, λstar, μ)
b0 = zeros(2N)
apply_dirichlet!(A0, b0, dirichlet_dofs, dirichlet_vals)
u0 = lu(A0) \ b0

err_ux = norm(u0[1:N]      .- [u_exact(p[1], p[2]) for p in points]) / norm([u_exact(p[1], p[2]) for p in points])
err_uy = norm(u0[(N+1):2N] .- [v_exact(p[1], p[2]) for p in points]) / norm([v_exact(p[1], p[2]) for p in points])
println(@sprintf("Timoshenko recovery: err_ux = %.2e  err_uy = %.2e", err_ux, err_uy))
@assert err_ux < 1e-8 && err_uy < 1e-8 "Timoshenko sanity check failed"

# ============================================================================
# Loss: state-only, L = sum(u .^ 2)  ⇒  ∂L/∂u = 2u
# ============================================================================

loss_from_u(u) = sum(abs2, u)
∂loss_∂u(u) = 2 .* u

# Closed-form loss as a function of pts_flat (for FD reference)
function loss_pts(pts_in::Vector{Float64})
    pts_v = [SVector{2}(pts_in[2i - 1], pts_in[2i]) for i in 1:N]
    W_d2x  = _build_weights(Partial(2, 1),      pts_v, pts_v, adjl, basis)
    W_d2y  = _build_weights(Partial(2, 2),      pts_v, pts_v, adjl, basis)
    W_d2xy = _build_weights(MixedPartial(1, 2), pts_v, pts_v, adjl, basis)
    A = assemble_elasticity_from_weights(W_d2x, W_d2y, W_d2xy, N, λstar, μ)
    b = zeros(2N)
    apply_dirichlet!(A, b, dirichlet_dofs, dirichlet_vals)
    u = lu(A) \ b
    return loss_from_u(u)
end

L0 = loss_pts(pts_flat)
println(@sprintf("Loss at reference: %.6e", L0))

# ============================================================================
# Manual-adjoint gradient (no rule-building needed — direct RBF backward)
# ============================================================================

println("\nManual-adjoint gradient (cold)...")
t_ad_cold = @elapsed begin
    result = shape_gradient(
        pts_flat, model, N, adjl, basis, active,
        dirichlet_dofs, dirichlet_vals, ∂loss_∂u,
    )
end
grad_ad = result.Δpts
println(@sprintf("  cold: %.3f s   ‖Δpts‖ = %.4e", t_ad_cold, norm(grad_ad)))

# Warm call
t_ad_warm = @elapsed shape_gradient(
    pts_flat, model, N, adjl, basis, active,
    dirichlet_dofs, dirichlet_vals, ∂loss_∂u,
)
println(@sprintf("  warm: %.4f s", t_ad_warm))

# Sanity: loss values agree
L_ad = loss_from_u(result.u)
@assert isapprox(L_ad, L0; rtol = 1e-12) "forward solve drift between loss_pts and shape_gradient"

# ============================================================================
# FD reference
# ============================================================================

println("\nComputing FD gradient (central, order-5)...")
t_fd = @elapsed begin
    grad_fd = FiniteDifferences.grad(FiniteDifferences.central_fdm(5, 1), loss_pts, pts_flat)[1]
end
println(@sprintf("  %.2f s   ‖grad_fd‖ = %.4e", t_fd, norm(grad_fd)))

# ============================================================================
# Validation
# ============================================================================

rel_err  = norm(grad_ad .- grad_fd) / norm(grad_fd)
max_abs  = maximum(abs, grad_ad .- grad_fd)
ratio    = norm(grad_ad) / norm(grad_fd)

# Boundary-only and interior-only slices (informational; both should match FD)
bnd_dofs = vcat(boundary_idx, boundary_idx .+ N)
int_dofs = vcat(interior_idx, interior_idx .+ N)
bnd_relerr = norm(grad_ad[bnd_dofs] .- grad_fd[bnd_dofs]) / norm(grad_fd[bnd_dofs])
int_relerr = norm(grad_ad[int_dofs] .- grad_fd[int_dofs]) / norm(grad_fd[int_dofs])

println("\n========================================")
println("Phase A: Manual-Adjoint Validation")
println("========================================")
println(@sprintf("N=%d  k=%d  poly_deg=3  interior=%d  boundary=%d",
                  N, k, length(interior_idx), length(boundary_idx)))
println(@sprintf("‖grad_AD‖ / ‖grad_FD‖    = %.6f  (1.0 → magnitudes match)", ratio))
println(@sprintf("AD/FD relative error     = %.4e", rel_err))
println(@sprintf("  boundary DOFs only     = %.4e", bnd_relerr))
println(@sprintf("  interior DOFs only     = %.4e", int_relerr))
println(@sprintf("max abs componentwise    = %.4e", max_abs))
println()
if rel_err < 1e-3
    println("PASS (rtol < 1e-3)")
else
    @warn "FAIL — rel_err = $rel_err  (target < 1e-3)"
    # Top offenders for triage
    diff = grad_ad .- grad_fd
    perm = sortperm(abs.(diff); rev = true)
    println("\nTop 10 component mismatches (dof, grad_ad, grad_fd, diff):")
    for k in 1:min(10, length(perm))
        i = perm[k]
        println(@sprintf("  dof %3d:  AD = %+10.3e   FD = %+10.3e   Δ = %+10.3e",
                          i, grad_ad[i], grad_fd[i], diff[i]))
    end
end
