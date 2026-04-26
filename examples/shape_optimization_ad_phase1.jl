# ============================================================================
# Phase 1 AD Validation: Differentiable PDE Solve via IFT
# ============================================================================
# Validates the full AD chain:
#   pts → make_system_differentiable → A(pts)
#       → apply_dirichlet! (BC rows, non-differentiable)
#       → PDESolveIFT → u(pts)
#       → compute_von_mises → σ_vm → L
#       → ∂L/∂pts
#
# Setup: regular grid on [0,L]×[-D,D]
#   - Boundary points: Dirichlet with Timoshenko exact displacement
#   - Interior points: active PDE equations (∂L/∂pts flows through solve)
#   - b_rhs: fixed at reference config (Phase 1 simplification)
#
# Sanity check: with poly_deg=3, Timoshenko (degree-3 polynomial) should be
# recovered to machine precision.
# ============================================================================
using Pkg
Pkg.activate(@__DIR__)

using Macchiato
using Mooncake
using RadialBasisFunctions
using StaticArrays
using SparseArrays
using LinearAlgebra
using FiniteDifferences

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

u_exact(x, y) = -P / (6E_val * I_val) * y * ((6L - 3x) * x + (2 + ν_val) * (y^2 - D^2))
v_exact(x, y) =  P / (6E_val * I_val) * (3ν_val * y^2 * (L - x) + (4 + 5ν_val) * D^2 * x + (3L - x) * x^2)

# ============================================================================
# Regular grid: boundary Dirichlet, interior PDE
# ============================================================================

nx, ny = 9, 5
xs = range(0.0, L;  length = nx)
ys = range(-D, D;   length = ny)
points = [SVector{2}(x, y) for x in xs for y in ys]
N = length(points)

on_boundary(p) = p[1] ≈ 0.0 || p[1] ≈ L || p[2] ≈ -D || p[2] ≈ D

boundary_idx = findall(on_boundary, points)
interior_idx = findall(!on_boundary, points)

println("Points: $N  (boundary: $(length(boundary_idx)), interior: $(length(interior_idx)))")

pts_flat = vcat([collect(p) for p in points]...)

basis = PHS(3; poly_deg = 3)
k     = 35
adjl  = RadialBasisFunctions.find_neighbors(points, k)

# DOF layout: [u_x(1..N); u_y(1..N)]
dirichlet_dofs = vcat(boundary_idx, boundary_idx .+ N)
dirichlet_vals = vcat(
    [u_exact(p[1], p[2]) for p in points[boundary_idx]],
    [v_exact(p[1], p[2]) for p in points[boundary_idx]],
)

active = make_active_dofs_elasticity(interior_idx, N)

# ============================================================================
# Sanity check: Timoshenko recovery at reference config
# ============================================================================

A0 = make_system_differentiable(model, pts_flat, N, adjl, basis, λstar, μ)
b0 = zeros(2N)
apply_dirichlet!(A0, b0, dirichlet_dofs, dirichlet_vals)

u0 = lu(A0) \ b0

err_ux = norm(u0[1:N]       .- [u_exact(p[1], p[2]) for p in points]) / norm([u_exact(p[1], p[2]) for p in points])
err_uy = norm(u0[(N+1):2N]  .- [v_exact(p[1], p[2]) for p in points]) / norm([v_exact(p[1], p[2]) for p in points])

println("Timoshenko recovery:  err_ux = $err_ux  err_uy = $err_uy")
if err_ux < 1e-8 && err_uy < 1e-8
    println("  PASS")
else
    @warn "Timoshenko recovery check failed — check poly_deg and k"
end

# ============================================================================
# Differentiable loss
# ============================================================================

solver = PDESolveIFT(active)

# Probe: von Mises at a few interior points
probe_idx = interior_idx[1:min(3, length(interior_idx))]

function loss(pts_in)
    A = make_system_differentiable(model, pts_in, N, adjl, basis, λstar, μ)
    b = zeros(eltype(pts_in), 2N)
    apply_dirichlet!(A, b, dirichlet_dofs, dirichlet_vals)
    u = solver(A, b)
    σ_vm = compute_von_mises(u, N, pts_in, adjl, basis, λstar, μ)
    return sum(σ_vm[probe_idx] .^ 2)
end

# Verify the loss computes without error at reference config
L0 = loss(pts_flat)
println("\nLoss at reference config: $L0")

# ============================================================================
# AD gradient (Mooncake)
# ============================================================================

println("\nBuilding Mooncake rrule...")
rule = Mooncake.build_rrule(loss, pts_flat)
_, (_, grad_ad) = Mooncake.value_and_gradient!!(rule, loss, pts_flat)
println("  done  (norm = $(norm(grad_ad)))")

# ============================================================================
# FD gradient (reference)
# ============================================================================

println("Computing FD gradient (this may take a moment)...")
grad_fd = FiniteDifferences.grad(FiniteDifferences.central_fdm(5, 1), loss, pts_flat)[1]
println("  done  (norm = $(norm(grad_fd)))")

# ============================================================================
# Validation
# ============================================================================

rel_err = norm(grad_ad .- grad_fd) / norm(grad_fd)

println("\n========================================")
println("Phase 1 AD Validation")
println("========================================")
println("N=$N  k=$k  poly_deg=3  interior=$(length(interior_idx))")
println("AD/FD relative error: $(round(rel_err; sigdigits=4))")
if rel_err < 1e-3
    println("PASS (rtol < 1e-3)")
else
    @warn "FAIL: rtol = $rel_err"
end
