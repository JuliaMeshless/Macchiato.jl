# ============================================================================
# Phase 2 AD Validation: Domain-Aware Differentiable Pipeline
# ============================================================================
# Validates the full AD chain using the Domain/Simulation API:
#   Domain → active_dofs, build_dirichlet_info (automatic)
#          → gradient(sim, loss; wrt=:pts) (user-facing API)
#
# Rebuilds the Phase 1 cantilever validation but derives all AD setup from
# the Domain object instead of manual index construction.
#
# Setup: rectangular domain, all sides Dirichlet (Timoshenko exact displacement)
#   - No Neumann BCs (deferred to future phase)
#   - b_rhs = 0 (no body force)
# ============================================================================
using Pkg
Pkg.activate(@__DIR__)

using Macchiato
import Macchiato as MM
using WhatsThePoint
import WhatsThePoint as WTP
using RadialBasisFunctions: PHS, find_neighbors
using Mooncake
using StaticArrays
using SparseArrays
using LinearAlgebra
using FiniteDifferences
using Unitful: m, °, ustrip

# ============================================================================
# Parameters (same as Phase 1)
# ============================================================================

L     = 8.0
D     = 1.0
P     = 1000.0
E_val = 1.0e7
ν_val = 0.3
I_val = 2D^3 / 3

model = LinearElasticity(E = E_val, ν = ν_val)
μ, λstar = lame_parameters(model)

u_exact(x, y) = -P / (6E_val * I_val) * y * ((6L - 3x) * x + (2 + ν_val) * (y^2 - D^2))
v_exact(x, y) =  P / (6E_val * I_val) * (3ν_val * y^2 * (L - x) + (4 + 5ν_val) * D^2 * x + (3L - x) * x^2)

# ============================================================================
# Domain Setup — all four sides Dirichlet (prescribed Timoshenko displacement)
# ============================================================================

dx = 0.5m  # Point spacing (coarser than Phase 1 for manageable AD compile time)

# Create rectangular boundary
rx = dx:dx:((L * m) - dx)
ry = dx:dx:((2D * m) - dx)

# Bottom (y = -D)
p_bot = [WTP.Point(i, -D * m) for i in rx]
n_bot = [WTP.Vec(0.0, -1.0) for _ in rx]

# Right (x = L)
p_right = [WTP.Point(L * m, -D * m + i) for i in ry]
n_right = [WTP.Vec(1.0, 0.0) for _ in ry]

# Top (y = D)
p_top = [WTP.Point(i, D * m) for i in reverse(rx)]
n_top = [WTP.Vec(0.0, 1.0) for _ in rx]

# Left (x = 0)
p_left = [WTP.Point(0.0m, -D * m + i) for i in reverse(ry)]
n_left = [WTP.Vec(-1.0, 0.0) for _ in ry]

pts = vcat(p_bot, p_right, p_top, p_left)
nrms = vcat(n_bot, n_right, n_top, n_left)
areas = fill(dx, length(pts))

part = PointBoundary(pts, nrms, areas)
split_surface!(part, 75°)
# surface1=bottom, surface2=right, surface3=top, surface4=left

Δ = dx
cloud = WTP.discretize(part, ConstantSpacing(Δ), alg = VanDerSandeFornberg())

# All four sides: Dirichlet with Timoshenko exact displacement
bc_func(x, t) = (u_exact(x[1], x[2]), v_exact(x[1], x[2]))

bcs = Dict(
    :surface1 => Displacement(bc_func),   # Bottom
    :surface2 => Displacement(bc_func),   # Right
    :surface3 => Displacement(bc_func),   # Top
    :surface4 => Displacement(bc_func),   # Left
)

domain = MM.Domain(cloud, bcs, model)
sim = Simulation(domain)

N = length(cloud)
println("Points: $N  (DOFs: $(2N))")

# ============================================================================
# Sanity check: Timoshenko recovery via mainstream make_system
# ============================================================================

println("\n--- Sanity check: mainstream make_system ---")
basis = PHS(3; poly_deg = 3)
k = 35

A_main, b_main = make_system(model, domain; k = k, basis = basis)

# Apply BCs manually (replicating what LinearProblem does internally)
dirichlet_dofs, dirichlet_vals = build_dirichlet_info(domain)
apply_dirichlet!(A_main, b_main, dirichlet_dofs, dirichlet_vals)

u_main = lu(A_main) \ b_main
ux_main = u_main[1:N]
uy_main = u_main[(N + 1):(2N)]

coords = MM._coords(cloud)
ux_ana = [u_exact(ustrip(pt[1]), ustrip(pt[2])) for pt in coords]
uy_ana = [v_exact(ustrip(pt[1]), ustrip(pt[2])) for pt in coords]

err_ux = norm(ux_main - ux_ana) / norm(ux_ana)
err_uy = norm(uy_main - uy_ana) / norm(uy_ana)
println("  err_ux = $err_ux")
println("  err_uy = $err_uy")
if err_ux < 1e-8 && err_uy < 1e-8
    println("  PASS (Timoshenko recovered to machine precision)")
else
    @warn "  FAIL — check poly_deg and k"
end

# ============================================================================
# Verify active_dofs and build_dirichlet_info
# ============================================================================

println("\n--- Domain-derived AD setup ---")
active = active_dofs(domain)
n_active = count(active)
n_total = length(active)
println("  active DOFs: $n_active / $n_total")
println("  dirichlet DOFs: $(length(dirichlet_dofs))")

# All boundary points should be Dirichlet → inactive
@assert n_active == 2N - length(dirichlet_dofs) "active_dofs count mismatch"

# ============================================================================
# Verify make_system_differentiable(domain) matches mainstream make_system
# ============================================================================

println("\n--- make_system_differentiable(domain) vs mainstream ---")
A_diff, b_diff = make_system_differentiable(model, domain; k = k, basis = basis)
apply_dirichlet!(A_diff, b_diff, dirichlet_dofs, dirichlet_vals)

u_diff = lu(A_diff) \ b_diff
err_A = norm(A_diff - A_main) / norm(A_main)
err_u = norm(u_diff - u_main) / norm(u_main)
println("  ‖A_diff - A_main‖ / ‖A_main‖ = $err_A")
println("  ‖u_diff - u_main‖ / ‖u_main‖ = $err_u")
if err_A < 1e-12 && err_u < 1e-12
    println("  PASS (differentiable assembly matches mainstream)")
else
    @warn "  mismatch between differentiable and mainstream assembly"
end

# ============================================================================
# AD/FD validation via gradient(sim, loss; wrt=:pts)
# ============================================================================

println("\n--- AD/FD validation via gradient(sim, loss; wrt=:pts) ---")

# Probe: von Mises at a few interior points
interior_coords = MM._coords(cloud.volume)
interior_N = length(interior_coords)
probe_local_idx = 1:min(3, interior_N)

# Map volume-local indices to global indices
vol_start = N - interior_N + 1  # boundary points come first, then volume
probe_idx = vol_start .+ (probe_local_idx .- 1)

# Extract flat coordinate vector (reference config)
coords_flat = vcat([ustrip.(collect(p)) for p in coords]...)

# Build the setup tuple (same as what `gradient` wrapper does internally)
adjl_ad = find_neighbors(MM._ustrip(coords), k)
solver = PDESolveIFT(active)

# Two-argument loss: `(pts_flat, setup)` — the signature `gradient()` expects
function loss_fn(pts_flat, setup)
    (; adjl, active, dirichlet_dofs, dirichlet_vals, basis, solver) = setup
    N_local = length(pts_flat) ÷ 2

    A = make_system_differentiable(model, pts_flat, N_local, adjl, basis, λstar, μ)
    b = zeros(eltype(pts_flat), 2N_local)
    apply_dirichlet!(A, b, dirichlet_dofs, dirichlet_vals)
    u = solver(A, b)
    σ_vm = compute_von_mises(u, N_local, pts_flat, adjl, basis, λstar, μ)
    return sum(σ_vm[probe_idx] .^ 2)
end

setup = (;
    adjl = adjl_ad,
    active,
    dirichlet_dofs,
    dirichlet_vals,
    basis,
    solver,
)

# Single-arg closure wrapping loss_fn for direct Mooncake / FD calls
loss_closure(pts_in) = loss_fn(pts_in, setup)

# Verify the loss computes at reference config
L0 = loss_closure(coords_flat)
println("  Loss at reference config: $L0")

# Verify the gradient() wrapper returns a valid result
grad_pts = gradient(sim, loss_fn; k = k, basis = basis)
grad_ad_flat = vcat([collect(g) for g in grad_pts]...)
println("  gradient() wrapper: norm = $(norm(grad_ad_flat))")

# AD gradient via Mooncake directly on the closure
println("  Building Mooncake rrule...")
rule = Mooncake.build_rrule(loss_closure, coords_flat)
_, (_, grad_ad) = Mooncake.value_and_gradient!!(rule, loss_closure, coords_flat)
println("  done  (norm = $(norm(grad_ad)))")

# Verify the gradient wrapper gives the same result as direct Mooncake call
grad_wrapper_err = norm(grad_ad_flat - grad_ad) / norm(grad_ad)
println("  ‖grad(gradient wrapper) - grad(direct)‖ / ‖grad(direct)‖ = $grad_wrapper_err")
if grad_wrapper_err > 1e-12
    @warn "  gradient() wrapper disagrees with direct Mooncake call"
end

# FD gradient (reference)
println("  Computing FD gradient (this may take a moment)...")
fdm = FiniteDifferences.central_fdm(5, 1)
grad_fd = FiniteDifferences.grad(fdm, loss_closure, coords_flat)[1]
println("  done  (norm = $(norm(grad_fd)))")

# Validation
rel_err = norm(grad_ad .- grad_fd) / norm(grad_fd)

println("\n========================================")
println("Phase 2 AD Validation")
println("========================================")
println("N=$N  k=$k  poly_deg=3")
println("AD/FD relative error: $(round(rel_err; sigdigits = 4))")
if rel_err < 1e-3
    println("PASS (rtol < 1e-3)")
else
    @warn "FAIL: rtol = $rel_err"
end
