# ============================================================================
# 2D Cantilever Beam - Linear Elasticity
# ============================================================================
# Validates against Timoshenko analytical solution for a cantilever beam
# under end-load (parabolic shear distribution).
#
# Geometry: L × 2D beam, x ∈ [0, L], y ∈ [-D, D]
# Left end (x=0): Clamped (prescribed displacement from exact solution)
# Right end (x=L): Parabolic shear traction
# Top/bottom (y=±D): Traction-free
#
# Timoshenko beam solution (plane stress):
#   u(x,y) = -P/(6EI) [y((6L-3x)x + (2+ν)(y²-D²))]
#   v(x,y) = P/(6EI) [3νy²(L-x) + (4+5ν)D²x/4 + (3L-x)x²]
#
# where I = 2D³/3 is the second moment of area.
# ============================================================================

using MeshlessMultiphysics
import MeshlessMultiphysics as MM
using WhatsThePoint
import WhatsThePoint as WTP
using Unitful: m, °, ustrip
using LinearAlgebra
using LinearSolve
using Statistics: mean

# ============================================================================
# Problem Parameters
# ============================================================================

L = 8.0    # Beam length
D = 1.0    # Half-height (beam goes from y=-D to y=D)
P = 1000.0 # Applied load (total shear force)
E_val = 1e7
ν_val = 0.3
I = 2D^3 / 3  # Second moment of area

# ============================================================================
# Timoshenko Analytical Solution
# ============================================================================

function u_exact(x, y)
    return -P / (6E_val * I) * y * ((6L - 3x) * x + (2 + ν_val) * (y^2 - D^2))
end

function v_exact(x, y)
    return P / (6E_val * I) * (3ν_val * y^2 * (L - x) + (4 + 5ν_val) * D^2 * x / 4 + (3L - x) * x^2)
end

# ============================================================================
# Domain Setup
# ============================================================================

dx = 0.2 * m  # Point spacing

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
cloud, _ = repel(cloud, ConstantSpacing(Δ); α = Δ / 20, max_iters = 500)

println("Total points: ", length(cloud))

# ============================================================================
# Boundary Conditions
# ============================================================================

# Left (clamped): exact displacement from Timoshenko solution
bc_left(x, t) = (u_exact(x[1], x[2]), v_exact(x[1], x[2]))

# Right: parabolic shear traction  τ = P(D²-y²)/(2I)
# The traction vector at x=L with n=(1,0) is:
#   tx = σ₁₁·nx = σ₁₁  (need to compute from exact solution)
#   ty = σ₁₂·nx = σ₁₂
# From Timoshenko: σ₁₂ = -P(D²-y²)/(2I), σ₁₁ = 0 at x=L
bc_right(x, t) = (0.0, -P * (D^2 - x[2]^2) / (2I))

bcs = Dict(
    :surface1 => TractionFree(),                   # Bottom: free surface
    :surface2 => Traction(bc_right),                # Right: parabolic shear
    :surface3 => TractionFree(),                   # Top: free surface
    :surface4 => Displacement(bc_left),            # Left: clamped (exact displacement)
)

# ============================================================================
# Model and Solve
# ============================================================================

model = LinearElasticity(E = E_val, ν = ν_val)
domain = MM.Domain(cloud, bcs, model)

sim = Simulation(domain)
set!(sim, ux = 0.0, uy = 0.0)
run!(sim)

# ============================================================================
# Compare with Analytical Solution
# ============================================================================

ux_sim, uy_sim = displacement(sim)
N = length(cloud)

coords = MM._coords(cloud)
ux_ana = [u_exact(ustrip(pt.x), ustrip(pt.y)) for pt in coords]
uy_ana = [v_exact(ustrip(pt.x), ustrip(pt.y)) for pt in coords]

err_ux = ux_sim .- ux_ana
err_uy = uy_sim .- uy_ana

L2_ux = norm(err_ux) / sqrt(N)
L2_uy = norm(err_uy) / sqrt(N)
Linf_ux = norm(err_ux, Inf)
Linf_uy = norm(err_uy, Inf)

println("\n========================================")
println("Cantilever Beam Results")
println("========================================")
println("Beam: L=$L, D=$D, P=$P, E=$E_val, ν=$ν_val")
println("Points: $N")
println()
println("Error Analysis (vs Timoshenko):")
println("  ux: L2 = $L2_ux, L∞ = $Linf_ux")
println("  uy: L2 = $L2_uy, L∞ = $Linf_uy")
println()

# Max tip deflection (analytical at x=L, y=0):
v_tip_exact = v_exact(L, 0.0)
println("Tip deflection (analytical): $v_tip_exact")

# Find rightmost point for comparison
max_x = maximum(ustrip(pt.x) for pt in coords)
tip_indices = findall(i -> abs(ustrip(coords[i].x) - max_x) < 0.01 &&
                           abs(ustrip(coords[i].y)) < 0.01 + ustrip(dx), 1:N)
if !isempty(tip_indices)
    v_tip_num = mean(uy_sim[tip_indices])
    println("Tip deflection (numerical): $v_tip_num")
    println("Relative error: $(abs(v_tip_num - v_tip_exact) / abs(v_tip_exact))")
end
