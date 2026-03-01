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
#   v(x,y) = P/(6EI) [3νy²(L-x) + (4+5ν)D²x + (3L-x)x²]
#
# where I = 2D³/3 is the second moment of area.
# ============================================================================
using Pkg
Pkg.activate(@__DIR__)

using Macchiato
import Macchiato as MM
using WhatsThePoint
import WhatsThePoint as WTP
using RadialBasisFunctions: PHS
using Unitful: m, °, ustrip
using LinearAlgebra
using LinearSolve
using SparseArrays: nnz
using Statistics: mean
using CairoMakie

# ============================================================================
# Problem Parameters
# ============================================================================

L = 8.0    # Beam length
D = 1.0    # Half-height (beam goes from y=-D to y=D)
P = 1000.0 # Applied load (total shear force)
E_val = 1.0e7
ν_val = 0.3
I = 2D^3 / 3  # Second moment of area

# ============================================================================
# Timoshenko Analytical Solution
# ============================================================================

function u_exact(x, y)
    return -P / (6E_val * I) * y * ((6L - 3x) * x + (2 + ν_val) * (y^2 - D^2))
end

function v_exact(x, y)
    return P / (6E_val * I) * (3ν_val * y^2 * (L - x) + (4 + 5ν_val) * D^2 * x + (3L - x) * x^2)
end

# ============================================================================
# Domain Setup
# ============================================================================

dx = 0.1 * m  # Point spacing

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
cloud, _ = repel(cloud, ConstantSpacing(Δ); α = Δ / 50, max_iters = 5000)

# ============================================================================
# Visualize Point Cloud
# ============================================================================

fig_cloud = Figure(; size = (1000, 400));
ax_cloud = Axis(fig_cloud[1, 1]; title = "Point Cloud", xlabel = "x", ylabel = "y", aspect = DataAspect())

for surf_name in WTP.names(cloud.boundary)
    pts_s = WTP.points(cloud[surf_name])
    c_s = WTP.coords.(pts_s)
    xs = [ustrip(c.x) for c in c_s]
    ys = [ustrip(c.y) for c in c_s]
    scatter!(ax_cloud, xs, ys; markersize = 10, label = string(surf_name))
end

pts_v = WTP.points(cloud.volume)
c_v = WTP.coords.(pts_v)
xv = [ustrip(c.x) for c in c_v]
yv = [ustrip(c.y) for c in c_v]
scatter!(ax_cloud, xv, yv; markersize = 8, color = :gray, label = "interior")
axislegend(ax_cloud; position = :rb)
fig_cloud

# ============================================================================
# Boundary Conditions
# ============================================================================

# Left (clamped): exact displacement from Timoshenko solution
bc_left(x, t) = (u_exact(x[1], x[2]), v_exact(x[1], x[2]))

# Right: parabolic shear traction  σ₁₂ = P(D²-y²)/(2I)
# The traction vector at x=L with n=(1,0) is:
#   tx = σ₁₁·nx = 0  (σ₁₁ = 0 at x=L)
#   ty = σ₁₂·nx = P(D²-y²)/(2I)
bc_right(x, t) = (0.0, P * (D^2 - x[2]^2) / (2I))

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

basis_kw = (; basis = PHS(3; poly_deg = 3))

# Warmup (JIT compilation)
prob = LinearSolve.LinearProblem(sim.domain; basis_kw...)
sol = LinearSolve.solve(prob)

# Timed runs
GC.gc()
t_assembly = @elapsed prob = LinearSolve.LinearProblem(sim.domain; basis_kw...)
t_solve = @elapsed sol = LinearSolve.solve(prob)

sim._solution = sol.u
sim.iteration = 1

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

abs_err_ux = abs.(err_ux)
abs_err_uy = abs.(err_uy)

mean_abs_ux = mean(abs_err_ux)
mean_abs_uy = mean(abs_err_uy)
max_abs_ux = maximum(abs_err_ux)
max_abs_uy = maximum(abs_err_uy)

println("\n========================================")
println("Performance Summary")
println("========================================")
println("Method:        Meshless (PHS RBF)")
println("Points:        $N")
println("DOFs:          $(2N)")
println("System nnz:    $(nnz(prob.A))")
println("Assembly time: $(round(t_assembly; digits = 4)) s")
println("Solve time:    $(round(t_solve; digits = 4)) s")
println("Total time:    $(round(t_assembly + t_solve; digits = 4)) s")
println()
println("========================================")
println("Cantilever Beam Results")
println("========================================")
println("Beam: L=$L, D=$D, P=$P, E=$E_val, ν=$ν_val")
println("Points: $N")
println()
println("Absolute Error (vs Timoshenko):")
println("  ux: mean = $(round(mean_abs_ux; digits = 8)), max = $(round(max_abs_ux; digits = 8))")
println("  uy: mean = $(round(mean_abs_uy; digits = 8)), max = $(round(max_abs_uy; digits = 8))")
println()

# Max tip deflection (analytical at x=L, y=0):
v_tip_exact = v_exact(L, 0.0)
println("Tip deflection (analytical): $v_tip_exact")

# Find rightmost point for comparison
max_x = maximum(ustrip(pt.x) for pt in coords)
tip_indices = findall(
    i -> abs(ustrip(coords[i].x) - max_x) < 0.01 &&
        abs(ustrip(coords[i].y)) < 0.01 + ustrip(dx), 1:N
)
if !isempty(tip_indices)
    v_tip_num = mean(uy_sim[tip_indices])
    tip_abs_err = abs(v_tip_num - v_tip_exact)
    println("Tip deflection (numerical): $v_tip_num")
    println("Tip deflection error: $(round(tip_abs_err; digits = 8))")
end

# ============================================================================
# Visualization
# ============================================================================

x = [ustrip(pt.x) for pt in coords]
y = [ustrip(pt.y) for pt in coords]

displacement_mag = sqrt.(ux_sim .^ 2 .+ uy_sim .^ 2)
ana_mag = sqrt.(ux_ana .^ 2 .+ uy_ana .^ 2)
error_mag = sqrt.(err_ux .^ 2 .+ err_uy .^ 2)

fig = Figure(; size = (1400, 1800));

# Row 1: analytical displacement components
ax1 = Axis(fig[1, 1]; title = "uₓ (analytical)", xlabel = "x", ylabel = "y", aspect = DataAspect())
sc1 = scatter!(ax1, x, y; color = ux_ana, colormap = :RdBu, markersize = 6)
Colorbar(fig[1, 2], sc1)

ax2 = Axis(fig[1, 3]; title = "uᵧ (analytical)", xlabel = "x", ylabel = "y", aspect = DataAspect())
sc2 = scatter!(ax2, x, y; color = uy_ana, colormap = :RdBu, markersize = 6)
Colorbar(fig[1, 4], sc2)

# Row 2: numerical displacement components
ax3 = Axis(fig[2, 1]; title = "uₓ (numerical)", xlabel = "x", ylabel = "y", aspect = DataAspect())
sc3 = scatter!(ax3, x, y; color = ux_sim, colormap = :RdBu, markersize = 6)
Colorbar(fig[2, 2], sc3)

ax4 = Axis(fig[2, 3]; title = "uᵧ (numerical)", xlabel = "x", ylabel = "y", aspect = DataAspect())
sc4 = scatter!(ax4, x, y; color = uy_sim, colormap = :RdBu, markersize = 6)
Colorbar(fig[2, 4], sc4)

# Row 3: displacement magnitude (analytical vs numerical)
ax5 = Axis(fig[3, 1]; title = "‖u‖ (analytical)", xlabel = "x", ylabel = "y", aspect = DataAspect())
sc5 = scatter!(ax5, x, y; color = ana_mag, colormap = :viridis, markersize = 6)
Colorbar(fig[3, 2], sc5)

ax6 = Axis(fig[3, 3]; title = "‖u‖ (numerical)", xlabel = "x", ylabel = "y", aspect = DataAspect())
sc6 = scatter!(ax6, x, y; color = displacement_mag, colormap = :viridis, markersize = 6)
Colorbar(fig[3, 4], sc6)

# Row 4: absolute error
ax7 = Axis(fig[4, 1]; title = "abs error uₓ", xlabel = "x", ylabel = "y", aspect = DataAspect())
sc7 = scatter!(ax7, x, y; color = abs_err_ux, colormap = :inferno, markersize = 6)
Colorbar(fig[4, 2], sc7)

ax8 = Axis(fig[4, 3]; title = "abs error uᵧ", xlabel = "x", ylabel = "y", aspect = DataAspect())
sc8 = scatter!(ax8, x, y; color = abs_err_uy, colormap = :inferno, markersize = 6)
Colorbar(fig[4, 4], sc8)
fig
