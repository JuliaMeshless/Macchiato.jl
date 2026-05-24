# ============================================================================
# Plate-with-hole — 3D STL → point cloud (validates the STL-native pipeline)
# ============================================================================
# Loads plate_with_hole.stl (3D thin plate, elliptical through-hole) via WTP and
# fills the 3D volume. The key check: WTP's 3D isinside is octree/ray-based (not
# the 2D single-loop winding test), so it should correctly leave the hole tunnel
# EMPTY. This is the cloud-generation step for the 3D ("semi-3D") lift.
#
# NOTE: this only exercises geometry. The 3D elasticity SOLVE is not here —
# Macchiato's LinearElasticity is 2D plane-stress only (see plan_plate_with_hole.md
# §3D lift); a 3D elasticity model + assembly is the next build.
#
# Run:  jlrun plate_with_hole/plate_with_hole_cloud3d.jl   (from examples/)
# ============================================================================
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using WhatsThePoint
import WhatsThePoint as WTP
using Unitful: m, °, ustrip
using Printf
using CairoMakie

const A = 0.4    # hole semi-axis x (must match the STL)
const B = 0.2    # hole semi-axis y

stl = joinpath(@__DIR__, "plate_with_hole.stl")
part = PointBoundary(stl)
println("Loaded STL: ", stl)
split_surface!(part, 75°)
println("surfaces after split_surface!(75°): ", WTP.names(part))

Δ = 0.05m
println("Discretizing (ConstantSpacing $Δ, VanDerSandeFornberg)...")
cloud = WTP.discretize(part, ConstantSpacing(Δ), alg = VanDerSandeFornberg())

# ---- volume points + hole-exclusion check ----------------------------------
vp = WTP.points(cloud.volume)
vc = WTP.coords.(vp)
xv = [ustrip(c.x) for c in vc]; yv = [ustrip(c.y) for c in vc]; zv = [ustrip(c.z) for c in vc]
ellipse_val(x, y) = (x / A)^2 + (y / B)^2          # <1 ⇔ inside the hole
ev = ellipse_val.(xv, yv)

println()
@printf("volume points: %d\n", length(vp))
@printf("bbox  x[%.3f,%.3f]  y[%.3f,%.3f]  z[%.3f,%.3f]\n",
        extrema(xv)..., extrema(yv)..., extrema(zv)...)
@printf("min ellipse-value over volume = %.3f   (≥1 ⇒ hole tunnel empty)\n", minimum(ev))
nin = count(<(0.9), ev)
println(nin == 0 ? "✓ hole correctly EMPTY (0 volume points inside)" :
                   "✗ $nin volume points fell INSIDE the hole — isinside failed")

# ---- surface breakdown + identify the hole wall ----------------------------
println()
println("boundary surfaces:")
for s in WTP.names(cloud.boundary)
    sp = WTP.points(cloud[s]); sc = WTP.coords.(sp)
    xs = [ustrip(c.x) for c in sc]; ys = [ustrip(c.y) for c in sc]; zs = [ustrip(c.z) for c in sc]
    evs = ellipse_val.(xs, ys)
    ishole = maximum(evs) < 1.4          # all points hug the ellipse ⇒ hole wall
    @printf("  %-12s %5d pts   z[%+.3f,%+.3f]  ellipse-val[%.2f,%.2f] %s\n",
            string(s), length(sp), extrema(zs)..., extrema(evs)..., ishole ? "← HOLE wall (design surface)" : "")
end

# ---- figure ----------------------------------------------------------------
fig = Figure(; size = (900, 700))
ax = Axis3(fig[1, 1]; title = "3D cloud from STL (volume + hole wall)",
           xlabel = "x", ylabel = "y", zlabel = "z", aspect = (4, 4, 1))
scatter!(ax, xv, yv, zv; markersize = 4, color = (:gray, 0.4), label = "volume")
for s in WTP.names(cloud.boundary)
    sp = WTP.points(cloud[s]); sc = WTP.coords.(sp)
    evs = ellipse_val.([ustrip(c.x) for c in sc], [ustrip(c.y) for c in sc])
    maximum(evs) < 1.4 || continue
    scatter!(ax, [ustrip(c.x) for c in sc], [ustrip(c.y) for c in sc], [ustrip(c.z) for c in sc];
             markersize = 6, color = :crimson)
end
figpath = joinpath(@__DIR__, "plate_with_hole_cloud3d.png")
save(figpath, fig)
println("\nFigure saved: ", figpath)
