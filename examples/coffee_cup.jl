# Coffee-mug conduction render — the docs home-page hero.
#
# A stylized mug (examples/assets/mug.stl, authored by examples/assets/make_mug.py)
# heated from the bottom: fixed hot temperature on the base underside, convective
# losses everywhere else. Steady-state solve, scattered points coloured by
# temperature in the Julia brand colors.
#
# Run from the repo root: julia --project=examples examples/coffee_cup.jl
# Outputs:
#   docs/src/public/hero.png        — VitePress hero, served as /hero.png (committed)
#   examples/coffee_cup_preview.png — QA render with axes + colorbar (gitignored)
#
# The hero renders into a ~250 px roughly-square slot, so the spacing is coarse
# and the markers fat on purpose — anything finer washes out to grey.

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using WhatsThePoint
import WhatsThePoint as WTP
using Macchiato
import Macchiato as MM
import Meshes
using Unitful: mm, m, ustrip, @u_str
using CairoMakie
using Random: seed!

# The Poisson-disk surface sampler and Bridson fill draw from the global RNG, so
# every run is a different cloud realization. The steady shadow-point BC rows are
# sensitive to the realization on this coarse thin-walled geometry — most clouds
# solve cleanly but some blow through the maximum principle at the bottom
# Dirichlet/Robin junction. Seed 3 is verified clean (T stays within [29.9, 100.7]);
# the assert after run! catches a bad realization if anything here changes.
seed!(3)

# Julia brand colors, matching docs/src/.vitepress/theme/style.css
const JULIA_PURPLE = "#9558B2"
const JULIA_RED = "#CB3C33"
const JULIA_GREEN = "#389826"

const dx = 4.0e-3m  # 4 mm

# ============================================================================
# Geometry: import the STL, promote to Float64, convert to metres
# ============================================================================
#
# Binary STL stores Float32 and `import_mesh` preserves the machine type, which
# would otherwise propagate all the way into the RBF weight assembly. The
# conversion to metres is load-bearing too: the Octree interior fill constructs
# its points from raw floats, which Meshes defaults to metres — a boundary in
# any other unit fails to combine with the generated volume.

mesh32 = import_mesh(joinpath(@__DIR__, "assets", "mug.stl"), mm)
verts = map(Meshes.vertices(mesh32)) do p
    xyz = Float64.(ustrip.(u"m", Meshes.to(p))) .* m
    Meshes.Point(xyz...)
end
mesh = Meshes.SimpleMesh(verts, Meshes.topology(mesh32))

part = PointBoundary(mesh, ConstantSpacing(dx))

# Partition the single sampled surface by height: the base underside sits at
# z = 0 and nothing else (the handle bottoms out at z = 10 mm) is below 2 mm.
surf = part[:surface1]
pts = WTP.points(surf)
nrm = WTP.normal(surf)
ar = WTP.area(surf)
bottom = [ustrip(u"m", Meshes.to(p)[3]) < 2.0e-3 for p in pts]
delete!(part, :surface1)
part[:bottom] = PointSurface(pts[bottom], nrm[bottom], ar[bottom])
part[:walls] = PointSurface(pts[.!bottom], nrm[.!bottom], ar[.!bottom])
@assert count(bottom) > 100 "suspiciously few bottom points: $(count(bottom))"
@assert count(!, bottom) > 1000 "suspiciously few wall points: $(count(!, bottom))"

cloud = WTP.discretize(part, ConstantSpacing(dx); alg = Octree(mesh; spacing = ConstantSpacing(dx)))
println("cloud: ", length(WTP.points(cloud)), " points ",
    "(bottom ", count(bottom), ", walls ", count(!, bottom), ")")

# ============================================================================
# Physics: steady conduction, hot base, convective losses
# ============================================================================
#
# Robin BC is h·T + k·∂T/∂n = h·T∞ with lengths in metres, so h/k = 4 m⁻¹
# puts the fin decay length √(k·t/2h) ≈ 35 mm ≈ half the mug height — the
# gradient spans the full mug instead of saturating hot or collapsing at the base.
# Don't lower h below ~4 to stretch the gradient further: the weaker Robin
# diagonal makes the system fragile near the Dirichlet/Robin junction ring and
# some cloud realizations blow through the maximum principle. The hero render
# stretches the colors instead (see `shade` below).

const T∞ = 20.0
const T_hot = 100.0
const k_th = 1.0

bcs = Dict(
    :bottom => MM.Temperature(T_hot),
    :walls => MM.Convection(4.0, k_th, T∞),
)

domain = MM.Domain(cloud, bcs, SolidEnergy(k = k_th, ρ = 1.0, cₚ = 1.0))
sim = Simulation(domain)
# The default steady scheme (one-sided directional derivative rows) is unstable on
# thin 3D shells like the mug wall — it violates the maximum principle by ±200°.
# Shadow points anchor ∂T/∂n on interpolated interior points instead, which keeps
# the Robin rows well-posed on walls only ~2.5 nodes thick.
run!(sim; scheme = ShadowPoints(ustrip(u"m", dx), 2))
T = temperature(sim)

@assert all(isfinite, T)
@assert 0.0 < minimum(T) && maximum(T) < 115.0 "T out of range: $(extrema(T))"
println("solved: extrema(T) = ", extrema(T))

# ============================================================================
# Render
# ============================================================================

coords = MM._coords(cloud)
x = [ustrip(u"m", c[1]) for c in coords]
y = [ustrip(u"m", c[2]) for c in coords]
z = [ustrip(u"m", c[3]) for c in coords]

az, el = 1.45π, π / 8
eye = (cos(el) * cos(az), cos(el) * sin(az), sin(el))
ord = sortperm(x .* eye[1] .+ y .* eye[2] .+ z .* eye[3])  # far → near for overdraw

heatmap_grad = cgrad([JULIA_GREEN, JULIA_PURPLE, JULIA_RED])

# iCloud's file provider intermittently stalls `close` on multi-MB PNGs written
# in place under the synced repo ("SystemError: close: Operation timed out").
# Render to a local temp file and move it into position in one bulk copy instead.
function save_png(path, fig; kwargs...)
    tmp = tempname() * ".png"
    save(tmp, fig; kwargs...)
    mv(tmp, path; force = true)
    return path
end

# Hero-only color stretch: the physical profile decays exponentially up the wall,
# which crowds most of the mug into the bottom fifth of the colormap. A gentle
# power stretch of normalized temperature spreads the green→purple→red sweep over
# the full height without touching the physics — 0.65 keeps a green cast at the
# rim and handle (√ washes it out to grey). The QA preview keeps the linear scale.
shade = clamp.((T .- T∞) ./ (T_hot - T∞), 0.0, 1.0) .^ 0.65

function mug_scatter!(ax, color, colorrange; markersize = 1.8e-3)  # marker radius in data units (m)
    return meshscatter!(
        ax,
        x[ord],
        y[ord],
        z[ord];
        color = color[ord],
        colorrange = colorrange,
        colormap = heatmap_grad,
        markersize = markersize,
    )
end

# --- hero: no decorations, transparent background --------------------------
fig = Figure(; size = (1000, 1000), backgroundcolor = :transparent)
ax = Axis3(
    fig[1, 1];
    azimuth = az,
    elevation = el,
    aspect = :data,
    viewmode = :fitzoom,
    protrusions = 0,
    xypanelvisible = false,
    xzpanelvisible = false,
    yzpanelvisible = false,
    backgroundcolor = :transparent,
)
hidedecorations!(ax)
hidespines!(ax)
mug_scatter!(ax, shade, (0.0, 1.0))

hero_path = joinpath(@__DIR__, "..", "docs", "src", "public", "hero.png")
save_png(hero_path, fig; px_per_unit = 2)
@assert isfile(hero_path) && filesize(hero_path) > 20_000
println("saved: ", abspath(hero_path))

# --- QA preview: axes + colorbar, white background -------------------------
qa = Figure(; size = (1200, 1000))
qax = Axis3(qa[1, 1]; azimuth = az, elevation = el, aspect = :data)
sc = mug_scatter!(qax, T, (T∞, T_hot))
Colorbar(qa[1, 2], sc; label = "T")

qa_path = joinpath(@__DIR__, "coffee_cup_preview.png")
save_png(qa_path, qa; px_per_unit = 2)
println("saved: ", abspath(qa_path))
