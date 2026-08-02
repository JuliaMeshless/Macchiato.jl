# generate_hero.jl
# One-time script to regenerate the VitePress hero image and navbar logo.
# Run from repo root: julia docs/generate_hero.jl
#
# Unlike `generate_assets.jl`, which produces labelled figures for the Gallery and
# Examples pages, this script produces *art*: no axes, no colorbar, no title, and a
# transparent background so the images sit cleanly on the VitePress hero gradient.
#
# Output goes to `docs/src/public/`, which VitePress serves at the site root — so
# `hero.png` is referenced as `/hero.png` in the `index.md` frontmatter.

using Pkg
Pkg.activate(joinpath(@__DIR__))
Pkg.instantiate()

using Macchiato
import Macchiato as MM
using WhatsThePoint
import WhatsThePoint as WTP
using Unitful: m, °, ustrip
using CairoMakie

const PUBLIC_DIR = joinpath(@__DIR__, "src", "public")
mkpath(PUBLIC_DIR)

# Julia brand colors, matching docs/src/.vitepress/theme/style.css
const JULIA_PURPLE = "#9558B2"
const JULIA_RED = "#CB3C33"
const JULIA_GREEN = "#389826"

"""
Build a rectangular boundary point cloud of size `Lx × Ly` with spacing `dx`.
Shared by both renders; mirrors the geometry helper in `generate_assets.jl`.
"""
function rectangle_cloud(Lx, Ly, dx)
    rx = dx:dx:(Lx - dx)
    ry = dx:dx:(Ly - dx)

    pts = vcat(
        [WTP.Point(i, zero(Ly)) for i in rx],
        [WTP.Point(Lx, i) for i in ry],
        [WTP.Point(i, Ly) for i in reverse(rx)],
        [WTP.Point(zero(Lx), i) for i in reverse(ry)],
    )
    nrms = vcat(
        [WTP.Vec(0.0, -1.0) for _ in rx],
        [WTP.Vec(1.0, 0.0) for _ in ry],
        [WTP.Vec(0.0, 1.0) for _ in rx],
        [WTP.Vec(-1.0, 0.0) for _ in ry],
    )
    areas = fill(dx, length(pts))

    part = PointBoundary(pts, nrms, areas)
    split_surface!(part, 75°)
    return WTP.discretize(part, ConstantSpacing(dx))
end

"""
Strip every decoration from an axis so only the data remains.
"""
function bare_axis!(ax)
    hidedecorations!(ax)
    hidespines!(ax)
    return ax
end

# ============================================================================
# hero.png — steady-state temperature on the unit square, rendered as art
# ============================================================================
#
# The unit square is deliberate: VitePress renders the hero image into a roughly
# square slot only ~250 px wide, so a square subject fills the frame where the
# 8:2 cantilever beam shrank to an illegible sliver. The spacing is also much
# coarser than the Examples page uses — at that display size anything finer
# turns into sub-pixel dots and washes out to grey.

function generate_hero()
    dx = 1 / 26 * m
    cloud = rectangle_cloud(1m, 1m, dx)

    bcs = Dict(
        :surface1 => MM.Temperature(0.0),
        :surface2 => MM.Temperature(0.0),
        :surface3 => MM.Temperature(100.0),
        :surface4 => MM.Temperature(0.0),
    )

    domain = MM.Domain(cloud, bcs, SolidEnergy(k = 1.0, ρ = 1.0, cₚ = 1.0))
    sim = Simulation(domain)
    run!(sim)
    T = temperature(sim)

    coords = MM._coords(cloud)
    x = [ustrip(pt.x) for pt in coords]
    y = [ustrip(pt.y) for pt in coords]

    fig = Figure(; size = (1000, 1000), backgroundcolor = :transparent)
    ax = Axis(fig[1, 1]; backgroundcolor = :transparent, aspect = DataAspect())
    bare_axis!(ax)
    scatter!(
        ax,
        x,
        y;
        color = T,
        colormap = cgrad([JULIA_GREEN, JULIA_PURPLE, JULIA_RED]),
        markersize = 24,
    )

    save(joinpath(PUBLIC_DIR, "hero.png"), fig; px_per_unit = 2)
    return println("Saved hero.png")
end

# ============================================================================
# logo.png — point-cloud disc coloured by a smooth radial field
# ============================================================================
#
# An RBF-FD stencil: one centre node and the neighbours its weights are built from.
# That is literally the unit of computation in Macchiato, and — unlike a dense
# scatter — it stays legible at the 24×24 navbar size, where anything with more
# than a handful of marks collapses into a coloured blob.

function generate_logo()
    n_neighbors = 6
    θs = [2π * j / n_neighbors + π / 6 for j in 0:(n_neighbors - 1)]
    nx = [cos(θ) for θ in θs]
    ny = [sin(θ) for θ in θs]

    fig = Figure(; size = (512, 512), backgroundcolor = :transparent)
    ax = Axis(fig[1, 1]; backgroundcolor = :transparent, aspect = DataAspect())
    bare_axis!(ax)

    # Stencil legs, drawn first so the nodes sit on top of them.
    for (x, y) in zip(nx, ny)
        lines!(ax, [0.0, x], [0.0, y]; color = (JULIA_PURPLE, 0.45), linewidth = 7)
    end

    # Neighbour nodes alternate green/red around the ring; centre node is purple.
    scatter!(
        ax,
        nx,
        ny;
        color = [isodd(j) ? JULIA_GREEN : JULIA_RED for j in 1:n_neighbors],
        markersize = 62,
    )
    scatter!(ax, [0.0], [0.0]; color = JULIA_PURPLE, markersize = 92)

    # Breathing room so the outer nodes are not clipped.
    limits!(ax, -1.35, 1.35, -1.35, 1.35)

    save(joinpath(PUBLIC_DIR, "logo.png"), fig; px_per_unit = 2)
    return println("Saved logo.png")
end

# ============================================================================
# Run
# ============================================================================

generate_hero()
generate_logo()
println("Hero assets generated in $(PUBLIC_DIR)")
