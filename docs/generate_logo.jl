# generate_logo.jl
# One-time script to regenerate the navbar logo.
# Run from repo root: julia docs/generate_logo.jl
#
# The hero image (`docs/src/public/hero.png`) is produced by
# `examples/coffee_cup.jl` — this script only generates the logo.
#
# Unlike `generate_assets.jl`, which produces labelled figures for the Gallery and
# Examples pages, this script produces *art*: no axes, no colorbar, no title, and a
# transparent background so the image sits cleanly on the VitePress hero gradient.
#
# Output goes to `docs/src/public/`, which VitePress serves at the site root.

using Pkg
Pkg.activate(joinpath(@__DIR__))
Pkg.instantiate()

using CairoMakie

const PUBLIC_DIR = joinpath(@__DIR__, "src", "public")
mkpath(PUBLIC_DIR)

# Julia brand colors, matching docs/src/.vitepress/theme/style.css
const JULIA_PURPLE = "#9558B2"
const JULIA_RED = "#CB3C33"
const JULIA_GREEN = "#389826"

"""
Strip every decoration from an axis so only the data remains.
"""
function bare_axis!(ax)
    hidedecorations!(ax)
    hidespines!(ax)
    return ax
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

generate_logo()
println("Logo generated in $(PUBLIC_DIR)")
