# Visualization for the Helmholtz cylinder-scattering example. Loads the
# solution saved by helmholtz_cylinder.jl (which also includes this script as
# its final step) and renders examples/helmholtz_cylinder.png (gitignored) —
# 1x2 figure: |E| amplitude with the shadow region and standing waves, and the
# pointwise |error| against the exact series on a log scale.
#
# The solve is minutes; the render is seconds. Tweak the figure here and
# re-render without re-solving:
#   julia --project=examples examples/helmholtz_cylinder_viz.jl

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using CairoMakie
using JLD2: load

data = load(joinpath(@__DIR__, "helmholtz_cylinder_solution.jld2"))
xs, ys = data["xs"], data["ys"]
E_num, E_ex = data["E_num"], data["E_ex"]
# named to avoid clashing with the solve script's consts when included from it
spacing, cyl_radius = data["h"], data["a_cyl"]

# Dirichlet rows sit near machine eps — far below the interior error band. Clamp
# to a floor just under that band so the log colorbar resolves the interior
# variation instead of stretching over the empty decades down to eps.
err_abs = abs.(E_num .- E_ex)
floor_val = 1.0e-8
err_plot = max.(err_abs, floor_val)
err_range = (floor_val, max(maximum(err_abs), 10floor_val))

# |E| gets a sequential map anchored at zero, built from the Julia logo colors
# to match the Maxwell showcase: white → green → purple → red.
const CMAP_SEQ = cgrad(["#ffffff", "#389826", "#9558B2", "#CB3C33"], [0.0, 0.35, 0.7, 1.0])

# iCloud's file provider intermittently stalls `close` on multi-MB PNGs written
# in place under the synced repo ("SystemError: close: Operation timed out").
# Render to a local temp file and move it into position in one bulk copy instead.
function save_png(path, fig; kwargs...)
    tmp = tempname() * ".png"
    save(tmp, fig; kwargs...)
    mv(tmp, path; force = true)
    return path
end

# 1.5x the default 14pt so the labels stay legible at slide scale.
fig = Figure(; size = (1500, 700), fontsize = 21)

function field_panel!(where_, color, colorrange; title, colormap, colorscale = identity)
    ax = Axis(
        where_; aspect = DataAspect(), title = title,
        xlabel = "x (m)", ylabel = "y (m)"
    )
    plt = scatter!(
        ax, xs, ys;
        color = color, colorrange = colorrange, colormap = colormap,
        colorscale = colorscale, markerspace = :data, markersize = 1.4spacing
    )
    # Fill the PEC cylinder so it reads as a solid object and hides the hole rim.
    poly!(ax, Circle(Point2f(0, 0), Float32(cyl_radius)); color = :gray25)
    return plt
end

p1 = field_panel!(
    fig[1, 1], abs.(E_num), (0, maximum(abs, E_num));
    title = "|E| — shadow and standing waves", colormap = CMAP_SEQ
)
Colorbar(fig[1, 2], p1)
p2 = field_panel!(
    fig[1, 3], err_plot, err_range;
    title = "|E − E_exact| (log scale)", colormap = :inferno, colorscale = log10
)
Colorbar(fig[1, 4], p2)

png_path = joinpath(@__DIR__, "helmholtz_cylinder.png")
save_png(png_path, fig; px_per_unit = 2)
@assert isfile(png_path) && filesize(png_path) > 20_000
println("saved: ", abspath(png_path))
