# Geometry-only "problem setup" renders for the JuliaCon deck (slides 21 & 24).
# Same visual style as the solution plots, but no field — just nodes + objects.
# Outputs land in handoffs/slide-assets/ next to the images they replace.

using Pkg
const EX = "/Users/kylebeggs/Library/Mobile Documents/com~apple~CloudDocs/pro/dev/Macchiato/examples"
const ASSETS = "/Users/kylebeggs/Library/Mobile Documents/com~apple~CloudDocs/pro/dev/Macchiato/handoffs/slide-assets"
Pkg.activate(EX)

using CairoMakie
using JLD2: load
using WhatsThePoint
import WhatsThePoint as WTP
using Macchiato: node_coordinates
using Unitful: m
using Random: seed!

# iCloud stalls close() on multi-MB PNGs written in place — temp file + mv.
function save_png(path, fig; kwargs...)
    tmp = tempname() * ".png"
    save(tmp, fig; kwargs...)
    mv(tmp, path; force = true)
    return path
end

const NODE_GRAY = :gray62

# ============================================================================
# Slide 21: Helmholtz cylinder — annulus of nodes around the PEC cylinder
# ============================================================================
if !isfile(joinpath(ASSETS, "helm_geometry_deck.png"))
data = load(joinpath(EX, "helmholtz_cylinder_solution.jld2"))
xs_h, ys_h = data["xs"], data["ys"]
spacing, a_cyl = data["h"], data["a_cyl"]
R_out = 4.0

fig1 = Figure(; size = (750, 700), fontsize = 21)
ax1 = Axis(fig1[1, 1]; aspect = DataAspect(),
    title = "the domain — an annulus of nodes",
    xlabel = "x (m)", ylabel = "y (m)")
scatter!(ax1, xs_h, ys_h;
    color = NODE_GRAY, markerspace = :data, markersize = 1.4spacing)
poly!(ax1, Circle(Point2f(0, 0), Float32(a_cyl)); color = :gray25)
lines!(ax1, Circle(Point2f(0, 0), Float32(R_out));
    color = (:gray25, 0.6), linestyle = :dash, linewidth = 1.5)
# incident plane wave, traveling +x
for y in (-1.2, 0.0, 1.2)
    arrows!(ax1, [-5.3], [y], [0.75], [0.0];
        color = "#4063D8", linewidth = 3, arrowsize = 14)
end
xlims!(ax1, -5.6, 4.35)
ylims!(ax1, -4.35, 4.35)
save_png(joinpath(ASSETS, "helm_geometry_deck.png"), fig1; px_per_unit = 2)
println("saved helm_geometry_deck.png")
end

# ============================================================================
# Slide 24: Maxwell Julia logo — rectangle of nodes, glass dots, source, sponge
# (geometry section copied from examples/maxwell_julia_logo.jl)
# ============================================================================
seed!(42)

const Lx, Ly = 3.2, 2.4
const h = 0.02
const r_dot = 0.32
const s_tri = 0.72
const cx0, cy0 = 2.0, 1.25
const x_src = (0.75, 0.95)
const L_sp = 0.3

const dots = (
    (label = "red", c = (cx0 - s_tri / 2, cy0 - s_tri / (2 * √3)), color = "#CB3C33"),
    (label = "green", c = (cx0, cy0 + s_tri / √3), color = "#389826"),
    (label = "purple", c = (cx0 + s_tri / 2, cy0 - s_tri / (2 * √3)), color = "#9558B2"),
)

dxu = h * m
rx = dxu:dxu:(Lx * m - dxu)
ry = dxu:dxu:(Ly * m - dxu)
p = vcat(
    map(i -> WTP.Point(i, 0m), rx),
    map(i -> WTP.Point(Lx * m, i), ry),
    map(i -> WTP.Point(i, Ly * m), reverse(rx)),
    map(i -> WTP.Point(0m, i), reverse(ry)),
)
n_out = vcat(
    fill(WTP.Vec(0.0, -1.0), length(rx)),
    fill(WTP.Vec(1.0, 0.0), length(ry)),
    fill(WTP.Vec(0.0, 1.0), length(rx)),
    fill(WTP.Vec(-1.0, 0.0), length(ry)),
)
part = PointBoundary(p, n_out, fill(dxu, length(p)))
cloud = WTP.discretize(part, ConstantSpacing(dxu))
coords = node_coordinates(cloud)
println("cloud: ", length(coords), " points")

xs2 = [x[1] for x in coords]
ys2 = [x[2] for x in coords]

fig2 = Figure(; size = (940, 740))
ax2 = Axis(fig2[1, 1]; aspect = DataAspect(),
    title = "the setup — glass dots, burst source, sponge walls",
    xlabel = "x (m)", ylabel = "y (m)")
scatter!(ax2, xs2, ys2;
    color = NODE_GRAY, markerspace = :data, markersize = 1.4h)
for d in dots
    poly!(ax2, Circle(Point2f(d.c...), r_dot);
        color = (d.color, 0.25), strokecolor = d.color, strokewidth = 2.5)
end
lines!(ax2, [L_sp, Lx - L_sp, Lx - L_sp, L_sp, L_sp],
    [L_sp, L_sp, Ly - L_sp, Ly - L_sp, L_sp];
    color = (:black, 0.75), linestyle = :dash, linewidth = 2.5)
text!(ax2, L_sp + 0.05, L_sp + 0.03;
    text = "sponge", color = (:black, 0.75), fontsize = 22.5)
scatter!(ax2, [x_src[1]], [x_src[2]];
    marker = :star8, markersize = 22, color = "#4063D8")
text!(ax2, x_src[1] + 0.06, x_src[2] - 0.02;
    text = "source", color = "#4063D8", fontsize = 24)
save_png(joinpath(ASSETS, "logo_geometry_deck.png"), fig2; px_per_unit = 2)
println("saved logo_geometry_deck.png")
println("DONE")
