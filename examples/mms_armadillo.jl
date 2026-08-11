# Custom-PDE verification on the Stanford armadillo — manufactured solutions.
#
# Solves the steady advection-diffusion equation ν∇²u − c⋅∇u = f on the Stanford
# armadillo (examples/assets/armadillo.stl, prepared by
# examples/assets/prep_armadillo.py from the "Armadillo" model in the Stanford
# Computer Graphics Laboratory 3D Scanning Repository,
# graphics.stanford.edu/data/3Dscanrep) with a manufactured solution
# u = sin(αx)sin(αy)sin(αz). The source term f and the Dirichlet boundary values
# come from the exact solution, so the numerical error is measurable at every
# node. The model is the AdvectionDiffusion pattern from docs/src/custom_pdes.md,
# exercised on real 3D geometry.
#
# Run from the repo root: julia --project=examples examples/mms_armadillo.jl
# Output: examples/mms_armadillo.png (gitignored) — a 2x2 figure: solution and
# pointwise |error|, each rendered full and cut open at the mid-y plane so the
# interior of the domain is visible.
#
# Requires WhatsThePoint ≥ 0.3 (the `Orthtree` volume-fill API).

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
using StaticArrays: SVector
using LinearAlgebra: norm
using Random: seed!

# The Poisson-disk surface sampler and Bridson fill draw from the global RNG, so
# every run is a different cloud realization. The asserts after the solve catch
# a bad realization if anything here changes.
seed!(42)

# 3 mm keeps the thin features (ears, fingers) sampled — the armadillo is a
# ~151 mm tall statue with much finer extremities than the moka pot.
const dx = 3.0e-3m

# ============================================================================
# Geometry: import the STL, promote to Float64, convert to metres
# ============================================================================
#
# Binary STL stores Float32 and `import_mesh` preserves the machine type, which
# would otherwise propagate all the way into the RBF weight assembly. The
# conversion to metres is load-bearing too: the Orthtree interior fill constructs
# its points from raw floats, which Meshes defaults to metres — a boundary in
# any other unit fails to combine with the generated volume.

# Turn the armadillo 30° about z so the render shows a 3/4 view instead of a
# head-on one — the y-normal cutaway plane then sections the body obliquely.
const θz = π / 6
const sinθ, cosθ = sincos(θz)

mesh32 = import_mesh(joinpath(@__DIR__, "assets", "armadillo.stl"), mm)
verts = map(Meshes.vertices(mesh32)) do p
    v = Float64.(ustrip.(u"m", Meshes.to(p)))
    xyz = SVector(cosθ * v[1] - sinθ * v[2], sinθ * v[1] + cosθ * v[2], v[3]) .* m
    Meshes.Point(xyz...)
end
mesh = Meshes.SimpleMesh(verts, Meshes.topology(mesh32))

# All boundary conditions are identical (Dirichlet from the exact solution), so
# the single sampled surface needs no splitting.
part = PointBoundary(mesh, ConstantSpacing(dx))
@assert length(WTP.points(part[:surface1])) > 1000 "suspiciously few boundary points"

cloud = WTP.discretize(part, ConstantSpacing(dx); alg = Orthtree(mesh; spacing = ConstantSpacing(dx)))
println("cloud: ", length(WTP.points(cloud)), " points ",
    "(", length(WTP.points(part[:surface1])), " boundary)")

# ============================================================================
# Manufactured solution: u = sin(αx)sin(αy)sin(αz)
# ============================================================================
#
# α is scaled so one full period spans the armadillo's largest extent (~151 mm)
# — about 50 points per wavelength at 3 mm spacing, comfortably resolved.

vert_xyz = [ustrip.(u"m", Meshes.to(p)) for p in verts]
bbox_lo = reduce((a, b) -> min.(a, b), vert_xyz)
bbox_hi = reduce((a, b) -> max.(a, b), vert_xyz)
const α = 2π / maximum(bbox_hi .- bbox_lo)

# Advection velocity: |c|·dx ≈ 0.24 keeps the grid Péclet number well below 1,
# so the centered RBF-FD discretization is stable without upwinding. Don't push
# |c| past ~300 at this spacing without adding stabilization.
const ν = 1.0
const c = SVector(60.0, 45.0, 30.0)

u_exact(x) = sin(α * x[1]) * sin(α * x[2]) * sin(α * x[3])
∇u_exact(x) = α .* SVector(
    cos(α * x[1]) * sin(α * x[2]) * sin(α * x[3]),
    sin(α * x[1]) * cos(α * x[2]) * sin(α * x[3]),
    sin(α * x[1]) * sin(α * x[2]) * cos(α * x[3]),
)
f_source(x) = -3ν * α^2 * u_exact(x) - sum(c .* ∇u_exact(x))

# ============================================================================
# Custom PDE model: ν∇²u − c⋅∇u = f (docs/src/custom_pdes.md pattern)
# ============================================================================

struct AdvectionDiffusion{T, V <: AbstractVector, S} <: Macchiato.AbstractModel
    ν::T          # diffusivity
    c::V          # advection velocity vector
    source::S     # source term f(x) -> value
end

Macchiato.num_vars(::AdvectionDiffusion, _) = 1

# The whole operator is fused into a single weight matrix. `k` is hard-coded
# because run! kwargs are forwarded to the BC assembly as well.
function Macchiato.make_system(model::AdvectionDiffusion, domain; kwargs...)
    (; ν, c) = model
    x = node_coordinates(domain)
    op = (@operator ν * ∇² - c ⋅ ∇)(x; k = 40)
    b = [model.source(xᵢ) for xᵢ in x]
    return weights(op), b
end

# ============================================================================
# Solve
# ============================================================================

bcs = Dict(:surface1 => PrescribedValue((x, t) -> u_exact(x)))
domain = MM.Domain(cloud, bcs, AdvectionDiffusion(ν, c, f_source))
sim = Simulation(domain)  # Steady() by default
run!(sim)
u = solution(sim)

@assert all(isfinite, u)
@assert maximum(abs, u) < 1.5 "solution outside [-1.5, 1.5]: destabilized solve?"

# ============================================================================
# Error report against the exact solution
# ============================================================================
#
# node_coordinates(domain) is ordered identically to solution(sim): boundary
# points first (surface by surface), then volume points.

coords = node_coordinates(domain)
u_ex = u_exact.(coords)
err = u .- u_ex

L2_error = norm(err, 2) / sqrt(length(err))  # RMS-normalized L2
Linf_error = norm(err, Inf)
relative_L2 = norm(err, 2) / norm(u_ex, 2)  # textbook ‖err‖₂/‖u‖₂, both un-normalized
boundary_error = norm(err[1:length(cloud.boundary)], Inf)

println("\nerror vs exact solution:")
for (name, val) in (
    "L2 (RMS)" => L2_error,
    "L∞" => Linf_error,
    "relative L2" => relative_L2,
    "boundary L∞" => boundary_error,
)
    println("  ", rpad(name, 14), round(val; sigdigits = 4))
end

# Dirichlet rows are exact identity rows, so the boundary error is solver eps.
# Thresholds are ~2x the values observed at dx = 3 mm with seed 42.
@assert boundary_error < 1.0e-10 "boundary rows not exact: $boundary_error"
@assert relative_L2 < 1.5e-3 "relative L2 error too large: $relative_L2"
@assert Linf_error < 2.0e-3 "L∞ error too large: $Linf_error"

# ============================================================================
# Render: 2x2 — solution and |error|, full and cut open at the mid-y plane
# ============================================================================

xs = [p[1] for p in coords]
ys = [p[2] for p in coords]
zs = [p[3] for p in coords]

az, el = 0.45π, π / 8
eye = (cos(el) * cos(az), cos(el) * sin(az), sin(el))

# Cutaway plane y = y_mid with normal ŷ: the camera sits mostly along +y, so
# keeping the y ≤ y_mid half turns the cut face toward the viewer and exposes
# the interior points.
y_mid = (minimum(ys) + maximum(ys)) / 2
half = ys .<= y_mid

# Dirichlet rows sit near machine eps — far below the interior error band. Clamp
# to a floor just under that band so the log colorbar resolves the interior
# variation instead of stretching over the empty decades down to eps.
err_abs = abs.(err)
floor_val = 1.0e-8
err_plot = max.(err_abs, floor_val)
u_range = extrema(u)
err_range = (floor_val, max(maximum(err_abs), 10floor_val))

# iCloud's file provider intermittently stalls `close` on multi-MB PNGs written
# in place under the synced repo ("SystemError: close: Operation timed out").
# Render to a local temp file and move it into position in one bulk copy instead.
function save_png(path, fig; kwargs...)
    tmp = tempname() * ".png"
    save(tmp, fig; kwargs...)
    mv(tmp, path; force = true)
    return path
end

function field_panel!(where_, mask, color, colorrange; title, colormap, colorscale = identity)
    ax = Axis3(where_; azimuth = az, elevation = el, aspect = :data, title = title)
    x, y, z, col = xs[mask], ys[mask], zs[mask], color[mask]
    ord = sortperm(x .* eye[1] .+ y .* eye[2] .+ z .* eye[3])  # far → near for overdraw
    plt = meshscatter!(ax, x[ord], y[ord], z[ord];
        color = col[ord], colorrange = colorrange, colormap = colormap,
        colorscale = colorscale, markersize = 1.0e-3)
    return plt
end

full = trues(length(u))
fig = Figure(; size = (1800, 1600))

s1 = field_panel!(fig[1, 1], full, u, u_range;
    title = "u (numerical)", colormap = :viridis)
field_panel!(fig[2, 1], half, u, u_range;
    title = "u — cutaway", colormap = :viridis)
Colorbar(fig[1:2, 2], s1; label = "u")

s2 = field_panel!(fig[1, 3], full, err_plot, err_range;
    title = "|u − u_exact| (log scale)", colormap = :inferno, colorscale = log10)
field_panel!(fig[2, 3], half, err_plot, err_range;
    title = "|u − u_exact| — cutaway", colormap = :inferno, colorscale = log10)
Colorbar(fig[1:2, 4], s2; label = "|u − u_exact|")

png_path = joinpath(@__DIR__, "mms_armadillo.png")
save_png(png_path, fig; px_per_unit = 2)
@assert isfile(png_path) && filesize(png_path) > 20_000
println("saved: ", abspath(png_path))
