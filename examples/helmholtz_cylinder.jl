# Optics verification: TM plane-wave scattering off a conducting cylinder.
#
# Solves the 2D Helmholtz equation ∇²E + κ²E = 0 for the total field E_z of a
# transverse-magnetic plane wave scattering off a perfectly conducting (PEC)
# cylinder — the canonical optics/electromagnetics benchmark with an exact
# analytical solution (Mie-type Bessel/Hankel series). The domain is the
# annulus between the cylinder surface (E = 0, PEC) and an outer truncation
# circle where the exact series supplies the Dirichlet data, so the numerical
# error is measurable at every node. The field is complex — the amplitude AND
# phase of the time-harmonic oscillation at each point — and the system is
# solved directly in ComplexF64: the operator weights are real, the complexity
# enters through b and the boundary data. The model is the custom-PDE Helmholtz
# pattern from docs/src/custom_pdes.md.
#
# Run from the repo root: julia --project=examples examples/helmholtz_cylinder.jl
# Output: examples/helmholtz_cylinder.png (gitignored) — 2x2 figure: Re(E) and
# Im(E) interference fringes, |E| amplitude with the shadow region, and the
# pointwise |error| against the exact series on a log scale.
#
# Requires RadialBasisFunctions ≥ 0.7.1 (dimension-aware PHS Laplacian — 0.7.0
# hardcodes the 3D constant and silently biases 2D solves).

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using WhatsThePoint
import WhatsThePoint as WTP
using Macchiato
import Macchiato as MM
using Unitful: m, ustrip
using OrderedCollections: LittleDict
using SpecialFunctions: besselj, hankelh1
using CairoMakie
using StaticArrays: SVector
using LinearAlgebra: norm
using Random: seed!

# The Orthtree Bridson interior fill draws from the global RNG, so every run is
# a different cloud realization. The asserts after the solve catch a bad
# realization if anything here changes.
seed!(42)

# ============================================================================
# Problem setup: wavelength λ = 1 m, cylinder diameter 2λ
# ============================================================================

const κ = 2π          # wavenumber (1/m); NOT named k — that's the stencil-size kwarg
const a_cyl = 1.0     # PEC cylinder radius: diameter 2λ gives a deep shadow
const R_out = 4.0     # outer truncation radius: 3λ of annulus around the cylinder
const h = 0.04        # nominal spacing: 25 points per wavelength, κh ≈ 0.25

# ============================================================================
# Exact solution: incident plane wave + scattered Hankel series
# ============================================================================
#
# Total field for e^{iκx} incident on a PEC cylinder of radius a:
#   E(r,θ) = e^{iκ·r·cosθ} − Σₙ εₙ iⁿ [Jₙ(κa)/Hₙ⁽¹⁾(κa)] Hₙ⁽¹⁾(κr) cos(nθ)
# The incident part is kept in closed form (Jacobi–Anger in reverse), so the
# series only carries the scattered field, whose terms are bounded by |Jₙ(κa)|
# — truncating at N = κa + 20 puts the series error near 1e-11.

const N_series = ceil(Int, κ * a_cyl) + 20

mie_coeffs(κ, a, N) = [
    -(n == 0 ? 1.0 : 2.0) * im^n * besselj(n, κ * a) / hankelh1(n, κ * a)
        for n in 0:N
]
const COEFFS = mie_coeffs(κ, a_cyl, N_series)

function E_exact(x)
    r = hypot(x[1], x[2])
    θ = atan(x[2], x[1])
    E = cis(κ * x[1])
    for (i, c) in pairs(COEFFS)
        E += c * hankelh1(i - 1, κ * r) * cos((i - 1) * θ)
    end
    return E
end

# The series must satisfy the PEC condition on its own — verifies the
# coefficients and truncation before any solving happens.
pec_check = maximum(
    abs(E_exact(SVector(a_cyl * cos(t), a_cyl * sin(t)))) for t in range(0, 2π, 361)
)
@assert pec_check < 1.0e-8 "series violates PEC condition: $pec_check"

# ============================================================================
# Geometry: annulus from two analytic circles (exact normals)
# ============================================================================
#
# The 2D Orthtree fill indexes the boundary loops with a segment quadtree that
# resolves the outer-loop-plus-hole topology by nesting parity, so the hole
# stays empty without special casing. The loops are kept consistently oriented
# (outer CCW, inner CW) with exact outward-from-the-material normals.

function circle_surface(r, n; ccw::Bool, outward::Bool)
    s = ccw ? 1.0 : -1.0
    θs = [s * 2π * (j - 1) / n for j in 1:n]
    pts = [WTP.Point(r * cos(t) * m, r * sin(t) * m) for t in θs]
    sgn = outward ? 1.0 : -1.0
    nrms = [WTP.Vec(sgn * cos(t), sgn * sin(t)) for t in θs]
    return PointSurface(pts, nrms, fill(2π * r / n * m, n))
end

part = PointBoundary(
    LittleDict(
        :cylinder => circle_surface(a_cyl, round(Int, 2π * a_cyl / h); ccw = false, outward = false),
        :outer => circle_surface(R_out, round(Int, 2π * R_out / h); ccw = true, outward = true),
    )
)

# 2D Orthtree: segment-quadtree geometry index + graded Bridson (Poisson-disk)
# placement. This is also what `discretize` defaults to in 2D.
cloud = WTP.discretize(part, ConstantSpacing(h * m); alg = Orthtree(part; spacing = ConstantSpacing(h * m)))
n_pts = length(WTP.points(cloud))
println("cloud: ", n_pts, " points (", length(cloud.boundary), " boundary)")
@assert 12_000 < n_pts < 40_000 "point count $n_pts out of range: check loop orientations"

# ============================================================================
# Custom PDE model: ∇²E + κ²E = 0 (docs/src/custom_pdes.md Helmholtz pattern)
# ============================================================================

struct Helmholtz{T} <: Macchiato.AbstractModel
    κ::T          # wavenumber
end

Macchiato.num_vars(::Helmholtz, _) = 1

# `f` inside @operator is the Identity symbol, and `k` is hard-coded because
# run! kwargs are forwarded to the BC assembly as well. Wave problems are
# pollution-limited: the phase error of the discrete operator accumulates over
# the ~6λ propagation path, so the operator needs high order — poly_deg = 6
# (28 monomials in 2D, k = 60 ≥ 2x that) keeps the accumulated dispersion
# error at the 1e-3 level at 25 points per wavelength.
function Macchiato.make_system(model::Helmholtz, domain; kwargs...)
    κ² = model.κ^2
    x = node_coordinates(domain)
    op = (@operator ∇² + κ² * f)(x; k = 60, basis = PHS(3; poly_deg = 6))
    # Solve uniformly in ComplexF64. The weights are real, but A is promoted at
    # assembly because a real factorization cannot back-substitute a complex
    # right-hand side — LinearSolve's default solver selection only lands on a
    # complex-capable factorization when the matrix eltype is complex.
    return ComplexF64.(weights(op)), zeros(ComplexF64, length(x))
end

# ============================================================================
# Solve — directly in ComplexF64
# ============================================================================

model = Helmholtz(κ)
bcs = Dict(
    :cylinder => PrescribedValue(0.0 + 0.0im),  # PEC: tangential E vanishes on a conductor
    :outer => PrescribedValue((x, t) -> E_exact(x)),
)
domain = MM.Domain(cloud, bcs, model)
sim = Simulation(domain)
run!(sim)
E_num = solution(sim)

@assert all(isfinite, abs.(E_num))
@assert maximum(abs, E_num) < 2.5 "field exceeds physical bound ~2: destabilized solve?"

# ============================================================================
# Error report against the exact series
# ============================================================================
#
# node_coordinates(domain) is ordered identically to solution(sim): boundary
# points first (surface by surface), then volume points.

coords = node_coordinates(domain)
E_ex = E_exact.(coords)
err = E_num .- E_ex

L2_error = norm(err, 2) / sqrt(length(err))  # RMS-normalized L2
Linf_error = maximum(abs, err)
relative_L2 = norm(err, 2) / norm(E_ex, 2)  # textbook ‖err‖₂/‖E‖₂
boundary_error = maximum(abs, err[1:length(cloud.boundary)])

println("\nerror vs exact series:")
for (name, val) in (
    "L2 (RMS)" => L2_error,
    "L∞" => Linf_error,
    "relative L2" => relative_L2,
    "boundary L∞" => boundary_error,
)
    println("  ", rpad(name, 14), round(val; sigdigits = 4))
end

# Both surfaces are Dirichlet identity rows, so the boundary error is solver eps.
# Thresholds are ~2.5x the values observed at h = 0.04, poly_deg = 6, seed 42.
@assert boundary_error < 1.0e-10 "boundary rows not exact: $boundary_error"
@assert relative_L2 < 2.0e-3 "relative L2 error too large: $relative_L2"
@assert Linf_error < 6.0e-3 "L∞ error too large: $Linf_error"

# ============================================================================
# Render: 2x2 — Re(E), Im(E), |E|, and |error| on a log scale
# ============================================================================

xs = [p[1] for p in coords]
ys = [p[2] for p in coords]

# Dirichlet rows sit near machine eps — far below the interior error band. Clamp
# to a floor just under that band so the log colorbar resolves the interior
# variation instead of stretching over the empty decades down to eps.
err_abs = abs.(err)
floor_val = 1.0e-8
err_plot = max.(err_abs, floor_val)
err_range = (floor_val, max(maximum(err_abs), 10floor_val))

# Shared symmetric range for the Re/Im fringe panels (diverging colormap
# centered on zero); |E| gets a sequential map anchored at zero.
M = max(maximum(abs, real.(E_num)), maximum(abs, imag.(E_num)))

# iCloud's file provider intermittently stalls `close` on multi-MB PNGs written
# in place under the synced repo ("SystemError: close: Operation timed out").
# Render to a local temp file and move it into position in one bulk copy instead.
function save_png(path, fig; kwargs...)
    tmp = tempname() * ".png"
    save(tmp, fig; kwargs...)
    mv(tmp, path; force = true)
    return path
end

fig = Figure(; size = (1500, 1400))

function field_panel!(where_, color, colorrange; title, colormap, colorscale = identity)
    ax = Axis(where_; aspect = DataAspect(), title = title,
        xlabel = "x (m)", ylabel = "y (m)")
    plt = scatter!(ax, xs, ys;
        color = color, colorrange = colorrange, colormap = colormap,
        colorscale = colorscale, markerspace = :data, markersize = 1.4h)
    # Fill the PEC cylinder so it reads as a solid object and hides the hole rim.
    poly!(ax, Circle(Point2f(0, 0), Float32(a_cyl)); color = :gray25)
    return plt
end

p1 = field_panel!(fig[1, 1], real.(E_num), (-M, M);
    title = "Re(E) — interference fringes", colormap = :RdBu)
Colorbar(fig[1, 2], p1)
p2 = field_panel!(fig[1, 3], imag.(E_num), (-M, M);
    title = "Im(E)", colormap = :RdBu)
Colorbar(fig[1, 4], p2)
p3 = field_panel!(fig[2, 1], abs.(E_num), (0, maximum(abs, E_num));
    title = "|E| — shadow and standing waves", colormap = :viridis)
Colorbar(fig[2, 2], p3)
p4 = field_panel!(fig[2, 3], err_plot, err_range;
    title = "|E − E_exact| (log scale)", colormap = :inferno, colorscale = log10)
Colorbar(fig[2, 4], p4)

png_path = joinpath(@__DIR__, "helmholtz_cylinder.png")
save_png(png_path, fig; px_per_unit = 2)
@assert isfile(png_path) && filesize(png_path) > 20_000
println("saved: ", abspath(png_path))
