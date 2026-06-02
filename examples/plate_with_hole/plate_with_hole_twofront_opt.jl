# ============================================================================
# plate_with_hole_twofront_opt.jl
#
# TWO-FRONT shape optimization: discrete adjoint + remeshing, closed-loop on
# point-cloud quality.  The design is a few smooth Fourier modes of the hole
# radius (this AVERAGES OUT the per-node Nyquist noise — failure mode A); the
# cloud is kept healthy by two cooperating fronts:
#
#   FRONT 1 — MORPH (within a remesh interval):
#       The interior is slaved to the boundary by a smooth, DIFFERENTIABLE
#       Laplace extension.  The discrete adjoint flows through it exactly
#       (transpose-corrected gradient).  Fixed stencils ⇒ a smooth objective.
#
#   FRONT 2 — REMESH + RE-ANCHOR (between intervals):
#       When a quality indicator trips, regenerate a FRESH high-quality cloud
#       for the CURRENT design and RE-ANCHOR (reset reference + refactorize the
#       morph).  Each remesh reads as a RE-INITIALIZATION, not a perturbed
#       continuation — this is what removes cloud degradation (B) and the
#       stale-cloud objective bias (C).  (See docs/boundary_gradient_noise.md
#       §UPDATE 2026-06-02 for why all three fronts are non-negotiable.)
#
# CLOSED-LOOP CONTROL: a registry of quality INDICATORS is measured every
# iteration and logged.  Each can be enabled/disabled and has its own threshold.
# Only ENABLED indicators trigger a remesh.  Toggle them in the CONFIG block to
# discover the minimal useful set and discard redundant ones.
#
# Descend on the GRADIENT (‖Jᵀg‖→0), not on the remesh-jumping C value.
#
# Run:  jlrun plate_with_hole/plate_with_hole_twofront_opt.jl   (from examples/)
# ============================================================================
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Macchiato
import RadialBasisFunctions: _build_weights, Partial, MixedPartial, find_neighbors
using RadialBasisFunctions: PHS
using StaticArrays, LinearAlgebra, Statistics, Printf
using CairoMakie

# ============================================================================
# CONFIG — problem, loop, and the toggleable quality-indicator registry.
# ============================================================================
const Lx = 4.0; const Ly = 4.0          # plate half-extents (square plate)
const a0 = 0.40; const b0 = 0.20        # starting ellipse semi-axes
const dx = 0.05                          # background lattice spacing
const σ∞ = 1.0                           # biaxial far-field traction
const k  = 35                            # RBF-FD stencil size

const MAX_ITER  = 60
const STEP_FRAC = 0.03                   # max radial boundary step / r₀ per iter
const SOB_P     = 2                      # Sobolev order (H^p); length sob_r is auto

model   = LinearElasticity(E = 1.0e7, ν = 0.3)
μ, λstar = lame_parameters(model)
basis   = PHS(3; poly_deg = 3)

# Indicator thresholds are first guesses — the whole point is to watch the log
# and prune.  `trip_when` is the side of the threshold that means "cloud is bad".
# Each measure returns a scalar from (morphed points, cloud state).
# (Indicator struct imported from Macchiato)

# ============================================================================
# Geometry helpers (background lattice, outer frame, Fourier hole).
# ============================================================================
const r_ref  = sqrt(a0 * b0)                                  # area-equiv circle radius
const nθ     = max(48, round(Int, 2π * sqrt((a0^2 + b0^2)/2) / dx))
const θvals  = [2π*(j-1)/nθ for j in 1:nθ]
const margin = 1 + 1.2 * dx / r_ref                          # interior cull margin
const xs_grid = (-Lx/2 + dx):dx:(Lx/2 - dx/2)
const ys_grid = (-Ly/2 + dx):dx:(Ly/2 - dx/2)
const A_target = π * a0 * b0

# Design = FourierModes from Macchiato.  modes/sob_r are calibrated below.
# radius_at, boundary_points, r0_for_area, contract_gradient, sob_weight
# are all imported from the framework.  θvals, nθ, A_target are problem constants.

flat(v) = reduce(vcat, [[p[1], p[2]] for p in v])

function build_outer_frame()
    pts = SVector{2,Float64}[]; tag = Symbol[]
    for y in (-Ly/2):dx:(Ly/2)
        push!(pts, SVector(-Lx/2, y)); push!(tag, :xlo)
        push!(pts, SVector(Lx/2, y));  push!(tag, :xhi)
    end
    for x in (-Lx/2 + dx):dx:(Lx/2 - dx)
        push!(pts, SVector(x, Ly/2));  push!(tag, :yhi)
        push!(pts, SVector(x, -Ly/2)); push!(tag, :ylo)
    end
    return pts, tag
end
outer_normal(tag) = tag === :xhi ? SVector(1.0,0.0) : tag === :xlo ? SVector(-1.0,0.0) :
                    tag === :yhi ? SVector(0.0,1.0) : SVector(0.0,-1.0)

# ============================================================================
# CloudState — everything tied to ONE cloud (rebuilt at each remesh/re-anchor).
# Holds the reference geometry the morph extends from, the factorized Laplace
# morph operator, the index ranges, and the (cloud-specific) boundary conditions.
# ============================================================================
struct CloudState
    pts::Vector{SVector{2,Float64}}      # the freshly-anchored cloud (morph reference)
    N::Int
    n_int::Int; n_outer::Int
    interior_idx::Vector{Int}
    outer_idx::Vector{Int}
    hole_idx::Vector{Int}
    boundary_idx::Vector{Int}            # [outer; hole] — morph boundary ordering
    neumann_idx::Vector{Int}
    outer_tag::Vector{Symbol}
    adjl::Vector{Vector{Int}}            # fixed adjacency for this interval
    ref_radius::Vector{Float64}          # per-node stencil radius AT ANCHOR (drift baseline)
    ref_interior::Vector{SVector{2,Float64}}
    ref_hole::Vector{SVector{2,Float64}}
    morph::LaplaceExtension               # FRONT 1: Laplace morph (from framework)
    dx::Float64                           # background lattice spacing (for indicator measures)
    dirichlet_dofs::Vector{Int}
    active::BitVector
    interior_rows::BitVector
end

# FRONT 2: build a fresh, quality-controlled cloud for `design` and anchor it.
# "Quality-controlled" here = a clean Cartesian interior freshly culled to the
# CURRENT hole, uniform boundary nodes, structured frame.  (A relaxation/
# resample step could slot in here later without changing anything downstream.)
function anchor(design::FourierModes)
    hole = boundary_points(design, θvals)
    inside(p) = hypot(p[1], p[2]) < margin * radius_at(design, atan(p[2], p[1]))
    interior = SVector{2,Float64}[]
    for x in xs_grid, y in ys_grid
        p = SVector(x, y); inside(p) || push!(interior, p)
    end
    outer, otag = build_outer_frame()
    n_int = length(interior); n_outer = length(outer)
    pts = vcat(interior, outer, hole); N = length(pts)
    interior_idx = collect(1:n_int)
    outer_idx    = collect((n_int+1):(n_int+n_outer))
    hole_idx     = collect((n_int+n_outer+1):N)
    boundary_idx = vcat(outer_idx, hole_idx)
    neumann_idx  = vcat(outer_idx, hole_idx)
    adjl = find_neighbors(pts, k)
    ref_radius = [maximum(hypot(pts[i][1]-pts[j][1], pts[i][2]-pts[j][2])
                          for j in adjl[i] if j != i) for i in 1:N]

    # Boundary conditions (3 pins remove rigid-body modes); recomputed per cloud.
    nrst(p0) = interior_idx[argmin([hypot(pts[i][1]-p0[1], pts[i][2]-p0[2]) for i in interior_idx])]
    pin1 = nrst((0.0, 0.8)); pin2 = nrst((0.0, -0.8)); pin3 = nrst((0.8, 0.0))
    dirichlet_dofs = [pin1, pin2, pin3 + N]
    active = let a = trues(2N); a[pin1] = a[pin2] = a[pin3 + N] = false; a end
    interior_rows = let r = falses(N); foreach(i -> r[i] = true, interior_idx); r end

    # Factorize the Laplace morph operator on THIS cloud (the re-anchor).
    ext = build_laplace_extension(pts, adjl, basis, interior_idx, boundary_idx,
                                  pts[interior_idx], hole, n_outer, nθ)

    return CloudState(pts, N, n_int, n_outer, interior_idx, outer_idx, hole_idx,
                      boundary_idx, neumann_idx, otag, adjl, ref_radius,
                      pts[interior_idx], hole, ext, dx,
                      dirichlet_dofs, active, interior_rows)
end

# FRONT 1: morph the interior to follow `design`, extending from the anchored
# reference by the TOTAL boundary displacement (so distortion never accumulates
# within an interval).  Returns the current point cloud.
morph_cloud(st::CloudState, design::FourierModes) =
    morph(st.morph, boundary_points(design, θvals))

# Transpose of the morph: carry the interior nodal gradient onto the boundary.
morph_transpose(st::CloudState, g_all) =
    Macchiato.morph_transpose(st.morph, g_all, st.hole_idx)

# ============================================================================
# Forward solve + discrete adjoint (exact nodal gradient at every node).
# ============================================================================
const _ZJV = SVector(0.0, 0.0)
# Outer (fixed-normal) nodes: zero normal-Jacobian.  i_prev/i_next carry the
# node's own index (unused since the Jacobian blocks are zero, but must be valid).
zero_njac(i) = NormalJacobian(i, i, _ZJV, _ZJV, _ZJV, _ZJV)

function build_layout(pts, st::CloudState)
    neumann_adjl = st.adjl[st.neumann_idx]
    hn, hjacs = polyline_normals(flat(pts), st.hole_idx, collect(1:nθ))
    njacs = vcat([zero_njac(i) for i in st.outer_idx], hjacs)
    normals = SVector{2,Float64}[]; tractions = SVector{2,Float64}[]
    for i in 1:st.n_outer
        nn = outer_normal(st.outer_tag[i]); push!(normals, nn); push!(tractions, σ∞ .* nn)
    end
    append!(normals, hn); append!(tractions, fill(SVector(0.0,0.0), nθ))
    layout = build_traction_layout(st.neumann_idx, neumann_adjl, normals, tractions, λstar, μ, st.N)
    return layout, njacs, neumann_adjl
end

function solve_adjoint(pts, st::CloudState)
    layout, njacs, neumann_adjl = build_layout(pts, st)
    b = zeros(2*st.N)
    for kk in eachindex(layout.rows); b[layout.rows[kk]] = layout.b_vals[kk]; end
    res = shape_gradient(flat(pts), model, st.N, st.adjl, basis, st.active,
                         st.dirichlet_dofs, zeros(3), _ -> b;
                         interior_rows = st.interior_rows, traction_layout = layout,
                         neumann_ids = st.neumann_idx, neumann_adjl = neumann_adjl,
                         traction_jacobians = nothing, normal_jacobians = njacs)
    C = dot(b, res.u)
    g_all = [SVector(res.Δpts[2*i-1], res.Δpts[2*i]) for i in 1:st.N]
    return C, g_all
end

# ============================================================================
# Contract the nodal gradient onto the area-constrained Fourier design modes.
# ============================================================================
function design_gradient(g_all, pts, st::CloudState, design::FourierModes)
    ĝ_hole = morph_transpose(st, g_all)                    # FRONT-1 transpose
    hole = pts[st.hole_idx]
    return contract_gradient(design, ĝ_hole, hole, θvals)
end

# ============================================================================
# Quality indicators — imported from Macchiato (Indicator, trips, assess,
# measure_morph_drift, measure_min_gap, measure_spacing_cv, measure_boundary_cv,
# measure_min_sep, measure_stencil_growth).
# ============================================================================

# ============================================================================
# Calibration — derive the design modes and Sobolev length from cloud geometry.
# ============================================================================
function calibrate()
    # Bootstrap: anchor a circle (zero oscillatory coefficients, empty modes)
    ds0 = FourierModes(r_ref, Float64[], Float64[], Int[], NaN, SOB_P)
    st0 = anchor(ds0)
    ds, ρ = calibrate_fourier(st0.pts, st0.adjl, r_ref; SOB_P = SOB_P)
    return ds.modes, ds.sob_r, ρ
end

# Fit the starting ellipse onto the design modes.
function fit_start(modes)
    return fit_start_fourier(a0, b0, θvals, nθ, modes, A_target)
end

# ============================================================================
# The closed-loop two-front optimizer.
# ============================================================================
function optimize(design::FourierModes, indicators)
    st = anchor(design)                                     # FRONT 2: initial anchor
    log = NamedTuple[]
    @printf("%4s %12s %8s %9s  %-26s %s\n", "it", "C", "r₀", "a₂", "indicators", "event")
    println("-"^96)
    for it in 1:MAX_ITER
        pts = morph_cloud(st, design)                       # FRONT 1
        vals, tripped = assess(indicators, pts, st)
        event = ""
        if !isempty(tripped)                                # FRONT 2: re-anchor
            st   = anchor(design)
            pts  = morph_cloud(st, design)                  # == fresh cloud, zero morph
            vals, _ = assess(indicators, pts, st)
            event = "REMESH ⟵ " * join(string.(tripped), ",")
        end

        C, g_all = solve_adjoint(pts, st)
        dCa, dCb = design_gradient(g_all, pts, st, design)

        # Sobolev-preconditioned, fixed normalized descent (max radial step = STEP_FRAC·r₀).
        da = [-dCa[i]/sob_weight(m, design.sob_r, design.SOB_P) for (i, m) in enumerate(design.modes)]
        db = [-dCb[i]/sob_weight(m, design.sob_r, design.SOB_P) for (i, m) in enumerate(design.modes)]
        δr = [sum(da[i]*cos(m*θvals[j]) + db[i]*sin(m*θvals[j]) for (i, m) in enumerate(design.modes)) for j in 1:nθ]
        maxδ = maximum(abs, δr)
        maxδ < 1e-14 && (println("  Converged (‖g‖≈0)."); break)
        s = STEP_FRAC * design.r0 / maxδ

        vstr = join([@sprintf("%s=%.2f", n, v) for (n, v) in vals], " ")
        @printf("%4d %12.6e %8.4f %+9.4f  %-26s %s\n", it, C, design.r0, design.a[1], vstr, event)
        push!(log, (it = it, C = C, r0 = design.r0, a2 = design.a[1],
                    vals = Dict(vals), remeshed = !isempty(tripped)))

        a = design.a .+ s .* da
        b = design.b .+ s .* db
        design = FourierModes(r0_for_area(A_target, a, b), a, b,
                              design.modes, design.sob_r, design.SOB_P)
    end
    return design, st, log
end

# ============================================================================
# Run.
# ============================================================================
modes, sob_r, ρ = calibrate()
@printf("calibration: ρ=%.4f (%.1f dx)  r_ref=%.4f  ⇒  modes=%s  sob_r=%.3f\n",
        ρ, ρ/dx, r_ref, string(collect(modes)), sob_r)

# Candidate indicator registry — toggle `enabled` to prune to the minimal set.
indicators = [
    Indicator(name = :morph_drift, measure = measure_morph_drift, threshold = 0.60, trip_when = :above, enabled = true),
    Indicator(name = :min_gap,     measure = measure_min_gap,     threshold = 0.30, trip_when = :below, enabled = true),
    Indicator(name = :spacing_cv,  measure = measure_spacing_cv,  threshold = 0.35, trip_when = :above, enabled = true),
    Indicator(name = :boundary_cv, measure = measure_boundary_cv, threshold = 0.40, trip_when = :above, enabled = true),
    Indicator(name = :min_sep,     measure = measure_min_sep,     threshold = 0.20, trip_when = :below, enabled = true),
    Indicator(name = :stencil_growth, measure = measure_stencil_growth, threshold = 1.40, trip_when = :above, enabled = true),
]
println("indicators enabled: ", join(string.(getfield.(filter(i->i.enabled, indicators), :name)), ", "), "\n")

start = fit_start(modes)
# fit_start_fourier returns a FourierModes with NaN sob_r; patch in calibrated value.
start = FourierModes(start.r0, start.a, start.b, start.modes, sob_r, SOB_P)
final, st_final, log = optimize(start, indicators)

r_circle = sqrt(A_target / π)
n_remesh = count(e -> e.remeshed, log)
println("\n--- Final design ---")
@printf("r₀:  %.4f → %.4f   (circle: %.4f)\n", start.r0, final.r0, r_circle)
@printf("a₂:  %+.4f → %+.4f   (circle: 0)\n", start.a[1], final.a[1])
@printf("C :  %.6e → %.6e\n", log[1].C, log[end].C)
@printf("remeshes triggered: %d / %d iters\n", n_remesh, length(log))

# ============================================================================
# Plot: shapes, C history (remesh events marked), a₂, indicator trajectories.
# ============================================================================
fig = Figure(size = (1500, 900))

ax1 = Axis(fig[1,1]; title = "Hole shapes", aspect = DataAspect(), xlabel = "x", ylabel = "y")
hs = boundary_points(start, θvals); lines!(ax1, vcat(getindex.(hs,1), hs[1][1]), vcat(getindex.(hs,2), hs[1][2]); color=:red, linewidth=2, label="start")
hf = boundary_points(final, θvals); lines!(ax1, vcat(getindex.(hf,1), hf[1][1]), vcat(getindex.(hf,2), hf[1][2]); color=:green, linewidth=2, label="final")
ts = range(0, 2π; length=200); lines!(ax1, r_circle.*cos.(ts), r_circle.*sin.(ts); color=:blue, linestyle=:dash, label="circle")
axislegend(ax1; position=:rb)

its = [e.it for e in log]
ax2 = Axis(fig[1,2]; title = "Compliance (remesh events ▼)", xlabel = "iter", ylabel = "C")
lines!(ax2, its, [e.C for e in log]; linewidth=2)
rm_its = [e.it for e in log if e.remeshed]
isempty(rm_its) || vlines!(ax2, rm_its; color=(:red,0.3), linestyle=:dash)

ax3 = Axis(fig[1,3]; title = "a₂ → 0 (circle)", xlabel = "iter", ylabel = "a₂")
lines!(ax3, its, [e.a2 for e in log]; linewidth=2)
hlines!(ax3, [0.0]; color=:gray, linestyle=:dash)

ax4 = Axis(fig[2,1:3]; title = "Quality indicators per iteration (remesh ▼)", xlabel = "iter", ylabel = "value")
for ind in filter(i -> i.enabled, indicators)
    lines!(ax4, its, [get(e.vals, ind.name, NaN) for e in log]; linewidth=2, label=string(ind.name))
end
isempty(rm_its) || vlines!(ax4, rm_its; color=(:red,0.3), linestyle=:dash)
axislegend(ax4; position=:rt)

save(joinpath(@__DIR__, "plate_with_hole_twofront_opt.png"), fig)
println("\nSaved: plate_with_hole_twofront_opt.png")
