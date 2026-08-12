# Monodomain electrophysiology on a segmented biventricular heart.
#
# Solves the Niederer et al. (2011) benchmark physics — the monodomain equation
# ∂V/∂t = D∇²V − I_ion/Cm with ten Tusscher–Panfilov 2006 epicardial cells and
# unsplit forward Euler — on a realistic biventricular geometry
# (examples/assets/biventricle.stl, prepared by examples/assets/prep_biventricle.py
# from the Bai et al. 2015 UK Biobank biventricular cardiac atlas, instance 010).
# The atlas surface is deliberately "re-segmented" at 1 mm with no smoothing, so
# the STL carries the authentic stair-step artifacts of a real CT/MR segmentation
# pipeline. Conductivity is isotropic — the RBF operator layer cannot express a
# per-node fiber tensor — using the Niederer longitudinal value D_L everywhere,
# so the wave travels at the physiological fiber-direction bulk speed. The stimulus is the
# Niederer corner-cube recipe transplanted to a 3 mm apex cap; the deliverable is
# the activation-time map.
#
# Units: the STL numbers are millimetres and are imported with an `m` unit label
# (the Orthtree volume fill emits raw-float points that Meshes defaults to
# metres, so the labels are lies everyone agrees on). All physics is mm/ms/mV.
#
# Two known coarse-discretization caveats, both shared with coarse FEM: the
# conduction velocity is overestimated at 1-2 mm spacing (same as the Niederer
# study's coarse grids), and k-nearest-neighbor stencils can couple across thin
# cavity gaps near the apex (the stencil radius ≈ 4·dx has no notion of the
# cavity between two walls).
#
# Run from the repo root:
#   julia --project=examples -t auto examples/biventricle_monodomain.jl
# BIV_DX=1.25 (mm) overrides the default 1.5 spacing for a finer (slower) run.
# Outputs: examples/biventricle_monodomain.png and examples/biventricle_result.vtu
#          (both gitignored).
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
using Unitful: m, ustrip, @u_str
# OrdinaryDiffEq 7.x no longer re-exports the low-order solvers
using OrdinaryDiffEqLowOrderRK: Euler, DiscreteCallback
using OhMyThreads: tforeach
using StaticArrays: SVector
using LinearAlgebra: mul!, norm
using SparseArrays: nnz, nonzeros
using Statistics: cor, mean
using CairoMakie
using Random: seed!

include(joinpath(@__DIR__, "niederer_benchmark", "ten_tusscher_2006.jl"))

# The Poisson-disk surface sampler and Bridson fill draw from the global RNG, so
# every run is a different cloud realization. The asserts after the solve catch
# a bad realization if anything here changes.
seed!(42)

# ============================================================================
# Physical parameters (Niederer 2011, Table 3; isotropic reduction)
# ============================================================================

const σ_L = 0.17 * 0.62 / (0.17 + 0.62)      # ≈ 0.13342 mS/mm (harmonic mean)
const σ_T = 0.019 * 0.24 / (0.019 + 0.24)    # ≈ 0.01761 mS/mm
const β = 140.0                              # surface-to-volume ratio (1/mm)
const Cm_tissue = 0.01                       # membrane capacitance (µF/mm²)
const D_L = σ_L / (β * Cm_tissue)            # ≈ 0.0953 mm²/ms along fiber
const D_T = σ_T / (β * Cm_tissue)            # ≈ 0.0126 mm²/ms transverse
# Isotropic reduction: use the longitudinal diffusivity everywhere, so the wave
# travels at the physiological fiber-direction bulk speed (CV ≈ 0.5 mm/ms at
# fine resolution). The CV-volume-preserving geometric mean (D_L·D_T²)^(1/3) is
# ~4x smaller; CV ∝ √D makes it half as fast, pushing full apex-paced
# activation past 1 s of simulated time — impractical for an example run.
const D_iso = D_L

const I_stim = 50.0                          # µA/mm³, Niederer stimulus amplitude
const stim_duration = 2.0                    # ms
const stim_dvdt_amp = I_stim / (β * Cm_tissue)   # ≈ 35.71 mV/ms
const stim_height = 3.0                      # apex cap: stimulate z < z_apex + 3 mm
const V_activation = 0.0                     # first crossing of V through 0 mV

const dx_mm = parse(Float64, get(ENV, "BIV_DX", "1.5"))   # node spacing (mm)
const dt = 0.01     # ms; reaction-limited (TT06 upstroke) — diffusion CFL has ~40x margin
# The recorded run at the default 1.5 mm spacing finishes activating at 275 ms
# (the coarse-grid CV slowdown is the same effect as coarse FEM in the Niederer
# study); 350 ms leaves ~75 ms of margin. Finer spacings activate earlier.
const t_end = 350.0

# ============================================================================
# Geometry: import the STL, promote to Float64, keep millimetre numbers
# ============================================================================
#
# Binary STL stores Float32 and `import_mesh` preserves the machine type, which
# would otherwise propagate all the way into the RBF weight assembly. Unlike the
# armadillo example there is no unit conversion: the numbers already are what the
# physics wants (mm), only the label is `m`.

mesh32 = import_mesh(joinpath(@__DIR__, "assets", "biventricle.stl"), m)
verts = map(Meshes.vertices(mesh32)) do p
    Meshes.Point((Float64.(ustrip.(u"m", Meshes.to(p))) .* m)...)
end
mesh = Meshes.SimpleMesh(verts, Meshes.topology(mesh32))

# Poisson-disk surface sampling is spacing-driven, so the stair-stepped
# tessellation does not force boundary points onto every terrace edge; keep
# dx at or above the 1 mm segmentation pitch.
spacing = ConstantSpacing(dx_mm * m)
part = PointBoundary(mesh, spacing)
n_surf = length(WTP.points(part[:surface1]))
@assert n_surf > 5_000 "suspiciously few boundary points ($n_surf)"

cloud = WTP.discretize(part, spacing; alg = Orthtree(mesh; spacing = spacing))
coords = node_coordinates(cloud)   # raw mm numbers, boundary-first ordering
N = length(coords)
n_bnd = length(cloud.boundary)
println("cloud: $N points ($n_bnd boundary + $(N - n_bnd) volume) at dx = $dx_mm mm")
@assert 30_000 < N < 400_000 "point count $N out of range: check units/spacing"

zs = getindex.(coords, 3)
z_apex = minimum(zs)
stim_mask = zs .< z_apex + stim_height
println("stimulus region: $(count(stim_mask)) nodes in the $(stim_height) mm apex cap")
@assert count(stim_mask) ≥ 10 "apex cap nearly empty; reduce dx"

# ============================================================================
# Custom PDE model: isotropic monodomain + TT06 reaction
# ============================================================================

struct Monodomain{T} <: Macchiato.AbstractModel
    D::T                     # isotropic diffusivity (mm²/ms)
    stim_mask::Vector{Bool}  # node-ordered apex stimulus region
    stim_dvdt::T             # I_stim/(β·Cm) (mV/ms)
    stim_duration::T         # ms
end

Macchiato.num_vars(::Monodomain, _) = NSTATES   # 19 states per node, SoA layout

# `k = 60` (not the default 40) for the same reason as the Niederer benchmark:
# boundary stencils on a scattered — here, choppy — surface go rank-deficient in
# the quadratic cross-terms unless forced to reach into the interior. Hard-coded
# because the transient path calls make_f with no kwargs.
function Macchiato.make_f(model::Monodomain, domain; kwargs...)
    (; D, stim_mask, stim_dvdt, stim_duration) = model
    n = length(domain.cloud)
    W, _, _ = MM.build_neumann_diffusion(domain; k = 60)
    WD = D .* W
    # Every surface is ZeroFlux, so the ghost flux source c(t) ≡ 0 and the G·c
    # term of the SolidEnergy pattern drops out entirely.
    max_w = maximum(abs, nonzeros(WD))
    println(
        "monodomain W: nnz = $(nnz(WD)), max|W| = $(round(max_w; digits = 3)), " *
            "forward-Euler dt_max ≈ $(round(2 / max_w; digits = 3)) ms"
    )

    return function f(du, u, p, t)
        # tt06_reaction_point! SETS du (du[i] = -I_ion + stim), so the reaction
        # must run first; diffusion then accumulates onto the V block.
        active = t < stim_duration ? stim_dvdt : 0.0
        tforeach(1:n) do i
            tt06_reaction_point!(du, u, i, n, stim_mask[i] ? active : 0.0)
        end
        mul!(view(du, 1:n), WD, view(u, 1:n), 1.0, 1.0)
        return nothing
    end
end

# ============================================================================
# Solve: Domain → Simulation → run! (forward Euler + activation callback)
# ============================================================================

model = Monodomain(D_iso, collect(stim_mask), stim_dvdt_amp, stim_duration)
bcs = Dict(:surface1 => ZeroFlux())
domain = MM.Domain(cloud, bcs, model)
sim = Simulation(domain, Transient(Δt = dt, stop_time = t_end, solver = Euler()))

# Custom 19-variable model: set! has no field map for it, so assign u0 directly
# (run!'s _ensure_u0_initialized! respects a pre-set u0).
sim.u0 = tt06_initial_conditions(N)

# Activation times: first V > 0 crossing, checked after every accepted Euler step.
# save_positions = (false, false) is load-bearing — the DiscreteCallback default
# (true, true) would force a 19N-float solution save at every step. The recorder
# closes over its arrays (rather than reading script globals) so the per-step
# scan stays type-stable.
function make_activation_recorder(activation, n; log_interval = 25.0)
    next_log = Ref(log_interval)
    return function record!(integrator)
        t = integrator.t
        V = view(integrator.u, 1:n)
        @inbounds for i in 1:n
            if isnan(activation[i]) && V[i] > V_activation
                activation[i] = t
            end
        end
        if t ≥ next_log[] - 1.0e-9
            Vmin, Vmax = extrema(V)
            println(
                "  t = $(round(t; digits = 1)) ms | V ∈ [$(round(Vmin; digits = 1)), " *
                    "$(round(Vmax; digits = 1))] mV | activated: " *
                    "$(count(!isnan, activation)) / $n"
            )
            next_log[] += log_interval
        end
        return nothing
    end
end

activation_time = fill(NaN, N)
always(u, t, integrator) = true
cb = DiscreteCallback(
    always, make_activation_recorder(activation_time, N);
    save_positions = (false, false),
)

println("Solving (dt = $dt ms, t_end = $t_end ms, $(round(Int, t_end / dt)) steps)...")
elapsed = @elapsed run!(sim; callback = cb)
println("solve: $(round(elapsed; digits = 1)) s for $N nodes")

u_final = solution(sim)
V_final = u_final[1:N]

# ============================================================================
# Probes and sanity report
# ============================================================================

"""
    nearest_point_index(pts, target) -> (Int, Float64)

Linear-search nearest neighbor of `target` in `pts`. O(N) per query; fine for
the handful of probe points we query.
"""
function nearest_point_index(pts, target)
    tx, ty, tz = target
    best_i = 1
    best_d2 = Inf
    @inbounds for i in eachindex(pts)
        p = pts[i]
        d2 = (p[1] - tx)^2 + (p[2] - ty)^2 + (p[3] - tz)^2
        if d2 < best_d2
            best_d2 = d2
            best_i = i
        end
    end
    return best_i, sqrt(best_d2)
end

# Geometric probes (the STL is centered in x/y, apex at z = 0, base at z = H).
# Anatomical labels are approximate — read them off the rendered map.
xs = getindex.(coords, 1)
ys = getindex.(coords, 2)
H = maximum(zs)
mid = 0.5 * H
mid_slab = findall(i -> abs(zs[i] - mid) < 0.05 * H, 1:N)
probes = [
    "apex" => Tuple(coords[argmin(zs)]),
    "base" => Tuple(coords[argmax(zs)]),
    "mid-height -x wall" => Tuple(coords[mid_slab[argmin(xs[mid_slab])]]),
    "mid-height +x wall" => Tuple(coords[mid_slab[argmax(xs[mid_slab])]]),
    "mid-height -y wall" => Tuple(coords[mid_slab[argmin(ys[mid_slab])]]),
    "mid-height +y wall" => Tuple(coords[mid_slab[argmax(ys[mid_slab])]]),
    "mid-septum (centroid proxy)" => Tuple(coords[mid_slab[argmin([norm(coords[i] - SVector(0.0, 0.0, mid)) for i in mid_slab])]]),
]

println("\nActivation times at geometric probes:")
println("  probe                        position (mm)             AT (ms)")
for (label, target) in probes
    idx, _ = nearest_point_index(coords, target)
    p = coords[idx]
    at = activation_time[idx]
    at_str = isnan(at) ? "not activated" : string(round(at; digits = 1))
    println(
        "  $(rpad(label, 28)) " *
            "($(rpad(round(p[1]; digits = 1), 6)), $(rpad(round(p[2]; digits = 1), 6)), $(rpad(round(p[3]; digits = 1), 5)))  $at_str"
    )
end

n_unactivated = count(isnan, activation_time)
apex_center = mean(coords[stim_mask])
dists = [norm(c - apex_center) for c in coords]
activated = findall(!isnan, activation_time)
spread_cor = cor(dists[activated], activation_time[activated])
t_first = minimum(activation_time[activated])
t_last = maximum(activation_time[activated])
println("\nactivation: first $t_first ms, last $t_last ms, unactivated $n_unactivated / $N")
println("distance-activation correlation (Euclidean from apex): $(round(spread_cor; digits = 3))")

@assert all(isfinite, V_final)
@assert all(v -> -120 < v < 120, V_final) "V outside physiological range: destabilized solve?"
@assert n_unactivated == 0 "$n_unactivated nodes not activated by $t_end ms — raise t_end or check CV"
@assert t_first ≤ stim_duration + 1.0 "first activation after the stimulus window"
@assert coords[argmin(activation_time)][3] < z_apex + stim_height + 2.0 "earliest activation not at the apex"
@assert t_last < t_end - 10.0 "wavefront barely finished — raise t_end"
# Euclidean distance is only a proxy for the geodesic (the wave wraps around the
# cavities), so demand a strong but not perfect correlation.
@assert spread_cor > 0.8 "activation does not spread outward from the apex (cor = $spread_cor)"

# ============================================================================
# Outputs: VTU + activation-map render (full and mid-y cutaway)
# ============================================================================

exportvtk(
    joinpath(@__DIR__, "biventricle_result"), WTP.points(cloud),
    [V_final, activation_time], ["V", "activation_time"],
)
println("Results written to biventricle_result.vtu")

az, el = 0.3π, π / 10
eye = (cos(el) * cos(az), cos(el) * sin(az), sin(el))
at_range = (0.0, t_last)

# iCloud's file provider intermittently stalls `close` on multi-MB PNGs written
# in place under the synced repo. Render to a local temp file and move it into
# position in one bulk copy instead.
function save_png(path, fig; kwargs...)
    tmp = tempname() * ".png"
    save(tmp, fig; kwargs...)
    mv(tmp, path; force = true)
    return path
end

function activation_panel!(where_, mask; title)
    ax = Axis3(where_; azimuth = az, elevation = el, aspect = :data, title = title)
    x, y, z, col = xs[mask], ys[mask], zs[mask], activation_time[mask]
    ord = sortperm(x .* eye[1] .+ y .* eye[2] .+ z .* eye[3])  # far → near for overdraw
    return meshscatter!(
        ax, x[ord], y[ord], z[ord];
        color = col[ord], colorrange = at_range, colormap = :viridis,
        markersize = 0.6dx_mm
    )
end

# The y-normal cut through the centered heart exposes both cavities side by side
# (a four-chamber-like section); the camera sits mostly along +y so the cut face
# turns toward the viewer.
half = ys .<= 0.0
fig = Figure(; size = (1800, 950))
s1 = activation_panel!(fig[1, 1], trues(N); title = "activation time — apex stimulus")
activation_panel!(fig[1, 3], half; title = "cutaway at y = 0")
Colorbar(fig[1, 2], s1; label = "activation time (ms)")

png_path = joinpath(@__DIR__, "biventricle_monodomain.png")
save_png(png_path, fig; px_per_unit = 2)
@assert isfile(png_path) && filesize(png_path) > 20_000
println("saved: ", abspath(png_path))
