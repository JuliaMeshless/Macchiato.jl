# Spiral (scroll) wave on the segmented biventricular heart — animated.
#
# Same geometry and monodomain + ten Tusscher-Panfilov 2006 setup as
# examples/biventricle_monodomain.jl, with the model pushed into sustained
# reentry and the voltage field recorded into an animation of the full heart.
#
# How the spiral is induced — two knobs, both required:
#
# 1. Cell-model settings (via the TT06_*_SCALE pre-include hooks of
#    ten_tusscher_2006.jl): G_CaL × 0.3, G_Kr × 2, G_Ks × 2 shorten APD90 from
#    ~280 ms to ~150 ms (verified by the single-cell run below). Reentry only
#    survives if the wavelength λ = CV·APD fits inside the tissue; together
#    with the slow isotropic diffusivity below, λ ≈ 20-30 mm ≪ heart size.
#    (ten Tusscher & Panfilov 2006 tune exactly these currents to study
#    spiral stability.)
# 2. S1-S2 cross-field protocol (the classic quadrant recipe from ten
#    Tusscher's spiral demos): S1 is the usual apex-cap stimulus at t = 0, so
#    its recovery line sweeps apex → base and its refractory tail dissolves at
#    ≈ H/CV + APD90. S2 stimulates one QUADRANT — a lateral half of the lower
#    heart (y < 0, z < 60 mm) — in the window between S1 exiting at the base
#    and its tail dissolving. The quadrant wave is blocked against the tail on
#    one side (broken front); as the tail dissolves the free end finds
#    excitable tissue and curls into a reentrant scroll. The double asymmetry
#    of the quadrant is load-bearing — symmetric S2 regions (a full half or an
#    all-height slab) collide with themselves and just chase the tail to the
#    base (both variants were tried and failed).
#
# Isotropic diffusivity is D_L/2 ≈ 0.048 mm²/ms — slow enough that the
# wavelength λ = CV·APD ≈ 50 mm fits the reentrant circuits (around the
# cavities and the break line), fast enough that the induction and several
# rotations fit an affordable simulated window.
#
# Run from the repo root (expect ~30-40 min solve + a few min of rendering):
#   julia --project=examples -t auto examples/biventricle_spiral.jl
# BIV_DX=1.25 (mm) overrides the default 1.5 spacing for a finer (slower) run.
# Output: examples/biventricle_spiral.mp4 (gitignored), full-heart voltage
# animation at 5 ms per frame, 30 fps.
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
using CairoMakie
using JLD2: jldsave
using Random: seed!

# APD-shortening cell-model settings — MUST be defined before the include, the
# cell file bakes them into its conductance constants (see its header).
const TT06_G_CaL_SCALE = 0.3
const TT06_G_Kr_SCALE = 2.0
const TT06_G_Ks_SCALE = 2.0

include(joinpath(@__DIR__, "niederer_benchmark", "ten_tusscher_2006.jl"))

seed!(42)   # same cloud realization as biventricle_monodomain.jl

# ============================================================================
# Single-cell check: measure APD90 of the modified model
# ============================================================================
#
# The S2 timing hangs off the actual APD, so measure it instead of trusting a
# hand-tuned number to survive future cell-file edits.

function single_cell_apd90(; dtc = 0.01, t_max = 600.0)
    u = tt06_initial_conditions(1)
    du = similar(u)
    V_rest = u[1]
    t_up = NaN
    t90 = NaN
    V_peak = -Inf
    for step in 1:round(Int, t_max / dtc)
        t = step * dtc
        stim = 10.0 < t < 11.0 ? 52.0 : 0.0
        tt06_reaction_point!(du, u, 1, 1, stim)
        u .+= dtc .* du
        V = u[1]
        V_peak = max(V_peak, V)
        if isnan(t_up) && V > 0.0
            t_up = t
        end
        if !isnan(t_up) && isnan(t90) && V < V_rest + 0.1 * (V_peak - V_rest)
            t90 = t
            break
        end
    end
    return V_peak, t90 - t_up
end

V_peak_cell, APD90 = single_cell_apd90()
println("modified TT06: V_peak = $(round(V_peak_cell; digits = 1)) mV, APD90 = $(round(APD90; digits = 1)) ms")
@assert 30.0 < V_peak_cell < 50.0 "single-cell upstroke looks wrong"
@assert 100.0 < APD90 < 220.0 "APD90 = $APD90 ms outside the spiral-friendly range"

# ============================================================================
# Physical parameters
# ============================================================================

const σ_L = 0.17 * 0.62 / (0.17 + 0.62)      # ≈ 0.13342 mS/mm (harmonic mean)
const σ_T = 0.019 * 0.24 / (0.019 + 0.24)    # ≈ 0.01761 mS/mm
const β = 140.0                              # surface-to-volume ratio (1/mm)
const Cm_tissue = 0.01                       # membrane capacitance (µF/mm²)
const D_L = σ_L / (β * Cm_tissue)            # ≈ 0.0953 mm²/ms along fiber
const D_T = σ_T / (β * Cm_tissue)            # ≈ 0.0126 mm²/ms transverse
const D_iso = D_L / 2                        # ≈ 0.048 mm²/ms — λ ≈ 50 mm (see header)

const I_stim = 50.0                          # µA/mm³, Niederer stimulus amplitude
const stim_dvdt_amp = I_stim / (β * Cm_tissue)   # ≈ 35.71 mV/ms
const stim_height = 3.0                      # S1 apex cap: z < z_apex + 3 mm
const s1_duration = 2.0                      # ms
# S1 CV at this D and spacing is ≈ 0.33 mm/ms (measured 0.47 at D_L, CV ∝ √D).
const CV_est = 0.33                          # mm/ms
const s2_duration = 4.0                      # ms; longer than S1 to capture partially recovered tissue
const s2_zcut = 60.0                         # mm; S2 quadrant: y < 0 AND z < s2_zcut
const V_activation = 0.0

const dx_mm = parse(Float64, get(ENV, "BIV_DX", "1.5"))   # node spacing (mm)
const dt = 0.01          # ms; reaction-limited (TT06 upstroke)
const snap_every = 5.0   # ms between animation frames

# ============================================================================
# Geometry (identical to biventricle_monodomain.jl)
# ============================================================================

mesh32 = import_mesh(joinpath(@__DIR__, "assets", "biventricle.stl"), m)
verts = map(Meshes.vertices(mesh32)) do p
    Meshes.Point((Float64.(ustrip.(u"m", Meshes.to(p))) .* m)...)
end
mesh = Meshes.SimpleMesh(verts, Meshes.topology(mesh32))

spacing = ConstantSpacing(dx_mm * m)
part = PointBoundary(mesh, spacing)
cloud = WTP.discretize(part, spacing; alg = Orthtree(mesh; spacing = spacing))
coords = node_coordinates(cloud)   # raw mm numbers, boundary-first ordering
N = length(coords)
n_bnd = length(cloud.boundary)
println("cloud: $N points ($n_bnd boundary) at dx = $dx_mm mm")
@assert 30_000 < N < 400_000 "point count $N out of range: check units/spacing"

xs = getindex.(coords, 1)
ys = getindex.(coords, 2)
zs = getindex.(coords, 3)
z_apex = minimum(zs)
H = maximum(zs)
s1_mask = zs .< z_apex + stim_height
# S2 quadrant: one lateral half (y < 0), lower heart only (z < s2_zcut). The
# double asymmetry is load-bearing: an S2 spanning all heights (or the full
# ring) produces a symmetric second beat that collides with itself and merely
# chases the S1 recovery line — which retreats at exactly CV, so the free end
# can never wrap while the tail exists. Both symmetric variants were tried and
# died at the base without a single reentrant beat.
s2_mask = (ys .< 0.0) .& (zs .< s2_zcut)
# Timing: S1 exits the base at ≈ H/CV (~315 ms) and its refractory tail fully
# dissolves ≈ APD90 later (~463 ms). Fire S2 between the two: the tail still
# blocks the quadrant wave on one side (broken front), and it dissolves just as
# the free end needs excitable territory to curl into.
const t_s2 = round(H / CV_est + 0.4 * APD90)
const t_end = t_s2 + 1000.0   # ≈ 5 spiral rotations after induction (period ≈ 200 ms)
println("S1: $(count(s1_mask)) apex nodes at t = 0; S2: $(count(s2_mask)) quadrant nodes at t = $t_s2 ms")
flush(stdout)
@assert count(s1_mask) ≥ 10 "apex cap nearly empty; reduce dx"
@assert 0.1 < count(s2_mask) / N < 0.6 "S2 quadrant mask looks wrong"

# ============================================================================
# Custom PDE model: isotropic monodomain + TT06 reaction, S1-S2 stimulus
# ============================================================================

struct SpiralMonodomain{T} <: Macchiato.AbstractModel
    D::T
    s1_mask::Vector{Bool}
    s2_mask::Vector{Bool}
    stim_dvdt::T
end

Macchiato.num_vars(::SpiralMonodomain, _) = NSTATES

# k = 60 for boundary-stencil conditioning on the choppy surface, hard-coded
# because the transient path calls make_f with no kwargs (see the other
# biventricle example).
function Macchiato.make_f(model::SpiralMonodomain, domain; kwargs...)
    (; D, s1_mask, s2_mask, stim_dvdt) = model
    n = length(domain.cloud)
    W, _, _ = MM.build_neumann_diffusion(domain; k = 60)
    WD = D .* W
    max_w = maximum(abs, nonzeros(WD))
    println(
        "monodomain W: nnz = $(nnz(WD)), max|W| = $(round(max_w; digits = 3)), " *
            "forward-Euler dt_max ≈ $(round(2 / max_w; digits = 3)) ms"
    )

    return function f(du, u, p, t)
        s1_active = t < s1_duration
        s2_active = t_s2 < t < t_s2 + s2_duration
        tforeach(1:n) do i
            stim = (s1_active && s1_mask[i]) || (s2_active && s2_mask[i]) ? stim_dvdt : 0.0
            tt06_reaction_point!(du, u, i, n, stim)
        end
        mul!(view(du, 1:n), WD, view(u, 1:n), 1.0, 1.0)
        return nothing
    end
end

# ============================================================================
# Solve, recording voltage snapshots and per-node upstroke counts
# ============================================================================

model = SpiralMonodomain(D_iso, collect(s1_mask), collect(s2_mask), stim_dvdt_amp)
bcs = Dict(:surface1 => ZeroFlux())
domain = MM.Domain(cloud, bcs, model)
sim = Simulation(domain, Transient(Δt = dt, stop_time = t_end, solver = Euler()))
sim.u0 = tt06_initial_conditions(N)

# Snapshots feed the animation; upstroke counts are the reentry evidence: S1
# gives every node one activation and the S2 beat a second, so a third upstroke
# can only come from a reentrant wave. Counting uses hysteresis (a node must
# repolarize below -40 mV to re-arm) — the TT06 spike-notch-dome shape crosses
# 0 mV twice per beat and would double-count otherwise.
snapshots = Vector{Float32}[]
snap_times = Float64[]
upstrokes = zeros(Int, N)
function make_recorder(snaps, times, counts, n; log_interval = 50.0)
    armed = trues(n)
    next_snap = Ref(0.0)
    next_log = Ref(log_interval)
    return function record!(integrator)
        t = integrator.t
        V = view(integrator.u, 1:n)
        @inbounds for i in 1:n
            if armed[i] && V[i] > V_activation
                counts[i] += 1
                armed[i] = false
            elseif !armed[i] && V[i] < -40.0
                armed[i] = true
            end
        end
        if t ≥ next_snap[] - 1.0e-9
            push!(snaps, Float32.(V))
            push!(times, t)
            next_snap[] += snap_every
        end
        if t ≥ next_log[] - 1.0e-9
            println(
                "  t = $(round(t; digits = 1)) ms | V ∈ [$(round(minimum(V); digits = 1)), " *
                    "$(round(maximum(V); digits = 1))] mV | ≥3 upstrokes: " *
                    "$(count(≥(3), counts)) / $n"
            )
            flush(stdout)
            next_log[] += log_interval
        end
        return nothing
    end
end

always(u, t, integrator) = true
cb = DiscreteCallback(
    always, make_recorder(snapshots, snap_times, upstrokes, N);
    save_positions = (false, false),
)

println("Solving (dt = $dt ms, t_end = $t_end ms, S2 at $t_s2 ms, $(round(Int, t_end / dt)) steps)...")
elapsed = @elapsed run!(sim; callback = cb)
println("solve: $(round(elapsed; digits = 1)) s for $N nodes, $(length(snapshots)) frames")

V_final = solution(sim)[1:N]
n_reentrant = count(≥(3), upstrokes)
println("upstrokes: max $(maximum(upstrokes)); nodes with ≥3 (reentry evidence): $n_reentrant / $N")
flush(stdout)

# ============================================================================
# Animation: full heart, voltage colormap, one frame per snapshot
# ============================================================================
#
# Rendered BEFORE the reentry asserts so that a failed induction still leaves
# the movie behind for diagnosis. The raw frames are also saved to a JLD2 next
# to this script first, so the animation can be re-rendered without re-solving.
#
# Only the surface nodes are drawn (the interior is occluded anyway), as 2D
# scatter sprites: CairoMakie meshscatter tessellates every point into a sphere
# mesh and software-rasterizes ~100M triangles per frame at this cloud size —
# measured at minutes per frame, vs ~0.3 s for sprites.

jldsave(
    joinpath(@__DIR__, "biventricle_spiral_frames.jld2");
    snap_times, V = reduce(hcat, snapshots), x = xs, y = ys, z = zs,
    n_bnd, dx_mm, t_s2, APD90,
)
println("frames saved to biventricle_spiral_frames.jld2")
flush(stdout)

az, el = 0.3π, π / 10
eye = (cos(el) * cos(az), cos(el) * sin(az), sin(el))
surf_ids = 1:n_bnd   # boundary-first ordering
depth = xs[surf_ids] .* eye[1] .+ ys[surf_ids] .* eye[2] .+ zs[surf_ids] .* eye[3]
ord = surf_ids[sortperm(depth)]  # far → near, camera fixed

# Custom "Julia dots" colormap — all four logo colors, each with a
# physiological job: Julia blue = resting, green = repolarizing tail,
# purple = depolarizing shoulder, red = plateau; the pale seam marks the
# wavefront. Green sits between blue and the seam so it paints the slow
# wave-back, not the fast front (voltage crosses the mid-range twice).
julia_dots = cgrad(
    ["#26418f", "#4063D8", "#389826", "#EFEAF2", "#9558B2", "#CB3C33", "#8f2622"],
    [0.0, 0.2, 0.4, 0.55, 0.66, 0.82, 1.0],
)

fig = Figure(; size = (1200, 1100))
frame_V = Observable(snapshots[1][ord])
frame_title = Observable("t = 0 ms")
ax = Axis3(
    fig[1, 1]; azimuth = az, elevation = el, aspect = :data,
    title = frame_title, titlesize = 22,
)
plt = scatter!(
    ax, xs[ord], ys[ord], zs[ord];
    color = frame_V, colorrange = (-90.0, 40.0), colormap = julia_dots,
    markersize = 11,
)
Colorbar(fig[1, 2], plt; label = "V (mV)")

# iCloud's file provider stalls large in-place writes under the synced repo;
# record to a local temp file and move it into position (same workaround as the
# PNG examples).
mp4_path = joinpath(@__DIR__, "biventricle_spiral.mp4")
tmp = tempname() * ".mp4"
record(fig, tmp, eachindex(snapshots); framerate = 30) do fi
    frame_V[] = snapshots[fi][ord]
    frame_title[] = "monodomain scroll wave — t = $(round(Int, snap_times[fi])) ms"
    return nothing
end
mv(tmp, mp4_path; force = true)
@assert isfile(mp4_path) && filesize(mp4_path) > 200_000
println("saved: ", abspath(mp4_path), " ($(round(filesize(mp4_path) / 1.0e6; digits = 1)) MB, $(length(snapshots)) frames)")
flush(stdout)

# ============================================================================
# Reentry verification
# ============================================================================

@assert all(isfinite, V_final)
@assert all(v -> -120 < v < 120, V_final) "V outside physiological range: destabilized solve?"
@assert n_reentrant > 0.02 * N "no sustained reentry — check t_s2 = $t_s2 against APD90 = $APD90"
@assert maximum(snapshots[end]) > -40.0 "activity died out before t_end — spiral not sustained"
println("reentry sustained through t_end = $t_end ms")
