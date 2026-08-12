# Niederer benchmark for cardiac electrophysiology — on the Macchiato API.
# Niederer et al. (2011) "Verification of cardiac tissue electrophysiology simulators
# using an N-version benchmark", Phil. Trans. R. Soc. A 369:4331-4351
#
# Solves the monodomain equation ∂V/∂t = ∇·(D∇V) − I_ion/Cm on a 20×7×3 mm cuboid
# with the ten Tusscher–Panfilov 2006 epicardial cell model (19 states per node),
# fibers along x (D = diag(D_L, D_T, D_T)), zero-flux boundaries everywhere, and a
# 2 ms corner-cube stimulus. Unsplit forward Euler through the standard Transient
# mode; the anisotropic operator is folded through the ghost-node machinery via
# `Macchiato.build_neumann_diffusion(domain; operator = ...)` — this script is the
# worked anisotropic example referenced by docs/src/custom_pdes.md.
#
# Units: coordinates carry an `m` unit label but the numbers are millimetres —
# the Orthtree volume fill emits raw-float points that Meshes defaults to metres,
# so the labels are lies everyone agrees on. All physics is mm/ms/mV.
#
# Run from the repo root:
#   julia --project=examples -t auto examples/niederer_benchmark/niederer_benchmark.jl
# Outputs: console activation table (P1–P9), niederer_diagonal.csv (tracked),
#          niederer_result.vtu and niederer_benchmark.png (both gitignored).
#
# Requires WhatsThePoint ≥ 0.3 (the `Orthtree` volume-fill API).

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using WhatsThePoint
import WhatsThePoint as WTP
using Macchiato
import Macchiato as MM
import Meshes
using Meshes: SimpleMesh, connect, Triangle
using Unitful: m
# OrdinaryDiffEq 7.x no longer re-exports the low-order solvers
using OrdinaryDiffEqLowOrderRK: Euler, DiscreteCallback
using OhMyThreads: tforeach
using StaticArrays: SVector
using LinearAlgebra: mul!, norm
using SparseArrays: nnz, nonzeros
using DelimitedFiles: writedlm, readdlm
using CairoMakie
using Random: seed!

include(joinpath(@__DIR__, "ten_tusscher_2006.jl"))

# The Poisson-disk surface sampler and Bridson fill draw from the global RNG, so
# every run is a different cloud realization. The asserts after the solve catch
# a bad realization if anything here changes.
seed!(42)

# ============================================================================
# Physical parameters (Niederer 2011, Table 3)
# ============================================================================

const Lx, Ly, Lz = 20.0, 7.0, 3.0   # domain size (mm); fibers along x

# Intra-/extra-cellular conductivities, σ in S/m == mS/mm (Table 3)
const σ_iL, σ_iT = 0.17, 0.019
const σ_eL, σ_eT = 0.62, 0.24
# Monodomain conductivities: σ = σ_i σ_e / (σ_i + σ_e)
const σ_L = σ_iL * σ_eL / (σ_iL + σ_eL)   # ≈ 0.13342 mS/mm
const σ_T = σ_iT * σ_eT / (σ_iT + σ_eT)   # ≈ 0.01761 mS/mm

const β = 140.0                       # surface-to-volume ratio (1/mm)
const Cm_tissue = 0.01                # membrane capacitance (µF/mm²)
const D_L = σ_L / (β * Cm_tissue)     # diffusion along fiber (mm²/ms)
const D_T = σ_T / (β * Cm_tissue)     # diffusion transverse (mm²/ms)

# Stimulus (Table 3): 50 000 µA/cm³ for 2 ms in a 1.5 mm cube at the corner.
# Converted to µA/mm³ so units line up with β [1/mm] and Cm [µF/mm²]:
# stim_dvdt = I_stim / (β Cm) = 50 / 1.4 ≈ 35.71 mV/ms.
const I_stim = 50.0                   # stimulus amplitude (µA/mm³ = 50 000 µA/cm³)
const stim_duration = 2.0             # stimulus duration (ms)
const stim_size = 1.5                 # stimulus cube side length (mm)

# Activation threshold per paper §3(f): first crossing of V through 0 mV.
const V_activation = 0.0

# Niederer benchmark reference points (Figure 1b). 8 corners + centre.
const P_REF = (
    P1 = (0.0, 0.0, 0.0),    # stimulus corner
    P2 = (0.0, 7.0, 0.0),
    P3 = (20.0, 7.0, 0.0),
    P4 = (20.0, 0.0, 0.0),
    P5 = (0.0, 0.0, 3.0),
    P6 = (0.0, 7.0, 3.0),
    P7 = (20.0, 0.0, 3.0),
    P8 = (20.0, 7.0, 3.0),   # far corner, last to activate
    P9 = (10.0, 3.5, 1.5),   # centre
)

# dx=0.5 is the shipping configuration: stable, activation times within ~2 ms of
# FD on the boundary reference points. Reaching Niederer's dx=0.1 with explicit
# time stepping needs semi-implicit diffusion, not available in this iteration.
const dx = 0.5    # node spacing (mm)
const dt = 0.005  # forward-Euler step (ms); dt·max|W| must stay ≲ 2
const t_end = 80.0

# ============================================================================
# Geometry: 12-triangle cuboid, Poisson-disk surface + Orthtree volume fill
# ============================================================================

vertices = Meshes.Point.(
    [
        (0.0, 0.0, 0.0), (Lx, 0.0, 0.0), (Lx, Ly, 0.0), (0.0, Ly, 0.0),
        (0.0, 0.0, Lz), (Lx, 0.0, Lz), (Lx, Ly, Lz), (0.0, Ly, Lz),
    ]
)
triangles = [
    connect((1, 3, 2), Triangle), connect((1, 4, 3), Triangle),
    connect((5, 6, 7), Triangle), connect((5, 7, 8), Triangle),
    connect((1, 2, 6), Triangle), connect((1, 6, 5), Triangle),
    connect((3, 4, 8), Triangle), connect((3, 8, 7), Triangle),
    connect((1, 5, 8), Triangle), connect((1, 8, 4), Triangle),
    connect((2, 3, 7), Triangle), connect((2, 7, 6), Triangle),
]
mesh = SimpleMesh(vertices, triangles)

# Poisson-disk surface sampling is tessellation-independent — no refinement loop,
# and per-point normals come from the sampled triangle, i.e. exact face normals.
# All six faces share the same ZeroFlux BC, so the single :surface1 needs no
# splitting.
spacing = ConstantSpacing(dx * m)
part = PointBoundary(mesh, spacing)
cloud = WTP.discretize(part, spacing; alg = Orthtree(mesh; spacing = spacing))

coords = node_coordinates(cloud)   # raw mm numbers, boundary-first ordering
N = length(coords)
n_bnd = length(cloud.boundary)
println("cloud: $N points ($n_bnd boundary + $(N - n_bnd) volume)")
@assert 2_500 < N < 12_000 "point count $N out of range: check units/spacing"
# Orthtree previously leaked points outside thin geometry (WhatsThePoint #76/#77);
# surface any recurrence instead of filtering it silently.
tol = 1.0e-6
@assert all(coords) do p
    -tol ≤ p[1] ≤ Lx + tol && -tol ≤ p[2] ≤ Ly + tol && -tol ≤ p[3] ≤ Lz + tol
end "volume fill emitted out-of-domain points"

# ============================================================================
# Custom PDE model: monodomain with anisotropic diffusion + TT06 reaction
# ============================================================================

struct Monodomain <: Macchiato.AbstractModel
    D::SVector{3, Float64}   # diagonal diffusion tensor (mm²/ms), fibers along x
    stim_dvdt::Float64       # I_stim/(β·Cm) (mV/ms)
    stim_duration::Float64   # ms
    stim_size::Float64       # stimulus corner-cube side (mm)
end

Macchiato.num_vars(::Monodomain, _) = NSTATES   # 19 states per node, SoA layout

# `k = 60` is deliberately larger than the k≈30 often quoted for 3D RBF-FD: on a
# scattered cloud, boundary-point stencils with only 40 neighbors tend to land
# entirely on a single face (quadratic cross-terms xy/xz/yz go near-rank-deficient
# → stencil weights blow up). k = 60 forces each stencil to reach into the
# interior. It is hard-coded because the transient path calls make_f with no
# kwargs — run! kwargs go to OrdinaryDiffEq.solve only.
function Macchiato.make_f(model::Monodomain, domain; kwargs...)
    D = model.D
    n = length(domain.cloud)
    W, _, _ = MM.build_neumann_diffusion(
        domain; k = 60,
        operator = (data; eval_points, k) -> custom(
            data, @operator(∇ ⋅ (D * ∇));
            eval_points = eval_points, k = k, basis = PHS(3; poly_deg = 2),
        ),
    )
    row_sum = maximum(abs, vec(sum(W; dims = 2)))
    max_w = maximum(abs, nonzeros(W))
    println(
        "monodomain W: nnz = $(nnz(W)), max|row-sum| = $row_sum, " *
            "max|W| = $(round(max_w; digits = 1)), forward-Euler dt_max ≈ " *
            "$(round(2 / max_w; digits = 4)) ms"
    )
    @assert row_sum < 1.0e-8 "ghost-folded operator does not annihilate constants"

    x = node_coordinates(domain)   # boundary-first, same ordering as u
    s = model.stim_size
    is_stim = [x[i][1] ≤ s && x[i][2] ≤ s && x[i][3] ≤ s for i in 1:n]
    println("stimulus region: $(count(is_stim)) nodes in the $s mm corner cube")
    @assert count(is_stim) > 10 "stimulus cube nearly empty; reduce dx"

    stim_amp = model.stim_dvdt
    stim_dur = model.stim_duration
    return function f(du, u, p, t)
        # tt06_reaction_point! SETS du (du[i] = -I_ion + stim), so the reaction
        # must run first; diffusion then accumulates onto the V block.
        active = t < stim_dur ? stim_amp : 0.0
        tforeach(1:n) do i
            tt06_reaction_point!(du, u, i, n, is_stim[i] ? active : 0.0)
        end
        mul!(view(du, 1:n), W, view(u, 1:n), 1.0, 1.0)
        return nothing
    end
end

# ============================================================================
# Solve: Domain → Simulation → run! (unsplit forward Euler + activation callback)
# ============================================================================

model = Monodomain(SVector(D_L, D_T, D_T), I_stim / (β * Cm_tissue), stim_duration, stim_size)
bcs = Dict(:surface1 => ZeroFlux())
domain = MM.Domain(cloud, bcs, model)
sim = Simulation(domain, Transient(Δt = dt, stop_time = t_end, solver = Euler()))

# Custom 19-variable model: set! has no field map for it, so assign u0 directly
# (run!'s _ensure_u0_initialized! respects a pre-set u0).
sim.u0 = tt06_initial_conditions(N)

# Activation times: first V > 0 crossing, checked after every accepted Euler step.
# save_positions = (false, false) is load-bearing — the DiscreteCallback default
# (true, true) would force a 19N-float solution save at all 16 000 steps.
# The recorder closes over its arrays (rather than reading script globals) so the
# per-step scan stays type-stable.
function make_activation_recorder(activation, n; log_interval = 5.0)
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

println("Solving (dt = $dt ms, t_end = $t_end ms)...")
@time run!(sim; callback = cb)

u_final = solution(sim)
V_final = u_final[1:N]
@assert all(isfinite, V_final)
@assert all(v -> -120 < v < 120, V_final) "V outside physiological range: destabilized solve?"

# ============================================================================
# Activation report vs the Niederer reference points
# ============================================================================

"""
    nearest_point_index(pts, target) -> (Int, Float64)

Linear-search nearest neighbor of `target` in `pts`. O(N) per query; fine for
the handful of benchmark points we query.
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

println("\nActivation times at Niederer reference points:")
println("  Pi   target (mm)           snapped (mm)                dist   AT (ms)")
at_pairs = [
    begin
            idx, dist = nearest_point_index(coords, target)
            p = coords[idx]
            t_act = activation_time[idx]
            at_str = isnan(t_act) ? "not activated" : string(round(t_act; digits = 2))
            println(
                "  $(rpad(name, 4)) ",
                "($(rpad(round(target[1]; digits = 2), 5)), $(rpad(round(target[2]; digits = 2), 4)), $(rpad(round(target[3]; digits = 2), 4))) ",
                "($(rpad(round(p[1]; digits = 3), 6)), $(rpad(round(p[2]; digits = 3), 5)), $(rpad(round(p[3]; digits = 3), 5))) ",
                "$(rpad(round(dist; digits = 3), 6)) $at_str"
            )
            name => t_act
        end
        for (name, target) in pairs(P_REF)
]
at = NamedTuple(at_pairs)

# Golden reference: the recorded dx=0.5 run of the pre-port script
# (niederer_run_ghost_dx0.5.log). The cloud realization differs (Poisson-disk +
# Bridson vs refined centroids + VdSF; ghost offset is now the local
# nearest-neighbor spacing rather than a fixed dx). The recorded run of this
# script (niederer_run_macchiato_dx0.5.log) lands within 0.8 ms of golden
# everywhere except P2 (2.5), P6 (1.6), and P5 (1.4); tolerances below leave
# ~1 ms of margin over those observations.
golden = (P1 = 1.3, P2 = 19.0, P3 = 41.6, P4 = 34.6, P5 = 5.5, P6 = 20.4, P7 = 34.7, P8 = 42.0, P9 = 21.1)
@assert count(isnan, activation_time) == 0 "not all nodes activated by $t_end ms"
@assert at.P1 < 5.0
@assert at.P1 < at.P5 < at.P2 && at.P2 < at.P3 "wavefront ordering violated"
@assert maximum(values(at)) < 50.0
for name in keys(golden)
    Δ = abs(at[name] - golden[name])
    tolr = name === :P2 ? 3.5 : 2.5
    @assert Δ ≤ tolr "activation time $name off golden $(golden[name]) by $(round(Δ; digits = 2)) ms (tol $tolr)"
end

# ============================================================================
# Outputs: diagonal CSV (tracked) + VTU (gitignored)
# ============================================================================

function write_diagonal_csv(filename, pts, activation; n_samples = 50)
    p1 = SVector(P_REF.P1...)
    p8 = SVector(P_REF.P8...)
    total_length = norm(p8 - p1)
    rows = Matrix{Float64}(undef, n_samples, 5)  # distance, x, y, z, AT
    for i in 1:n_samples
        s = (i - 1) / (n_samples - 1)
        target = p1 + s * (p8 - p1)
        idx, _ = nearest_point_index(pts, Tuple(target))
        p = pts[idx]
        rows[i, 1] = s * total_length
        rows[i, 2] = p[1]
        rows[i, 3] = p[2]
        rows[i, 4] = p[3]
        rows[i, 5] = activation[idx]
    end
    open(filename, "w") do io
        println(io, "distance_mm,x_mm,y_mm,z_mm,activation_time_ms")
        writedlm(io, rows, ',')
    end
    println("Diagonal P1→P8 activation curve written to $filename")
    return rows
end

diag_rows = write_diagonal_csv(joinpath(@__DIR__, "niederer_diagonal.csv"), coords, activation_time)

exportvtk(
    joinpath(@__DIR__, "niederer_result"), WTP.points(cloud),
    [V_final, activation_time], ["V", "activation_time"],
)
println("Results written to niederer_result.vtu")

# ============================================================================
# Render: activation map (full + mid-y cutaway) and the diagonal curve vs FD
# ============================================================================

xs = [p[1] for p in coords]
ys = [p[2] for p in coords]
zs = [p[3] for p in coords]

az, el = 0.35π, π / 7
eye = (cos(el) * cos(az), cos(el) * sin(az), sin(el))
y_mid = Ly / 2
half = ys .<= y_mid
at_range = (0.0, maximum(activation_time))

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
        markersize = 0.5dx
    )
end

fig = Figure(; size = (1800, 1200))
s1 = activation_panel!(fig[1, 1], trues(N); title = "activation time (ms)")
activation_panel!(fig[1, 3], half; title = "activation time — cutaway at y = $(y_mid) mm")
Colorbar(fig[1, 2], s1; label = "activation time (ms)")

ax = Axis(
    fig[2, 1:3]; xlabel = "distance along P1→P8 diagonal (mm)",
    ylabel = "activation time (ms)", title = "diagonal activation curve"
)
fd_csv = joinpath(@__DIR__, "niederer_diagonal_fd.csv")
if isfile(fd_csv)
    fd = readdlm(fd_csv, ','; skipstart = 1)
    lines!(
        ax, Float64.(fd[:, 1]), Float64.(fd[:, 5]);
        color = :gray, linestyle = :dash, label = "FD reference (dx = 0.1 mm)"
    )
end
scatterlines!(
    ax, diag_rows[:, 1], diag_rows[:, 5];
    color = :crimson, markersize = 6, label = "RBF-FD (dx = $dx mm)"
)
axislegend(ax; position = :lt)

png_path = joinpath(@__DIR__, "niederer_benchmark.png")
save_png(png_path, fig; px_per_unit = 2)
@assert isfile(png_path) && filesize(png_path) > 20_000
println("saved: ", abspath(png_path))
