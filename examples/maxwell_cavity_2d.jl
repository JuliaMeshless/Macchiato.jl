# Electromagnetics verification: full time-domain Maxwell in a PEC cavity.
#
# Solves the full first-order Maxwell curl system in 2D TMz form,
#   ∂Ez/∂t =  ∂Hy/∂x − ∂Hx/∂y      (Ampère)
#   ∂Hx/∂t = −∂Ez/∂y               (Faraday, x)
#   ∂Hy/∂t =  ∂Ez/∂x               (Faraday, y)
# in normalized units (ε₀ = μ₀ = c = 1) inside a unit-square cavity with
# perfectly conducting (PEC) walls — the canonical time-domain electromagnetics
# benchmark. The initial condition seeds the exact TM₁₁ standing mode, which
# oscillates in place at ω = π√2 indefinitely, so the numerical solution is
# checkable against a closed-form field at every node and every time:
#   Ez =  sin(πx)·sin(πy)·cos(ωt)
#   Hx = −(π/ω)·sin(πx)·cos(πy)·sin(ωt)
#   Hy =  (π/ω)·cos(πx)·sin(πy)·sin(ωt)
# PEC walls require tangential E to vanish; in TMz that is Ez = 0, a plain
# Dirichlet condition on the first variable block — exactly what the framework's
# transient path pins (H stays dynamic on the walls, as it should).
#
# Centered RBF-FD first-derivative matrices generically carry eigenvalues with
# small positive real parts, so an undamped wave system slowly self-excites.
# The standard remedy (Fornberg & Flyer) is a weak hyperviscosity: here γ∇²
# on every field block with γ = 0.05·c·h². Grid-scale noise (|k| ≈ π/h) decays
# ~14 e-folds over the run while the physical mode (|k|² = 2π²) loses only
# ~2e-3 of its amplitude — below the error asserts. The h² scaling vanishes
# under refinement. If energy grows, raise the 0.05 prefactor; γ = 0 disables.
#
# Run from the repo root: julia --project=examples examples/maxwell_cavity_2d.jl
# Output: examples/maxwell_cavity_2d.gif (gitignored) — left: the Ez mode
# breathing in place; right: numeric-vs-exact probe trace with a time cursor.

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using WhatsThePoint
import WhatsThePoint as WTP
using Macchiato
import Macchiato as MM
using Unitful: m
using OrdinaryDiffEqLowOrderRK: RK4, DiscreteCallback
using CairoMakie
using LinearAlgebra: mul!, norm, dot
using Random: seed!

# The Orthtree Bridson interior fill draws from the global RNG, so every run is
# a different cloud realization. The asserts after the solve catch a bad
# realization if anything here changes.
seed!(42)

# ============================================================================
# Problem setup: unit-square cavity, TM₁₁ mode
# ============================================================================

const Lx, Ly = 1.0, 1.0
const h = 0.02                     # nominal spacing: λ = √2 → ~70 pts/wavelength
const kx, ky = π / Lx, π / Ly      # mode wavenumbers
const ω = hypot(kx, ky)            # cavity dispersion: ω = π√2
const T_period = 2π / ω            # oscillation period = √2
const γ_hv = 0.05 * h^2            # Laplacian hyperviscosity (c = 1)
const dt = T_period / 240          # c·dt/h ≈ 0.295, well inside RK4's CFL limit
const t_end = 3.125 * T_period     # 750 steps; ωt = 6.25π so sin = cos = √2/2
const Δt_frame = T_period / 30     # 8 steps per frame → GIF plays 1 period/sec

# t_end deliberately avoids integer multiples of T: at t = nT the exact H
# fields vanish identically and their relative errors are ill-defined. At
# 3.125·T all three fields sit at ~71% amplitude.

# ============================================================================
# Exact solution (substitute into the curl equations to verify)
# ============================================================================

Ez_exact(x, t) = sin(kx * x[1]) * sin(ky * x[2]) * cos(ω * t)
Hx_exact(x, t) = -(ky / ω) * sin(kx * x[1]) * cos(ky * x[2]) * sin(ω * t)
Hy_exact(x, t) = (kx / ω) * cos(kx * x[1]) * sin(ky * x[2]) * sin(ω * t)

# ============================================================================
# Geometry: unit square with exact axis-aligned normals
# ============================================================================
#
# Corners are excluded (each edge runs h..1-h) to avoid normal ambiguity. The
# four edges stay one closed-loop surface (:surface1): the 2D Orthtree's
# segment quadtree takes each named surface as a closed loop, so splitting the
# edges apart would hand it four zero-area "loops" of collinear points — and
# all four walls share the PEC condition anyway.

dxu = h * m
rx = dxu:dxu:(Lx * m - dxu)
ry = dxu:dxu:(Ly * m - dxu)

p_bot = map(i -> WTP.Point(i, 0m), rx)
p_right = map(i -> WTP.Point(Lx * m, i), ry)
p_top = map(i -> WTP.Point(i, Ly * m), reverse(rx))
p_left = map(i -> WTP.Point(0m, i), reverse(ry))

n_bot = map(i -> WTP.Vec(0.0, -1.0), rx)
n_right = map(i -> WTP.Vec(1.0, 0.0), ry)
n_top = map(i -> WTP.Vec(0.0, 1.0), rx)
n_left = map(i -> WTP.Vec(-1.0, 0.0), ry)

p = vcat(p_bot, p_right, p_top, p_left)
n = vcat(n_bot, n_right, n_top, n_left)
part = PointBoundary(p, n, fill(dxu, length(p)))

cloud = WTP.discretize(part, ConstantSpacing(dxu))
N = length(WTP.points(cloud))
n_bnd = length(cloud.boundary)
println("cloud: ", N, " points (", n_bnd, " boundary)")
@assert 1200 < N < 4500 "point count $N out of range: check the edge construction"

# ============================================================================
# Custom PDE model: first-order Maxwell curl system (3 coupled fields)
# ============================================================================

struct MaxwellTMz{T} <: Macchiato.AbstractModel
    γ::T   # Laplacian hyperviscosity coefficient (0 disables)
end

# u = [Ez(1:N); Hx(N+1:2N); Hy(2N+1:3N)] — Macchiato's variable-major layout.
Macchiato.num_vars(::MaxwellTMz, _) = 3

# The transient path calls make_f with no kwargs, so the stencil parameters are
# hard-coded here. First derivatives are much better conditioned than the
# Helmholtz example's fused 2nd-order operator: poly_deg = 4 (15 monomials in
# 2D, k = 40 ≥ 2x that) is plenty at ~70 points per wavelength. One shared
# adjacency list feeds all three operators.
function Macchiato.make_f(model::MaxwellTMz, domain; kwargs...)
    x = node_coordinates(domain)
    n = length(x)
    basis = PHS(3; poly_deg = 4)
    k = 40
    adjl = find_neighbors(x, k)
    Dx = weights(partial(x, 1, 1; k, adjl, basis))
    Dy = weights(partial(x, 1, 2; k, adjl, basis))
    L = model.γ * weights(laplacian(x; k, adjl, basis))

    # Degenerate-stencil tripwire: poly_deg = 4 must reproduce linears exactly.
    xs = [p[1] for p in x]
    ys = [p[2] for p in x]
    @assert maximum(abs, Dx * xs .- 1) < 1.0e-8 "Dx fails ∂x(x) = 1"
    @assert maximum(abs, Dy * ys .- 1) < 1.0e-8 "Dy fails ∂y(y) = 1"
    @assert maximum(abs, Dx * ones(n)) < 1.0e-8 "Dx does not annihilate constants"

    # The Dirichlet closures run after this and overwrite the wall rows of dEz
    # with 0 − Ez; since Ez starts exactly zero there, the walls stay pinned
    # through every RK4 stage.
    return function f(du, u, p, t)
        Ez = view(u, 1:n)
        Hx = view(u, (n + 1):(2n))
        Hy = view(u, (2n + 1):(3n))
        dEz = view(du, 1:n)
        dHx = view(du, (n + 1):(2n))
        dHy = view(du, (2n + 1):(3n))
        mul!(dEz, Dx, Hy)               #  ∂Hy/∂x
        mul!(dEz, Dy, Hx, -1, 1)        #    − ∂Hx/∂y
        mul!(dHx, Dy, Ez, -1, 0)        # −∂Ez/∂y
        mul!(dHy, Dx, Ez)               #  ∂Ez/∂x
        mul!(dEz, L, Ez, 1, 1)          # γ∇² stabilization on every block
        mul!(dHx, L, Hx, 1, 1)
        mul!(dHy, L, Hy, 1, 1)
        return nothing
    end
end

# ============================================================================
# Domain, initial condition, frame recorder
# ============================================================================

# PEC on all four walls: tangential E (= Ez) vanishes on a conductor.
bcs = Dict(:surface1 => PrescribedValue(0.0))
domain = MM.Domain(cloud, bcs, MaxwellTMz(γ_hv))
sim = Simulation(domain, Transient(Δt = dt, stop_time = t_end, solver = RK4()))

# node_coordinates(domain) is ordered identically to u's blocks: boundary
# points first (surface by surface), then volume points.
coords = node_coordinates(domain)
Ez0 = [Ez_exact(x, 0.0) for x in coords]
@assert maximum(abs, Ez0[1:n_bnd]) < 1.0e-12 "IC violates PEC on the walls"

# Custom multi-var model: set! has no field map for it, assign u0 directly
# (run!'s _ensure_u0_initialized! respects a pre-set u0). H starts at zero.
sim.u0 = vcat(Ez0, zeros(N), zeros(N))

# run! keeps only the endpoint of the ODE solution, so animation frames are
# captured in a callback. Discrete energy ½Σ(Ez² + Hx² + Hy²) is recorded with
# each frame — a PEC cavity conserves it, so drift is the instability tripwire.
frames = Vector{Vector{Float64}}()
frame_t = Float64[]
energies = Float64[]

function make_frame_recorder(frames, frame_t, energies, n)
    next_frame = Ref(Δt_frame)
    return function record!(integrator)
        t = integrator.t
        if t ≥ next_frame[] - 1.0e-9
            push!(frames, copy(view(integrator.u, 1:n)))
            push!(frame_t, t)
            push!(energies, 0.5 * sum(abs2, integrator.u))
            next_frame[] += Δt_frame
            if length(frame_t) % 30 == 0
                println(
                    "  t = ", round(t; digits = 2), " (", round(t / T_period; digits = 2),
                    " T) | energy = ", round(energies[end]; sigdigits = 6),
                    " | max|Ez| = ", round(maximum(abs, frames[end]); sigdigits = 3),
                )
            end
        end
        return nothing
    end
end

push!(frames, sim.u0[1:N])
push!(frame_t, 0.0)
push!(energies, 0.5 * sum(abs2, sim.u0))

always(u, t, integrator) = true
# save_positions = (false, false) is load-bearing: the default saves 3N floats
# at every one of the 750 steps.
cb = DiscreteCallback(
    always, make_frame_recorder(frames, frame_t, energies, N);
    save_positions = (false, false),
)

# ============================================================================
# Solve — fixed-step RK4 (adaptive = false is load-bearing: it keeps the CFL
# guarantee and lands frames exactly on the T/30 cadence)
# ============================================================================

@time run!(sim; adaptive = false, callback = cb)
@assert length(frames) ≥ 90 "recorder captured too few frames"

# ============================================================================
# Error report against the exact mode
# ============================================================================

u_f = solution(sim)
t_f = sim.time
Ez_num = u_f[1:N]
Hx_num = u_f[(N + 1):(2N)]
Hy_num = u_f[(2N + 1):(3N)]
@assert all(isfinite, u_f)
@assert maximum(abs, Ez_num) ≤ 1.05 "Ez exceeds mode amplitude 1: destabilized?"
@assert max(maximum(abs, Hx_num), maximum(abs, Hy_num)) ≤ 0.75 "H exceeds 1/√2"

Ez_err = Ez_num .- [Ez_exact(x, t_f) for x in coords]
Hx_err = Hx_num .- [Hx_exact(x, t_f) for x in coords]
Hy_err = Hy_num .- [Hy_exact(x, t_f) for x in coords]
rel_Ez = norm(Ez_err) / norm([Ez_exact(x, t_f) for x in coords])
rel_Hx = norm(Hx_err) / norm([Hx_exact(x, t_f) for x in coords])
rel_Hy = norm(Hy_err) / norm([Hy_exact(x, t_f) for x in coords])
wall_Ez = maximum(abs, Ez_num[1:n_bnd])
energy_drift = energies[end] / energies[1] - 1

# Probe trace: frequency/phase agreement over the whole run in one number.
probe_i = argmin([hypot(x[1] - 0.25, x[2] - 0.25) for x in coords])
probe_num = [f[probe_i] for f in frames]
probe_ex = [Ez_exact(coords[probe_i], t) for t in frame_t]
probe_corr = dot(probe_num, probe_ex) / (norm(probe_num) * norm(probe_ex))

println("\nerror vs exact TM₁₁ mode at t = ", round(t_f; digits = 4), ":")
for (name, val) in (
        "rel L2 Ez" => rel_Ez,
        "rel L2 Hx" => rel_Hx,
        "rel L2 Hy" => rel_Hy,
        "wall max|Ez|" => wall_Ez,
        "energy drift" => energy_drift,
        "probe corr" => probe_corr,
    )
    println("  ", rpad(name, 14), round(val; sigdigits = 4))
end

# Thresholds are ~2.5x the values observed at h = 0.02, poly_deg = 4, seed 42
# (rel L2 ≈ 1.7e-3 on all three fields, drift ≈ +6.7e-3). The discrete energy
# wobbles within a period because the scattered-node sum is an imperfect
# quadrature whose error differs between the Ez² and H² integrands — growth
# beyond the bound, not the wobble, is the instability signal.
@assert wall_Ez < 1.0e-10 "PEC rows not pinned: $wall_Ez"
@assert rel_Ez < 5.0e-3 "relative L2 error on Ez too large: $rel_Ez"
@assert rel_Hx < 5.0e-3 "relative L2 error on Hx too large: $rel_Hx"
@assert rel_Hy < 5.0e-3 "relative L2 error on Hy too large: $rel_Hy"
@assert abs(energy_drift) < 0.02 "energy drift $energy_drift: instability or overdamping"
@assert probe_corr > 0.999 "probe trace decorrelated from exact: $probe_corr"

# ============================================================================
# Render: animated GIF — Ez field + probe trace with time cursor
# ============================================================================

xs = [x[1] for x in coords]
ys = [x[2] for x in coords]
probe_pts = Point2f.(frame_t ./ T_period, probe_num)

c_obs = Observable(frames[1])
probe_obs = Observable(probe_pts[1:1])
cursor_obs = Observable([0.0])
title_obs = Observable("Ez — t = 0.0 (0.0 T)")

fig = Figure(; size = (1180, 520))
ax_f = Axis(fig[1, 1]; aspect = DataAspect(), title = title_obs,
    xlabel = "x (m)", ylabel = "y (m)")
plt = scatter!(ax_f, xs, ys;
    color = c_obs, colorrange = (-1.0, 1.0), colormap = :RdBu,
    markerspace = :data, markersize = 1.4h)
Colorbar(fig[1, 2], plt; label = "Ez")
ax_p = Axis(fig[1, 3]; title = "probe near (0.25, 0.25)",
    xlabel = "t / T", ylabel = "Ez")
ts_fine = range(0.0, t_end; length = 600)
lines!(ax_p, collect(ts_fine ./ T_period), [Ez_exact(coords[probe_i], t) for t in ts_fine];
    color = :gray35, linewidth = 2, label = "exact")
scatter!(ax_p, probe_obs; color = :crimson, markersize = 7, label = "numeric")
vlines!(ax_p, cursor_obs; color = (:gray, 0.4))
ylims!(ax_p, -0.65, 0.65)
axislegend(ax_p; position = :rt)

# iCloud's file provider intermittently stalls `close` on multi-MB files written
# in place under the synced repo. Render to a local temp file and move it into
# position in one bulk copy instead.
gif_path = joinpath(@__DIR__, "maxwell_cavity_2d.gif")
tmp_gif = tempname() * ".gif"
record(fig, tmp_gif, eachindex(frames); framerate = 30) do i
    c_obs[] = frames[i]
    probe_obs[] = probe_pts[1:i]
    cursor_obs[] = [frame_t[i] / T_period]
    title_obs[] = "Ez — t = $(round(frame_t[i]; digits = 2)) ($(round(frame_t[i] / T_period; digits = 2)) T)"
end
mv(tmp_gif, gif_path; force = true)
@assert isfile(gif_path) && filesize(gif_path) > 100_000
println("saved: ", abspath(gif_path))
