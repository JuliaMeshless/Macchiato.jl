# Electromagnetics showcase: a pulse scattering off the Julia logo.
#
# The three Julia logo dots become dielectric "glass" disks (ε = 4, so the wave
# speed halves inside) in the same first-order Maxwell TMz system validated
# against the exact cavity mode in examples/maxwell_cavity_2d.jl:
#   ∂Ez/∂t = (1/ε(x))·(∂Hy/∂x − ∂Hx/∂y) + γ∇²Ez − σEz + J
#   ∂Hx/∂t = −∂Ez/∂y + γ∇²Hx − σHx
#   ∂Hy/∂t =  ∂Ez/∂x + γ∇²Hy − σHy
# in normalized units (ε₀ = μ₀ = c = 1, μ = 1 everywhere). A short 2-cycle
# burst fired below-left of the logo expands, reaches the red dot first, then
# green, then purple — each dot refracts and briefly traps the wave
# (whispering-gallery ring-down) before re-radiating. Graded-conductivity
# sponges backed by PEC outer walls absorb the outgoing field.
#
# New capability vs the cavity example: inhomogeneous permittivity ε(x). The
# dot interfaces are tanh-smoothed over w_ε = 1.5h — a sharp jump would ring
# with centered RBF-FD stencils.
#
# Headline diagnostic: causality. The per-dot stored EM energy must peak in
# source-distance order red → green → purple.
#
# Run from the repo root: julia --project=examples examples/maxwell_julia_logo.jl
# Output: examples/maxwell_julia_logo.gif (gitignored) — |Ez| through a white →
# green → purple → red ramp built from the Julia logo colors, each frame
# normalized to its own maximum so the wavefronts stay bold through the whole
# decay; the dots are thin outline circles in the brand colors.

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using WhatsThePoint
import WhatsThePoint as WTP
using Macchiato
import Macchiato as MM
using Unitful: m
using OrdinaryDiffEqLowOrderRK: DiscreteCallback
using OrdinaryDiffEqSSPRK: SSPRK43
using CairoMakie
using LinearAlgebra: mul!, norm
using Random: seed!

seed!(42)  # Bridson interior fill draws from the global RNG

# ============================================================================
# Problem setup
# ============================================================================

const Lx, Ly = 3.2, 2.4
const h = 0.02                      # λ₀/h = 25 points per vacuum wavelength
const λ0 = 0.5                      # burst carrier wavelength (c = 1)
const ω0 = 2π / λ0

# The three logo dots: point-up equilateral triangle, side s_tri (edge gaps
# 0.08, like the real logo), centroid (2.0, 1.25). Wave speed inside is
# c/√ε = c/2, so the internal wavelength λ₀/2 = 12.5h stays resolved.
const r_dot = 0.32
const s_tri = 0.72
const cx0, cy0 = 2.0, 1.25
const ε_d = 4.0                     # dot permittivity (refractive index 2)
const w_ε = 1.5h                    # tanh interface half-width

# Listed in causal order: source-to-center distances are red ≈ 0.89,
# green ≈ 1.44, purple ≈ 1.61.
const dots = (
    (label = "red", c = (cx0 - s_tri / 2, cy0 - s_tri / (2 * √3)), color = "#CB3C33"),
    (label = "green", c = (cx0, cy0 + s_tri / √3), color = "#389826"),
    (label = "purple", c = (cx0 + s_tri / 2, cy0 - s_tri / (2 * √3)), color = "#9558B2"),
)

const x_src = (0.75, 0.95)          # burst center, below-left of the logo
const w_src = 0.04                  # spatial Gaussian half-width (2h)
const A_src = 100.0                 # calibrated: peak |Ez| ≈ 1.5 (0.42 at A = 28)
const t0_src = 0.9                  # burst center time
const τ_p = 0.25                    # burst envelope width (~2 cycles)

const L_sp = 0.3                    # sponge width on all four walls
const σ_max = 40.0                  # sponge peak conductivity (quadratic ramp)

const γ_hv = 0.05 * h^2             # Laplacian hyperviscosity (see cavity example)
const dt = 0.006                    # c·dt/h = 0.3, inside SSPRK43's CFL limit
const t_end = 6.0                   # 1000 steps
const Δt_frame = 5dt                # 200 frames + t = 0

dist(x, c) = hypot(x[1] - c[1], x[2] - c[2])

# ============================================================================
# Geometry: plain rectangle, one closed CCW loop (:surface1)
# ============================================================================
# The dots are NOT boundaries — their nodes are ordinary volume nodes that
# happen to carry ε = 4, so no interior loops are needed. Corners are excluded
# (each edge runs h..L−h) to avoid normal ambiguity, like the cavity example.

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
N = length(WTP.points(cloud))
n_bnd = length(cloud.boundary)
println("cloud: ", N, " points (", n_bnd, " boundary)")
@assert 16_000 < N < 34_000 "point count $N out of range: check the rectangle construction"

# ============================================================================
# Custom PDE model: Maxwell TMz + inhomogeneous ε + sponge + soft burst source
# ============================================================================

struct Maxwell{T, V <: AbstractVector{T}} <: Macchiato.AbstractModel
    γ::T        # Laplacian hyperviscosity coefficient
    inv_ε::V    # per-node 1/ε(x)
    σ::V        # per-node sponge conductivity
    src::V      # per-node source spatial profile
    ω0::T       # burst carrier frequency
    t0::T       # burst center time
    τ_p::T      # burst envelope width
end

Macchiato.num_vars(::Maxwell, _) = 3   # u = [Ez; Hx; Hy] blocks of N

# The transient path calls make_f with no kwargs — stencil parameters are
# hard-coded. One shared adjacency list feeds all three operators.
function Macchiato.make_f(model::Maxwell, domain; kwargs...)
    x = node_coordinates(domain)
    n = length(x)
    basis = PHS(3; poly_deg = 4)
    k = 40
    adjl = find_neighbors(x, k)
    Dx = weights(partial(x, 1, 1; k, adjl, basis))
    Dy = weights(partial(x, 1, 2; k, adjl, basis))
    L = model.γ * weights(laplacian(x; k, adjl, basis))

    # Degenerate-stencil tripwire: poly_deg = 4 must reproduce linears exactly.
    xs_ = [q[1] for q in x]
    ys_ = [q[2] for q in x]
    @assert maximum(abs, Dx * xs_ .- 1) < 1.0e-8 "Dx fails ∂x(x) = 1"
    @assert maximum(abs, Dy * ys_ .- 1) < 1.0e-8 "Dy fails ∂y(y) = 1"
    @assert maximum(abs, Dx * ones(n)) < 1.0e-8 "Dx does not annihilate constants"

    (; inv_ε, σ, src, ω0, t0, τ_p) = model
    return function f(du, u, p, t)
        Ez = view(u, 1:n)
        Hx = view(u, (n + 1):(2n))
        Hy = view(u, (2n + 1):(3n))
        dEz = view(du, 1:n)
        dHx = view(du, (n + 1):(2n))
        dHy = view(du, (2n + 1):(3n))
        mul!(dEz, Dx, Hy)               #  (∇×H)z = ∂Hy/∂x − ∂Hx/∂y
        mul!(dEz, Dy, Hx, -1, 1)
        @. dEz *= inv_ε                 #  the new physics: ∂Ez/∂t = (1/ε)(∇×H)z
        mul!(dHx, Dy, Ez, -1, 0)        # −∂Ez/∂y
        mul!(dHy, Dx, Ez)               #  ∂Ez/∂x
        mul!(dEz, L, Ez, 1, 1)          # γ∇² stabilization on every block
        mul!(dHx, L, Hx, 1, 1)
        mul!(dHy, L, Hy, 1, 1)
        s = sin(ω0 * t) * exp(-((t - t0) / τ_p)^2)
        @. dEz += src * s - σ * Ez
        @. dHx -= σ * Hx
        @. dHy -= σ * Hy
        return nothing
    end
end

# ============================================================================
# Domain, material/sponge/source profiles, per-dot masks
# ============================================================================

bcs = Dict(:surface1 => PrescribedValue(0.0))   # PEC outer walls behind the sponges

# node_coordinates(cloud) matches u's block ordering: boundary first, volume after.
coords = node_coordinates(cloud)

ε_at(x) = 1 + (ε_d - 1) * sum(0.5 * (1 - tanh((dist(x, d.c) - r_dot) / w_ε)) for d in dots)
ε_vec = [ε_at(x) for x in coords]
inv_ε_vec = 1 ./ ε_vec

ξ(d) = max(0.0, (L_sp - d) / L_sp)
sponge_at(x) = σ_max * max(ξ(x[1]), ξ(Lx - x[1]), ξ(x[2]), ξ(Ly - x[2]))^2
σ_vec = [sponge_at(x) for x in coords]

src_vec = [A_src * exp(-(dist(x, x_src) / w_src)^2) for x in coords]

model = Maxwell(γ_hv, inv_ε_vec, σ_vec, src_vec, ω0, t0_src, τ_p)
domain = MM.Domain(cloud, bcs, model)
sim = Simulation(domain, Transient(Δt = dt, stop_time = t_end, solver = SSPRK43()))
# Custom multi-var model: set! has no field map for it — assign u0 directly.
sim.u0 = zeros(3N)   # fields start at rest; the burst source excites them

masks = [findall(x -> dist(x, d.c) < r_dot, coords) for d in dots]
@assert all(length.(masks) .> 500) "dot masks unexpectedly small: $(length.(masks))"

# ============================================================================
# Frame + energy recorder
# ============================================================================
# run! keeps only the endpoint, so frames and diagnostics are captured in a
# DiscreteCallback. Total EM energy needs the ε weight: ½Σ(ε·Ez² + Hx² + Hy²).

frames = Vector{Vector{Float64}}()
frame_t = Float64[]
energies = Float64[]
dot_E = [Float64[] for _ in dots]

function make_recorder(frames, frame_t, energies, dot_E, n)
    next_frame = Ref(Δt_frame)
    return function record!(integrator)
        t = integrator.t
        if t ≥ next_frame[] - 1.0e-9
            Ez = view(integrator.u, 1:n)
            Hx = view(integrator.u, (n + 1):(2n))
            Hy = view(integrator.u, (2n + 1):(3n))
            push!(frames, copy(Ez))
            push!(frame_t, t)
            E_tot = 0.0
            @inbounds for i in 1:n
                E_tot += ε_vec[i] * abs2(Ez[i]) + abs2(Hx[i]) + abs2(Hy[i])
            end
            push!(energies, 0.5 * E_tot)
            for (j, mask) in enumerate(masks)
                Ed = 0.0
                @inbounds for i in mask
                    Ed += ε_vec[i] * abs2(Ez[i]) + abs2(Hx[i]) + abs2(Hy[i])
                end
                push!(dot_E[j], 0.5 * Ed)
            end
            next_frame[] += Δt_frame
            if length(frame_t) % 40 == 0
                println("  t = ", round(t; digits = 2),
                    " | energy = ", round(energies[end]; sigdigits = 5),
                    " | max|Ez| = ", round(maximum(abs, Ez); sigdigits = 3))
            end
        end
        return nothing
    end
end

push!(frames, sim.u0[1:N])
push!(frame_t, 0.0)
push!(energies, 0.0)
foreach(E -> push!(E, 0.0), dot_E)

always(u, t, integrator) = true
# save_positions = (false, false) is load-bearing: the default saves 3N floats
# at every one of the 1000 steps.
cb = DiscreteCallback(always, make_recorder(frames, frame_t, energies, dot_E, N);
    save_positions = (false, false))

# ============================================================================
# Solve — fixed-step SSPRK43 (adaptive = false is load-bearing)
# ============================================================================

@time run!(sim; adaptive = false, callback = cb)
@assert length(frames) ≥ 150 "recorder captured too few frames"

# ============================================================================
# Diagnostics: causal arrival order, penetration, ring-down, absorption
# ============================================================================

u_f = solution(sim)
@assert all(isfinite, u_f)
wall_Ez = maximum(abs, u_f[1:n_bnd])
@assert wall_Ez < 1.0e-10 "PEC rows not pinned: $wall_Ez"

peak_Ez = maximum(maximum(abs, fr) for fr in frames)
t_pk = [frame_t[argmax(E)] for E in dot_E]
E_pk = [maximum(E) for E in dot_E]
E_fin = [E[end] for E in dot_E]
E_max, E_end = maximum(energies), energies[end]

println("\npeak |Ez| over the run: ", round(peak_Ez; sigdigits = 3))
for (d, tp, ep, ef) in zip(dots, t_pk, E_pk, E_fin)
    println("  ", rpad(d.label, 7), "t_peak = ", round(tp; digits = 2),
        " | E_peak = ", round(ep; sigdigits = 4),
        " | E_final/E_peak = ", round(ef / ep; sigdigits = 3))
end
println("  total energy: max = ", round(E_max; sigdigits = 4),
    " | final/max = ", round(E_end / E_max; sigdigits = 3))

# Causal arrival order — the headline check: source-to-dot distances are
# red 0.89 < green 1.44 < purple 1.61, so stored energy must peak in that order.
@assert t_pk[1] < t_pk[2] < t_pk[3] "dot energies peaked out of causal order: $t_pk"

# Thresholds are ~0.4× (floor) / ~2.5× (ceilings) the values observed at
# h = 0.02, seed 42, A_src = 100: E_pk ≈ [42, 7.3, 33], t_pk = [2.25, 2.79, 3.51],
# ring-down ratios E_fin/E_pk ≈ [0.043, 0.19, 0.066], total final/max ≈ 0.028.
# A PEC dot would store ~0 — the penetration floor proves ε(x) is engaged.
@assert all(E_pk .> 3.0) "a dot stored almost no energy: $E_pk — dielectric physics not engaged?"
@assert all(E_fin .< 0.5 .* E_pk) "a dot has not rung down by t_end: $(E_fin ./ E_pk)"
@assert E_end < 0.07 * E_max "sponges failed to absorb: final/max = $(E_end / E_max)"

# ============================================================================
# Render: animated GIF — |Ez| over the logo dots
# ============================================================================
# Color is |Ez| through a white → green → purple → red ramp built from the
# Julia logo colors, normalized per frame: each frame's own maximum maps to the
# red end, so the wavefronts stay bold while the field decays by orders of
# magnitude. Near-zero is pinned to white (green only starts at 35%) so the
# background stays clean, and the floor keeps the near-empty early frames white
# instead of amplifying numerical noise. The per-dot energy story lives in the
# asserts and printed peaks above; the animation shows the field.

xs = [x[1] for x in coords]
ys = [x[2] for x in coords]
m_floor = 0.02 * peak_Ez
shade_frame(fr) = (m = max(maximum(abs, fr), m_floor); clamp.(abs.(fr) ./ m, 0, 1))

c_obs = Observable(shade_frame(frames[1]))
title_obs = Observable("Ez — t = 0.0")

fig = Figure(; size = (1060, 740))
ax_f = Axis(fig[1, 1]; aspect = DataAspect(), title = title_obs,
    xlabel = "x (m)", ylabel = "y (m)")
plt = scatter!(ax_f, xs, ys;
    color = c_obs, colorrange = (0, 1),
    colormap = cgrad(["#ffffff", "#389826", "#9558B2", "#CB3C33"], [0.0, 0.35, 0.7, 1.0]),
    markerspace = :data, markersize = 1.4h)
for d in dots
    poly!(ax_f, Circle(Point2f(d.c...), r_dot);
        color = :transparent, strokecolor = d.color, strokewidth = 2)
end
Colorbar(fig[1, 2], plt; label = "|Ez| / frame max")
lines!(ax_f, [L_sp, Lx - L_sp, Lx - L_sp, L_sp, L_sp],
    [L_sp, L_sp, Ly - L_sp, Ly - L_sp, L_sp];
    color = (:gray, 0.4), linestyle = :dash)

# iCloud's file provider intermittently stalls `close` on multi-MB files;
# render to a temp file and mv into position. Every other frame at 24 fps keeps
# the palettized GIF a reasonable size (full cadence hit ~80 MB on the double slit).
gif_path = joinpath(@__DIR__, "maxwell_julia_logo.gif")
tmp_gif = tempname() * ".gif"
record(fig, tmp_gif, 1:2:length(frames); framerate = 24) do i
    c_obs[] = shade_frame(frames[i])
    title_obs[] = "Ez — t = $(round(frame_t[i]; digits = 2))"
end
mv(tmp_gif, gif_path; force = true)
@assert isfile(gif_path) && filesize(gif_path) > 100_000
println("saved: ", abspath(gif_path))
