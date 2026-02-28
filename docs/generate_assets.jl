# generate_assets.jl
# One-time script to regenerate documentation images.
# Run from repo root: julia docs/generate_assets.jl

using Pkg
Pkg.activate(joinpath(@__DIR__))
Pkg.instantiate()

using Macchiato
import Macchiato as MM
using WhatsThePoint
import WhatsThePoint as WTP
using RadialBasisFunctions: PHS
using Unitful: m, °, ustrip
using LinearAlgebra
using LinearSolve
using Statistics: mean
using CairoMakie

const ASSETS_DIR = joinpath(@__DIR__, "src", "assets")
mkpath(ASSETS_DIR)

# ============================================================================
# 1. 2D Heat Conduction — Temperature Field
# ============================================================================

function generate_heat_2d()
    dx = 1 / 50 * m
    rx = dx:dx:(1m - dx)
    ry = dx:dx:(1m - dx)

    p_bot = [WTP.Point(i, 0m) for i in rx]
    p_right = [WTP.Point(1m, i) for i in ry]
    p_top = [WTP.Point(i, 1m) for i in reverse(rx)]
    p_left = [WTP.Point(0m, i) for i in reverse(ry)]

    n_bot = [WTP.Vec(0.0, -1.0) for _ in rx]
    n_right = [WTP.Vec(1.0, 0.0) for _ in ry]
    n_top = [WTP.Vec(0.0, 1.0) for _ in rx]
    n_left = [WTP.Vec(-1.0, 0.0) for _ in ry]

    pts = vcat(p_bot, p_right, p_top, p_left)
    nrms = vcat(n_bot, n_right, n_top, n_left)
    areas = fill(dx, length(pts))

    part = PointBoundary(pts, nrms, areas)
    split_surface!(part, 75°)

    cloud = WTP.discretize(part, ConstantSpacing(dx), alg=VanDerSandeFornberg())
    cloud, _ = repel(cloud, ConstantSpacing(dx); α=dx / 20, max_iters=500)

    bcs = Dict(
        :surface1 => MM.Temperature(0.0),
        :surface2 => MM.Temperature(0.0),
        :surface3 => MM.Temperature(100.0),
        :surface4 => MM.Temperature(0.0),
    )

    domain = MM.Domain(cloud, bcs, SolidEnergy(k=1.0, ρ=1.0, cₚ=1.0))
    sim = Simulation(domain)
    run!(sim)
    T = temperature(sim)

    coords = MM._coords(cloud)
    x = [ustrip(pt.x) for pt in coords]
    y = [ustrip(pt.y) for pt in coords]

    fig = Figure(; size=(800, 700), backgroundcolor=:white)
    ax = Axis(fig[1, 1];
        title="Steady-State Temperature",
        xlabel="x [m]",
        ylabel="y [m]",
        aspect=DataAspect(),
    )
    sc = scatter!(ax, x, y; color=T, colormap=:inferno, markersize=12)
    Colorbar(fig[1, 2], sc; label="T")

    save(joinpath(ASSETS_DIR, "heat_2d.png"), fig; px_per_unit=2)
    println("Saved heat_2d.png")
end

# ============================================================================
# 2. 2D Cantilever Beam — Displacement Magnitude
# ============================================================================

function generate_cantilever_beam_2d()
    L = 8.0
    D = 1.0
    P = 1000.0
    E_val = 1e7
    ν_val = 0.3
    I = 2D^3 / 3

    u_exact(x, y) = -P / (6E_val * I) * y * ((6L - 3x) * x + (2 + ν_val) * (y^2 - D^2))
    v_exact(x, y) = P / (6E_val * I) * (3ν_val * y^2 * (L - x) + (4 + 5ν_val) * D^2 * x + (3L - x) * x^2)

    dx = 0.1 * m
    rx = dx:dx:((L * m) - dx)
    ry = dx:dx:((2D * m) - dx)

    p_bot = [WTP.Point(i, -D * m) for i in rx]
    n_bot = [WTP.Vec(0.0, -1.0) for _ in rx]
    p_right = [WTP.Point(L * m, -D * m + i) for i in ry]
    n_right = [WTP.Vec(1.0, 0.0) for _ in ry]
    p_top = [WTP.Point(i, D * m) for i in reverse(rx)]
    n_top = [WTP.Vec(0.0, 1.0) for _ in rx]
    p_left = [WTP.Point(0.0m, -D * m + i) for i in reverse(ry)]
    n_left = [WTP.Vec(-1.0, 0.0) for _ in ry]

    pts = vcat(p_bot, p_right, p_top, p_left)
    nrms = vcat(n_bot, n_right, n_top, n_left)
    areas = fill(dx, length(pts))

    part = PointBoundary(pts, nrms, areas)
    split_surface!(part, 75°)

    cloud = WTP.discretize(part, ConstantSpacing(dx), alg=VanDerSandeFornberg())
    cloud, _ = repel(cloud, ConstantSpacing(dx); α=dx / 50, max_iters=2000)

    bc_left(x, t) = (u_exact(x[1], x[2]), v_exact(x[1], x[2]))
    bc_right(x, t) = (0.0, P * (D^2 - x[2]^2) / (2I))

    bcs = Dict(
        :surface1 => TractionFree(),
        :surface2 => Traction(bc_right),
        :surface3 => TractionFree(),
        :surface4 => Displacement(bc_left),
    )

    model = LinearElasticity(E=E_val, ν=ν_val)
    domain = MM.Domain(cloud, bcs, model)

    sim = Simulation(domain)
    set!(sim, ux=0.0, uy=0.0)

    basis_kw = (; basis=PHS(3; poly_deg=3))
    prob = LinearSolve.LinearProblem(sim.domain; basis_kw...)
    sol = LinearSolve.solve(prob)

    sim._solution = sol.u
    sim.iteration = 1

    ux_sim, uy_sim = displacement(sim)
    displacement_mag = sqrt.(ux_sim .^ 2 .+ uy_sim .^ 2)

    coords = MM._coords(cloud)
    x = [ustrip(pt.x) for pt in coords]
    y = [ustrip(pt.y) for pt in coords]

    fig = Figure(; size=(1000, 500), backgroundcolor=:white)
    ax = Axis(fig[1, 1];
        title="Displacement Magnitude ‖u‖",
        xlabel="x",
        ylabel="y",
        aspect=DataAspect(),
    )
    sc = scatter!(ax, x, y; color=displacement_mag, colormap=:viridis, markersize=8)
    Colorbar(fig[1, 2], sc; label="‖u‖")

    save(joinpath(ASSETS_DIR, "cantilever_beam_2d.png"), fig; px_per_unit=2)
    println("Saved cantilever_beam_2d.png")
end

# ============================================================================
# Run
# ============================================================================

generate_heat_2d()
generate_cantilever_beam_2d()
println("All assets generated.")
