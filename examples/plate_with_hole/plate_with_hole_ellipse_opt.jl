# ============================================================================
# Plate-with-hole — Ellipse-mode shape optimization (a, b) at fixed area.
# The raw ℓ² nodal gradient is Nyquist noise (±7e-5 sawtooth), but when
# CONTACTED onto smooth ellipse modes, the noise integrates out:
#   dC/da = Σ_j g_j,x * cos(θ_j)      dC/db = Σ_j g_j,y * sin(θ_j)
# With area constraint a*b=const, this is a 1D problem in aspect ratio ρ=a/b.
# Each iteration: compute nodal gradient, project to get dC/da and dC/db,
# take a step that reduces ρ toward 1 (the circle).
# ============================================================================
# Run:  jlrun plate_with_hole/plate_with_hole_ellipse_opt.jl   (from examples/)
# ============================================================================
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Macchiato
import RadialBasisFunctions: _build_weights, Partial, MixedPartial, find_neighbors
using RadialBasisFunctions: PHS
using StaticArrays
using LinearAlgebra
using Printf
using CairoMakie

# ---- parameters ------------------------------------------------------------
const Lx = 4.0; const Ly = 4.0
const a0 = 0.40; const b0 = 0.20
const dx = 0.05
const σ∞ = 1.0
const n_iter = 30
const step_ρ = 0.05              # step in log aspect ratio per iteration

model = LinearElasticity(E = 1.0e7, ν = 0.3)
μ, λstar = lame_parameters(model)
basis = PHS(3; poly_deg = 3); const k = 35

# ---- cloud generation (FIXED interior — no morphing, no remeshing) ---------
ellipse_val(p, a, b) = (p[1] / a)^2 + (p[2] / b)^2
const margin = 1 + 1.2 * dx / min(a0, b0)
const nθ = max(48, round(Int, 2π * sqrt((a0^2 + b0^2) / 2) / dx))

base_pts = SVector{2, Float64}[]; base_tag = Symbol[]
let xs = (-Lx / 2 + dx):dx:(Lx / 2 - dx / 2),
    ys = (-Ly / 2 + dx):dx:(Ly / 2 - dx / 2)
    for x in xs, y in ys
        ellipse_val(SVector(x, y), a0, b0) > margin^2 || continue
        push!(base_pts, SVector(x, y)); push!(base_tag, :interior)
    end
end
for y in (-Ly / 2):dx:(Ly / 2)
    push!(base_pts, SVector(-Lx / 2, y)); push!(base_tag, :xlo)
    push!(base_pts, SVector(Lx / 2, y));  push!(base_tag, :xhi)
end
for x in (-Lx / 2 + dx):dx:(Lx / 2 - dx)
    push!(base_pts, SVector(x, Ly / 2));  push!(base_tag, :yhi)
    push!(base_pts, SVector(x, -Ly / 2)); push!(base_tag, :ylo)
end

const n_fixed = length(base_pts)
function build_hole(a, b, nθ)
    [SVector(a * cos(2π * j / nθ), b * sin(2π * j / nθ)) for j in 0:(nθ - 1)]
end
hole0 = build_hole(a0, b0, nθ)
append!(base_pts, hole0); append!(base_tag, fill(:hole, nθ))

const N = length(base_pts)
const hole_rng = (n_fixed + 1):N
const hole_idx = collect(hole_rng)

idx_of(t) = findall(==(t), base_tag)
const interior_idx = idx_of(:interior)
const outer_idx = vcat(idx_of(:xlo), idx_of(:xhi), idx_of(:yhi), idx_of(:ylo))
const neumann_idx = vcat(outer_idx, hole_idx)

const adjl0 = find_neighbors(base_pts, k)
neumann_adjl0 = adjl0[neumann_idx]

nearest(p0, pool) = pool[argmin([hypot(base_pts[i][1] - p0[1], base_pts[i][2] - p0[2]) for i in pool])]
const pin_ux1 = nearest((0.0, 0.8), interior_idx)
const pin_ux2 = nearest((0.0, -0.8), interior_idx)
const pin_uy1 = nearest((0.8, 0.0), interior_idx)
const dirichlet_dofs = [pin_ux1, pin_ux2, pin_uy1 + N]
const active = let a = trues(2N)
    a[pin_ux1] = a[pin_ux2] = a[pin_uy1 + N] = false; a
end
const interior_rows = let r = falses(N); for i in interior_idx; r[i] = true; end; r end

# ---- forward solve + gradient ----------------------------------------------
hole_centroid(hole) = sum(hole) / length(hole)
function hole_area(a, b, nθ)
    hole = build_hole(a, b, nθ)
    A2 = 0.0; n = length(hole)
    for j in 1:n; p=hole[j]; q=hole[mod1(j+1,n)]; A2+=p[1]*q[2]-q[1]*p[2]; end
    return abs(A2) / 2
end

const A_target = hole_area(a0, b0, nθ)
const r_circle = sqrt(A_target / π)

# For a given ellipse (a,b) satisfying a*b = A_target/π,
# we only need ONE design variable (ρ = a/b or equivalently log ρ).
# But we compute dC/da and dC/db separately, then combine with the area constraint.

const hole_loop = collect(hole_rng)
const hole_pos  = collect(1:length(hole_loop))
const _ZJV = SVector(0.0, 0.0)
zero_njac(g) = NormalJacobian(g, g, _ZJV, _ZJV, _ZJV, _ZJV)
hole_polyline(hole) = polyline_normals(
    reduce(vcat, [[p[1], p[2]] for p in vcat(base_pts[1:n_fixed], hole)]),
    hole_loop, hole_pos)

outer_normal(i) = base_tag[i] === :xhi ? SVector(1.0, 0.0) :
                  base_tag[i] === :xlo ? SVector(-1.0, 0.0) :
                  base_tag[i] === :yhi ? SVector(0.0, 1.0) : SVector(0.0, -1.0)

function compute_gradient(a, b, nθ)
    hole = build_hole(a, b, nθ)
    pts = vcat(base_pts[1:n_fixed], hole)
    hn, hjacs = hole_polyline(hole)
    njacs = vcat(NormalJacobian[zero_njac(i) for i in outer_idx], hjacs)

    normals = SVector{2, Float64}[]; tractions = SVector{2, Float64}[]
    for i in outer_idx
        n = outer_normal(i); push!(normals, n); push!(tractions, σ∞ .* n)
    end
    append!(normals, hn); append!(tractions, fill(SVector(0.0, 0.0), length(hole)))
    layout = build_traction_layout(neumann_idx, neumann_adjl0, normals, tractions, λstar, μ, N)

    b = zeros(2N)
    for kk in eachindex(layout.rows); b[layout.rows[kk]] = layout.b_vals[kk]; end

    res = shape_gradient(reduce(vcat, [[p[1], p[2]] for p in pts]),
                         model, N, adjl0, basis, active,
                         dirichlet_dofs, zeros(3), _ -> b;
                         interior_rows = interior_rows, traction_layout = layout,
                         neumann_ids = neumann_idx, neumann_adjl = neumann_adjl0,
                         traction_jacobians = nothing, normal_jacobians = njacs)

    C = dot(b, res.u)
    g_hole = [SVector(res.Δpts[2*gi - 1], res.Δpts[2*gi]) for gi in hole_idx]
    return (C = C, g_hole = g_hole)
end

# Contract raw nodal gradient onto ellipse semi-axis modes.
# For x_j = a*cos(θ_j):  ∂x_j/∂a = cos(θ_j), ∂y_j/∂a = 0
# For y_j = b*sin(θ_j):  ∂x_j/∂b = 0,           ∂y_j/∂b = sin(θ_j)
function contract_ellipse_gradient(g_hole, a, b, nθ)
    dCda = 0.0; dCdb = 0.0
    for j in 1:nθ
        θ = 2π * (j - 1) / nθ
        dCda += g_hole[j][1] * cos(θ)   # g_x * ∂x/∂a
        dCdb += g_hole[j][2] * sin(θ)   # g_y * ∂y/∂b
    end
    return (dCda = dCda, dCdb = dCdb)
end

# ============================================================================
# Optimization: reduce aspect ratio ρ = a/b toward 1 (circle)
# Constraint: a*b = A_target/π (= r_circle²)
# Free variable: log ρ = log(a/b)
# Derivatives: da/d(log ρ) = a/2,  db/d(log ρ) = -b/2
# dC/d(log ρ) = (a/2) * dC/da + (-b/2) * dC/db = (a*dCda - b*dCdb)/2
# ============================================================================

function optimize_step(a, b)
    # Compute gradient
    local res = compute_gradient(a, b, nθ)
    local mg = contract_ellipse_gradient(res.g_hole, a, b, nθ)

    # Gradient with respect to log aspect ratio
    local dC_dlogρ = (a * mg.dCda - b * mg.dCdb) / 2

    # Area constraint check: actual vs target
    local area_now = π * a * b  # approximate for ellipse
    local area_err = area_now - A_target

    return (C = res.C, dCda = mg.dCda, dCdb = mg.dCdb,
            dC_dlogρ = dC_dlogρ, area_err = area_err)
end

# ---- initial state ----
a_cur, b_cur = a0, b0
result = optimize_step(a_cur, b_cur)

println("=== Ellipse-mode optimization ===\n")
println("Target: circle r=$(round(r_circle, digits=4))  (area=$(round(A_target, digits=6)))")
println("Start:  a=$(a0), b=$(b0), ρ=$(round(a0/b0, digits=2)), C=$(@sprintf("%.4e", result.C))")
@printf("  dC/da = %+.4e   dC/db = %+.4e\n", result.dCda, result.dCdb)
@printf("  dC/d(log ρ) = %+.4e  (should be >0: decreasing ρ decreases C)\n\n", result.dC_dlogρ)

@printf("%5s %10s %8s %8s %8s %12s %12s\n", "iter", "C", "a", "b", "ρ=a/b", "dC/d(logρ)", "area_err")
@printf("%5d %10s %8.4f %8.4f %8.2f %12s %12s\n", 0, "", a_cur, b_cur, a_cur/b_cur, "", "")

hist_C = Float64[]; hist_a = Float64[]; hist_b = Float64[]; hist_ρ = Float64[]

for it in 1:n_iter
    result = optimize_step(a_cur, b_cur)
    push!(hist_C, result.C); push!(hist_a, a_cur); push!(hist_b, b_cur)
    push!(hist_ρ, a_cur / b_cur)

    dC_dlogρ = result.dC_dlogρ
    logρ = log(a_cur / b_cur)

    # Descent: move logρ toward 0 (circle), direction given by dC/d(logρ)
    if abs(dC_dlogρ) < 1e-12 || abs(logρ) < 1e-6
        println("  Converged.")
        break
    end

    # Normalized step: reduce |logρ| by step_ρ, with sign from gradient
    Δlogρ = -sign(dC_dlogρ) * step_ρ
    # Don't overshoot zero
    if abs(Δlogρ) > abs(logρ)
        Δlogρ = -logρ  # go exactly to circle
    end
    logρ_new = logρ + Δlogρ
    ρ_new = exp(logρ_new)

    # New semi-axes satisfying area constraint
    a_new = sqrt(A_target / π * ρ_new)
    b_new = sqrt(A_target / (π * ρ_new))

    @printf("%5d %10.5e %8.4f %8.4f %8.2f %+12.4e %+12.2e\n",
            it, result.C, a_new, b_new, ρ_new, result.dC_dlogρ, π*a_new*b_new - A_target)

    global a_cur, b_cur = a_new, b_new
end

# ---- final state ----
result = optimize_step(a_cur, b_cur)
push!(hist_C, result.C); push!(hist_a, a_cur); push!(hist_b, b_cur)
push!(hist_ρ, a_cur / b_cur)

println("\n--- Final ---")
@printf("a: %.3f → %.3f  (target: %.3f)\n", a0, a_cur, r_circle)
@printf("b: %.3f → %.3f  (target: %.3f)\n", b0, b_cur, r_circle)
@printf("ρ: %.2f → %.4f  (target: 1.0)\n", a0/b0, a_cur/b_cur)
@printf("C: %.4e → %.4e  (target circle: ~5.0e-5)\n", hist_C[1], result.C)

# ---- plot ----
fig = Figure(size = (1200, 450))
ax1 = Axis(fig[1, 1]; title = "Hole shapes", aspect = DataAspect(), xlabel = "x", ylabel = "y")
h_init = build_hole(a0, b0, nθ)
hx = [p[1] for p in h_init]; hy = [p[2] for p in h_init]
lines!(ax1, vcat(hx, hx[1]), vcat(hy, hy[1]); color = :red, linewidth = 2, label = "start")
h_final = build_hole(a_cur, b_cur, nθ)
hx = [p[1] for p in h_final]; hy = [p[2] for p in h_final]
lines!(ax1, vcat(hx, hx[1]), vcat(hy, hy[1]); color = :green, linewidth = 2, label = "final")
ts = range(0, 2π; length = 200)
lines!(ax1, r_circle .* cos.(ts), r_circle .* sin.(ts);
       color = :blue, linestyle = :dash, linewidth = 2, label = "target circle")
axislegend(ax1)

ax2 = Axis(fig[1, 2]; title = "Aspect ratio convergence", xlabel = "iter", ylabel = "ρ = a/b")
lines!(ax2, [a0/b0; hist_ρ]; linewidth = 2)
hlines!(ax2, [1.0]; linestyle = :dash, color = :gray, label = "circle")
axislegend(ax2)

ax3 = Axis(fig[1, 3]; title = "Compliance", xlabel = "iter", ylabel = "C")
lines!(ax3, hist_C; linewidth = 2)
hlines!(ax3, [4.88e-5]; linestyle = :dash, color = :gray, label = "circle C (est)")

save(joinpath(@__DIR__, "plate_with_hole_ellipse_opt.png"), fig)
println("Saved: plate_with_hole_ellipse_opt.png")
