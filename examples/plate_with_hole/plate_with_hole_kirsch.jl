# ============================================================================
# STEP 0 — manufactured-solution Kirsch test (polar annulus mesh)
# ============================================================================
# Apply the EXACT Kirsch traction at the outer boundary of a circular annulus
# (hole at r=R, outer boundary at r=R_max). If the solver is correct, the
# interior stress field MUST match the Kirsch solution — no finite-plate
# corrections, no far-field assumptions. A direct test of RBF-FD accuracy.
#
# Run:  jlrun plate_with_hole/plate_with_hole_kirsch.jl   (from examples/)
# ============================================================================
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Macchiato
import RadialBasisFunctions: _build_weights, Partial, MixedPartial, find_neighbors
using RadialBasisFunctions: PHS
using StaticArrays
using LinearAlgebra
using SparseArrays: nnz
using Printf
using CairoMakie

# ---- parameters ------------------------------------------------------------
const R     = 0.30                     # hole radius
const Rmax  = 2.00                     # outer boundary radius (~6.7R)
const σ∞    = 1.0                      # far-field uniaxial tension
const Nr    = 50                       # radial layers (hole → outer)
const Nθ    = 120                      # angular points
const Em    = 1.0e7; const νm = 0.3
model = LinearElasticity(E = Em, ν = νm); μ, λstar = lame_parameters(model)
basis = PHS(3; poly_deg = 3); const k = 35

# ---- polar annulus mesh (r ∈ [R, Rmax], uniform in θ, graded in r) ---------
# Cluster radial layers slightly toward the hole for better stress resolution.
grade = 1.7
svals = [R + (Rmax - R) * (i / Nr)^grade for i in 0:Nr]  # r values

pts = SVector{2, Float64}[]; tag = Symbol[]
for (i, r) in enumerate(svals)
    for j in 0:(Nθ - 1)
        θ = 2π * j / Nθ
        push!(pts, SVector(r * cos(θ), r * sin(θ)))
        push!(tag, i == 1 ? :hole : i == Nr + 1 ? :outer : :interior)
    end
end

const N = length(pts)
adjl = find_neighbors(pts, k)

idx_of(t) = findall(==(t), tag)
const interior_idx = idx_of(:interior)
const hole_idx     = idx_of(:hole)
const outer_idx    = idx_of(:outer)
const neumann_idx  = vcat(hole_idx, outer_idx)
const neumann_adjl = adjl[neumann_idx]

@printf("Polar annulus: N=%d  (interior=%d  hole=%d  outer=%d)\n",
        N, length(interior_idx), length(hole_idx), length(outer_idx))

# ---- Kirsch analytic functions ---------------------------------------------
kirsch_σrr(r, θ) = σ∞ / 2 * (1 - (R / r)^2) +
                   σ∞ / 2 * (1 + 3 * (R / r)^4 - 4 * (R / r)^2) * cos(2θ)
kirsch_σrθ(r, θ) = -σ∞ / 2 * (1 - 3 * (R / r)^4 + 2 * (R / r)^2) * sin(2θ)
kirsch_σθθ(r, θ) = σ∞ / 2 * (1 + (R / r)^2) -
                   σ∞ / 2 * (1 + 3 * (R / r)^4) * cos(2θ)

# Cartesian traction: t = σ·e_r = σ_rr e_r + σ_rθ e_θ
function kirsch_traction(r, θ)
    σrr = kirsch_σrr(r, θ)
    σrθ = kirsch_σrθ(r, θ)
    return SVector(σrr * cos(θ) - σrθ * sin(θ),
                   σrr * sin(θ) + σrθ * cos(θ))
end

# ---- BCs: hole traction-free; outer boundary = exact Kirsch traction -------
normal_of(i) = (tag[i] === :outer ? pts[i] / hypot(pts[i]...) :     # outward radial
                -pts[i] / hypot(pts[i]...))                          # hole inward

traction_of(i) = tag[i] === :outer ?
    kirsch_traction(hypot(pts[i]...), atan(pts[i][2], pts[i][1])) :
    SVector(0.0, 0.0)                                                # hole traction-free

neumann_normals   = [normal_of(i)   for i in neumann_idx]
neumann_tractions = [traction_of(i) for i in neumann_idx]

# ---- rigid-body pins (3 DOFs; stress field is unaffected by the choice) ----
# Pin two interior points to remove x/y translation + rotation.
pin1 = interior_idx[argmin([hypot(pts[i][1] - 1.0, pts[i][2]) for i in interior_idx])]
pin2 = interior_idx[argmin([hypot(pts[i][1] + 0.5, pts[i][2] - 1.0) for i in interior_idx])]
const dirichlet_dofs = [pin1, pin1 + N, pin2 + N]   # ux,uy at pin1; uy at pin2
const dirichlet_vals = zeros(3)

# ---- assemble + solve ------------------------------------------------------
W_d2x  = _build_weights(Partial(2, 1), pts, pts, adjl, basis)
W_d2y  = _build_weights(Partial(2, 2), pts, pts, adjl, basis)
W_d2xy = _build_weights(MixedPartial(1, 2), pts, pts, adjl, basis)
neu = pts[neumann_idx]
W_dx = _build_weights(Partial(1, 1), pts, neu, neumann_adjl, basis)
W_dy = _build_weights(Partial(1, 2), pts, neu, neumann_adjl, basis)

A = assemble_elasticity_from_weights(W_d2x, W_d2y, W_d2xy, N, λstar, μ)
b = zeros(2N)
apply_dirichlet!(A, b, dirichlet_dofs, dirichlet_vals)
tl = build_traction_layout(neumann_idx, neumann_adjl, neumann_normals,
                           neumann_tractions, λstar, μ, N)
apply_traction!(A, b, tl, W_dx, W_dy)

@printf("System: 2N=%d DOFs, nnz=%d\n", 2N, nnz(A))
u = lu(A) \ b
ux = u[1:N]; uy = u[(N + 1):2N]

# ---- stress recovery (plane stress) ----------------------------------------
Dx = _build_weights(Partial(1, 1), pts, pts, adjl, basis)
Dy = _build_weights(Partial(1, 2), pts, pts, adjl, basis)
εxx = Dx * ux; εyy = Dy * uy; εxy = 0.5 .* (Dy * ux .+ Dx * uy)
cc = Em / (1 - νm^2)
σxx = cc .* (εxx .+ νm .* εyy)
σyy = cc .* (εyy .+ νm .* εxx)
σxy = (Em / (1 + νm)) .* εxy

# ---- verification: compare σθθ along θ=90° vs exact Kirsch ------------------
# Primary validation: σθθ along θ=90° (y-axis), where σθθ = σxx and the stress
# concentration lives. Use interior points only (accurate RBF derivative recovery).
ray90 = sort([i for i in interior_idx
              if abs(pts[i][1]) < 0.03 && pts[i][2] > R],
             by = i -> pts[i][2])
# Also check θ=0° (x-axis), where σθθ = σyy, as a secondary check.
ray0  = sort([i for i in interior_idx
              if abs(pts[i][2]) < 0.03 && pts[i][1] > R],
             by = i -> pts[i][1])

# Pick one representative per radial layer (the one closest to the axis)
function dedupe_by_layer(ray)
    layers = Float64[]
    chosen = Int[]
    for i in ray
        r = hypot(pts[i][1], pts[i][2])
        if isempty(layers) || r - layers[end] > 0.01
            push!(layers, r); push!(chosen, i)
        end
    end
    return chosen
end

println()
println(repeat("=", 72))
println("MANUFACTURED-SOLUTION KIRSCH TEST (annulus R=$R → Rmax=$Rmax)")
println(repeat("=", 72))

# θ=90° profile (primary validation)
println("\n  σθθ along θ=90° (y-axis) vs Kirsch:")
@printf("  %6s  %10s  %10s  %8s\n", "r/R", "σθθ (FD)", "Kirsch", "rel")
rels90 = Float64[]
for i in dedupe_by_layer(ray90)
    r = hypot(pts[i][1], pts[i][2])
    fd = σxx[i]; exact = kirsch_σθθ(r, π / 2)
    rel = (fd - exact) / exact; push!(rels90, rel)
    @printf("  %6.2f  %10.4f  %10.4f  %+7.1f%%\n", r / R, fd, exact, 100rel)
end
worst90 = maximum(abs, rels90)

# θ=0° profile (secondary; has a zero crossing where relative error is meaningless)
println("\n  σθθ along θ=0° (x-axis) vs Kirsch (excl. zero-crossing |Kirsch|<0.05):")
@printf("  %6s  %10s  %10s  %8s\n", "r/R", "σθθ (FD)", "Kirsch", "rel")
rels0 = Float64[]
for i in dedupe_by_layer(ray0)
    r = hypot(pts[i][1], pts[i][2])
    fd = σyy[i]; exact = kirsch_σθθ(r, 0.0)
    abs(exact) > 0.05 || continue    # skip zero-crossing region
    rel = (fd - exact) / exact; push!(rels0, rel)
    @printf("  %6.2f  %10.4f  %10.4f  %+7.1f%%\n", r / R, fd, exact, 100rel)
end
worst0 = isempty(rels0) ? 0.0 : maximum(abs, rels0)

# Pass/fail on θ=90° profile (the stress concentration direction)
println()
if worst90 < 0.02
    @printf("  ✓ θ=90° worst |rel| = %.1f%% — SOLVER VALIDATED (Kirsch recovered)\n", 100 * worst90)
elseif worst90 < 0.05
    @printf("  ~ θ=90° worst |rel| = %.1f%% — marginal\n", 100 * worst90)
else
    @printf("  ✗ θ=90° worst |rel| = %.1f%% — SOLVER ISSUE\n", 100 * worst90)
end
if worst0 > 0
    @printf("    θ=0° worst |rel| = %.1f%% (secondary, excl. zero-crossing)\n", 100 * worst0)
end

# Global L2 error over all interior points (all three stress components)
let err_σxx = 0.0, err_σyy = 0.0, err_σxy = 0.0, norm_exact = 0.0
    for i in interior_idx
        r = hypot(pts[i][1], pts[i][2]); θ = atan(pts[i][2], pts[i][1])
        σrr_ex = kirsch_σrr(r, θ); σθθ_ex = kirsch_σθθ(r, θ); σrθ_ex = kirsch_σrθ(r, θ)
        c, s = cos(θ), sin(θ)
        σxx_ex = σrr_ex * c^2 + σθθ_ex * s^2 - 2 * σrθ_ex * c * s
        σyy_ex = σrr_ex * s^2 + σθθ_ex * c^2 + 2 * σrθ_ex * c * s
        σxy_ex = (σrr_ex - σθθ_ex) * c * s + σrθ_ex * (c^2 - s^2)
        err_σxx += (σxx[i] - σxx_ex)^2
        err_σyy += (σyy[i] - σyy_ex)^2
        err_σxy += (σxy[i] - σxy_ex)^2
        norm_exact += σxx_ex^2 + σyy_ex^2 + σxy_ex^2
    end
    global err_L2 = sqrt(err_σxx + err_σyy + err_σxy) / sqrt(norm_exact)
end
@printf("  Global L2 stress error (interior)  = %.2f%%\n", 100 * err_L2)

# ---- K_t estimate: extrapolate interior σθθ(r) at θ=90° to r=R -------------
# Fit σθθ(r) = c1 + c2*(R/r)² + c3*(R/r)⁴ using interior points only.
rr = [hypot(pts[i][1], pts[i][2]) / R for i in ray90]
rinv2 = (R ./ (rr .* R)) .^ 2; rinv4 = (R ./ (rr .* R)) .^ 4
X = hcat(ones(length(ray90)), rinv2, rinv4)
c = X \ [σxx[i] for i in ray90]
K_t_fit = (c[1] + c[2] + c[3]) / σ∞
@printf("\n  K_t (interior extrapolation) = %.3f   (Kirsch exact: 3.0)\n", K_t_fit)
@printf("  Fitted far-field σ∞           = %.4f   (expected: %.3f)\n", c[1], σ∞)

# ---- figure: σxx field + radial profiles vs Kirsch -------------------------
fig = Figure(; size = (1200, 460))
ax1 = Axis(fig[1, 1]; title = "σxx (FD)", xlabel = "x", ylabel = "y",
           aspect = DataAspect(), limits = (-2.2, 2.2, -2.2, 2.2))
sc = scatter!(ax1, [p[1] for p in pts], [p[2] for p in pts]; color = σxx,
              colormap = :RdBu, colorrange = (-1, 3), markersize = 6)
Colorbar(fig[1, 2], sc)

ax2 = Axis(fig[1, 3]; title = "σθθ along θ=90°", xlabel = "r/R", ylabel = "σθθ / σ∞")
rr_smooth = range(1.0, Rmax / R; length = 100)
lines!(ax2, rr_smooth, [kirsch_σθθ(rr * R, π / 2) / σ∞ for rr in rr_smooth];
       linestyle = :dash, label = "Kirsch")
scatter!(ax2, [hypot(pts[i][1], pts[i][2]) / R for i in ray90],
         [σxx[i] / σ∞ for i in ray90]; color = :crimson, label = "FD")

ax3 = Axis(fig[1, 4]; title = "σθθ along θ=0°", xlabel = "r/R", ylabel = "σθθ / σ∞")
lines!(ax3, rr_smooth, [kirsch_σθθ(rr * R, 0.0) / σ∞ for rr in rr_smooth];
       linestyle = :dash, label = "Kirsch")
scatter!(ax3, [hypot(pts[i][1], pts[i][2]) / R for i in ray0],
         [σyy[i] / σ∞ for i in ray0]; color = :crimson, label = "FD")

figpath = joinpath(@__DIR__, "plate_with_hole_kirsch.png")
save(figpath, fig)
println("Figure saved: ", figpath)
