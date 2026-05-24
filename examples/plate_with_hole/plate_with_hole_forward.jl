# ============================================================================
# Plate-with-hole — Stage 1, step 1: forward solve + Kirsch verification
# ============================================================================
# Goal (plan_plate_with_hole.md): build the plate-with-hole forward problem and
# verify the analytic stress concentration BEFORE any optimization.
#
# Model: 2D plane-stress plate with a small central circular hole, loaded in
# uniaxial tension (σ∞ in x). Clamped left edge / tension right edge removes
# rigid-body modes and reuses the validated cantilever BC structure (Dirichlet
# + Traction + traction-free). The hole boundary is a traction-free Neumann
# loop — exactly the design surface the shape optimizer will later move.
#
# Why 2D here: the validated elasticity solver AND the manual adjoint are 2D
# (2N DOFs, Partial(2,d) operators). The 3D STL (plate_with_hole.stl) is the
# geometry artifact for the eventual 3D lift; Stage 1 validates the method in
# the plane-stress cross-section.
#
# Cloud + assembly are built manually (not via WTP discretize, whose 2D
# isinside is single-loop and won't carve a hole), using the SAME assembly path
# shape_gradient differentiates: _build_weights → assemble_elasticity_from_weights
# → apply_dirichlet! / apply_traction! → lu solve.
#
# Analytic check (Kirsch, infinite plate, uniaxial σ∞):
#   hoop stress σθθ(θ) = σ∞(1 − 2cos2θ)  on the hole;  max = 3σ∞ at θ=±90°
#   (crown, perpendicular to load), and −σ∞ at θ=0,π (sides). K_t = 3.
# Finite width lowers the peak slightly; we report the measured ratio.
#
# Run:  jlrun plate_with_hole/plate_with_hole_forward.jl   (from examples/)
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
const Lx = 4.0      # plate length (x), domain [-Lx/2, Lx/2]
const Ly = 2.0      # plate height (y), domain [-Ly/2, Ly/2]
const R  = 0.15     # circular hole radius (small vs plate ⇒ ~infinite-plate Kirsch)
const dx = 0.05     # nominal node spacing
const σ∞ = 1.0      # applied uniaxial tension (x)

model    = LinearElasticity(E = 1.0e7, ν = 0.3)
const Em = 1.0e7
const νm = 0.3
μ, λstar = lame_parameters(model)
basis    = PHS(3; poly_deg = 3)
const k  = 35

# ---- build the 2D cloud: interior grid (hole carved) + boundary rings ------
pts = SVector{2, Float64}[]
tag = Symbol[]                                   # :interior / :left / :right / :top / :bot / :hole

# interior background grid, strictly inside the plate and outside hole(+buffer)
let xs = (-Lx / 2 + dx):dx:(Lx / 2 - dx / 2),
    ys = (-Ly / 2 + dx):dx:(Ly / 2 - dx / 2)
    for x in xs, y in ys
        hypot(x, y) > R + 0.5 * dx || continue   # carve the hole
        push!(pts, SVector(x, y)); push!(tag, :interior)
    end
end

# outer boundary points (corners assigned to left/right so each node appears once)
ys_edge = (-Ly / 2):dx:(Ly / 2)
xs_edge_in = (-Lx / 2 + dx):dx:(Lx / 2 - dx)     # top/bottom interior of the span
for y in ys_edge
    push!(pts, SVector(-Lx / 2, y)); push!(tag, :left)    # clamped
    push!(pts, SVector(Lx / 2, y));  push!(tag, :right)   # tension
end
for x in xs_edge_in
    push!(pts, SVector(x, Ly / 2));  push!(tag, :top)     # free
    push!(pts, SVector(x, -Ly / 2)); push!(tag, :bot)     # free
end

# hole ring (traction-free Neumann loop = future design boundary)
let nθ = max(48, round(Int, 2π * R / (0.5 * dx)))
    for j in 0:(nθ - 1)
        θ = 2π * j / nθ
        push!(pts, SVector(R * cos(θ), R * sin(θ))); push!(tag, :hole)
    end
end

const N = length(pts)
adjl = find_neighbors(pts, k)

idx_of(t)      = findall(==(t), tag)
interior_idx   = idx_of(:interior)
left_idx       = idx_of(:left)
right_idx      = idx_of(:right)
top_idx        = idx_of(:top)
bot_idx        = idx_of(:bot)
hole_idx       = idx_of(:hole)
neumann_idx    = vcat(right_idx, top_idx, bot_idx, hole_idx)   # everything not clamped/interior

@printf("Cloud: N=%d  (interior=%d  left=%d  right=%d  top=%d  bot=%d  hole=%d)\n",
        N, length(interior_idx), length(left_idx), length(right_idx),
        length(top_idx), length(bot_idx), length(hole_idx))

# ---- Neumann normals + tractions (aligned with neumann_idx order) ----------
# :hole normal points inward (toward centre) = out of the material
normal_of(i) =
    tag[i] === :right ? SVector(1.0, 0.0) :
    tag[i] === :top   ? SVector(0.0, 1.0) :
    tag[i] === :bot   ? SVector(0.0, -1.0) :
    -pts[i] / hypot(pts[i][1], pts[i][2])

traction_of(i) = tag[i] === :right ? SVector(σ∞, 0.0) : SVector(0.0, 0.0)

neumann_normals   = [normal_of(i)   for i in neumann_idx]
neumann_tractions = [traction_of(i) for i in neumann_idx]
neumann_adjl      = adjl[neumann_idx]

# ---- assemble + solve (manual path, = what shape_gradient differentiates) --
W_d2x  = _build_weights(Partial(2, 1),      pts, pts, adjl, basis)
W_d2y  = _build_weights(Partial(2, 2),      pts, pts, adjl, basis)
W_d2xy = _build_weights(MixedPartial(1, 2), pts, pts, adjl, basis)
neu_pts = pts[neumann_idx]
W_dx = _build_weights(Partial(1, 1), pts, neu_pts, neumann_adjl, basis)
W_dy = _build_weights(Partial(1, 2), pts, neu_pts, neumann_adjl, basis)

A = assemble_elasticity_from_weights(W_d2x, W_d2y, W_d2xy, N, λstar, μ)
b = zeros(2N)

dirichlet_dofs = vcat(left_idx, left_idx .+ N)
dirichlet_vals = zeros(length(dirichlet_dofs))
apply_dirichlet!(A, b, dirichlet_dofs, dirichlet_vals)

tlayout = build_traction_layout(neumann_idx, neumann_adjl, neumann_normals,
                                neumann_tractions, λstar, μ, N)
apply_traction!(A, b, tlayout, W_dx, W_dy)

@printf("System: 2N=%d DOFs, nnz=%d\n", 2N, nnz(A))
u = lu(A) \ b
ux = u[1:N]; uy = u[(N + 1):2N]
@printf("‖u‖ = %.4e   max|u| = %.4e\n", norm(u), maximum(abs, u))

# ---- stress recovery via RBF first derivatives (plane stress) --------------
Dx = _build_weights(Partial(1, 1), pts, pts, adjl, basis)
Dy = _build_weights(Partial(1, 2), pts, pts, adjl, basis)
εxx = Dx * ux
εyy = Dy * uy
εxy = 0.5 .* (Dy * ux .+ Dx * uy)
c   = Em / (1 - νm^2)
σxx = c .* (εxx .+ νm .* εyy)
σyy = c .* (εyy .+ νm .* εxx)
σxy = (Em / (1 + νm)) .* εxy            # = 2μ εxy
σvm = sqrt.(σxx .^ 2 .- σxx .* σyy .+ σyy .^ 2 .+ 3 .* σxy .^ 2)

# ---- Kirsch verification on the hole ring ----------------------------------
# Hoop stress σθθ at a hole point: σθθ = t̂ᵀσt̂ with t̂ the tangent (−sinθ, cosθ).
hoop(i) = let θ = atan(pts[i][2], pts[i][1]), t1 = -sin(θ), t2 = cos(θ)
    t1^2 * σxx[i] + 2 * t1 * t2 * σxy[i] + t2^2 * σyy[i]
end
ring = sort(hole_idx; by = i -> atan(pts[i][2], pts[i][1]))
hoops = [hoop(i) for i in ring]
imax = argmax(hoops)
θmax_deg = rad2deg(atan(pts[ring[imax]][2], pts[ring[imax]][1]))

# far-field σxx reference: by equilibrium the cross-sectional mean σxx on a
# hole-free section = σ∞. Probe three ways to separate a global recovery bias
# from a free-edge bias: (a) a single clean interior point, (b) strip away from
# the free edges (|y|<0.5), (c) the full strip.
strip      = [i for i in interior_idx if 1.0 < pts[i][1] < 1.3]
strip_core = [i for i in strip if abs(pts[i][2]) < 0.5]
i_probe    = argmin([hypot(pts[i][1] - 1.15, pts[i][2]) for i in interior_idx])
i_probe    = interior_idx[i_probe]
σ_probe    = σxx[i_probe]
σff_core   = sum(σxx[i] for i in strip_core) / length(strip_core)
σff        = sum(σxx[i] for i in strip) / length(strip)
@printf("  [recovery probe]  σxx@(%.2f,%.2f)=%.4f   strip|y|<0.5 mean=%.4f   full-strip mean=%.4f\n",
        pts[i_probe][1], pts[i_probe][2], σ_probe, σff_core, σff)

println()
println(repeat("=", 70))
println("KIRSCH VERIFICATION (uniaxial σ∞ = $σ∞ in x, circular hole R=$R)")
println(repeat("=", 70))
@printf("  far-field σxx (measured)      = %.4f   (applied σ∞ = %.4f)\n", σff, σ∞)
@printf("  peak hoop σθθ on hole         = %.4f  at θ = %.1f°  (Kirsch: 3σ∞ at ±90°)\n",
        hoops[imax], θmax_deg)
@printf("  hoop at crown (θ≈90°)         = %.4f\n",
        hoop(ring[argmin(abs.(rad2deg.(atan.(getindex.(pts[ring], 2), getindex.(pts[ring], 1))) .- 90))]))
@printf("  hoop at side  (θ≈0°)          = %.4f   (Kirsch: −σ∞)\n",
        hoop(ring[argmin(abs.(rad2deg.(atan.(getindex.(pts[ring], 2), getindex.(pts[ring], 1))) .- 0))]))
@printf("  K_t = peak σθθ / σ∞           = %.3f   (Kirsch ∞-plate: 3.0; finite width lowers it)\n",
        hoops[imax] / σ∞)
@printf("  K_t = peak σθθ / far-field    = %.3f   (boundary recovery underestimates)\n", hoops[imax] / σff)
println(repeat("=", 70))

# ---- INTERIOR verification vs Kirsch (no boundary recovery) ----------------
# Along the θ=90° line (the y-axis, x≈0) the hoop direction is x, so σθθ = σxx,
# and Kirsch (uniaxial σ∞ in x) gives σθθ(r) = σ∞(1 + ½(R/r)² + 1½(R/r)⁴).
# These are INTERIOR nodes ⇒ accurate recovery ⇒ a clean correctness test.
println()
println("Interior check — σxx along y-axis (θ=90°) vs exact Kirsch profile:")
@printf("  %6s  %10s  %10s  %8s\n", "r/R", "σxx (FD)", "Kirsch", "rel")
axis_nodes = sort([i for i in interior_idx if abs(pts[i][1]) < 0.6dx && pts[i][2] > R],
                  by = i -> pts[i][2])
axis_nodes = axis_nodes[1:min(8, length(axis_nodes))]
kirsch(r) = σ∞ * (1 + 0.5 * (R / r)^2 + 1.5 * (R / r)^4)
rels = [(σxx[i] - kirsch(pts[i][2])) / kirsch(pts[i][2]) for i in axis_nodes]
for (i, rel) in zip(axis_nodes, rels)
    r = pts[i][2]
    @printf("  %6.2f  %10.4f  %10.4f  %+7.1f%%\n", r / R, σxx[i], kirsch(r), 100rel)
end
@printf("  worst |rel| over the profile = %.1f%%   (small ⇒ the SOLVE is correct;\n",
        100 * maximum(abs, rels))
println("  the K_t deficit above is boundary-stress-recovery, not the physics)")
println(repeat("=", 70))

# ---- figure: von Mises field + hole-ring hoop stress -----------------------
fig = Figure(; size = (1100, 460))
ax1 = Axis(fig[1, 1]; title = "von Mises σ", xlabel = "x", ylabel = "y", aspect = DataAspect())
sc = scatter!(ax1, [p[1] for p in pts], [p[2] for p in pts]; color = σvm,
              colormap = :viridis, markersize = 5)
Colorbar(fig[1, 2], sc)
ax2 = Axis(fig[1, 3]; title = "hoop σθθ around hole vs Kirsch",
           xlabel = "θ (deg)", ylabel = "σθθ / σ∞")
θs = rad2deg.(atan.(getindex.(pts[ring], 2), getindex.(pts[ring], 1)))
perm = sortperm(θs)
lines!(ax2, θs[perm], (hoops ./ σ∞)[perm]; label = "FD-RBF")
lines!(ax2, -180:1:180, [1 - 2cosd(2θ) for θ in -180:1:180]; linestyle = :dash, label = "Kirsch")
axislegend(ax2; position = :rb)
figpath = joinpath(@__DIR__, "plate_with_hole_forward.png")
save(figpath, fig)
println("Figure saved: ", figpath)
