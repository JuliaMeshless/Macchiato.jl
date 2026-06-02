# ============================================================================
# probe_remesh_unbiased.jl
#
# THE decisive test for the "discrete adjoint + free remeshing" route:
# is the true circle (a₂=0) an UNBIASED stationary point when we REMESH?
#
# The objective-bias finding (docs/boundary_gradient_noise.md): on a SINGLE fixed
# cloud the discrete-compliance optimum sits at a₂≈±0.07 (sign brackets between
# morph/no-morph), NOT at the true circle.  Remeshing lets us average over clouds.
# Whether that averaging REACHES the circle depends on whether the m=2 design
# gradient at a₂=0 is ZERO-MEAN across clouds (debiases) or a STABLE bias.
#
# The earlier probe_node_ensemble jittered ONLY the interior with a FIXED boundary
# and found a stable bias (−18%).  Here we vary the thing the bias was anchored to:
# BOTH the interior lattice offset AND the boundary-node phase — a full remesh of
# the SAME domain, each cloud high-quality (offset+phase preserve uniformity).
#
#   dC/da₂ = Σⱼ g_rad,ⱼ cos(2θⱼ)   (area-constrained ≡ raw at a₂=0, since aₖ=0)
#   metric: ‖mean vector‖ / mean‖per-cloud‖
#       → 0  : zero-mean ⇒ remesh+descent CONVERGES to the circle
#       → 1  : stable bias ⇒ remeshing alone does NOT reach the circle
#
# Run:  jlrun plate_with_hole/probe_remesh_unbiased.jl   (from examples/)
# ============================================================================
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Macchiato
import RadialBasisFunctions: _build_weights, Partial, MixedPartial, find_neighbors
using RadialBasisFunctions: PHS
using StaticArrays, LinearAlgebra, Printf, Random

const Lx = 4.0; const Ly = 4.0; const a0 = 0.40; const b0 = 0.20
const dx = 0.05; const σ∞ = 1.0
model = LinearElasticity(E = 1.0e7, ν = 0.3)
μ, λstar = lame_parameters(model)
basis = PHS(3; poly_deg = 3); const k = 35
const r_circle = sqrt(a0 * b0)
const nθ = max(48, round(Int, 2π * sqrt((a0^2 + b0^2) / 2) / dx))
const margin = 1 + 1.2 * dx / r_circle

flat(v) = reduce(vcat, [[p[1], p[2]] for p in v])
const _ZJV = SVector(0.0, 0.0)
zero_njac(g) = NormalJacobian(g, g, _ZJV, _ZJV, _ZJV, _ZJV)

# One high-quality cloud of the circular-hole domain.  (ox,oy): interior lattice
# offset; φ: boundary-node phase.  Both preserve node-spacing uniformity, so every
# cloud is equally "good" — only the boundary-vs-interior registration changes.
function build_cloud(ox, oy, φ)
    interior = SVector{2,Float64}[]
    for x in (-Lx/2 + dx + ox):dx:(Lx/2 - dx/2), y in (-Ly/2 + dx + oy):dx:(Ly/2 - dx/2)
        (abs(x) < Lx/2 - dx/2 && abs(y) < Ly/2 - dx/2) || continue
        (x^2 + y^2) > (margin * r_circle)^2 || continue
        push!(interior, SVector(x, y))
    end
    outer = SVector{2,Float64}[]; otag = Symbol[]
    for y in (-Ly/2):dx:(Ly/2)
        push!(outer, SVector(-Lx/2, y)); push!(otag, :xlo)
        push!(outer, SVector(Lx/2, y));  push!(otag, :xhi)
    end
    for x in (-Lx/2 + dx):dx:(Lx/2 - dx)
        push!(outer, SVector(x, Ly/2));  push!(otag, :yhi)
        push!(outer, SVector(x, -Ly/2)); push!(otag, :ylo)
    end
    θ = [φ + 2π*(j-1)/nθ for j in 1:nθ]
    hole = [SVector(r_circle*cos(t), r_circle*sin(t)) for t in θ]
    return (pts = vcat(interior, outer, hole), n_int = length(interior),
            n_outer = length(outer), otag = otag, θ = θ)
end

# Validated discrete adjoint → nodal gradient at the hole nodes.
function adjoint_hole_grad(cloud)
    pts = cloud.pts; n_int = cloud.n_int; n_outer = cloud.n_outer; otag = cloud.otag
    N = length(pts)
    interior_idx = collect(1:n_int)
    outer_idx    = collect((n_int+1):(n_int+n_outer))
    hole_idx     = collect((n_int+n_outer+1):N)
    neumann_idx  = vcat(outer_idx, hole_idx)
    hole_pos     = collect(1:nθ)
    adjl = find_neighbors(pts, k)
    neumann_adjl = adjl[neumann_idx]
    nrst(p0) = interior_idx[argmin([hypot(pts[i][1]-p0[1], pts[i][2]-p0[2]) for i in interior_idx])]
    pin1 = nrst((0.0, 0.8)); pin2 = nrst((0.0,-0.8)); pin3 = nrst((0.8, 0.0))
    dirichlet_dofs = [pin1, pin2, pin3 + N]
    active = let a = trues(2N); a[pin1]=a[pin2]=a[pin3+N]=false; a end
    interior_rows = let r = falses(N); for i in interior_idx; r[i]=true; end; r end
    onormal(i) = otag[i]===:xhi ? SVector(1.0,0.0) : otag[i]===:xlo ? SVector(-1.0,0.0) :
                 otag[i]===:yhi ? SVector(0.0,1.0) : SVector(0.0,-1.0)
    hn, hjacs = polyline_normals(flat(pts), hole_idx, hole_pos)
    njacs = vcat(NormalJacobian[zero_njac(i) for i in outer_idx], hjacs)
    normals = SVector{2,Float64}[]; tractions = SVector{2,Float64}[]
    for i in 1:n_outer; nn = onormal(i); push!(normals, nn); push!(tractions, σ∞ .* nn); end
    append!(normals, hn); append!(tractions, fill(SVector(0.0,0.0), nθ))
    layout = build_traction_layout(neumann_idx, neumann_adjl, normals, tractions, λstar, μ, N)
    b = zeros(2N)
    for kk in eachindex(layout.rows); b[layout.rows[kk]] = layout.b_vals[kk]; end
    res = shape_gradient(flat(pts), model, N, adjl, basis, active,
                         dirichlet_dofs, zeros(3), _ -> b;
                         interior_rows = interior_rows, traction_layout = layout,
                         neumann_ids = neumann_idx, neumann_adjl = neumann_adjl,
                         traction_jacobians = nothing, normal_jacobians = njacs)
    g_hole = [SVector(res.Δpts[2*gi-1], res.Δpts[2*gi]) for gi in hole_idx]
    return dot(b, res.u), g_hole
end

# m=2 design gradient (radial projection) on one cloud.
function m2_gradient(cloud)
    _, g_hole = adjoint_hole_grad(cloud)
    θ = cloud.θ
    g_rad = [g_hole[j][1]*cos(θ[j]) + g_hole[j][2]*sin(θ[j]) for j in 1:nθ]
    da2 = sum(g_rad[j]*cos(2θ[j]) for j in 1:nθ)
    db2 = sum(g_rad[j]*sin(2θ[j]) for j in 1:nθ)
    return da2, db2
end

# ============================================================================
Random.seed!(1)
const K = 12
println("=== Remesh unbiasedness test at the circle (a₂=0) ===")
@printf("r_circle=%.4f  nθ=%d  K=%d clouds (interior offset + boundary phase)\n\n",
        r_circle, nθ, K)

# nominal (unshifted) cloud — connects to the optimizer's a₂*=+0.078 finding
da0, db0 = m2_gradient(build_cloud(0.0, 0.0, 0.0))
@printf("nominal cloud:  dC/da₂ = %+.4e   dC/db₂ = %+.4e   (sign ⇒ discrete optimum side)\n\n", da0, db0)

@printf("%4s %14s %14s %14s\n", "k", "dC/da₂", "dC/db₂", "|m2|")
das = Float64[]; dbs = Float64[]
for kk in 1:K
    ox = dx*rand(); oy = dx*rand(); φ = (2π/nθ)*rand()
    da2, db2 = m2_gradient(build_cloud(ox, oy, φ))
    push!(das, da2); push!(dbs, db2)
    @printf("%4d %+14.4e %+14.4e %14.4e\n", kk, da2, db2, hypot(da2, db2))
end

mda = sum(das)/K; mdb = sum(dbs)/K
mean_vec = hypot(mda, mdb)
mean_amp = sum(hypot.(das, dbs))/K
println("\n" * "-"^60)
@printf("mean dC/da₂      = %+.4e\n", mda)
@printf("mean dC/db₂      = %+.4e\n", mdb)
@printf("‖mean vector‖    = %.4e\n", mean_vec)
@printf("mean ‖per-cloud‖ = %.4e\n", mean_amp)
@printf("ratio ‖mean‖/mean‖·‖ = %.3f\n", mean_vec/max(mean_amp, 1e-30))
println("\nReading:")
println("  ratio → 0 : m=2 gradient at the circle is ZERO-MEAN across clouds ⇒")
println("              remesh + gradient descent CONVERGES to the true circle.")
println("  ratio → 1 : STABLE bias survives remeshing ⇒ need bracket/refinement too.")
