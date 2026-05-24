# ============================================================================
# DIAGNOSTIC (read-only): are the hole normals the reason the optimization
# gradients look small? Replicates optimize.jl's setup on a COARSE/fast cloud.
#
# Checks two independent normal problems:
#   (1) FORWARD: optimize.jl uses radial-about-centroid normals. For a 2:1
#       ellipse the TRUE outward normal is not radial — measure the angle error.
#   (2) GRADIENT: optimize.jl calls shape_gradient with normal_jacobians=nothing
#       (frozen normals). Compare that AD gradient to a full central-difference
#       gradient (which re-solves and so includes the normal-change effect).
#
# Run:  jlrun plate_with_hole/normals_check.jl   (from examples/)
# ============================================================================
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Macchiato
import RadialBasisFunctions: _build_weights, Partial, MixedPartial, find_neighbors
using RadialBasisFunctions: PHS
using StaticArrays, LinearAlgebra, Printf

# ---- coarse setup (same structure as optimize.jl) --------------------------
const Lx = 3.0; const Ly = 3.0; const a0 = 0.40; const b0 = 0.20
const dx = 0.08; const σ∞ = 1.0
model = LinearElasticity(E = 1.0e7, ν = 0.3); μ, λstar = lame_parameters(model)
basis = PHS(3; poly_deg = 3); const k = 35
ellipse_val(p, a, b) = (p[1] / a)^2 + (p[2] / b)^2

base_pts = SVector{2, Float64}[]; base_tag = Symbol[]
let margin = 1 + 1.2 * dx / min(a0, b0)
    for x in (-Lx / 2 + dx):dx:(Lx / 2 - dx / 2), y in (-Ly / 2 + dx):dx:(Ly / 2 - dx / 2)
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
const nθ = max(48, round(Int, 2π * sqrt((a0^2 + b0^2) / 2) / dx))
hole0 = [SVector(a0 * cos(2π * j / nθ), b0 * sin(2π * j / nθ)) for j in 0:(nθ - 1)]
append!(base_pts, hole0); append!(base_tag, fill(:hole, nθ))
const N = length(base_pts); const hole_rng = (n_fixed + 1):N
adjl = find_neighbors(base_pts, k)
idx_of(t) = findall(==(t), base_tag)
const interior_idx = idx_of(:interior)
const outer_idx = vcat(idx_of(:xlo), idx_of(:xhi), idx_of(:yhi), idx_of(:ylo))
const hole_idx = collect(hole_rng); const neumann_idx = vcat(outer_idx, hole_idx)
const neumann_adjl = adjl[neumann_idx]
nearest(p0, pool) = pool[argmin([hypot(base_pts[i][1]-p0[1], base_pts[i][2]-p0[2]) for i in pool])]
const pinA = nearest((1.0, 0.8), interior_idx); const pinB = nearest((-1.0, 0.8), interior_idx)
const dirichlet_dofs = [pinA, pinA + N, pinB + N]; const dirichlet_vals = zeros(3)
const active = let a = trues(2N); a[pinA] = a[pinA+N] = a[pinB+N] = false; a end
const interior_rows = let r = falses(N); for i in interior_idx; r[i] = true; end; r end

make_pts(hole) = vcat(base_pts[1:n_fixed], hole)
hole_centroid(hole) = sum(hole) / length(hole)
hole_normals(hole) = (c = hole_centroid(hole); [(-(p - c)) / hypot((p-c)...) for p in hole])
const hole_loop = collect(hole_rng); const hole_pos = collect(1:length(hole_loop))
const _ZJV = SVector(0.0, 0.0)
zero_njac(g) = NormalJacobian(g, g, _ZJV, _ZJV, _ZJV, _ZJV)
hole_polyline(hole) = polyline_normals(reduce(vcat, [[p[1], p[2]] for p in make_pts(hole)]), hole_loop, hole_pos)
outer_normal(i) = base_tag[i] === :xhi ? SVector(1.0,0.0) : base_tag[i] === :xlo ? SVector(-1.0,0.0) :
                  base_tag[i] === :yhi ? SVector(0.0,1.0) : SVector(0.0,-1.0)
function build_layout(hole)
    normals = SVector{2,Float64}[]; tractions = SVector{2,Float64}[]
    for i in outer_idx; n = outer_normal(i); push!(normals, n); push!(tractions, σ∞ .* n); end
    append!(normals, hole_polyline(hole)[1]); append!(tractions, fill(SVector(0.0,0.0), length(hole)))
    return build_traction_layout(neumann_idx, neumann_adjl, normals, tractions, λstar, μ, N)
end
function forward(hole)
    pts = make_pts(hole)
    W_d2x = _build_weights(Partial(2,1), pts, pts, adjl, basis)
    W_d2y = _build_weights(Partial(2,2), pts, pts, adjl, basis)
    W_d2xy = _build_weights(MixedPartial(1,2), pts, pts, adjl, basis)
    neu = pts[neumann_idx]
    W_dx = _build_weights(Partial(1,1), pts, neu, neumann_adjl, basis)
    W_dy = _build_weights(Partial(1,2), pts, neu, neumann_adjl, basis)
    A = assemble_elasticity_from_weights(W_d2x, W_d2y, W_d2xy, N, λstar, μ)
    b = zeros(2N); apply_dirichlet!(A, b, dirichlet_dofs, dirichlet_vals)
    apply_traction!(A, b, build_layout(hole), W_dx, W_dy)
    return (u = lu(A) \ b, b = b)
end
compliance(hole) = (s = forward(hole); dot(s.b, s.u))
function compliance_grad(hole)
    pts = make_pts(hole); layout = build_layout(hole)
    _, hjacs = hole_polyline(hole)
    njacs = vcat(NormalJacobian[zero_njac(i) for i in outer_idx], hjacs)
    b = zeros(2N); for kk in eachindex(layout.rows); b[layout.rows[kk]] = layout.b_vals[kk]; end
    res = shape_gradient(reduce(vcat, [[p[1], p[2]] for p in pts]), model, N, adjl, basis, active,
                         dirichlet_dofs, dirichlet_vals, _ -> b;
                         interior_rows = interior_rows, traction_layout = layout,
                         neumann_ids = neumann_idx, neumann_adjl = neumann_adjl,
                         traction_jacobians = nothing, normal_jacobians = njacs)
    return [SVector(res.Δpts[2g-1], res.Δpts[2g]) for g in hole_rng]
end

@printf("Coarse cloud N=%d (interior=%d hole=%d)\n", N, length(interior_idx), length(hole_idx))

# ---- CHECK 1: radial-about-centroid vs TRUE ellipse normal -----------------
# True outward normal of (x/a)²+(y/b)²=1 is ∝ (x/a², y/b²). optimize.jl uses
# radial -(p-centroid). Report the angle error (this is a FORWARD-solve error:
# the traction-free condition σ·n=0 is imposed on the wrong n).
println("\nCHECK 1 — hole-normal direction error vs TRUE ellipse normal (unsigned):")
pn, _ = hole_polyline(hole0)
ang_rad = Float64[]; ang_pl = Float64[]
for (j, p) in enumerate(hole0)
    nt = SVector(p[1] / a0^2, p[2] / b0^2); nt = nt / hypot(nt...)   # true ellipse normal
    nr = -p / hypot(p...)                                            # radial-about-centroid
    push!(ang_rad, acosd(clamp(abs(dot(nr, nt)), 0, 1)))
    push!(ang_pl,  acosd(clamp(abs(dot(pn[j], nt)), 0, 1)))
end
@printf("  radial-about-centroid : max %.1f°  mean %.1f°   (the OLD approximation)\n",
        maximum(ang_rad), sum(ang_rad) / length(ang_rad))
@printf("  polyline chord-tangent: max %.1f°  mean %.1f°   (the FIX)\n",
        maximum(ang_pl), sum(ang_pl) / length(ang_pl))

# ---- CHECK 2: frozen-normal AD gradient vs full FD -------------------------
println("\nCHECK 2 — compliance gradient: frozen-normal AD vs full central-FD:")
g_ad = compliance_grad(hole0)
probe = round.(Int, range(1, nθ; length = 8))
hh = 1.0e-6
@printf("  %4s %7s %7s | %11s %11s %11s %11s | %6s\n",
        "node","x","y","AD_x","FD_x","AD_y","FD_y","|AD|/|FD|")
ratios = Float64[]; cosaligns = Float64[]
for j in probe
    fdx = (compliance([i==j ? hole0[i]+SVector(hh,0) : hole0[i] for i in 1:nθ]) -
           compliance([i==j ? hole0[i]-SVector(hh,0) : hole0[i] for i in 1:nθ])) / 2hh
    fdy = (compliance([i==j ? hole0[i]+SVector(0,hh) : hole0[i] for i in 1:nθ]) -
           compliance([i==j ? hole0[i]-SVector(0,hh) : hole0[i] for i in 1:nθ])) / 2hh
    ad = g_ad[j]; fd = SVector(fdx, fdy)
    r = hypot(ad...) / max(hypot(fd...), 1e-30); push!(ratios, r)
    push!(cosaligns, dot(ad, fd) / max(hypot(ad...) * hypot(fd...), 1e-30))
    @printf("  %4d %7.3f %7.3f | %11.3e %11.3e %11.3e %11.3e | %6.3f\n",
            j, hole0[j][1], hole0[j][2], ad[1], fdx, ad[2], fdy, r)
end
@printf("\n  median |AD|/|FD| = %.3f   (≪1 ⇒ frozen-normal AD misses most of the gradient)\n",
        sort(ratios)[cld(length(ratios), 2)])
@printf("  mean cos(angle AD,FD) = %.3f   (<1 ⇒ AD points the wrong way too)\n",
        sum(cosaligns)/length(cosaligns))
