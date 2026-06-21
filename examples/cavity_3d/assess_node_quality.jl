# ============================================================================
# assess_node_quality.jl — quantify node-generation quality for the twofront
# cavity cloud (deterministic Cartesian-lattice interior + SPHERICAL outer
# icosphere shell + icosphere cavity).
#
# Mirrors diagnose_flat_boundary.jl's stencil-conditioning probe, but on the
# CURRENT twofront generator (not the octree/cube one).  Answers two questions:
#   (1) NODE QUALITY  — separation, near-duplicates, spacing uniformity per
#       class (interior / outer-shell / cavity), stencil radius spread.
#   (2) CONDITIONING  — per-node degree-3 3D Vandermonde σ_min/σ_max (the
#       quantity bunchkaufman! throws on); does the NEW spherical outer shell
#       obey the h≥Δ rule (off-shell interior support ⇒ non-singular)?
#
# Reconstructs the same generators as cavity_sphere_recovery_twofront.jl (kept
# standalone, as diagnose_flat_boundary.jl does, so it never runs the optimizer).
#
# Run:  jlrun cavity_3d/assess_node_quality.jl              (sphere cavity)
#       jlrun cavity_3d/assess_node_quality.jl ellipsoid    (ellipsoid cavity)
# ============================================================================
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Macchiato
import RadialBasisFunctions: find_neighbors
using StaticArrays, LinearAlgebra, Printf
using Statistics: mean, std, median

# ---- CONFIG (mirrors the example) -------------------------------------------
const L_OUT = 1.0
const Δ     = 0.08
const OUTER_NSUB = 3
const NSUB  = 2
const DEGREES = [0, 2]
const k     = 50
const CASE  = (length(ARGS) >= 1 && ARGS[1] == "ellipsoid") ? :ellipsoid : :sphere
const ax, ay, az = CASE === :ellipsoid ? (0.62, 0.48, 0.55) : (0.547, 0.547, 0.547)
const r_ref = (ax*ay*az)^(1/3)

# ---- generators (copied verbatim from the example) --------------------------
function outer_sphere(R, n_sub)
    dirs, faces = icosphere(n_sub)
    return [R * d for d in dirs], copy(dirs), faces
end

function interior_lattice(ds)
    cav_dirs = directions(ds); cav_r = radii(ds)
    margin = 1 + 1.2 * Δ / r_ref
    r_outer = L_OUT - 1.2 * Δ
    rng = (-L_OUT + Δ/2):Δ:(L_OUT - Δ/2)
    pts = SVector{3,Float64}[]
    for x in rng, y in rng, z in rng
        p = SVector(x, y, z); np = norm(p)
        np < 1e-9 && continue
        np > r_outer && continue
        d = p / np
        j = argmax(i -> dot(d, cav_dirs[i]), eachindex(cav_dirs))
        np < margin * cav_r[j] && continue
        push!(pts, p)
    end
    return pts
end

# ---- build the cloud (interior, then outer shell, then cavity — as anchor) ---
ds = fit_ellipsoid_sh(SphericalHarmonicModes(DEGREES, NSUB), ax, ay, az)
cav_pts = boundary_points(ds)
out_pts, out_nrm, _ = outer_sphere(L_OUT, OUTER_NSUB)
vol = interior_lattice(ds)
nvol = length(vol); nout = length(out_pts); ncav = length(cav_pts)
pts = vcat(vol, out_pts, cav_pts); N = length(pts)
cls = vcat(fill(:int, nvol), fill(:out, nout), fill(:cav, ncav))   # per-node class
@printf("CASE=%s  cavity (%.3f,%.3f,%.3f)  r_ref=%.4f\n", CASE, ax, ay, az, r_ref)
@printf("cloud: %d interior + %d outer-shell + %d cavity = %d nodes\n", nvol, nout, ncav, N)

# outer-shell + cavity surface spacing (mean nearest-neighbor within the class)
shell_h = let
    adj = find_neighbors(out_pts, 7)
    mean(minimum(norm(out_pts[i]-out_pts[j]) for j in adj[i] if j != i) for i in 1:nout)
end
cav_h = let
    adj = find_neighbors(cav_pts, 7)
    mean(minimum(norm(cav_pts[i]-cav_pts[j]) for j in adj[i] if j != i) for i in 1:ncav)
end
@printf("spacings: Δ(volume)=%.3f  outer-shell h≈%.3f (%.2f·Δ)  cavity h≈%.3f (%.2f·Δ)\n",
        Δ, shell_h, shell_h/Δ, cav_h, cav_h/Δ)
@printf("rule check: outer-shell h≥Δ ? %s   cavity h≥Δ ? %s   (boundary never finer than volume)\n",
        shell_h ≥ Δ ? "✓" : "✗ VIOLATED", cav_h ≥ Δ ? "✓" : "✗ VIOLATED")

# ---- separation / near-duplicates -------------------------------------------
adjl = find_neighbors(pts, k)
mindist = [minimum(norm(pts[i]-pts[j]) for j in adjl[i] if j != i) for i in 1:N]
@printf("\n-- separation --\n")
@printf("min nearest-neighbor distance: %.3e (%.3f·Δ)   coincident(<1e-9): %d\n",
        minimum(mindist), minimum(mindist)/Δ, count(<(1e-9), mindist))
@printf("near-duplicates  <0.25Δ: %d   <0.5Δ: %d   (octree's pain point was ~6e-3=%.3fΔ)\n",
        count(<(0.25Δ), mindist), count(<(0.5Δ), mindist), 6e-3/Δ)
for c in (:int, :out, :cav)
    idx = findall(==(c), cls)
    d = [minimum(norm(pts[i]-pts[j]) for j in adjl[i] if j != i) for i in idx]
    @printf("  %-4s nn-dist: min=%.3f median=%.3f (·Δ: %.2f / %.2f)\n",
            c, minimum(d), median(d), minimum(d)/Δ, median(d)/Δ)
end

# ---- spacing uniformity (CV) + stencil-radius spread ------------------------
@printf("\n-- uniformity (nn-dist CV per class; lower = more uniform) --\n")
for c in (:int, :out, :cav)
    idx = findall(==(c), cls)
    d = [minimum(norm(pts[i]-pts[j]) for j in adjl[i] if j != i) for i in idx]
    @printf("  %-4s spacing_cv=%.3f\n", c, std(d)/mean(d))
end
ρ = [maximum(norm(pts[i]-pts[j]) for j in adjl[i] if j != i) for i in 1:N]
@printf("stencil radius ρ (k=%d): min=%.3f median=%.3f max=%.3f  (ρ/Δ median=%.2f)\n",
        k, minimum(ρ), median(ρ), maximum(ρ), median(ρ)/Δ)
ρ_cav = maximum(ρ[i] for i in findall(==(:cav), cls))
@printf("ρ at cavity = %.3f  (ρ_cav/r_ref = %.2f; <1 ⇒ cavity well-resolved vs stencil)\n",
        ρ_cav, ρ_cav/r_ref)

# ---- per-stencil degree-3 3D unisolvency (the SingularException source) -----
function vander3(P)
    n = length(P); V = Matrix{Float64}(undef, n, 20)
    for (r,p) in enumerate(P)
        x,y,z = p
        V[r,:] .= (1, x,y,z, x^2,y^2,z^2, x*y,x*z,y*z,
                   x^3,y^3,z^3, x^2*y,x^2*z,y^2*x,y^2*z,z^2*x,z^2*y, x*y*z)
    end
    return V
end
σ = fill(NaN, N)
for i in 1:N
    P = pts[adjl[i]]; c = sum(P)/length(P); h = maximum(norm(p-c) for p in P)
    σ[i] = (s = svd(vander3([(p-c)/h for p in P])).S; s[end]/s[1])
end
@printf("\n-- stencil conditioning σ_min/σ_max (smaller = closer to singular) --\n")
@printf("global: min=%.3e median=%.3e max=%.3e\n", minimum(σ), median(σ), maximum(σ))
for c in (:int, :out, :cav)
    idx = findall(==(c), cls)
    @printf("  %-4s min=%.3e median=%.3e\n", c, minimum(σ[idx]), median(σ[idx]))
end
SING = 1e-8
nsing = count(<(SING), σ)
@printf("stencils below σ=%.0e (would throw SingularException): %d / %d\n", SING, nsing, N)

# ---- worst stencils, with off-class support (the h≥Δ mechanism) -------------
worst = sortperm(σ)
@printf("\n-- 8 worst stencils (off-class support = the 3D-cloud escape) --\n")
@printf("  %6s %5s %10s %9s %12s\n", "node", "class", "σmin/σmax", "#offClass", "minNbrDist")
for i in worst[1:8]
    noff = count(j -> cls[j] != cls[i], adjl[i])
    @printf("  %6d %5s %10.2e %9d %12.3e\n", i, cls[i], σ[i], noff, mindist[i])
end

@printf("\nVERDICT: %s\n", nsing == 0 ?
    "no singular stencils — spherical-outer cloud is well-conditioned for poly_deg=3." :
    "$(nsing) singular stencils — investigate the worst class above.")
