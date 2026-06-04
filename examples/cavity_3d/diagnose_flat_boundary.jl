# ============================================================================
# diagnose_flat_boundary.jl — reproduce + localize the cube-face SingularException.
#
# Hypothesis (PROJECT_DIARY 2026-06-03): a flat outer boundary should NOT be
# intrinsically singular; a face node's k-NN should reach OFF-PLANE interior
# support.  The crash is a BUG (octree gap behind faces / dense face grid /
# duplicate nodes), not a law.  This script builds the SAME octree pipeline as
# cavity_sphere_recovery_sh.jl but with a CUBE outer boundary, then for every
# node checks the degree-3 3D Vandermonde rank of its stencil (unisolvency) and
# reports WHY the worst stencils fail.
#
# Run:  jlrun cavity_3d/diagnose_flat_boundary.jl   (from examples/)
# ============================================================================
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using WhatsThePoint; import WhatsThePoint as WTP
import RadialBasisFunctions: find_neighbors
using StaticArrays, LinearAlgebra, Printf
using Unitful: m, ustrip

const L  = 1.0          # cube half-extent: domain [-L,L]^3
const Δ  = 0.10         # volume spacing
const k  = 50           # RBF-FD stencil size (matches example)
const HFACE = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : Δ   # face-grid spacing

mp(p) = WTP.Point(p[1]*m, p[2]*m, p[3]*m)

# ---- cube boundary: structured face grid (deduped) + axis normals -----------
function cube_boundary(L, h)
    n = round(Int, 2L/h) + 1
    ts = range(-L, L; length = n)
    pts = SVector{3,Float64}[]; nrm = SVector{3,Float64}[]
    seen = Dict{NTuple{3,Int},Int}()          # dedupe shared edges/corners
    key(p) = (round(Int, p[1]/h*2), round(Int, p[2]/h*2), round(Int, p[3]/h*2))
    addpt(p, nv) = begin
        kk = key(p)
        if !haskey(seen, kk)
            push!(pts, p); push!(nrm, nv); seen[kk] = length(pts)
        end
    end
    for (axis, s) in ((1,-1.0),(1,1.0),(2,-1.0),(2,1.0),(3,-1.0),(3,1.0))
        nv = setindex(SVector(0.0,0.0,0.0), s, axis)
        for a in ts, b in ts
            p = if axis == 1; SVector(s*L, a, b)
                elseif axis == 2; SVector(a, s*L, b)
                else; SVector(a, b, s*L) end
            addpt(p, nv)
        end
    end
    return pts, nrm
end

# coarse triangulated cube surface (8 corners, 12 outward triangles) for octree
function cube_mesh(L)
    c = [SVector(x,y,z) for z in (-L,L) for y in (-L,L) for x in (-L,L)]
    # index: 1:(-,-,-) 2:(+,-,-) 3:(-,+,-) 4:(+,+,-) 5:(-,-,+) 6:(+,-,+) 7:(-,+,+) 8:(+,+,+)
    F = [
        (1,3,4),(1,4,2),   # z=-L  (outward normal -z)
        (5,6,8),(5,8,7),   # z=+L  (+z)
        (1,2,6),(1,6,5),   # y=-L  (-y)
        (3,7,8),(3,8,4),   # y=+L  (+y)
        (1,5,7),(1,7,3),   # x=-L  (-x)
        (2,4,8),(2,8,6),   # x=+L  (+x)
    ]
    return WTP.SimpleMesh([mp(p) for p in c], [WTP.connect(t, WTP.Triangle) for t in F])
end

# ---- build cloud via the real octree pipeline -------------------------------
bnd_pts, bnd_nrm = cube_boundary(L, HFACE)
mesh = cube_mesh(L)
bnd  = PointBoundary([mp(p) for p in bnd_pts], bnd_nrm, fill(Δ^2*m^2, length(bnd_pts)))
n_target = round(Int, (2L)^3 / Δ^3)
alg = WTP.Octree(mesh; spacing = ConstantSpacing(Δ*m), alpha = 1.0, placement = :jittered)
cloud = WTP.discretize(bnd, ConstantSpacing(Δ*m); alg = alg, max_points = n_target)

svec(p) = (cc = WTP.coords(p); SVector{3,Float64}(ustrip(cc.x), ustrip(cc.y), ustrip(cc.z)))
vol = svec.(WTP.points(cloud.volume))
nvol = length(vol); nb = length(bnd_pts)
pts = vcat(vol, bnd_pts)            # interior first, then boundary (as in anchor())
N = length(pts)
is_bnd = vcat(falses(nvol), trues(nb))
@printf("cube cloud: %d volume + %d boundary = %d nodes (face h=%.3f, Δ=%.3f)\n",
        nvol, nb, N, HFACE, Δ)

# ---- duplicate / near-duplicate check ---------------------------------------
adjl = find_neighbors(pts, k)
mindist = fill(Inf, N)
for i in 1:N, j in adjl[i]
    j == i && continue
    d = norm(pts[i]-pts[j]); mindist[i] = min(mindist[i], d)
end
ndup = count(<(1e-9), mindist)
@printf("nearest-neighbor distance: min=%.3e  (#coincident<1e-9: %d)\n", minimum(mindist), ndup)

# ---- per-stencil degree-3 3D unisolvency (the SingularException source) -----
# monomials of total degree ≤ 3 in (x,y,z): 20 of them.
function vander3(P)                          # P: Vector{SVector3}, returns k×20
    n = length(P)
    V = Matrix{Float64}(undef, n, 20)
    for (r,p) in enumerate(P)
        x,y,z = p
        V[r,:] .= (1, x,y,z, x^2,y^2,z^2, x*y,x*z,y*z,
                   x^3,y^3,z^3, x^2*y,x^2*z,y^2*x,y^2*z,z^2*x,z^2*y, x*y*z)
    end
    return V
end

# local coords (centered + scaled) so σ are geometry, not magnitude, driven
σmin = fill(NaN, N)
for i in 1:N
    nb_i = adjl[i]
    P = pts[nb_i]
    c = sum(P)/length(P)
    h = maximum(norm(p-c) for p in P)
    Ploc = [(p - c)/h for p in P]
    s = svd(vander3(Ploc)).S
    σmin[i] = s[end]/s[1]            # rank gap: ~0 ⇒ unisolvency fails ⇒ singular
end

worst = sortperm(σmin)
@printf("\nstencil unisolvency σ_min/σ_max (smaller = closer to singular):\n")
@printf("  global  min=%.3e  median=%.3e  max=%.3e\n",
        minimum(σmin), sort(σmin)[N÷2], maximum(σmin))
@printf("  boundary-node min=%.3e   interior-node min=%.3e\n",
        minimum(σmin[is_bnd]), minimum(σmin[.!is_bnd]))

# ---- localize the worst stencils --------------------------------------------
function plane_split(i)
    # for a boundary node, classify its neighbors by perpendicular distance to
    # the node's face plane (using the boundary normal); report off-plane support.
    nb_i = adjl[i]
    nrm = is_bnd[i] ? bnd_nrm[i - nvol] : SVector(0.0,0.0,0.0)
    offp = [abs(dot(pts[j]-pts[i], nrm)) for j in nb_i]
    nv_neigh = count(j -> !is_bnd[j], nb_i)          # how many volume neighbors
    nearest_vol = isempty(filter(j->!is_bnd[j], nb_i)) ? Inf :
                  minimum(norm(pts[j]-pts[i]) for j in nb_i if !is_bnd[j])
    return nrm, offp, nv_neigh, nearest_vol
end

println("\n--- 8 worst stencils ---")
@printf("  %6s %4s %10s %8s %9s %12s %12s\n",
        "node","bnd","σmin/σmax","#volNbr","maxOffPl","nearestVol","minNbrDist")
for i in worst[1:8]
    nrm, offp, nvn, nvd = plane_split(i)
    @printf("  %6d %4s %10.2e %8d %9.3f %12.3f %12.3e\n",
            i, is_bnd[i] ? "Y" : "n", σmin[i], nvn, maximum(offp), nvd, mindist[i])
end

# ---- summary verdict --------------------------------------------------------
SING = 1e-8
nsing = count(<(SING), σmin)
@printf("\n%d / %d stencils below σ=%.0e (would throw SingularException)\n", nsing, N, SING)
if nsing > 0
    bsing = count(i -> is_bnd[i] && σmin[i] < SING, 1:N)
    @printf("   of those, %d are boundary nodes, %d interior\n", bsing, nsing-bsing)
    # for boundary singular nodes: how much off-plane support do they have?
    bad = [i for i in 1:N if σmin[i] < SING && is_bnd[i]]
    if !isempty(bad)
        offmax = [maximum(plane_split(i)[2]) for i in bad]
        nvols  = [plane_split(i)[3] for i in bad]
        @printf("   singular boundary nodes: median #volume-neighbors=%d, median max-off-plane=%.3f Δ\n",
                sort(nvols)[length(nvols)÷2+1], sort(offmax)[length(offmax)÷2+1]/Δ)
    end
end
