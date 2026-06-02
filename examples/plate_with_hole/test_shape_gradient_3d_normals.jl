# ============================================================================
# test_shape_gradient_3d_normals.jl — FD-validate the 3D discrete adjoint with
# mixed Dirichlet + traction BCs and *live* (differentiable) normals: the 3D
# Phase-D-L2 analogue of test_polyline_normals.jl + the 2D L2 path.
#
# Same uniaxial-tension block as test_shape_gradient_3d_traction.jl:
#   - face x=-Lx/2 : clamped (Dirichlet u=v=w=0)
#   - face x=+Lx/2 : loaded  (frozen applied traction t = σ∞·(1,0,0))
#   - other 4 faces: traction-free (t = 0)
# but now the *outward normals* in the σ·n = t rows are area-weighted vertex
# normals of a FIXED boundary triangle mesh, recomputed from the (perturbed)
# coordinates each evaluation.  The adjoint must therefore capture ∂n/∂pts via
# extract_normal_sensitivities_3d! (the triangle-normal Jacobian) on top of the
# ∂W/∂pts terms.  The triangle connectivity is fixed (discrete-adjoint
# contract); only vertex coordinates move.
#
# Two checks:
#   A. triangle_normals analytic Jacobian vs central FD (unit test).
#   B. shape_gradient_3d (live normals) vs central FD of the same discrete
#      problem.
#
# Run:  jlrun plate_with_hole/test_shape_gradient_3d_normals.jl   (from examples/)
# ============================================================================
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Macchiato
using WhatsThePoint
import WhatsThePoint as WTP
import RadialBasisFunctions: find_neighbors
using RadialBasisFunctions: PHS
using StaticArrays, LinearAlgebra, Printf
using Unitful: m, ustrip

const Lx, Ly, Lz = 0.6, 0.4, 0.3
const Δ  = 0.1
const k  = 50
const σ∞ = 1.0e4

model = LinearElasticity3D(E = 1.0e7, ν = 0.3)
μ, λ  = lame_parameters_3d(model)
basis = PHS(3; poly_deg = 3)

# ---- WTP cloud: box surface → irregular volume fill -------------------------
function box_surface(hx, hy, hz, Δ)
    pts = WTP.Point[]; nrm = SVector{3,Float64}[]
    seen = Set{NTuple{3,Int}}()
    key(x, y, z) = (round(Int, x/Δ*1e3), round(Int, y/Δ*1e3), round(Int, z/Δ*1e3))
    add(x, y, z, n) = (k0 = key(x,y,z); k0 in seen || (push!(seen, k0);
                       push!(pts, WTP.Point(x*m, y*m, z*m)); push!(nrm, n)))
    rng(h) = range(-h, h; length = max(2, round(Int, 2h/Δ) + 1))
    for x in rng(hx), y in rng(hy)
        add(x, y, +hz, SVector(0.0,0.0,1.0)); add(x, y, -hz, SVector(0.0,0.0,-1.0))
    end
    for x in rng(hx), z in rng(hz)
        add(x, +hy, z, SVector(0.0,1.0,0.0)); add(x, -hy, z, SVector(0.0,-1.0,0.0))
    end
    for y in rng(hy), z in rng(hz)
        add(+hx, y, z, SVector(1.0,0.0,0.0)); add(-hx, y, z, SVector(-1.0,0.0,0.0))
    end
    return PointBoundary(identity.(pts), nrm, fill(Δ^2 * m^2, length(pts))), length(pts)
end

bnd_surf, _ = box_surface(Lx/2, Ly/2, Lz/2, Δ)
cloud = WTP.discretize(bnd_surf, ConstantSpacing(Δ * m); alg = SlakKosec())
svec(p) = (c = WTP.coords(p); SVector{3,Float64}(ustrip(c.x), ustrip(c.y), ustrip(c.z)))
vol_pts = svec.(WTP.points(cloud.volume))
bnd_pts = svec.(WTP.points(cloud.boundary))
pts = vcat(vol_pts, bnd_pts)
N = length(pts)
n_int = length(vol_pts)
interior_idx = collect(1:n_int)
boundary_idx = collect((n_int+1):N)
adjl = find_neighbors(pts, k)
@printf("WTP cloud: %d volume + %d boundary = %d total\n", n_int, length(bnd_pts), N)

# ---- classify boundary nodes by dominant face ------------------------------
const half = (Lx/2, Ly/2, Lz/2)
function face_of(p)
    r = ntuple(d -> abs(p[d]) / half[d], 3)
    ax = argmax(r)
    return ax, sign(p[ax])
end

# ---- FIXED boundary triangle mesh ------------------------------------------
# Each of the 6 box faces is a regular grid (box_surface points are preserved by
# discretize).  Triangulate each face's grid into outward-wound triangles; edge
# and corner vertices are shared across faces (conformal) so their area-weighted
# normals blend the incident faces — exactly the chord-corner blend the 2D
# polyline normals produce.  Connectivity stays fixed under perturbation.
const inplane = Dict(1 => (2, 3), 2 => (3, 1), 3 => (1, 2))  # e_u × e_v = +e_ax

function build_box_triangulation(pts, boundary_idx, half; tol = 1e-6)
    faces = NTuple{3,Int}[]
    vertex_faces = Dict{Int, Vector{Int}}()
    push_face!(t) = (push!(faces, t); fi = length(faces);
                     for v in t; push!(get!(vertex_faces, v, Int[]), fi); end)
    for ax in 1:3, sg in (-1.0, 1.0)
        on_face = [i for i in boundary_idx if isapprox(pts[i][ax], sg*half[ax]; atol = tol)]
        uax, vax = inplane[ax]
        rnd(x) = round(x; digits = 6)
        us = sort(unique(rnd(pts[i][uax]) for i in on_face))
        vs = sort(unique(rnd(pts[i][vax]) for i in on_face))
        uidx = Dict(u => k for (k, u) in enumerate(us))
        vidx = Dict(v => k for (k, v) in enumerate(vs))
        grid = Dict{Tuple{Int,Int}, Int}()
        for i in on_face
            grid[(uidx[rnd(pts[i][uax])], vidx[rnd(pts[i][vax])])] = i
        end
        for iu in 1:length(us)-1, iv in 1:length(vs)-1
            v00 = grid[(iu,   iv  )]; v10 = grid[(iu+1, iv  )]
            v11 = grid[(iu+1, iv+1)]; v01 = grid[(iu,   iv+1)]
            if sg > 0                      # outward = +e_ax
                push_face!((v00, v10, v11)); push_face!((v00, v11, v01))
            else                           # outward = -e_ax (flip winding)
                push_face!((v00, v11, v10)); push_face!((v00, v01, v11))
            end
        end
    end
    return faces, vertex_faces
end

faces, vertex_faces = build_box_triangulation(pts, boundary_idx, half)
@printf("Boundary mesh: %d triangles over %d boundary vertices\n",
        length(faces), length(vertex_faces))

# ---- Neumann classification + reference normals (from the mesh) -------------
flat3(P) = reduce(vcat, [[p[1], p[2], p[3]] for p in P])
pts_flat0 = flat3(pts)

dirichlet_dofs = Int[]; dirichlet_vals = Float64[]
neumann_ids = Int[]
loaded_flag = Bool[]
for i in boundary_idx
    ax, sg = face_of(pts[i])
    if ax == 1 && sg < 0
        for c in 0:2; push!(dirichlet_dofs, i + c*N); push!(dirichlet_vals, 0.0); end
    else
        push!(neumann_ids, i)
        push!(loaded_flag, ax == 1 && sg > 0)
    end
end
neumann_adjl = adjl[neumann_ids]
vertex_faces_neu = [vertex_faces[g] for g in neumann_ids]

# Live normals at the reference cloud (mesh-based, area-weighted).
normals0, _ = triangle_normals(pts_flat0, faces, neumann_ids, vertex_faces_neu)

# Frozen applied traction: σ∞·(1,0,0) on the loaded face, 0 elsewhere — does NOT
# depend on the live normal (so b_vals stay fixed under perturbation ⇒ the only
# normal sensitivity is the LHS σ·n term).
tractions = [loaded_flag[i] ? SVector(σ∞, 0.0, 0.0) : SVector(0.0,0.0,0.0)
             for i in eachindex(neumann_ids)]

active = let a = trues(3N); for d in dirichlet_dofs; a[d] = false; end; a end
interior_rows = let r = falses(N); for i in interior_idx; r[i] = true; end; r end
@printf("BCs: %d Dirichlet dofs · %d Neumann nodes (%d loaded)\n",
        length(dirichlet_dofs), length(neumann_ids), count(loaded_flag))

# Sanity: interior-face normals should come out ≈ axis-aligned outward.
let nfree = findfirst(i -> !loaded_flag[i], eachindex(neumann_ids))
    g = neumann_ids[nfree]; n = normals0[nfree]; p = pts[g]
    @printf("sanity normal @ free node %d  p=(% .2f,% .2f,% .2f)  n=(% .3f,% .3f,% .3f)\n",
            g, p[1], p[2], p[3], n[1], n[2], n[3])
end

# Build the traction layout ONCE (structure fixed); coeffs refreshed each eval.
layout = build_traction_layout_3d(neumann_ids, neumann_adjl, normals0, tractions, λ, μ, N)

# ============================================================================
# Check A — triangle_normals analytic Jacobian vs central FD
# ============================================================================
println("\n=== Check A: triangle_normals Jacobian vs central FD ===")
# Perturb the cloud so normals are off-axis (non-trivial Jacobian entries).
pts_pert = copy(pts_flat0)
for i in boundary_idx
    pts_pert[3i-2] += 0.012 * sin(1.7i)
    pts_pert[3i-1] += 0.011 * cos(1.3i)
    pts_pert[3i]   += 0.010 * sin(0.9i + 1.0)
end
normals_p, jacs_p = triangle_normals(pts_pert, faces, neumann_ids, vertex_faces_neu)

nrm_comp(pf, i_local, c) = triangle_normals(pf, faces, neumann_ids, vertex_faces_neu)[1][i_local][c]
worst_A, worst_loc = let worst = 0.0, loc = "", hA = 1.0e-6
    for i_local in 1:length(neumann_ids)
        nj = jacs_p[i_local]
        for (kk, j) in enumerate(nj.cols)
            B = nj.blocks[kk]              # ∂N_i/∂p_j (rows = comp, cols = coord)
            for cc in 1:3                  # perturb coordinate cc of vertex j
                δ = zeros(3N); δ[3j - 3 + cc] = hA
                for rr in 1:3              # normal component rr
                    fd = (nrm_comp(pts_pert + δ, i_local, rr) -
                          nrm_comp(pts_pert - δ, i_local, rr)) / (2hA)
                    e = abs(B[rr, cc] - fd)
                    if e > worst
                        worst = e
                        loc = "i_local=$i_local (g=$(neumann_ids[i_local])) col=$j comp=$rr coord=$cc"
                    end
                end
            end
        end
    end
    (worst, loc)
end
@printf("  worst |analytic - FD| = %.3e   at %s\n", worst_A, worst_loc)
println(worst_A < 1e-7 ? "  ✓ Check A PASS (triangle-normal Jacobian)" :
                         "  ✗ Check A FAIL")

# ============================================================================
# Check B — shape_gradient_3d with LIVE normals vs central FD
# ============================================================================
println("\n=== Check B: shape_gradient_3d (live normals) vs central FD ===")

# Solve with normals recomputed from the (possibly perturbed) coordinates and
# the layout coeffs refreshed to match (frozen connectivity + frozen b_vals).
function solve_live(pf, dLdu)
    nrm, jacs = triangle_normals(pf, faces, neumann_ids, vertex_faces_neu)
    update_traction_coeffs_3d!(layout, nrm, λ, μ)
    return shape_gradient_3d(pf, model, N, adjl, basis, active,
                             dirichlet_dofs, dirichlet_vals, dLdu;
                             interior_rows = interior_rows,
                             traction_layout = layout,
                             neumann_ids = neumann_ids, neumann_adjl = neumann_adjl,
                             normal_jacobians = jacs)
end
compliance(pf)      = (u = solve_live(pf, identity).u; dot(u, u) / 2)
compliance_grad(pf) = solve_live(pf, u -> u).Δpts

g_ad = compliance_grad(pts_flat0)
@printf("‖u‖ = %.4e   (sanity: nonzero, finite)\n", norm(solve_live(pts_flat0, identity).u))

fd_at(i, c, hh) = (δ = zeros(3N); δ[3i - 3 + c] = hh;
                   (compliance(pts_flat0 + δ) - compliance(pts_flat0 - δ)) / (2hh))

# Probe a mix: interior nodes (no normal dependence) + Neumann nodes + their
# 1-ring boundary neighbours (where ∂n/∂pts is the new term being tested).
neu_probe = neumann_ids[1:min(6, end)]
ring_probe = unique(vcat([jacs.cols for jacs in
             triangle_normals(pts_flat0, faces, neumann_ids, vertex_faces_neu)[2][1:3]]...))
probe = unique(vcat(1:3, (n_int÷2):(n_int÷2+1), neu_probe, ring_probe))
ratios = Float64[]
println("\n  pt   c          AD       FD(1e-6)       FD(3e-5)   AD/FD(3e-5)")
for i in probe, c in 1:3
    fd1 = fd_at(i, c, 1.0e-6)
    fd2 = fd_at(i, c, 3.0e-5)
    ad  = g_ad[3i - 3 + c]
    abs(fd2) > 1e-8 && push!(ratios, abs(ad) / abs(fd2))
    @printf("  %4d  %1d   %+13.6e %+13.6e %+13.6e %10.5f\n",
            i, c, ad, fd1, fd2, abs(ad)/max(abs(fd2),1e-30))
end
med = sort(ratios)[max(1, length(ratios) ÷ 2)]
n_good = count(r -> abs(r - 1.0) < 0.05, ratios)
println("\n--- Summary (ratio vs FD(3e-5)) ---")
@printf("  median |AD|/|FD| = %.6f   (over %d entries, |FD|>1e-8)\n", med, length(ratios))
@printf("  within 5%% of 1.0: %d / %d\n", n_good, length(ratios))
println(abs(med - 1.0) < 0.02 ?
    "  ✓ Check B PASS — shape_gradient_3d VALIDATED (mixed BC, LIVE normals)" :
    "  ✗ Check B FAIL — inspect")

println()
println((worst_A < 1e-7 && abs(med - 1.0) < 0.02) ?
        "ALL 3D DIFFERENTIABLE-NORMAL TESTS PASS" : "TESTS FAILED")
