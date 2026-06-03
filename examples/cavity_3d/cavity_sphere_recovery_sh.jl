# ============================================================================
# cavity_sphere_recovery_sh.jl — 3D shape-optimization SHAKEDOWN.
#
# Benchmark: a large SPHERE with a central ellipsoidal cavity under HYDROSTATIC
# far-field tension (radial traction σ∞·n on the outer sphere, cavity traction-
# free).  The compliance-optimal cavity at fixed volume is a SPHERE (Eshelby/3D-
# Kirsch: uniform boundary stress).  Start from an ellipsoid, recover the sphere.
#
# A spherical (curved) outer boundary is used rather than a cube: flat cube faces
# at fine spacing give coplanar — hence unisolvency-singular — RBF-FD stencils;
# a curved boundary avoids that entirely, and hydrostatic load is just σ∞·n.
#
# This is the GROUND-TRUTH rung of the parametrization-comparison facility: the
# only problem with an exact analytic optimum, used to certify the metrics
# before climbing to flexibility benchmarks (Vigdergauz / fillet / drag).
#
# Pipeline (all validated pieces):
#   - geometry: octree volume fill (cavity = SH icosphere template, preserved)
#   - design  : SphericalHarmonicModes (radial r(θ,φ)=Σ c_lm Y_lm), volume-fixed
#   - forward+adjoint: shape_gradient_3d with LIVE cavity normals
#   - morph   : 3D LaplaceExtension (interior slaved to the moving cavity)
#   - gradient: morph_transpose → contract (Bᵀ) → volume-project → Sobolev-precond
#
# Run:  jlrun cavity_3d/cavity_sphere_recovery_sh.jl   (from examples/)
# ============================================================================
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Macchiato
using WhatsThePoint; import WhatsThePoint as WTP
import RadialBasisFunctions: find_neighbors
using RadialBasisFunctions: PHS
using StaticArrays, LinearAlgebra, Printf
using Unitful: m, ustrip

# ---- CONFIG -----------------------------------------------------------------
# Resolution: the cavity must be well-resolved vs the stencil (ρ ≲ r_ref) or the
# RBF-FD operators near it are inaccurate and the discrete optimum drifts off the
# sphere.  ρ ≈ k^(1/3)·Δ ≈ 3.7Δ, so keep r_ref ≳ several·Δ.
const R_OUT = 1.3                      # outer sphere radius (far-field frame)
const NSUB_OUT = 3                     # outer icosphere subdivision (642 nodes)
const Δ    = 0.10                      # volume spacing
const NSUB = 2                         # cavity icosphere subdivision (162 nodes;
                                       #  spacing ≳ Δ so stencils aren't coplanar-
                                       #  degenerate — denser-than-volume breaks unisolvency)
const DEGREES = [0, 2]                 # SH degrees: base radius + quadrupole only
                                       # (ellipsoid↔sphere subspace; excludes the
                                       #  center-shifting l=1 and noisy l≥3 modes)
const k    = 50                        # RBF-FD stencil
const σ∞   = 1.0e4
const ax, ay, az = 0.48, 0.34, 0.41    # start ellipsoid semi-axes (r_ref≈0.40)
const MAX_ITER  = 30
const STEP_FRAC = 0.04                 # max radial step / r_ref per iter
const SOB_P     = 2
const RUN_FD_CHECK = true

model = LinearElasticity3D(E = 1.0e7, ν = 0.3)
μ, λ  = lame_parameters_3d(model)
basis = PHS(3; poly_deg = 3)

flat3(P) = reduce(vcat, [[p[1], p[2], p[3]] for p in P])
const ZERO_NJAC3 = NormalJacobian3D(Int[], SMatrix{3,3,Float64,9}[])

# ---- outer sphere boundary (icosphere, outward radial normals) --------------
function outer_sphere(R, n_sub)
    dirs, faces = icosphere(n_sub)
    pts = [R * d for d in dirs]
    nrm = copy(dirs)                       # outward radial
    return pts, nrm, faces
end

# ---- octree mesh (outer icosphere outward + cavity icosphere inward) ---------
mp(p) = WTP.Point(p[1]*m, p[2]*m, p[3]*m)
function octree_mesh(out_pts, out_faces, cav_pts, cav_faces)
    no = length(out_pts)
    verts = vcat(out_pts, cav_pts)
    out_tri = collect(out_faces)                                  # already outward
    cav_in  = [(no+a, no+cc, no+b) for (a,b,cc) in cav_faces]     # reverse → inward
    tris = vcat(out_tri, cav_in)
    return WTP.SimpleMesh([mp(p) for p in verts], [WTP.connect(t, WTP.Triangle) for t in tris])
end

# ---- CloudState (one anchored cloud) ----------------------------------------
struct CloudState
    pts::Vector{SVector{3,Float64}}
    N::Int; n_int::Int; n_outer::Int; nv::Int
    interior_idx::Vector{Int}; outer_idx::Vector{Int}; cavity_idx::Vector{Int}
    boundary_idx::Vector{Int}
    out_nrm::Vector{SVector{3,Float64}}
    cav_faces_g::Vector{NTuple{3,Int}}
    cav_vfaces::Vector{Vector{Int}}
    neumann_ids::Vector{Int}; neumann_adjl::Vector{Vector{Int}}
    adjl::Vector{Vector{Int}}
    dirichlet_dofs::Vector{Int}
    active::Vector{Bool}; interior_rows::BitVector
    morph::LaplaceExtension
end

function anchor(ds::SphericalHarmonicModes)
    cav_pts = boundary_points(ds)
    cav_faces = surface_faces(ds)
    out_pts, out_nrm, out_faces = outer_sphere(R_OUT, NSUB_OUT)
    bnd_pts = vcat(out_pts, cav_pts)
    bnd_nrm = vcat(out_nrm, [-normalize(p) for p in cav_pts])
    bnd = PointBoundary([mp(p) for p in bnd_pts], bnd_nrm, fill(Δ^2*m^2, length(bnd_pts)))
    mesh = octree_mesh(out_pts, out_faces, cav_pts, cav_faces)
    V_solid = (4/3)*π*R_OUT^3 - cavity_volume(ds)
    n_target = round(Int, V_solid / Δ^3)
    alg = WTP.Octree(mesh; spacing = ConstantSpacing(Δ*m), alpha = 1.0, placement = :jittered)
    cloud = WTP.discretize(bnd, ConstantSpacing(Δ*m); alg = alg, max_points = n_target)
    svec(p) = (cc=WTP.coords(p); SVector{3,Float64}(ustrip(cc.x),ustrip(cc.y),ustrip(cc.z)))
    vol = svec.(WTP.points(cloud.volume))

    nvol = length(vol); n_outer = length(out_pts); nv = length(cav_pts)
    pts = vcat(vol, out_pts, cav_pts)
    N = length(pts)
    interior_idx = collect(1:nvol)
    outer_idx    = collect((nvol+1):(nvol+n_outer))
    cavity_idx   = collect((nvol+n_outer+1):N)
    boundary_idx = vcat(outer_idx, cavity_idx)

    # cavity faces in GLOBAL indices + per-cavity-vertex incident faces
    cav_faces_g = [(cavity_idx[a], cavity_idx[b], cavity_idx[c]) for (a,b,c) in cav_faces]
    vf = Dict{Int,Vector{Int}}()
    for (fi,(a,b,c)) in enumerate(cav_faces_g), v in (a,b,c)
        push!(get!(vf, v, Int[]), fi)
    end
    cav_vfaces = [vf[cavity_idx[j]] for j in 1:nv]

    adjl = find_neighbors(pts, k)
    neumann_ids = vcat(outer_idx, cavity_idx)
    neumann_adjl = adjl[neumann_ids]

    # pins (3-2-1) on interior nodes to remove rigid-body modes
    nrst(p0) = interior_idx[argmin([norm(pts[i]-p0) for i in interior_idx])]
    rmid = 0.6 * R_OUT
    A = nrst(SVector(rmid,0.,0.)); B = nrst(SVector(-rmid,0.,0.)); C = nrst(SVector(0.,rmid,0.))
    dirichlet_dofs = [A, A+N, A+2N,  B+N, B+2N,  C+2N]
    active = let a = trues(3N); for d in dirichlet_dofs; a[d]=false; end; collect(a) end
    interior_rows = let r=falses(N); for i in interior_idx; r[i]=true; end; r end

    ext = build_laplace_extension(pts, adjl, basis, interior_idx, boundary_idx,
                                  pts[interior_idx], cav_pts, n_outer, nv)
    return CloudState(pts, N, nvol, n_outer, nv, interior_idx, outer_idx, cavity_idx,
                      boundary_idx, out_nrm, cav_faces_g, cav_vfaces,
                      neumann_ids, neumann_adjl, adjl, dirichlet_dofs, active,
                      interior_rows, ext)
end

morph_cloud(st, ds) = morph(st.morph, boundary_points(ds))

# ---- forward solve + discrete adjoint ---------------------------------------
function solve_adjoint(pts, st::CloudState)
    pf = flat3(pts)
    cav_g_ids = st.cavity_idx
    cav_n, cav_j = triangle_normals(pf, st.cav_faces_g, cav_g_ids, st.cav_vfaces)
    normals = vcat(st.out_nrm, cav_n)
    njacs   = vcat(fill(ZERO_NJAC3, st.n_outer), cav_j)
    tractions = vcat([σ∞ .* n for n in st.out_nrm], fill(SVector(0.,0.,0.), st.nv))
    layout = build_traction_layout_3d(st.neumann_ids, st.neumann_adjl, normals, tractions, λ, μ, st.N)

    b = zeros(3*st.N)
    for kk in eachindex(layout.rows); b[layout.rows[kk]] = layout.b_vals[kk]; end
    res = shape_gradient_3d(pf, model, st.N, st.adjl, basis, st.active,
                            st.dirichlet_dofs, zeros(length(st.dirichlet_dofs)), _ -> b;
                            interior_rows = st.interior_rows, traction_layout = layout,
                            neumann_ids = st.neumann_ids, neumann_adjl = st.neumann_adjl,
                            normal_jacobians = njacs)
    C = dot(b, res.u)
    g_all = [SVector(res.Δpts[3i-2], res.Δpts[3i-1], res.Δpts[3i]) for i in 1:st.N]
    return C, g_all
end

# total design gradient dC/dc (raw: before volume-projection / preconditioning)
function design_grad_raw(g_all, st::CloudState, ds::SphericalHarmonicModes)
    ĝ = Macchiato.morph_transpose(st.morph, g_all, st.cavity_idx)   # length nv
    return contract_gradient(ds, ĝ)
end

# ---- run --------------------------------------------------------------------
const r_ref = (ax*ay*az)^(1/3)                      # equal-volume sphere radius
ds0 = fit_ellipsoid_sh(SphericalHarmonicModes(DEGREES, NSUB), ax, ay, az)
st  = anchor(ds0)
ρ   = maximum(norm(st.pts[i]-st.pts[j]) for i in 1:st.N for j in st.adjl[i] if j != i)
ds0 = calibrate_sph(ds0, ρ, r_ref)
V0  = cavity_volume(ds0)
@printf("cloud: %d int + %d cube + %d cavity = %d · ρ=%.3f (%.1fdx) · sob_l=%.3f\n",
        st.n_int, st.n_outer, st.nv, st.N, ρ, ρ/Δ, ds0.sob_l)
@printf("start ellipsoid (%.2f,%.2f,%.2f) · target sphere r=%.4f · V=%.5e\n",
        ax, ay, az, r_ref, V0)

# ---- full-chain FD gradient check -------------------------------------------
if RUN_FD_CHECK
    println("\nFD check — design_grad_raw vs central FD of compliance (fixed cloud):")
    C0, g_all0 = solve_adjoint(morph_cloud(st, ds0), st)
    g_ad = design_grad_raw(g_all0, st, ds0)
    Ccoef(c) = solve_adjoint(morph_cloud(st, with_coeffs(ds0, c)), st)[1]
    hh = 1e-7
    @printf("  %3s %12s %12s %10s\n", "k", "AD", "FD", "AD/FD")
    ratios = Float64[]
    for kidx in 1:min(8, n_design_vars(ds0))
        δ = zeros(n_design_vars(ds0)); δ[kidx] = hh
        fd = (Ccoef(ds0.coeffs + δ) - Ccoef(ds0.coeffs - δ)) / (2hh)
        abs(fd) > 1e-4 && push!(ratios, g_ad[kidx]/fd)
        @printf("  %3d %12.4e %12.4e %10.4f\n", kidx, g_ad[kidx], fd, g_ad[kidx]/max(abs(fd),1e-30)*sign(fd))
    end
    med = isempty(ratios) ? NaN : sort(ratios)[max(1,length(ratios)÷2)]
    @printf("  median AD/FD = %.5f   %s\n", med,
            abs(med-1) < 0.03 ? "✓ gradient validated" : "✗ inspect")
end

# ---- optimization loop ------------------------------------------------------
asph(ds) = (r = radii(ds); (maximum(r)-minimum(r)) / (sum(r)/length(r)))

# Volume-restored design from a coefficient step (exact: V(t·c)=t³V(c)).
function stepped(ds, c_new)
    d = with_coeffs(ds, c_new)
    return with_coeffs(d, d.coeffs .* (V0 / cavity_volume(d))^(1/3))
end
compliance_of(ds) = solve_adjoint(morph_cloud(st, ds), st)[1]

ds = ds0
println("\n  it        C        asph(%)     ‖g_c‖     step     Vol/V0")
println("-"^62)
log = NamedTuple[]
for it in 1:MAX_ITER
    global ds
    pts = morph_cloud(st, ds)
    C, g_all = solve_adjoint(pts, st)
    g_c    = design_grad_raw(g_all, st, ds)
    g_proj = project_volume(ds, g_c)
    g_pc   = [g_proj[i] / sph_sob_weight(ds.lm[i][1], ds.sob_l, SOB_P) for i in eachindex(g_proj)]
    gnorm  = norm(g_proj)
    gnorm < 1e-8 && (println("  converged (‖g‖≈0)"); push!(log,(it=it,C=C,asph=asph(ds),g=gnorm)); break)

    # backtracking line search on the normalized descent step (guarantees descent;
    # rejects folded-morph / invalid solves where C ≤ 0 or NaN).
    maxδ = maximum(abs, ds.Ymat * g_pc)
    s0 = STEP_FRAC * r_ref / max(maxδ, 1e-30)
    accepted = false; s_used = 0.0; ds_try = ds
    for bt in 0:5
        s = s0 / 2^bt
        dt = stepped(ds, ds.coeffs .- s .* g_pc)
        Ct = compliance_of(dt)
        if isfinite(Ct) && Ct > 0 && Ct < C
            ds_try = dt; s_used = s; accepted = true; break
        end
    end
    @printf("  %3d  %11.5e  %7.3f   %9.3e  %.2e  %.4f%s\n",
            it, C, 100*asph(ds), gnorm, s_used, cavity_volume(ds)/V0,
            accepted ? "" : "  (no descent — stop)")
    push!(log, (it=it, C=C, asph=asph(ds), g=gnorm))
    accepted || break
    ds = ds_try
end

@printf("\n--- result ---\n")
@printf("asphericity: %.3f%% → %.3f%%   (sphere = 0)\n", 100*log[1].asph, 100*asph(ds))
@printf("compliance:  %.5e → %.5e\n", log[1].C, log[end].C)
r = radii(ds)
@printf("cavity radius: [%.4f, %.4f]  (sphere r=%.4f)\n", minimum(r), maximum(r), r_ref)
println(asph(ds) < 0.05 ? "✓ SPHERE RECOVERED" : "… not yet spherical (see trajectory)")
