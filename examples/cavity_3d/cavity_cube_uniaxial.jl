# ============================================================================
# cavity_cube_uniaxial.jl — Cube with ellipsoidal cavity, UNIAXIAL tension.
#
# Literature benchmark (Allaire 1992, Holes & Kienzler 2004): cube under
# uniaxial far-field stress σ₀e_x⊗e_x, cavity traction-free.  The optimal
# cavity is an ellipsoid with major axis aligned with the loading direction
# (aspect ratio depends on Poisson's ratio).  Large compliance gap ⇒ steep
# gradient ⇒ clean convergence.
#
# Node generation: Bridson runs to NATURAL SATURATION (no max_points cap).
# The prior truncation ("front truncated by max_points") left unfilled regions
# that degraded stencil quality.  Here we set max_points high enough that
# Bridson saturates before hitting it.
#
# NO STL file, NO remesh.  Mesh built programmatically, cloud generated once,
# morph-only optimization loop.
#
# Run:  jlrun cavity_3d/cavity_cube_uniaxial.jl   (from examples/)
# ============================================================================
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Macchiato
import WhatsThePoint as WTP
const _M = WTP.Meshes
const VTKPointData = WTP.VTKPointData
import RadialBasisFunctions: find_neighbors
using RadialBasisFunctions: PHS
using StaticArrays, LinearAlgebra, Printf
using Statistics: mean, std, median
using Unitful: m, ustrip

# ---- CONFIG -----------------------------------------------------------------
const L_OUT  = 1.0
const Δ      = 0.10
const H_OUT  = Δ
const NSUB   = 2
const DEGREES = [0, 2]
const k      = 50
const σ₀     = 1.0e4                     # uniaxial stress magnitude
const ax, ay, az = 0.35, 0.50, 0.50      # start OBLATE (misaligned with loading)
const MAX_ITER  = 50
const STEP_FRAC = 0.005
const ARMIJO_C1 = 1e-4
const ARMIJO_MAX_BT = 15
const SOB_P     = 2

model = LinearElasticity3D(E = 1.0e7, ν = 0.3)
μ, λ  = lame_parameters_3d(model)
basis = PHS(3; poly_deg = 3)

flat3(P) = reduce(vcat, [[p[1], p[2], p[3]] for p in P])
const ZERO_NJAC3 = NormalJacobian3D(Int[], SMatrix{3,3,Float64,9}[])
svec(p) = (cc = WTP.coords(p); SVector{3,Float64}(ustrip(cc.x), ustrip(cc.y), ustrip(cc.z)))
mp(p) = WTP.Point(p[1] * m, p[2] * m, p[3] * m)

# ---- CloudState -------------------------------------------------------------
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

# ---- outer cube boundary ----------------------------------------------------
function _wind(pts, tri, nv)
    a, b, c = tri
    g = cross(pts[b] - pts[a], pts[c] - pts[a])
    return dot(g, nv) ≥ 0 ? tri : (a, c, b)
end

function outer_cube(L, h)
    n  = round(Int, 2L / h) + 1
    ts = range(-L, L; length = n)
    pts = SVector{3,Float64}[]; nacc = SVector{3,Float64}[]
    seen = Dict{NTuple{3,Int},Int}()
    key(p) = (round(Int, p[1]/h*2), round(Int, p[2]/h*2), round(Int, p[3]/h*2))
    function idx!(p, nv)
        kk = key(p); id = get(seen, kk, 0)
        id != 0 && (nacc[id] += nv; return id)
        push!(pts, p); push!(nacc, nv); seen[kk] = length(pts); return length(pts)
    end
    faces = NTuple{3,Int}[]
    for (axis, s) in ((1,-1.0),(1,1.0),(2,-1.0),(2,1.0),(3,-1.0),(3,1.0))
        nv  = setindex(SVector(0.0,0.0,0.0), s, axis)
        gid = Matrix{Int}(undef, n, n)
        for ia in 1:n, ib in 1:n
            a = ts[ia]; b = ts[ib]
            p = axis == 1 ? SVector(s*L, a, b) :
                axis == 2 ? SVector(a, s*L, b) : SVector(a, b, s*L)
            gid[ia, ib] = idx!(p, nv)
        end
        for ia in 1:(n-1), ib in 1:(n-1)
            v1,v2,v3,v4 = gid[ia,ib], gid[ia+1,ib], gid[ia+1,ib+1], gid[ia,ib+1]
            push!(faces, _wind(pts, (v1,v2,v3), nv))
            push!(faces, _wind(pts, (v1,v3,v4), nv))
        end
    end
    return pts, [normalize(v) for v in nacc], faces
end

# ---- annular mesh (cube outward + cavity inward) ----------------------------
function annular_mesh(out_pts, out_faces, cav_pts, cav_faces)
    no = length(out_pts)
    verts = vcat(out_pts, cav_pts)
    out_tri = collect(out_faces)
    cav_in  = [(no + a, no + c, no + b) for (a, b, c) in cav_faces]
    tris = vcat(out_tri, cav_in)
    return WTP.SimpleMesh([mp(p) for p in verts], [WTP.connect(t, WTP.Triangle) for t in tris])
end

# ---- anchor: build mesh ONCE, cloud ONCE, Bridson to saturation -------------
function anchor(ds::SphericalHarmonicModes)
    cav_pts   = boundary_points(ds)
    cav_faces = surface_faces(ds)
    out_pts, out_nrm, out_faces = outer_cube(L_OUT, H_OUT)

    bnd_pts = vcat(out_pts, cav_pts)
    bnd_nrm = vcat(out_nrm, [-normalize(p) for p in cav_pts])
    bnd = WTP.PointBoundary([mp(p) for p in bnd_pts], bnd_nrm, fill(Δ^2 * m^2, length(bnd_pts)))

    mesh = annular_mesh(out_pts, out_faces, cav_pts, cav_faces)

    # Bridson to NATURAL SATURATION: set max_points high so it never truncates.
    # The warning "front truncated by max_points" means unfilled regions — avoid it.
    spacing = WTP.ConstantSpacing(Δ * m)
    alg = WTP.Octree(mesh; spacing, alpha = 1.0, placement = :bridson)
    V_solid = (2 * L_OUT)^3 - cavity_volume(ds)
    n_max = round(Int, 3 * V_solid / Δ^3)     # 3× headroom — Bridson saturates before this
    cloud = WTP.discretize(bnd, spacing; alg, max_points = n_max)
    vol = svec.(WTP.points(WTP.volume(cloud)))

    nvol = length(vol); n_outer = length(out_pts); nv = length(cav_pts)
    pts = vcat(vol, out_pts, cav_pts)
    N = length(pts)
    interior_idx = collect(1:nvol)
    outer_idx    = collect((nvol+1):(nvol+n_outer))
    cavity_idx   = collect((nvol+n_outer+1):N)
    boundary_idx = vcat(outer_idx, cavity_idx)

    cav_faces_g = [(cavity_idx[a], cavity_idx[b], cavity_idx[c]) for (a,b,c) in cav_faces]
    vf = Dict{Int,Vector{Int}}()
    for (fi,(a,b,c)) in enumerate(cav_faces_g), v in (a,b,c)
        push!(get!(vf, v, Int[]), fi)
    end
    cav_vfaces = [vf[cavity_idx[j]] for j in 1:nv]

    adjl = find_neighbors(pts, k)
    neumann_ids = vcat(outer_idx, cavity_idx)
    neumann_adjl = adjl[neumann_ids]

    # 3-2-1 pins on interior nodes to remove rigid-body modes
    nrst(p0) = interior_idx[argmin([norm(pts[i]-p0) for i in interior_idx])]
    rmid = 0.6 * L_OUT
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
compliance_of(st, ds) = solve_adjoint(morph_cloud(st, ds), st)[1]

# ---- VTU export (per-iteration snapshots for ParaView) ----------------------
function save_vtu(path, pts, st::CloudState; fields...)
    p = reduce(hcat, [[p[1], p[2], p[3]] for p in pts])
    cells = [WTP.createvtkcells(p, false)...]
    vtk = WTP.createvtkfile(path, p, cells)
    # node class: 1=interior, 2=outer, 3=cavity
    cls = fill(1, length(pts))
    for i in st.outer_idx;  cls[i] = 2; end
    for i in st.cavity_idx; cls[i] = 3; end
    vtk["node_class", VTKPointData()] = cls
    for (name, vals) in pairs(fields)
        vtk[String(name), VTKPointData()] = vals isa AbstractVector{<:AbstractVector} ?
            reduce(hcat, vals) : vals
    end
    WTP.savevtk!(vtk)
end

# ---- cloud quality diagnostic -----------------------------------------------
function cloud_quality(pts, st::CloudState)
    # Nearest-neighbor distance for each interior node (any neighbor class)
    nn_dists = [minimum(norm(pts[i] - pts[j]) for j in st.adjl[i] if j != i)
                for i in st.interior_idx]
    nn_med = median(nn_dists)
    nn_min = minimum(nn_dists)
    nn_cv  = std(nn_dists) / nn_med
    n_dup  = count(d -> d < 0.3 * nn_med, nn_dists)
    n_gap  = count(d -> d > 2.5 * nn_med, nn_dists)
    return (nn_min=nn_min, nn_med=nn_med, nn_cv=nn_cv, n_dup=n_dup, n_gap=n_gap)
end

# ---- forward solve + discrete adjoint (UNIAXIAL traction) --------------------
# σ∞ = σ₀ e_x ⊗ e_x  →  traction on face with normal n: t = σ₀·n_x·e_x
# Returns (C, g_all, u_norm) — u_norm for sanity checking solver health.
function solve_adjoint(pts, st::CloudState)
    pf = flat3(pts)
    cav_n, cav_j = triangle_normals(pf, st.cav_faces_g, st.cavity_idx, st.cav_vfaces)
    normals = vcat(st.out_nrm, cav_n)
    njacs   = vcat(fill(ZERO_NJAC3, st.n_outer), cav_j)
    # Uniaxial: t = σ₀·n_x·(1,0,0) = (σ₀·n_x, 0, 0)
    tractions = vcat(
        [SVector(σ₀ * n[1], 0.0, 0.0) for n in st.out_nrm],
        fill(SVector(0.,0.,0.), st.nv),
    )
    layout = build_traction_layout_3d(st.neumann_ids, st.neumann_adjl, normals, tractions, λ, μ, st.N)

    b = zeros(3*st.N)
    for kk in eachindex(layout.rows); b[layout.rows[kk]] = layout.b_vals[kk]; end
    res = shape_gradient_3d(pf, model, st.N, st.adjl, basis, st.active,
                            st.dirichlet_dofs, zeros(length(st.dirichlet_dofs)), _ -> b;
                            interior_rows = st.interior_rows, traction_layout = layout,
                            neumann_ids = st.neumann_ids, neumann_adjl = st.neumann_adjl,
                            normal_jacobians = njacs)
    C = dot(b, res.u)
    u_norm = norm(res.u)
    g_all = [SVector(res.Δpts[3i-2], res.Δpts[3i-1], res.Δpts[3i]) for i in 1:st.N]
    return C, g_all, u_norm
end

design_grad_raw(g_all, st::CloudState, ds::SphericalHarmonicModes) =
    contract_gradient(ds, Macchiato.morph_transpose(st.morph, g_all, st.cavity_idx))

compliance_of(st, ds) = solve_adjoint(morph_cloud(st, ds), st)[1]

# ---- run --------------------------------------------------------------------
const r_ref = (ax*ay*az)^(1/3)
ds0 = fit_ellipsoid_sh(SphericalHarmonicModes(DEGREES, NSUB), ax, ay, az)
st  = anchor(ds0)
ρ   = maximum(norm(st.pts[i]-st.pts[j]) for i in 1:st.N for j in st.adjl[i] if j != i)
ds0 = calibrate_sph(ds0, ρ, r_ref)
const V0 = cavity_volume(ds0)
@printf("cloud: %d int + %d cube + %d cavity = %d · ρ=%.3f (%.1fdx) · sob_l=%.3f\n",
        st.n_int, st.n_outer, st.nv, st.N, ρ, ρ/Δ, ds0.sob_l)
@printf("start ellipsoid (%.2f,%.2f,%.2f) · target r_ref=%.4f · V=%.5e\n",
        ax, ay, az, r_ref, V0)
@printf("loading: UNIAXIAL σ₀=%.1e in x-direction\n", σ₀)

asph(ds) = (r = radii(ds); (maximum(r)-minimum(r)) / (sum(r)/length(r)))
stepped(ds, c) = (d = with_coeffs(ds, c); with_coeffs(d, d.coeffs .* (V0/cavity_volume(d))^(1/3)))

# ---- FD check ---------------------------------------------------------------
const FD_CHECK = false
if FD_CHECK
    println("\nFD check — design_grad_raw vs central FD of compliance (fixed cloud):")
    C0, g_all0, _ = solve_adjoint(morph_cloud(st, ds0), st)
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

# ---- morph-only optimization (Armijo backtracking, no Sobolev weighting) ----
const VTU_DIR = joinpath(@__DIR__, "vtu_uniaxial")
mkpath(VTU_DIR)

ds = ds0
println("\n  it        C        asph(%)     ‖g_proj‖    step     Vol/V0  bt    ‖u‖       nn_cv   dup gap")
println("-"^100)
log = NamedTuple[]
for it in 1:MAX_ITER
    global ds
    pts = morph_cloud(st, ds)
    C, g_all, u_norm = solve_adjoint(pts, st)
    g_c    = design_grad_raw(g_all, st, ds)
    g_proj = project_volume(ds, g_c)
    gnorm  = norm(g_proj)

    # Cloud quality diagnostic
    cq = cloud_quality(pts, st)

    # Save VTU snapshot
    g_vecs = [SVector(g_all[i][1], g_all[i][2], g_all[i][3]) for i in eachindex(g_all)]
    save_vtu(joinpath(VTU_DIR, @sprintf("iter_%03d.vtu", it)), pts, st;
             displacement = [SVector(0.0,0.0,0.0) for _ in pts],  # placeholder
             gradient = g_vecs)

    # Sanity check: detect degenerate solves
    if !isfinite(C) || C ≤ 0 || !isfinite(u_norm)
        @printf("  %3d  DEGENERATE  C=%.3e  ‖u‖=%.3e — stopping\n", it, C, u_norm)
        break
    end
    if cq.n_dup > 0
        @printf("  %3d  WARNING: %d near-duplicates detected (nn_min=%.4f, med=%.4f)\n",
                it, cq.n_dup, cq.nn_min, cq.nn_med)
    end

    gnorm < 1e-8 && (println("  converged (‖g‖≈0)"); push!(log,(it=it,C=C,asph=asph(ds),g=gnorm)); break)

    maxδ = maximum(abs, ds.Ymat * g_proj)
    s0 = STEP_FRAC * r_ref / max(maxδ, 1e-30)
    # Armijo backtracking: directional derivative ≈ ‖g_proj‖² (positive for descent)
    descent = gnorm^2
    accepted = false
    s = s0
    ds_new = ds
    C_new = C
    for bt in 0:ARMIJO_MAX_BT
        ds_try = stepped(ds, ds.coeffs .- s .* g_proj)
        C_try  = compliance_of(st, ds_try)
        if isfinite(C_try) && C_try > 0 && C_try ≤ C - ARMIJO_C1 * s * descent
            ds_new = ds_try; C_new = C_try; accepted = true
            @printf("  %3d  %11.5e  %7.3f   %9.3e  %.2e  %.4f  %d  %.3e  %.4f  %3d %3d\n",
                    it, C, 100*asph(ds), gnorm, s, cavity_volume(ds_new)/V0, bt, u_norm,
                    cq.nn_cv, cq.n_dup, cq.n_gap)
            break
        end
        s *= 0.5
    end
    if !accepted
        @printf("  %3d  %11.5e  %7.3f   %9.3e  %.2e  %.4f  ✗ Armijo failed  ‖u‖=%.3e  nn_cv=%.4f\n",
                it, C, 100*asph(ds), gnorm, s0, cavity_volume(ds)/V0, u_norm, cq.nn_cv)
        break
    end
    ds = ds_new
    push!(log, (it=it, C=C, asph=asph(ds), g=gnorm))
end

@printf("\n--- result ---\n")
@printf("asphericity: %.3f%% → %.3f%%\n", 100*asph(ds0), 100*asph(ds))
@printf("compliance:  %.5e → %.5e\n", log[1].C, log[end].C)
r = radii(ds)
@printf("cavity radii: ax=%.4f  ay=%.4f  az=%.4f  (sphere r=%.4f)\n", r[1], r[2], r[3], r_ref)
@printf("aspect ratio x/y: %.3f  (literature: depends on ν=%.1f)\n", r[1]/r[2], model.ν)
