# ============================================================================
# cavity_cube_twofront.jl — Cube outer boundary, ellipsoidal cavity, SH design.
#
# Pipeline: SH coefficients → cavity vertices → programmatic mesh → WTP Octree
# + Bridson → node cloud (generated ONCE) → morph-only optimization loop.
#
# NO STL file is generated.  The mesh is built programmatically and passed
# directly to WTP's Octree(mesh) API.  The node cloud is generated once at
# init; subsequent steps only move cavity vertices (SH coefficients) and
# morph interior nodes (LaplaceExtension).  The cloud is never regenerated.
#
# Run:  jlrun cavity_3d/cavity_cube_twofront.jl   (from examples/)
# ============================================================================
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Macchiato
import WhatsThePoint as WTP
import RadialBasisFunctions: find_neighbors
using RadialBasisFunctions: PHS
using StaticArrays, LinearAlgebra, Printf
using Statistics: mean, std
using Unitful: m, ustrip

# ---- CONFIG -----------------------------------------------------------------
const L_OUT  = 1.0                     # cube half-extent: domain [-L,L]³
const Δ      = 0.10                    # volume spacing
const H_OUT  = Δ                       # outer face-grid spacing (≥ Δ for flat-face conditioning)
const NSUB   = 2                       # cavity icosphere subdivision (162 vertices)
const DEGREES = [0, 2]                 # SH degrees
const k      = 50
const σ∞     = 1.0e4
const ax, ay, az = 0.48, 0.34, 0.41   # start ellipsoid (r_ref≈0.40, significant asphericity)
const MAX_ITER  = 30
const STEP_FRAC = 0.01
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
    rigid_modes::Matrix{Float64}
    active::Vector{Bool}; interior_rows::BitVector
    morph::LaplaceExtension
end

# ---- outer cube boundary (from cavity_sphere_recovery_sh.jl) ----------------
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
    cav_in  = [(no + a, no + c, no + b) for (a, b, c) in cav_faces]  # reversed → inward
    tris = vcat(out_tri, cav_in)
    return WTP.SimpleMesh([mp(p) for p in verts], [WTP.connect(t, WTP.Triangle) for t in tris])
end

# ---- anchor: build mesh ONCE, generate cloud ONCE ---------------------------
function anchor(ds::SphericalHarmonicModes)
    cav_pts   = boundary_points(ds)
    cav_faces = surface_faces(ds)
    out_pts, out_nrm, out_faces = outer_cube(L_OUT, H_OUT)

    # Boundary: programmatic points with normals (no STL)
    bnd_pts = vcat(out_pts, cav_pts)
    bnd_nrm = vcat(out_nrm, [-normalize(p) for p in cav_pts])
    bnd = WTP.PointBoundary([mp(p) for p in bnd_pts], bnd_nrm, fill(Δ^2 * m^2, length(bnd_pts)))

    # Mesh for Octree geometry classification
    mesh = annular_mesh(out_pts, out_faces, cav_pts, cav_faces)

    # WTP pipeline: Octree + Bridson, ONE-SHOT
    spacing = WTP.ConstantSpacing(Δ * m)
    alg = WTP.Octree(mesh; spacing, alpha = 1.0, placement = :bridson)
    V_solid = (2 * L_OUT)^3 - cavity_volume(ds)
    n_target = round(Int, V_solid / Δ^3)
    cloud = WTP.discretize(bnd, spacing; alg, max_points = n_target)
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

    rigid_modes    = rigid_body_modes_3d(pts)
    dirichlet_dofs = Int[]
    active         = trues(3N)
    interior_rows  = let r=falses(N); for i in interior_idx; r[i]=true; end; r end

    ext = build_laplace_extension(pts, adjl, basis, interior_idx, boundary_idx,
                                  pts[interior_idx], cav_pts, n_outer, nv)
    return CloudState(pts, N, nvol, n_outer, nv, interior_idx, outer_idx, cavity_idx,
                      boundary_idx, out_nrm, cav_faces_g, cav_vfaces,
                      neumann_ids, neumann_adjl, adjl, dirichlet_dofs, rigid_modes, active,
                      interior_rows, ext)
end

morph_cloud(st, ds) = morph(st.morph, boundary_points(ds))

# ---- forward solve + discrete adjoint ---------------------------------------
function solve_adjoint(pts, st::CloudState)
    pf = flat3(pts)
    cav_n, cav_j = triangle_normals(pf, st.cav_faces_g, st.cavity_idx, st.cav_vfaces)
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
                            normal_jacobians = njacs, rigid_modes = st.rigid_modes)
    C = dot(b, res.u)
    g_all = [SVector(res.Δpts[3i-2], res.Δpts[3i-1], res.Δpts[3i]) for i in 1:st.N]
    return C, g_all
end

design_grad_raw(g_all, st::CloudState, ds::SphericalHarmonicModes) =
    contract_gradient(ds, Macchiato.morph_transpose(st.morph, g_all, st.cavity_idx))

# ---- run --------------------------------------------------------------------
const r_ref = (ax*ay*az)^(1/3)
ds0 = fit_ellipsoid_sh(SphericalHarmonicModes(DEGREES, NSUB), ax, ay, az)
st  = anchor(ds0)
ρ   = maximum(norm(st.pts[i]-st.pts[j]) for i in 1:st.N for j in st.adjl[i] if j != i)
ds0 = calibrate_sph(ds0, ρ, r_ref)
const V0 = cavity_volume(ds0)
@printf("cloud: %d int + %d cube + %d cavity = %d · ρ=%.3f (%.1fdx) · sob_l=%.3f\n",
        st.n_int, st.n_outer, st.nv, st.N, ρ, ρ/Δ, ds0.sob_l)
@printf("start ellipsoid (%.2f,%.2f,%.2f) · target sphere r=%.4f · V=%.5e\n",
        ax, ay, az, r_ref, V0)

asph(ds) = (r = radii(ds); (maximum(r)-minimum(r)) / (sum(r)/length(r)))
stepped(ds, c) = (d = with_coeffs(ds, c); with_coeffs(d, d.coeffs .* (V0/cavity_volume(d))^(1/3)))

# ---- FD check ---------------------------------------------------------------
const FD_CHECK = false
if FD_CHECK
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

compliance_of(st, ds) = solve_adjoint(morph_cloud(st, ds), st)[1]

# ---- morph-only optimization (no remesh) ------------------------------------
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

    maxδ = maximum(abs, ds.Ymat * g_pc)
    s = STEP_FRAC * r_ref / max(maxδ, 1e-30)
    ds_new = stepped(ds, ds.coeffs .- s .* g_pc)
    C_new = compliance_of(st, ds_new)
    if isfinite(C_new) && C_new > 0
        ds = ds_new
        @printf("  %3d  %11.5e  %7.3f   %9.3e  %.2e  %.4f%s\n",
                it, C, 100*asph(ds), gnorm, s, cavity_volume(ds)/V0,
                C_new >= C ? "  (↑)" : "")
    else
    end
    push!(log, (it=it, C=C, asph=asph(ds), g=gnorm))
end

@printf("\n--- result ---\n")
@printf("asphericity: %.3f%% → %.3f%%   (sphere = 0)\n", 100*asph(ds0), 100*asph(ds))
@printf("compliance:  %.5e → %.5e\n", log[1].C, log[end].C)
r = radii(ds)
@printf("cavity radius: [%.4f, %.4f]  (sphere r=%.4f)\n", minimum(r), maximum(r), r_ref)
println(asph(ds) < 0.05 ? "✓ SPHERE RECOVERED" : "… not yet spherical (see trajectory)")
