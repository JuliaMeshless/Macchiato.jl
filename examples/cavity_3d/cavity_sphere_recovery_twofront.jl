# ============================================================================
# cavity_sphere_recovery_twofront.jl — 3D TWO-FRONT shape optimization.
#
# Node generation: WTP Octree + Bridson Poisson-disk sampling.  The annular
# cavity mesh (outer sphere − inner cavity) is built as a WTP SimpleMesh and
# loaded through WTP's standard pipeline (PointBoundary → Octree → discretize).
#
# Run:  jlrun cavity_3d/cavity_sphere_recovery_twofront.jl   (from examples/)
# ============================================================================
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Macchiato
import WhatsThePoint as WTP
const _M = WTP.Meshes
import RadialBasisFunctions: find_neighbors
using RadialBasisFunctions: PHS
using StaticArrays, LinearAlgebra, Printf
using Statistics: mean, std
using Unitful: m, ustrip

# ---- CONFIG -----------------------------------------------------------------
const L_OUT  = 1.0
const Δ      = 0.08
const OUTER_NSUB = 3
const NSUB   = 2
const DEGREES = [0, 2]
const k      = 50
const σ∞    = 1.0e4
const ax, ay, az = 0.575, 0.522, 0.545
const MAX_ITER  = 15
const STEP_FRAC = 0.04
const SOB_P     = 2

model = LinearElasticity3D(E = 1.0e7, ν = 0.3)
μ, λ  = lame_parameters_3d(model)
basis = PHS(3; poly_deg = 3)

flat3(P) = reduce(vcat, [[p[1], p[2], p[3]] for p in P])
const ZERO_NJAC3 = NormalJacobian3D(Int[], SMatrix{3,3,Float64,9}[])
svec(p) = (cc = WTP.coords(p); SVector{3,Float64}(ustrip(cc.x), ustrip(cc.y), ustrip(cc.z)))

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
    ref_radius::Vector{Float64}
    ref_interior::Vector{SVector{3,Float64}}
    dx::Float64
end

# ---- sphere mesh builder (matches WTP validate_cavity.jl conventions) -------
function _make_sphere_mesh(R, nθ, nφ)
    pts = [WTP.Point(R * sin(π * i / nθ) * cos(2π * j / nφ) * m,
                     R * sin(π * i / nθ) * sin(2π * j / nφ) * m,
                     R * cos(π * i / nθ) * m)
           for j in 0:(nφ - 1), i in 0:nθ][:]
    conn = WTP.Connectivity{WTP.Triangle}[]
    for i in 0:(nθ - 1), j in 0:(nφ - 1)
        a = i * nφ + j + 1; b = i * nφ + (j + 1) % nφ + 1
        c = (i + 1) * nφ + j + 1; d = (i + 1) * nφ + (j + 1) % nφ + 1
        i > 0    && push!(conn, WTP.connect((a, c, b)))
        i < nθ-1 && push!(conn, WTP.connect((b, c, d)))
    end
    return WTP.SimpleMesh(pts, conn)
end

function _make_annular_cavity_mesh(R_outer, r_inner, nθ, nφ)
    outer = _make_sphere_mesh(R_outer, nθ, nφ)
    inner = _make_sphere_mesh(r_inner, nθ, nφ)
    outer_v = collect(_M.vertices(outer))
    inner_v = collect(_M.vertices(inner))
    n_outer = length(outer_v)
    all_conn = WTP.Connectivity{WTP.Triangle}[]
    for c in _M.topology(outer); push!(all_conn, c); end
    for c in _M.topology(inner)
        p = _M.indices(c)
        push!(all_conn, WTP.connect((p[1] + n_outer, p[3] + n_outer, p[2] + n_outer)))
    end
    return WTP.SimpleMesh(vcat(outer_v, inner_v), all_conn)
end

# ---- anchor: build annular mesh → WTP pipeline → cloud ----------------------
function anchor(ds::SphericalHarmonicModes)
    r_cav = (ax * ay * az)^(1/3)
    nθ = 2 * 2^NSUB; nφ = 2 * nθ

    # 1. Build annular cavity mesh as WTP SimpleMesh
    mesh = _make_annular_cavity_mesh(L_OUT, r_cav, nθ, nφ)

    # 2. Spacing guidance pre-flight — validate Δ against geometry limits
    g = WTP.suggest_spacing(mesh; name = "annular_cavity", bridson_factor = 0.75)
    Δ_input = Δ * m
    if ustrip(Δ_input) >= ustrip(g.h_ceiling)
        @warn "Δ=$(Δ) ≥ h_ceiling=$(round(ustrip(g.h_ceiling); sigdigits=3)) — interior will be empty!"
    elseif ustrip(Δ_input) > ustrip(g.h_baseline)
        @warn "Δ=$(Δ) > h_baseline=$(round(ustrip(g.h_baseline); sigdigits=3)) — coarser than recommended, expect sparse cloud"
    end

    # 3. WTP standard pipeline: PointBoundary → Octree → discretize
    spacing = WTP.ConstantSpacing(Δ * m)
    bnd = WTP.PointBoundary(mesh, spacing)          # Poisson-disk boundary sampling
    alg = WTP.Octree(mesh; spacing, alpha = 1.0, placement = :bridson)
    V_solid = (4 / 3) * π * L_OUT^3 - cavity_volume(ds)
    n_target = round(Int, V_solid / Δ^3)
    cloud = WTP.discretize(bnd, spacing; alg, max_points = n_target)

    # 3. Extract volume points (strip units) and boundary info
    vol = svec.(WTP.points(WTP.volume(cloud)))
    out_pts_bnd = svec.(WTP.points(WTP.boundary(cloud)))
    out_nrm_bnd = WTP.normal(WTP.boundary(cloud))
    out_nrm = [SVector{3,Float64}(ustrip(n[1]), ustrip(n[2]), ustrip(n[3]))
               for n in out_nrm_bnd]

    cav_pts = boundary_points(ds)           # SH icosphere vertices (design surface)
    out_pts = out_pts_bnd                   # already SVectors from svec

    nvol = length(vol); n_outer = length(out_pts); nv = length(cav_pts)
    pts = vcat(vol, out_pts, cav_pts)
    N = length(pts)
    interior_idx = collect(1:nvol)
    outer_idx    = collect((nvol+1):(nvol+n_outer))
    cavity_idx   = collect((nvol+n_outer+1):N)
    boundary_idx = vcat(outer_idx, cavity_idx)

    cav_faces = surface_faces(ds)
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
    ref_radius = [maximum(norm(pts[i]-pts[j]) for j in adjl[i] if j != i) for i in 1:N]
    return CloudState(pts, N, nvol, n_outer, nv, interior_idx, outer_idx, cavity_idx,
                      boundary_idx, out_nrm, cav_faces_g, cav_vfaces,
                      neumann_ids, neumann_adjl, adjl, dirichlet_dofs, rigid_modes, active,
                      interior_rows, ext, ref_radius, vol, Δ)
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

# ---- quality measures -------------------------------------------------------
nn3(pts, adjl, i) = minimum(norm(pts[i] - pts[j]) for j in adjl[i] if j != i)
m_drift(pts, st)  = maximum(norm(pts[st.interior_idx[c]] - st.ref_interior[c]) for c in 1:st.n_int) / st.dx
m_gap(pts, st)    = minimum(norm(pts[i] - pts[j]) for i in st.interior_idx, j in st.cavity_idx) / st.dx
m_spacing_cv(pts, st) = (d = [nn3(pts, st.adjl, i) for i in st.interior_idx]; std(d) / mean(d))
m_min_sep(pts, st)    = minimum(nn3(pts, st.adjl, i) for i in 1:st.N) / st.dx
m_growth(pts, st)     = maximum(maximum(norm(pts[i]-pts[j]) for j in st.adjl[i] if j != i) / st.ref_radius[i]
                                for i in 1:st.N)
m_cavity_cv(pts, st)  = (d = [nn3(pts, st.adjl, i) for i in st.cavity_idx]; std(d) / mean(d))

# ---- run setup --------------------------------------------------------------
const r_ref = (ax*ay*az)^(1/3)
ds0 = fit_ellipsoid_sh(SphericalHarmonicModes(DEGREES, NSUB), ax, ay, az)
st0 = anchor(ds0)
ρ_cav  = maximum(maximum(norm(st0.pts[i]-st0.pts[j]) for j in st0.adjl[i] if j != i)
                 for i in st0.cavity_idx)
ρ_bulk = maximum(norm(st0.pts[i]-st0.pts[j]) for i in 1:st0.N for j in st0.adjl[i] if j != i)
ds0 = calibrate_sph(ds0, ρ_cav, r_ref)
const V0 = cavity_volume(ds0)
@printf("cloud: %d int + %d sphere + %d cavity = %d · ρ_cav=%.3f (%.2f·r_ref) ρ_bulk=%.3f · sob_l=%.3f\n",
        st0.n_int, st0.n_outer, st0.nv, st0.N, ρ_cav, ρ_cav/r_ref, ρ_bulk, ds0.sob_l)
@printf("start ellipsoid (%.2f,%.2f,%.2f) · target sphere r=%.4f · V=%.5e\n",
        ax, ay, az, r_ref, V0)

asph(ds) = (r = radii(ds); (maximum(r)-minimum(r)) / (sum(r)/length(r)))
stepped(ds, c) = (d = with_coeffs(ds, c); with_coeffs(d, d.coeffs .* (V0/cavity_volume(d))^(1/3)))

# ---- quality indicators -----------------------------------------------------
indicators = [
    Indicator(name = :morph_drift,    measure = m_drift,      threshold = 0.60, trip_when = :above, enabled = true),
    Indicator(name = :min_gap,        measure = m_gap,        threshold = 0.30, trip_when = :below, enabled = true),
    Indicator(name = :spacing_cv,     measure = m_spacing_cv, threshold = 0.50, trip_when = :above, enabled = true),
    Indicator(name = :min_sep,        measure = m_min_sep,    threshold = 0.15, trip_when = :below, enabled = true),
    Indicator(name = :stencil_growth, measure = m_growth,     threshold = 1.50, trip_when = :above, enabled = true),
    Indicator(name = :cavity_cv,      measure = m_cavity_cv,  threshold = 0.40, trip_when = :above, enabled = true),
]
println("indicators: ", join(string.(getfield.(filter(i->i.enabled, indicators), :name)), ", "))

# ---- FD check ---------------------------------------------------------------
const FD_CHECK = false
if FD_CHECK
    C0, g_all0 = solve_adjoint(morph_cloud(st0, ds0), st0)
    g_c0 = design_grad_raw(g_all0, st0, ds0)
    h = 1e-6
    @printf("\nFD check · C0=%.6e\n", C0)
    @printf("  %4s %4s %14s %14s %9s\n", "k", "l", "AD", "FD", "AD/FD")
    for kk in eachindex(ds0.coeffs)
        cp = copy(ds0.coeffs); cp[kk] += h
        cm = copy(ds0.coeffs); cm[kk] -= h
        Cp, _ = solve_adjoint(morph_cloud(st0, with_coeffs(ds0, cp)), st0)
        Cm, _ = solve_adjoint(morph_cloud(st0, with_coeffs(ds0, cm)), st0)
        fd = (Cp - Cm) / (2h)
        @printf("  %4d %4d %14.5e %14.5e %9.4f\n", kk, ds0.lm[kk][1], g_c0[kk], fd,
                abs(fd) > 1e-8 ? g_c0[kk]/fd : NaN)
    end
end

# ---- optimizer --------------------------------------------------------------
function optimize(ds, st, indicators)
    log = NamedTuple[]
    @printf("\n%4s %12s %9s %10s  %-30s %s\n", "it", "C", "asph(%)", "‖g‖", "indicators", "event")
    println("-"^92)
    for it in 1:MAX_ITER
        pts = morph_cloud(st, ds)
        vals, tripped = assess(indicators, pts, st)
        event = ""
        if !isempty(tripped)
            st  = anchor(ds)
            pts = morph_cloud(st, ds)
            vals, _ = assess(indicators, pts, st)
            event = "REMESH ⟵ " * join(string.(tripped), ",")
        end

        C, g_all = solve_adjoint(pts, st)
        g_c    = design_grad_raw(g_all, st, ds)
        g_proj = project_volume(ds, g_c)
        g_pc   = [g_proj[i] / sph_sob_weight(ds.lm[i][1], ds.sob_l, SOB_P) for i in eachindex(g_proj)]
        gnorm  = norm(g_proj)
        maxδ   = maximum(abs, ds.Ymat * g_pc)

        vstr = join([@sprintf("%s=%.2f", n, v) for (n, v) in vals], " ")
        @printf("%4d %12.5e %9.3f %10.3e  %-30s %s\n", it, C, 100*asph(ds), gnorm, vstr, event)
        push!(log, (it=it, C=C, asph=asph(ds), g=gnorm, remeshed=!isempty(tripped)))

        (gnorm < 1e-6 || maxδ < 1e-14) && (println("  Converged (‖g‖≈0)."); break)
        s = STEP_FRAC * r_ref / max(maxδ, 1e-30)
        ds = stepped(ds, ds.coeffs .- s .* g_pc)
    end
    return ds, st, log
end

final, st_final, log = optimize(ds0, st0, indicators)

# ---- summary ----------------------------------------------------------------
n_remesh = count(e -> e.remeshed, log)
@printf("\n--- result ---\n")
@printf("asphericity: %.3f%% → %.3f%%   (sphere = 0)\n", 100*log[1].asph, 100*asph(final))
@printf("compliance:  %.5e → %.5e\n", log[1].C, log[end].C)
r = radii(final)
@printf("cavity radius: [%.4f, %.4f]  (sphere r=%.4f)\n", minimum(r), maximum(r), r_ref)
@printf("remeshes triggered: %d / %d iters\n", n_remesh, length(log))
println(asph(final) < 0.05 ? "✓ SPHERE RECOVERED" : "… not yet spherical (see trajectory)")
