# ============================================================================
# cavity_sphere_recovery_twofront.jl — 3D TWO-FRONT shape optimization.
#
# The 3D lift of plate_with_hole_twofront_opt.jl.  Same closed-loop control:
#
#   FRONT 1 — MORPH (within a remesh interval):
#       The interior is slaved to the moving cavity by a smooth, DIFFERENTIABLE
#       3D Laplace extension.  The discrete adjoint flows through it exactly
#       (morph-transpose-corrected gradient).  Fixed stencils ⇒ smooth objective.
#
#   FRONT 2 — REMESH + RE-ANCHOR (between intervals):
#       When a quality indicator trips, regenerate a FRESH lattice cloud for the
#       CURRENT design and RE-ANCHOR (reset reference + refactorize the morph).
#       Each remesh is a RE-INITIALIZATION — this removes cloud degradation AND
#       the stale-cloud objective bias that stalled the single-anchor shakedown
#       (cavity_sphere_recovery_sh.jl: C dropped 5× but asph barely moved, ‖g‖
#       blew up 1.4e4→1.5e7 over 4 morph-only iters).
#
#   DESIGN-SPACE CONTRACTION (failure mode A cure, already in place):
#       The design is a few smooth spherical-harmonic radial modes (l=0,2), so
#       the per-node Nyquist noise integrates out in `contract_gradient`.
#
# Descend on the GRADIENT (‖g‖→0), fixed normalized step — NOT a C line search
# (C jumps across remeshes).  Benchmark: cube + ellipsoidal cavity under
# hydrostatic σ∞·n; exact optimum is a SPHERE (Eshelby).
#
# Run:  jlrun cavity_3d/cavity_sphere_recovery_twofront.jl   (from examples/)
# ============================================================================
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Macchiato
import RadialBasisFunctions: find_neighbors
using RadialBasisFunctions: PHS
using StaticArrays, LinearAlgebra, Printf
using Statistics: mean, std

# ---- CONFIG -----------------------------------------------------------------
# Node generation is a SIMPLE deterministic Cartesian lattice culled around the
# cavity (no octree/jitter) — clean uniform stencils (spacing_cv≈0, min_sep≈Δ, no
# near-duplicates), so the absolute quality indicators don't trip every iter and
# FRONT-1 morphing actually engages (the proper two-front regime, as in 2D).
# Determinism removes the remesh jitter noise too, so no common-random-numbers
# hack is needed.  (Graded spacing — BoundaryLayerSpacing — stays available in the
# framework for the harder geometries to come; it's just overkill for this rung.)
const L_OUT  = 1.0                     # outer CUBE half-extent: domain [-L_OUT,L_OUT]³
const Δ      = 0.08                    # uniform lattice spacing ⇒ ρ≈k^⅓·Δ ≈ 0.42
const H_OUT  = 0.10                    # outer face-grid spacing — MUST be ≥ Δ
const NSUB   = 2                       # cavity icosphere subdivision (162 nodes)
const DEGREES = [0, 2]                 # SH degrees: base radius + quadrupole only
const k      = 50                      # RBF-FD stencil — keep ≥50 for poly_deg=3 3D
                                       #  conditioning (do NOT lower to shrink ρ)
const σ∞    = 1.0e4
# LARGER hole: a big cavity is well-resolved vs the stencil (ρ/r_ref<1) on a COARSE
# uniform mesh — the cheap, simple way to escape the under-resolution bias.  r_ref≈0.55
# ⇒ ρ/r_ref≈0.76, ~16k nodes.  (Wall L_OUT−r_ref≈0.45; far-field is tighter than a
# tiny hole but ample for this shakedown.)
const ax, ay, az = 0.62, 0.48, 0.55    # start ellipsoid semi-axes (r_ref≈0.55)
const MAX_ITER  = 25
const STEP_FRAC = 0.04                 # max radial step / r_ref per iter
const SOB_P     = 2

model = LinearElasticity3D(E = 1.0e7, ν = 0.3)
μ, λ  = lame_parameters_3d(model)
basis = PHS(3; poly_deg = 3)

flat3(P) = reduce(vcat, [[p[1], p[2], p[3]] for p in P])
const ZERO_NJAC3 = NormalJacobian3D(Int[], SMatrix{3,3,Float64,9}[])

# ---- outer cube boundary (structured face grid, outward axis normals) -------
_wind(pts, tri, nv) = (a=tri[1]; b=tri[2]; c=tri[3];
    dot(cross(pts[b]-pts[a], pts[c]-pts[a]), nv) ≥ 0 ? tri : (a, c, b))

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

# ---- interior lattice culled around the cavity ------------------------------
# NOTE (tooling direction): this Cartesian lattice is a TEMPORARY simplification for
# this shakedown rung — NOT a rejection of WTP's Octree, which is the SOTA node
# generator (and far better-performing) and the intended path for the harder
# geometries.  The octree just needs development for shape-opt use: (a) avoid the
# near-surface near-duplicate volume points that forced remesh-every-iteration here,
# and (b) lean on its graded spacing (BoundaryLayerSpacing).  Return to Octree once
# those are addressed; the lattice only buys clean uniform stencils cheaply for now.
#
# Structured Cartesian grid over the cube, offset by Δ/2 so it never lands on the
# cube faces; cull any node inside (margin·) the cavity.  The cavity radius in a
# given direction is read off the SH template (nearest of the `nv` unit directions
# — the design is smooth, l=0,2 only, so nearest-direction is plenty for culling).
function interior_lattice(ds)
    cav_dirs = directions(ds); cav_r = radii(ds)
    margin = 1 + 1.2 * Δ / r_ref                       # leave a ≳1.2Δ gap to the cavity
    rng = (-L_OUT + Δ/2):Δ:(L_OUT - Δ/2)
    pts = SVector{3,Float64}[]
    for x in rng, y in rng, z in rng
        p = SVector(x, y, z); np = norm(p)
        np < 1e-9 && continue                          # origin: inside the cavity
        d = p / np
        j = argmax(i -> dot(d, cav_dirs[i]), eachindex(cav_dirs))
        np < margin * cav_r[j] && continue             # inside cavity → cull
        push!(pts, p)
    end
    return pts
end

# ---- CloudState (one anchored cloud) — + quality-indicator reference fields --
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
    ref_radius::Vector{Float64}                  # per-node stencil radius at anchor
    ref_interior::Vector{SVector{3,Float64}}     # interior reference (drift baseline)
    dx::Float64
end

function anchor(ds::SphericalHarmonicModes)
    cav_pts   = boundary_points(ds)
    cav_faces = surface_faces(ds)
    out_pts, out_nrm, _ = outer_cube(L_OUT, H_OUT)
    vol = interior_lattice(ds)

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

    nrst(p0) = interior_idx[argmin([norm(pts[i]-p0) for i in interior_idx])]
    rmid = 0.6 * L_OUT
    A = nrst(SVector(rmid,0.,0.)); B = nrst(SVector(-rmid,0.,0.)); C = nrst(SVector(0.,rmid,0.))
    dirichlet_dofs = [A, A+N, A+2N,  B+N, B+2N,  C+2N]
    active = let a = trues(3N); for d in dirichlet_dofs; a[d]=false; end; collect(a) end
    interior_rows = let r=falses(N); for i in interior_idx; r[i]=true; end; r end

    ext = build_laplace_extension(pts, adjl, basis, interior_idx, boundary_idx,
                                  pts[interior_idx], cav_pts, n_outer, nv)
    ref_radius = [maximum(norm(pts[i]-pts[j]) for j in adjl[i] if j != i) for i in 1:N]
    return CloudState(pts, N, nvol, n_outer, nv, interior_idx, outer_idx, cavity_idx,
                      boundary_idx, out_nrm, cav_faces_g, cav_vfaces,
                      neumann_ids, neumann_adjl, adjl, dirichlet_dofs, active,
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
                            normal_jacobians = njacs)
    C = dot(b, res.u)
    g_all = [SVector(res.Δpts[3i-2], res.Δpts[3i-1], res.Δpts[3i]) for i in 1:st.N]
    return C, g_all
end

design_grad_raw(g_all, st::CloudState, ds::SphericalHarmonicModes) =
    contract_gradient(ds, Macchiato.morph_transpose(st.morph, g_all, st.cavity_idx))

# ---- 3D quality measures (problem-specific; same (pts,st)->Float64 contract) -
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
# ρ AT THE CAVITY (where the design acts and the bias lives) — the global ρ is
# bulk-dominated and irrelevant.  Use ρ_cav to calibrate the Sobolev length.
ρ_cav  = maximum(maximum(norm(st0.pts[i]-st0.pts[j]) for j in st0.adjl[i] if j != i)
                 for i in st0.cavity_idx)
ρ_bulk = maximum(norm(st0.pts[i]-st0.pts[j]) for i in 1:st0.N for j in st0.adjl[i] if j != i)
ds0 = calibrate_sph(ds0, ρ_cav, r_ref)
const V0 = cavity_volume(ds0)
@printf("cloud: %d int + %d cube + %d cavity = %d · ρ_cav=%.3f (%.2f·r_ref) ρ_bulk=%.3f · sob_l=%.3f\n",
        st0.n_int, st0.n_outer, st0.nv, st0.N, ρ_cav, ρ_cav/r_ref, ρ_bulk, ds0.sob_l)
@printf("start ellipsoid (%.2f,%.2f,%.2f) · target sphere r=%.4f · V=%.5e\n",
        ax, ay, az, r_ref, V0)

asph(ds) = (r = radii(ds); (maximum(r)-minimum(r)) / (sum(r)/length(r)))
# Volume-restored design from a coefficient step (exact: V(t·c)=t³V(c)).
stepped(ds, c) = (d = with_coeffs(ds, c); with_coeffs(d, d.coeffs .* (V0/cavity_volume(d))^(1/3)))

# ---- toggleable quality-indicator registry (watch the log, prune) -----------
indicators = [
    Indicator(name = :morph_drift,    measure = m_drift,      threshold = 0.60, trip_when = :above, enabled = true),
    Indicator(name = :min_gap,        measure = m_gap,        threshold = 0.30, trip_when = :below, enabled = true),
    Indicator(name = :spacing_cv,     measure = m_spacing_cv, threshold = 0.35, trip_when = :above, enabled = true),
    Indicator(name = :min_sep,        measure = m_min_sep,    threshold = 0.20, trip_when = :below, enabled = true),
    Indicator(name = :stencil_growth, measure = m_growth,     threshold = 1.40, trip_when = :above, enabled = true),
    Indicator(name = :cavity_cv,      measure = m_cavity_cv,  threshold = 0.40, trip_when = :above, enabled = true),
]
println("indicators: ", join(string.(getfield.(filter(i->i.enabled, indicators), :name)), ", "))

# ---- the closed-loop two-front optimizer ------------------------------------
function optimize(ds, st, indicators)
    log = NamedTuple[]
    @printf("\n%4s %12s %9s %10s  %-30s %s\n", "it", "C", "asph(%)", "‖g‖", "indicators", "event")
    println("-"^92)
    for it in 1:MAX_ITER
        pts = morph_cloud(st, ds)                              # FRONT 1
        vals, tripped = assess(indicators, pts, st)
        event = ""
        if !isempty(tripped)                                   # FRONT 2: re-anchor
            st  = anchor(ds)
            pts = morph_cloud(st, ds)                          # fresh cloud, zero morph
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
        s = STEP_FRAC * r_ref / max(maxδ, 1e-30)               # fixed normalized step
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
