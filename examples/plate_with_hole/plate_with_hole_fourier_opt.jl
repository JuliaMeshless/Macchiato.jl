# ============================================================================
# Plate-with-hole — robust multi-DOF shape optimization
#
#   geometry-from-design  →  exact adjoint gradient  →  area constraint
#   →  Sobolev (Hᵖ) frequency preconditioner  →  reference-based interior morph
#   →  backtracking line search (monotone, positive compliance)
#
# WHY this design (see docs/boundary_gradient_noise.md, §"rising spectrum"):
#   The adjoint gradient is EXACT (FD-validated mode-by-mode to 1e-5). The
#   difficulty is that the discrete-compliance gradient SPECTRUM rises toward
#   the Nyquist mode — a near-Nyquist boundary wiggle has ~40× the gradient of
#   the circle-seeking m=2 mode. Raw/ℓ² steepest descent therefore chases
#   roughness. The cure is steepest descent in a Sobolev metric: precondition
#   each Fourier mode's gradient by 1/(1+(r·m)²)ᵖ, which decays faster than the
#   gradient spectrum rises, so the smooth signal leads. This is mesh-
#   independent (m is physical frequency) and lifts directly to 3D (spherical
#   harmonics: 1/(1+r²ℓ(ℓ+1))ᵖ).
#
# Three structural rules that keep it from tangling (the v1 failure modes):
#   1. Geometry is a PURE function of the design: hole = fourier_hole(coeffs).
#      No incremental-displacement bookkeeping → no design/geometry desync,
#      no angular aliasing.
#   2. The interior is morphed from the FIXED REFERENCE cloud by the TOTAL
#      boundary displacement each iteration (operator precomputed once), so
#      mesh distortion cannot accumulate.
#   3. A backtracking line search rejects any step that raises C, makes C
#      non-positive, or pushes interior nodes too close to the hole.
#
# Run:  jlrun plate_with_hole/plate_with_hole_fourier_opt.jl   (from examples/)
# ============================================================================
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Macchiato
import RadialBasisFunctions: _build_weights, Partial, MixedPartial, find_neighbors
using RadialBasisFunctions: PHS
using StaticArrays
using LinearAlgebra
using Printf
using CairoMakie

# ---- parameters ------------------------------------------------------------
const Lx        = 4.0
const Ly        = 4.0
const a0        = 0.40
const b0        = 0.20
const dx        = 0.05
const σ∞        = 1.0
const n_iter    = 40
# The per-mode probe (probe_fourier_modes.jl) showed the gradient spectrum is
# signal at m=2 (ellipse) and m=4 (finite-square plate) but noise-dominated for
# m≳6 (near-Nyquist).  Giving the design those high channels lets the fixed-step
# descent random-walk them up until the boundary wiggle breaks a fixed stencil
# (C→negative).  A cloud-derived cap closes them by construction.
# m_modes and sob_r are NOT hand-set — they are CALIBRATED from cloud geometry
# (stencil radius ρ) just after the reference cloud is built.  See the
# "parameter-free calibration" block below.  sob_p stays a structural choice.
const sob_p     = 2              # Sobolev order (p=2 ⇒ H², crushes high modes)
const step_frac = 0.03           # target max radial step as a fraction of r₀
const min_gap   = 0.08 * dx      # halt only when interior/hole actually touch
const RUN_FD_CHECK = true        # FD-validate the design gradient (incl. morph)
# MORPH=false keeps the interior FIXED (the ellipse-optimizer regime: fixed adjl
# + fixed interior is stable because boundary motion stays within stencil
# tolerance).  MORPH=true slaves the interior to the boundary via the reference
# Laplacian and back-propagates through its transpose — correct gradient
# (FD-validated), but fixed stencils on a drifting interior eventually
# ill-condition the discrete operator.  See docs/boundary_gradient_noise.md.
const MORPH = true

model   = LinearElasticity(E = 1.0e7, ν = 0.3)
μ, λstar = lame_parameters(model)
basis   = PHS(3; poly_deg = 3)
const k = 35

# ============================================================================
# Fourier radius parameterization
#   r(θ) = r₀ + Σ_{m∈modes} [aₘ cos(mθ) + bₘ sin(mθ)]
#   node j sits at parameter angle θⱼ (FIXED), position r(θⱼ)·(cosθⱼ, sinθⱼ)
# ============================================================================

function fourier_hole(r₀, a, b, θ, modes)
    hole = Vector{SVector{2,Float64}}(undef, length(θ))
    for j in eachindex(θ)
        r = r₀
        for (kk, m) in enumerate(modes)
            r += a[kk]*cos(m*θ[j]) + b[kk]*sin(m*θ[j])
        end
        hole[j] = SVector(r*cos(θ[j]), r*sin(θ[j]))
    end
    return hole
end

# Truncated-series area:  A = π r₀² + (π/2) Σ(aₘ² + bₘ²)
fourier_area(r₀, a, b) = π*r₀^2 + (π/2)*(sum(abs2, a) + sum(abs2, b))

# Hold area fixed by solving for r₀ given the oscillatory part.
function r0_for_area(a, b, A_target)
    osc = (π/2)*(sum(abs2, a) + sum(abs2, b))
    return sqrt(max(0.0, (A_target - osc)/π))
end

# Radial (outward) component of the nodal gradient.
function radial_component(g_hole, hole)
    [(g_hole[j][1]*hole[j][1] + g_hole[j][2]*hole[j][2]) / hypot(hole[j]...)
     for j in eachindex(g_hole)]
end

# Project radial gradient onto the Fourier modes (exact chain rule, since
# ∂xⱼ/∂aₘ = cos(mθⱼ)·r̂ⱼ ⇒ dC/daₘ = Σⱼ g_rad,ⱼ cos(mθⱼ)).
function fourier_gradient(g_rad, θ, modes)
    nθ = length(g_rad)
    dC_dr0 = sum(g_rad)
    dC_da = zeros(length(modes)); dC_db = zeros(length(modes))
    for (kk, m) in enumerate(modes)
        dC_da[kk] = sum(g_rad[j]*cos(m*θ[j]) for j in 1:nθ)
        dC_db[kk] = sum(g_rad[j]*sin(m*θ[j]) for j in 1:nθ)
    end
    return (dC_dr0 = dC_dr0, dC_da = dC_da, dC_db = dC_db)
end

# Project out the area direction: r₀ = r₀(a,b) ⇒ dr₀/daₖ = -aₖ/(2r₀).
# dC/daₖ|_A = dC/daₖ + dC/dr₀ · (-aₖ/(2r₀)).
function area_constrained_gradient(dC_dr0, dC_da, dC_db, a, b, r₀)
    factor = dC_dr0 / (2.0*r₀)
    return (dC_da .- factor .* a, dC_db .- factor .* b)
end

# Sobolev preconditioner weight per mode.
sob_weight(m) = (1.0 + (sob_r*m)^2)^sob_p

# ============================================================================
# Reference cloud (FIXED interior + outer; hole regenerated from design)
# ============================================================================
ellipse_val(p, a, b) = (p[1]/a)^2 + (p[2]/b)^2
const margin = 1 + 1.2*dx/min(a0, b0)
const nθ = max(48, round(Int, 2π*sqrt((a0^2 + b0^2)/2)/dx))

# NOTE: cull the reference interior around the ELLIPSE only.  Culling around a
# larger circle removes the handful of near-hole interior points that dominate
# the sensitivity and FLIPS the sign of dC/da₂ — the discrete gradient is
# FD-exact either way, but the near-boundary point layout is itself part of the
# design sensitivity (see docs/boundary_gradient_noise.md, fragility §).
ref_interior = SVector{2,Float64}[]
let xs = (-Lx/2+dx):dx:(Lx/2-dx/2), ys = (-Ly/2+dx):dx:(Ly/2-dx/2)
    for x in xs, y in ys
        ellipse_val(SVector(x,y), a0, b0) > margin^2 || continue
        push!(ref_interior, SVector(x,y))
    end
end
const n_int = length(ref_interior)

outer_pts = SVector{2,Float64}[]; outer_tag = Symbol[]
for y in (-Ly/2):dx:(Ly/2)
    push!(outer_pts, SVector(-Lx/2, y)); push!(outer_tag, :xlo)
    push!(outer_pts, SVector(Lx/2, y));  push!(outer_tag, :xhi)
end
for x in (-Lx/2+dx):dx:(Lx/2-dx)
    push!(outer_pts, SVector(x, Ly/2));  push!(outer_tag, :yhi)
    push!(outer_pts, SVector(x, -Ly/2)); push!(outer_tag, :ylo)
end
const n_outer = length(outer_pts)

# parameter angles + initial ellipse hole (defines reference boundary)
const θ_vals = [2π*(j-1)/nθ for j in 1:nθ]
hole0 = [SVector(a0*cos(t), b0*sin(t)) for t in θ_vals]

const ref_pts = vcat(ref_interior, outer_pts, hole0)
const N = length(ref_pts)
const interior_idx = collect(1:n_int)
const outer_idx    = collect((n_int+1):(n_int+n_outer))
const hole_idx     = collect((n_int+n_outer+1):N)
const neumann_idx  = vcat(outer_idx, hole_idx)
const hole_pos     = collect(1:nθ)
const boundary_idx = vcat(outer_idx, hole_idx)

nearest(p0, pool) = pool[argmin([hypot(ref_pts[i][1]-p0[1], ref_pts[i][2]-p0[2]) for i in pool])]
const pin_ux1 = nearest((0.0, 0.8), interior_idx)
const pin_ux2 = nearest((0.0,-0.8), interior_idx)
const pin_uy1 = nearest((0.8, 0.0), interior_idx)
const dirichlet_dofs = [pin_ux1, pin_ux2, pin_uy1 + N]
const active = let a = trues(2N); a[pin_ux1]=a[pin_ux2]=a[pin_uy1+N]=false; a end
const interior_rows = let r = falses(N); for i in interior_idx; r[i]=true; end; r end

const _ZJV = SVector(0.0, 0.0)
zero_njac(g) = NormalJacobian(g, g, _ZJV, _ZJV, _ZJV, _ZJV)
flat(v) = reduce(vcat, [[p[1], p[2]] for p in v])
outer_normal(i) = outer_tag[i]===:xhi ? SVector(1.0,0.0) :
                  outer_tag[i]===:xlo ? SVector(-1.0,0.0) :
                  outer_tag[i]===:yhi ? SVector(0.0,1.0) : SVector(0.0,-1.0)

# ============================================================================
# Reference-based interior morph operator (precomputed ONCE).
#   Solve ∇²(Δx)=0 on the REFERENCE cloud, Δx prescribed on the boundary loop,
#   Δx=0 on the outer frame.  Morph map:  Δx_int = -L_int⁻¹ L_int,bnd · Δx_bnd.
#   Applying it to the TOTAL boundary displacement (current − reference) each
#   iteration means distortion never accumulates.
# ============================================================================
const adjl_ref = find_neighbors(ref_pts, k)

# ---- parameter-free calibration from cloud geometry ------------------------
# The ∂W/∂x artifact decorrelates at the STENCIL RADIUS ρ (CLAUDE.md Phase D L3).
# Both design knobs are DERIVED from ρ — no hand-tuning, mesh-independent, lifts
# to 3D unchanged (ρ is a length, m is a physical frequency):
#   • Sobolev length = ρ  ⇒  weight (1+(ρ/r·m)²)^p   (Vertex-Morphing filter rule,
#     Antonau et al. 2024: filter radius = local discretization scale).
#   • mode cap m_cap = ⌊π·r/ρ⌋  ⇒  admit only modes whose half-wavelength spans
#     at least one stencil radius (2 stencils per wavelength = stencil Nyquist).
#     Modes finer than this are in the broadband ∂W/∂x band and are pure noise.
const stencil_radius = maximum(
    hypot(ref_pts[i][1] - ref_pts[j][1], ref_pts[i][2] - ref_pts[j][2])
    for i in 1:N for j in adjl_ref[i])
const r_ref   = sqrt(a0 * b0)                       # area-equivalent circle radius
const m_cap   = max(2, floor(Int, π * r_ref / stencil_radius))
const m_modes = 2:m_cap
const sob_r   = stencil_radius / r_ref
@printf("calibration: ρ=%.4f (%.1f dx)  r_ref=%.4f  ⇒  m_cap=%d  m_modes=%s  sob_r=%.3f\n\n",
        stencil_radius, stencil_radius / dx, r_ref, m_cap, string(collect(m_modes)), sob_r)

const morph = let
    Wxx = _build_weights(Partial(2,1), ref_pts, ref_pts, adjl_ref, basis)
    Wyy = _build_weights(Partial(2,2), ref_pts, ref_pts, adjl_ref, basis)
    Wlap = Wxx + Wyy
    L_int     = Wlap[interior_idx, interior_idx] + 1e-8*I
    L_int_bnd = Wlap[interior_idx, boundary_idx]
    (fact = lu(L_int), L_int_bnd = L_int_bnd)
end

# Given the current hole nodes, return the full point cloud.
# MORPH=false ⇒ interior frozen at the reference (stable regime).
function morph_cloud(hole)
    if !MORPH
        return vcat(ref_interior, outer_pts, hole)
    end
    cur_bnd = vcat(outer_pts, hole)            # outer fixed, hole moves
    ref_bnd = vcat(outer_pts, hole0)
    Δbx = [cur_bnd[j][1] - ref_bnd[j][1] for j in eachindex(cur_bnd)]
    Δby = [cur_bnd[j][2] - ref_bnd[j][2] for j in eachindex(cur_bnd)]
    Δix = morph.fact \ (-morph.L_int_bnd * Δbx)
    Δiy = morph.fact \ (-morph.L_int_bnd * Δby)
    pts = Vector{SVector{2,Float64}}(undef, N)
    @inbounds for (kk, i) in enumerate(interior_idx)
        pts[i] = ref_interior[kk] + SVector(Δix[kk], Δiy[kk])
    end
    @inbounds for (kk, i) in enumerate(outer_idx);  pts[i] = outer_pts[kk]; end
    @inbounds for (kk, i) in enumerate(hole_idx);   pts[i] = hole[kk];      end
    return pts
end

# Geometric guard: closest interior node to any hole node.
function min_interior_hole_gap(pts)
    g = Inf
    @inbounds for i in interior_idx, j in hole_idx
        d = hypot(pts[i][1]-pts[j][1], pts[i][2]-pts[j][2])
        d < g && (g = d)
    end
    return g
end

# ============================================================================
# Forward solve (+ adjoint).  adjl rebuilt on the current morphed cloud.
# ============================================================================
function build_layout(pts, adjl)
    neumann_adjl = adjl[neumann_idx]
    hole = pts[hole_idx]
    hn, hjacs = polyline_normals(flat(pts), hole_idx, hole_pos)
    njacs = vcat(NormalJacobian[zero_njac(i) for i in outer_idx], hjacs)
    normals = SVector{2,Float64}[]; tractions = SVector{2,Float64}[]
    for i in 1:n_outer
        nn = outer_normal(i); push!(normals, nn); push!(tractions, σ∞ .* nn)
    end
    append!(normals, hn); append!(tractions, fill(SVector(0.0,0.0), nθ))
    layout = build_traction_layout(neumann_idx, neumann_adjl, normals, tractions, λstar, μ, N)
    return layout, njacs, neumann_adjl
end

# compliance only (for line search + FD).  FIXED adjacency (adjl_ref): the morph
# preserves topology, so freezing the stencils makes C(design) SMOOTH — without
# this, find_neighbors flips connectivity as the boundary moves and the ~2%
# remeshing jumps swamp the descent signal (the v1 line-search stall).
function solve_C(pts)
    adjl = adjl_ref
    layout, _, neumann_adjl = build_layout(pts, adjl)
    Wx  = _build_weights(Partial(2,1),     pts, pts, adjl, basis)
    Wy  = _build_weights(Partial(2,2),     pts, pts, adjl, basis)
    Wxy = _build_weights(MixedPartial(1,2), pts, pts, adjl, basis)
    neu = pts[neumann_idx]
    Wdx = _build_weights(Partial(1,1), pts, neu, neumann_adjl, basis)
    Wdy = _build_weights(Partial(1,2), pts, neu, neumann_adjl, basis)
    A = assemble_elasticity_from_weights(Wx, Wy, Wxy, N, λstar, μ)
    b = zeros(2N)
    apply_dirichlet!(A, b, dirichlet_dofs, zeros(length(dirichlet_dofs)))
    apply_traction!(A, b, layout, Wdx, Wdy)
    return dot(b, lu(A) \ b)
end

# compliance + FULL nodal gradient (adjoint) at every node.  FIXED adjl_ref.
function solve_C_grad(pts)
    adjl = adjl_ref
    layout, njacs, neumann_adjl = build_layout(pts, adjl)
    b = zeros(2N)
    for kk in eachindex(layout.rows); b[layout.rows[kk]] = layout.b_vals[kk]; end
    res = shape_gradient(flat(pts), model, N, adjl, basis, active,
                         dirichlet_dofs, zeros(3), _ -> b;
                         interior_rows = interior_rows, traction_layout = layout,
                         neumann_ids = neumann_idx, neumann_adjl = neumann_adjl,
                         traction_jacobians = nothing, normal_jacobians = njacs)
    C = dot(b, res.u)
    g_all = [SVector(res.Δpts[2*i-1], res.Δpts[2*i]) for i in 1:N]
    return C, g_all
end

# Correct the boundary (hole) nodal gradient for the interior morph.
#   interior is slaved to the boundary by  Δx_int = M·Δx_bnd,  M = -L_int⁻¹L_int,bnd
#   ⇒ the total design gradient at the boundary is  ĝ_bnd = g_bnd + Mᵀ g_int,
#   Mᵀ g_int = -L_int,bndᵀ (L_intᵀ \ g_int).  Without this term the boundary-only
#   adjoint is wrong by O(1) once the interior moves (the 3D-critical lesson).
function hole_design_gradient(g_all)
    if !MORPH                                  # interior fixed ⇒ no morph term
        return [g_all[i] for i in hole_idx]
    end
    gix = [g_all[i][1] for i in interior_idx]
    giy = [g_all[i][2] for i in interior_idx]
    cbx = -(morph.L_int_bnd' * (morph.fact' \ gix))   # length n_bnd, boundary_idx order
    cby = -(morph.L_int_bnd' * (morph.fact' \ giy))
    ĝ = Vector{SVector{2,Float64}}(undef, nθ)
    for (kk, i) in enumerate(hole_idx)
        bpos = n_outer + kk                            # hole follows outer in boundary_idx
        ĝ[kk] = SVector(g_all[i][1] + cbx[bpos], g_all[i][2] + cby[bpos])
    end
    return ĝ
end

# ============================================================================
# Initialize design from the starting ellipse, then re-fix r₀ for exact area.
# ============================================================================
const A_target = π*a0*b0
const r_circle = sqrt(A_target/π)

a_cur = zeros(length(m_modes)); b_cur = zeros(length(m_modes))
let r_vals = [hypot(hole0[j]...) for j in 1:nθ], r0fit = sum(hypot.(getindex.(hole0,1), getindex.(hole0,2)))/nθ
    for (kk, m) in enumerate(m_modes)
        a_cur[kk] = (2/nθ)*sum((r_vals[j]-r0fit)*cos(m*θ_vals[j]) for j in 1:nθ)
        b_cur[kk] = (2/nθ)*sum((r_vals[j]-r0fit)*sin(m*θ_vals[j]) for j in 1:nθ)
    end
end
r₀_cur = r0_for_area(a_cur, b_cur, A_target)
const a_init = copy(a_cur)

println("=== Robust multi-DOF shape optimization ===\n")
println("Modes: $(collect(m_modes))   nθ=$nθ   N=$N")
println("Sobolev: weight = 1/(1+($sob_r·m)²)^$sob_p   |  step_frac=$step_frac")
@printf("Target circle r=%.4f  (area=%.6f)\n", r_circle, A_target)
@printf("Initial:  r₀=%.4f   a₂=%+.4f\n\n", r₀_cur, a_cur[1])

# ============================================================================
# Optional FD validation of the design gradient (INCLUDING the morph).
# Perturb aₖ with area enforced (r₀ recomputed), regenerate boundary, morph,
# re-solve.  Compare to the area-constrained adjoint projection.
# ============================================================================
if RUN_FD_CHECK
    println("--- FD check of area-constrained design gradient (morph on) ---")
    pts_c = morph_cloud(fourier_hole(r₀_cur, a_cur, b_cur, θ_vals, m_modes))
    _, g_all = solve_C_grad(pts_c)
    hole_c = pts_c[hole_idx]
    ĝ_hole = hole_design_gradient(g_all)
    fg = fourier_gradient(radial_component(ĝ_hole, hole_c), θ_vals, m_modes)
    dCa_A, _ = area_constrained_gradient(fg.dC_dr0, fg.dC_da, fg.dC_db, a_cur, b_cur, r₀_cur)
    εfd = 1e-5
    for (kk, m) in enumerate(m_modes)   # validate every admitted mode
        ap = copy(a_cur); ap[kk] += εfd; r0p = r0_for_area(ap, b_cur, A_target)
        am = copy(a_cur); am[kk] -= εfd; r0m = r0_for_area(am, b_cur, A_target)
        Cp = solve_C(morph_cloud(fourier_hole(r0p, ap, b_cur, θ_vals, m_modes)))
        Cm = solve_C(morph_cloud(fourier_hole(r0m, am, b_cur, θ_vals, m_modes)))
        fd = (Cp - Cm)/(2εfd)
        @printf("  a_%d : adjoint|_A = %+.4e   FD|_A = %+.4e   rel.err = %.2e\n",
                m, dCa_A[kk], fd, abs(dCa_A[kk]-fd)/max(abs(fd),1e-30))
    end
    println()
end

# ============================================================================
# Optimization loop
# ============================================================================
hist_C = Float64[]; hist_r0 = Float64[]; hist_a2 = Float64[]
hist_him = Float64[]; hist_gap = Float64[]

@printf("%4s %12s %8s %8s %10s %10s %8s\n",
        "it", "C", "r₀", "a₂", "‖m≥3‖", "step", "gap")
println("-"^70)

# Fixed-size normalized Sobolev descent — NO C-gated line search.  The validated
# gradient gives a trustworthy descent DIRECTION; gating acceptance on the
# discrete C lets the optimizer walk into near-degenerate pockets of the
# fixed-stencil operator (C collapses to ~1e-9 while the design freezes). The
# ellipse optimizer marches on gradient sign for exactly this reason. We only
# halt on a geometric (gap) violation or non-finite C.
function optimize(a_cur, b_cur, r₀_cur)
    for it in 1:n_iter
        hole_c = fourier_hole(r₀_cur, a_cur, b_cur, θ_vals, m_modes)
        pts_c  = morph_cloud(hole_c)
        gapv   = min_interior_hole_gap(pts_c)
        gapv < min_gap && (println("  Geometric stop (gap=$(round(gapv,digits=4)))."); break)

        C_cur, g_all = solve_C_grad(pts_c)
        (!isfinite(C_cur) || C_cur <= 0) && (println("  Non-physical C — stopping."); break)
        ĝ_hole = hole_design_gradient(g_all)
        fg = fourier_gradient(radial_component(ĝ_hole, hole_c), θ_vals, m_modes)
        dCa_A, dCb_A = area_constrained_gradient(fg.dC_dr0, fg.dC_da, fg.dC_db,
                                                 a_cur, b_cur, r₀_cur)

        # Sobolev-preconditioned descent direction, normalized so the max radial
        # boundary step equals step_frac·r₀.
        da = [-dCa_A[kk]/sob_weight(m) for (kk,m) in enumerate(m_modes)]
        db = [-dCb_A[kk]/sob_weight(m) for (kk,m) in enumerate(m_modes)]
        δr = [sum(da[kk]*cos(m*θ_vals[j]) + db[kk]*sin(m*θ_vals[j])
                  for (kk,m) in enumerate(m_modes)) for j in 1:nθ]
        maxδ = maximum(abs, δr)
        maxδ < 1e-14 && (println("  Converged (gradient ≈ 0)."); break)
        # Fixed normalized step (max radial boundary step = step_frac·r₀).  A
        # gradient-decaying "settling" step was tried and REJECTED: under MORPH it
        # lingers in the drifting-interior regime where the fixed stencils
        # ill-condition, corrupting the gradient (b₂ grows, C non-monotone).  The
        # fixed step marches through that region fast.  See docs/boundary_gradient_noise.md.
        s = step_frac * r₀_cur / maxδ

        him = sqrt(sum(abs2, a_cur[2:end]) + sum(abs2, b_cur[2:end]))  # non-m2 energy
        push!(hist_C, C_cur); push!(hist_r0, r₀_cur); push!(hist_a2, a_cur[1])
        push!(hist_him, him); push!(hist_gap, gapv)
        @printf("%4d %12.6e %8.4f %+8.4f %10.3e %10.3e %8.4f\n",
                it, C_cur, r₀_cur, a_cur[1], him, step_frac*r₀_cur, gapv)

        # take the step; enforce area via r₀
        a_cur = a_cur .+ s .* da
        b_cur = b_cur .+ s .* db
        r₀_cur = r0_for_area(a_cur, b_cur, A_target)
    end
    # final compliance
    C_fin, _ = solve_C_grad(morph_cloud(fourier_hole(r₀_cur, a_cur, b_cur, θ_vals, m_modes)))
    return a_cur, b_cur, r₀_cur, C_fin
end

a_cur, b_cur, r₀_cur, C_cur = optimize(a_cur, b_cur, r₀_cur)

# ============================================================================
# Final report
# ============================================================================
println("\n--- Final design ---")
@printf("r₀:  %.4f → %.4f   (circle: %.4f)\n", r0_for_area(a_init, zeros(length(m_modes)), A_target), r₀_cur, r_circle)
@printf("a₂:  %+.4f → %+.4f   (circle: 0)\n", a_init[1], a_cur[1])
@printf("C :  %.6e → %.6e\n", isempty(hist_C) ? C_cur : C_cur, C_cur)
println("\n  mode   aₘ            bₘ        Sobolev-wt")
for (kk, m) in enumerate(m_modes)
    @printf("   %2d   %+.6e  %+.6e   %.2f\n", m, a_cur[kk], b_cur[kk], sob_weight(m))
end

# ============================================================================
# Plot
# ============================================================================
fig = Figure(size = (1400, 420))

ax1 = Axis(fig[1,1]; title="Hole shapes", aspect=DataAspect(), xlabel="x", ylabel="y")
hx = getindex.(hole0,1); hy = getindex.(hole0,2)
lines!(ax1, vcat(hx,hx[1]), vcat(hy,hy[1]); color=:red, linewidth=2, label="start (ellipse)")
hf = fourier_hole(r₀_cur, a_cur, b_cur, θ_vals, m_modes)
hx = getindex.(hf,1); hy = getindex.(hf,2)
lines!(ax1, vcat(hx,hx[1]), vcat(hy,hy[1]); color=:green, linewidth=2, label="final")
ts = range(0, 2π; length=200)
lines!(ax1, r_circle.*cos.(ts), r_circle.*sin.(ts); color=:blue, linestyle=:dash, linewidth=2, label="circle")
axislegend(ax1; position=:rb)

ax2 = Axis(fig[1,2]; title="Compliance (monotone)", xlabel="iter", ylabel="C")
lines!(ax2, hist_C; linewidth=2)

ax3 = Axis(fig[1,3]; title="a₂ → 0 and high-mode energy", xlabel="iter", ylabel="value")
lines!(ax3, hist_a2; linewidth=2, label="a₂")
lines!(ax3, hist_him; linewidth=2, color=:orange, label="‖m≥6‖")
hlines!(ax3, [0.0]; linestyle=:dash, color=:gray)
axislegend(ax3)

save(joinpath(@__DIR__, "plate_with_hole_fourier_opt.png"), fig)
println("\nSaved: plate_with_hole_fourier_opt.png")
