# ============================================================================
# Plate-with-hole — Stage 1 optimization: compliance minimization, equibiaxial
# ============================================================================
# Elliptical hole (2:1) in a square plate under equibiaxial tension. The hole
# boundary is the design surface. Expected: ellipse → circle (min-compliance shape
# at fixed area under equibiaxial load). Status: WORKING but PARTIAL (ellipse rounds
# toward the circle, compliance ↓ — directional only); see plan_plate_with_hole.md.
#
# Interior re-spacing: each design step the interior nodes are relaxed to spacing
# equilibrium with WhatsThePoint's SpacingEquilibriumForce (the hole + outer nodes
# are fixed repulsion sources), so a growing hole pushes the interior shell outward —
# no engulfing, no stale-stencil collapse — letting the hole reach the circle (b≈0.283).
# We use the boundary-only shape gradient (no differentiable morph); the dropped
# interior pullback ‖Δpts_interior‖ is reported each iter and stays small when the
# spacing is clean (it is a discrete stencil artifact, →0 with mesh quality).
#
# Solver validated via manufactured-solution Kirsch test (plate_with_hole_kirsch.jl):
# global L2 stress error 0.62%, K_t = 2.966 vs exact 3.0.
#
# Hole normals: TRUE polyline normals (`hole_polyline` → polyline_normals, Phase D
# L2) in BOTH the forward (build_layout) and the gradient (normal_jacobians). The
# old radial-about-centroid normals were 37° wrong on the ellipse — see normals_check.jl.
#
# Mode: adjoint-only (RUN_FD_CHECK=false; normalized step, no line-search solves) ⇒
# one shape_gradient call per iteration. Set RUN_FD_CHECK=true to re-validate the
# gradient against FD (slow: 24 solves). dx=0.05 is a fast first-look; use 0.03 for fine.
#
# Run:  jlrun plate_with_hole/plate_with_hole_optimize.jl   (from examples/)
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
using WhatsThePoint: SpacingEquilibriumForce, compute_force

# ---- parameters ------------------------------------------------------------
const Lx = 4.0; const Ly = 4.0          # plate [-2,2] × [-2,2]
const a0 = 0.40; const b0 = 0.20        # initial hole semi-axes (2:1 ellipse)
const dx = 0.05                          # node spacing (fast first-look; use 0.03 for a fine background run)
const σ∞ = 1.0                           # equibiaxial tension
const n_iter = 40
const α0 = 50.0                          # (unused in adjoint-only mode below; kept for reference)
const filt_r = 0.10                      # Helmholtz filter radius (physical length)
const max_move = 0.007                   # adjoint-only: target max hole-node motion per iter

model = LinearElasticity(E = 1.0e7, ν = 0.3)
μ, λstar = lame_parameters(model)
basis = PHS(3; poly_deg = 3); const k = 35

# ---- cloud: interior grid (carve ellipse) + outer edges + hole ring --------
ellipse_val(p, a, b) = (p[1] / a)^2 + (p[2] / b)^2
margin = 1 + 1.2 * dx / min(a0, b0)

base_pts = SVector{2, Float64}[]; base_tag = Symbol[]
let xs = (-Lx / 2 + dx):dx:(Lx / 2 - dx / 2),
    ys = (-Ly / 2 + dx):dx:(Ly / 2 - dx / 2)
    for x in xs, y in ys
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

const n_fixed = length(base_pts)          # everything before the hole ring is fixed
const nθ = max(48, round(Int, 2π * sqrt((a0^2 + b0^2) / 2) / dx))
hole0 = [SVector(a0 * cos(2π * j / nθ), b0 * sin(2π * j / nθ)) for j in 0:(nθ - 1)]
append!(base_pts, hole0); append!(base_tag, fill(:hole, nθ))

const N = length(base_pts)
const hole_rng = (n_fixed + 1):N          # global indices of hole nodes
adjl = find_neighbors(base_pts, k)

idx_of(t) = findall(==(t), base_tag)
const interior_idx = idx_of(:interior)
const outer_idx = vcat(idx_of(:xlo), idx_of(:xhi), idx_of(:yhi), idx_of(:ylo))
const hole_idx = collect(hole_rng)
const neumann_idx = vcat(outer_idx, hole_idx)
neumann_adjl = adjl[neumann_idx]   # refreshed each iter as interior re-spaces

# rigid-body pins: 3 DOFs consistent with symmetry.
# Equibiaxial loading + centered hole → ux=0 on x=0 (y-axis), uy=0 on y=0 (x-axis).
nearest(p0, pool) = pool[argmin([hypot(base_pts[i][1] - p0[1], base_pts[i][2] - p0[2]) for i in pool])]
# Points on the symmetry axes (above/below hole for y-axis, left/right for x-axis)
const pin_ux1 = nearest((0.0,  0.8), interior_idx)   # ux=0 on +y axis (correct by symmetry)
const pin_ux2 = nearest((0.0, -0.8), interior_idx)   # ux=0 on -y axis
const pin_uy1 = nearest((0.8,  0.0), interior_idx)   # uy=0 on +x axis (correct by symmetry)
const dirichlet_dofs = [pin_ux1, pin_ux2, pin_uy1 + N]   # 2× ux + 1× uy = 3 DOFs
const dirichlet_vals = zeros(3)
const active = let a = trues(2N)
    a[pin_ux1] = a[pin_ux2] = a[pin_uy1 + N] = false
    a
end
const interior_rows = let r = falses(N); for i in interior_idx; r[i] = true; end; r end

const pinA = pin_ux1; const pinB = pin_uy1   # keep names for reaction check below
@printf("Cloud N=%d  (interior=%d outer=%d hole=%d)\n",
        N, length(interior_idx), length(outer_idx), length(hole_idx))
@printf("Pins: ux=0 at %d (0,%.2f) %d (0,%.2f)  uy=0 at %d (%.2f,0)\n",
        pin_ux1, base_pts[pin_ux1][2], pin_ux2, base_pts[pin_ux2][2],
        pin_uy1, base_pts[pin_uy1][1])

# ---- geometry helpers ------------------------------------------------------
# cur_fixed = the non-design nodes (interior + outer), MUTABLE: interior nodes are
# re-spaced by the WTP repel relaxation as the hole moves; outer nodes stay anchored.
# make_pts rebuilds the full cloud (current fixed part + current hole).
cur_fixed = collect(base_pts[1:n_fixed])
make_pts(hole) = vcat(cur_fixed, hole)
hole_centroid(hole) = sum(hole) / length(hole)
function hole_area(hole)
    A2 = 0.0; n = length(hole)
    for j in 1:n
        p = hole[j]; q = hole[mod1(j + 1, n)]
        A2 += p[1] * q[2] - q[1] * p[2]
    end
    return abs(A2) / 2
end
function rescale_to_area!(hole, target)
    c = hole_centroid(hole); s = sqrt(target / hole_area(hole))
    for j in eachindex(hole); hole[j] = c + s * (hole[j] - c); end
    return hole
end
# (radial-about-centroid — kept for reference; replaced by the polyline normals below)
function hole_normals(hole)
    c = hole_centroid(hole)
    return [(-(p - c)) / hypot((p - c)[1], (p - c)[2]) for p in hole]
end

# Phase D L2: TRUE hole normals via chord-tangent polyline (analytic Jacobian).
# Replaces radial-about-centroid, which was up to ~37° wrong on the 2:1 ellipse.
const hole_loop = collect(hole_rng)             # ordered ring (global indices, CCW)
const hole_pos  = collect(1:length(hole_loop))  # each hole node's position in the loop
const _ZJV = SVector(0.0, 0.0)
zero_njac(g) = NormalJacobian(g, g, _ZJV, _ZJV, _ZJV, _ZJV)   # for fixed (outer) normals
hole_polyline(hole) =                                          # → (normals, NormalJacobians)
    polyline_normals(reduce(vcat, [[p[1], p[2]] for p in make_pts(hole)]), hole_loop, hole_pos)

# ---- forward solve + compliance (equibiaxial; hole traction-free) ----------
outer_normal(i) = base_tag[i] === :xhi ? SVector(1.0, 0.0) :
                  base_tag[i] === :xlo ? SVector(-1.0, 0.0) :
                  base_tag[i] === :yhi ? SVector(0.0, 1.0) : SVector(0.0, -1.0)

function build_layout(hole)
    hn, _ = hole_polyline(hole)
    normals = SVector{2, Float64}[]; tractions = SVector{2, Float64}[]
    for i in outer_idx
        n = outer_normal(i); push!(normals, n); push!(tractions, σ∞ .* n)
    end
    append!(normals, hn); append!(tractions, fill(SVector(0.0, 0.0), length(hole)))
    return build_traction_layout(neumann_idx, neumann_adjl, normals, tractions, λstar, μ, N)
end

function forward(hole)
    pts = make_pts(hole)
    W_d2x  = _build_weights(Partial(2, 1), pts, pts, adjl, basis)
    W_d2y  = _build_weights(Partial(2, 2), pts, pts, adjl, basis)
    W_d2xy = _build_weights(MixedPartial(1, 2), pts, pts, adjl, basis)
    neu = pts[neumann_idx]
    W_dx = _build_weights(Partial(1, 1), pts, neu, neumann_adjl, basis)
    W_dy = _build_weights(Partial(1, 2), pts, neu, neumann_adjl, basis)
    A = assemble_elasticity_from_weights(W_d2x, W_d2y, W_d2xy, N, λstar, μ)
    b = zeros(2N)
    apply_dirichlet!(A, b, dirichlet_dofs, dirichlet_vals)
    apply_traction!(A, b, build_layout(hole), W_dx, W_dy)
    u = lu(A) \ b
    return (u = u, b = b, pts = pts)
end
compliance(hole) = (s = forward(hole); dot(s.b, s.u))

# ---- iter-0 forward-solve sanity check -------------------------------------
println("\n--- Iter-0 forward solve check ---")
s0 = forward(hole0)
u0 = s0.u; bvec0 = s0.b; pts0 = s0.pts
C0 = dot(bvec0, u0)
@printf("  Compliance C = %.4e\n", C0)
@printf("  ‖u‖ = %.4e   max|u| = %.4e\n", norm(u0), maximum(abs, u0))

# Check force balance: sum of b (before Dirichlet) on pinned DOFs should be ~0
# Actually, check that pin reactions are small (self-equilibrated load)
A0 = let pts0_l = make_pts(hole0)
    W_d2x  = _build_weights(Partial(2, 1), pts0_l, pts0_l, adjl, basis)
    W_d2y  = _build_weights(Partial(2, 2), pts0_l, pts0_l, adjl, basis)
    W_d2xy = _build_weights(MixedPartial(1, 2), pts0_l, pts0_l, adjl, basis)
    A0_l = assemble_elasticity_from_weights(W_d2x, W_d2y, W_d2xy, N, λstar, μ)
    A0_l
end
pin_reactions = A0 * u0   # full residual before Dirichlet application
@printf("  Pin reactions: ux[%d]=%.2e  ux[%d]=%.2e  uy[%d]=%.2e  (expect ~0)\n",
        pin_ux1, pin_reactions[pin_ux1], pin_ux2, pin_reactions[pin_ux2],
        pin_uy1 + N, pin_reactions[pin_uy1 + N])

# Quick stress check on a few interior points
Dx = _build_weights(Partial(1, 1), pts0, pts0, adjl, basis)
Dy = _build_weights(Partial(1, 2), pts0, pts0, adjl, basis)
ux0 = u0[1:N]; uy0 = u0[(N + 1):2N]
εxx0 = Dx * ux0; εyy0 = Dy * uy0
cc = 1.0e7 / (1 - 0.3^2)
σxx0 = cc .* (εxx0 .+ 0.3 .* εyy0)
σyy0 = cc .* (εyy0 .+ 0.3 .* εxx0)
# Check far-field: pick a few interior points away from the hole
i1 = interior_idx[argmin([hypot(pts0[i][1] - 1.5, pts0[i][2]) for i in interior_idx])]
i2 = interior_idx[argmin([hypot(pts0[i][1] - 0.0, pts0[i][2] - 1.5) for i in interior_idx])]
@printf("  σxx at (%.2f,%.2f) = %.4f  (equibiaxial → expect σ∞=%.1f)\n",
        pts0[i1][1], pts0[i1][2], σxx0[i1], σ∞)
@printf("  σyy at (%.2f,%.2f) = %.4f  (equibiaxial → expect σ∞=%.1f)\n",
        pts0[i2][1], pts0[i2][2], σyy0[i2], σ∞)

# ---- shape gradient of compliance (LIVE polyline normals; ∂b/∂pts = 0) -----
# Hole is traction-free (b=0 there) and outer normals are fixed ⇒ no load term;
# the only normal contribution is ∂A/∂n·∂n/∂pts on the hole, fed via normal_jacobians.
function compliance_grad(hole)
    pts = make_pts(hole)
    layout = build_layout(hole)
    _, hjacs = hole_polyline(hole)
    njacs = vcat(NormalJacobian[zero_njac(i) for i in outer_idx], hjacs)  # outer fixed, hole live
    b = zeros(2N)
    for k in eachindex(layout.rows); b[layout.rows[k]] = layout.b_vals[k]; end
    res = shape_gradient(reduce(vcat, [[p[1], p[2]] for p in pts]),
                         model, N, adjl, basis, active,
                         dirichlet_dofs, dirichlet_vals, _ -> b;
                         interior_rows = interior_rows, traction_layout = layout,
                         neumann_ids = neumann_idx, neumann_adjl = neumann_adjl,
                         traction_jacobians = nothing, normal_jacobians = njacs)
    return res.Δpts, res.u, b
end

hole_grad(Δpts) = [SVector(Δpts[2g - 1], Δpts[2g]) for g in hole_rng]

# ---- Helmholtz filter on the closed hole loop (physical radius) ------------
function helmholtz_loop(gvecs, hole, r)
    n = length(hole)
    h = [hypot((hole[mod1(j + 1, n)] - hole[j])[1],
               (hole[mod1(j + 1, n)] - hole[j])[2]) for j in 1:n]
    A = zeros(n, n)
    for j in 1:n
        jm = mod1(j - 1, n); hm = h[jm]; hp = h[j]; hbar = (hm + hp) / 2
        A[j, jm] += -r^2 / (hbar * hm)
        A[j, mod1(j + 1, n)] += -r^2 / (hbar * hp)
        A[j, j] += 1 + r^2 / (hbar * hm) + r^2 / (hbar * hp)
    end
    gx = A \ [g[1] for g in gvecs]; gy = A \ [g[2] for g in gvecs]
    return [SVector(gx[j], gy[j]) for j in 1:n]
end

# ---- interior re-spacing via WTP's repel force law (Option A) --------------
# Every design step, relax the interior nodes to equilibrium at the target spacing
# using WhatsThePoint's SpacingEquilibriumForce (F(u)=(1−u²)/(u²+β)², zero at r=s).
# The hole + outer nodes are FIXED repulsion sources, so a growing hole pushes the
# interior shell outward — no engulfing, no stale-stencil collapse — without a
# differentiable morph. Mutates `cur_fixed` (interior only); boundary untouched.
const force_model = SpacingEquilibriumForce(0.2)
const relax_s  = dx        # target spacing (ConstantSpacing)
const relax_α  = 0.08      # damping: step = s·α·F per sweep
const relax_kk = 15        # neighbours used for the repulsion sum
const relax_sweeps = 200
const relax_tol = 5.0e-4   # stop when max |step| / s < tol
const band_w = 4 * dx      # re-space ONLY interior nodes within band_w of the hole;
                           # the bulk grid stays frozen — it is well-conditioned and
                           # anchors the band, so the relaxation can't drift or clump
                           # (moving every node lets the truncated-kNN attraction term
                           # net an inward pull → bulk collapse → singular stencils).
mindist_to_hole(p, hole) = minimum(hypot((p - h)[1], (p - h)[2]) for h in hole)

# point-in-polygon (signed-angle winding) on the hole loop — mirrors WTP's 2D
# isinside without the unitful/validation overhead; cheap enough for the guard.
function inside_hole(p, hole)
    n = length(hole); s = 0.0
    @inbounds for j in 1:n
        a = hole[j] - p; b = hole[mod1(j + 1, n)] - p
        s += atan(a[1] * b[2] - a[2] * b[1], a[1] * b[1] + a[2] * b[2])
    end
    return abs(s) > π
end

# safety net: shove an escaped interior node radially back out, ~one spacing
# beyond the hole node nearest in angle (ray from the hole centroid).
function project_out(p, hole)
    c = hole_centroid(hole)
    d = p - c; r = hypot(d[1], d[2]); r < 1.0e-12 && return c + SVector(relax_s, 0.0)
    ang = atan(d[2], d[1])
    j = argmin([abs(rem2pi(atan((hole[m] - c)[2], (hole[m] - c)[1]) - ang, RoundNearest))
                for m in eachindex(hole)])
    rh = hypot((hole[j] - c)[1], (hole[j] - c)[2])
    return c + (rh + relax_s) * d / r
end

function relax_interior!(hole)
    all_pts = make_pts(hole)
    # band = interior nodes near the hole; only these move (bulk + boundary are
    # fixed repulsion sources). Recomputed each step since the hole moves.
    band = [gi for gi in interior_idx if mindist_to_hole(all_pts[gi], hole) < band_w]
    nbrs = find_neighbors(all_pts, relax_kk + 1)
    moves = Vector{SVector{2, Float64}}(undef, length(band))
    for _ in 1:relax_sweeps
        for (loc, gi) in enumerate(band)
            xi = all_pts[gi]; f = SVector(0.0, 0.0)
            for nj in nbrs[gi]
                nj == gi && continue
                dd = xi - all_pts[nj]; r = hypot(dd[1], dd[2])
                r < 1.0e-12 && continue
                f = f + compute_force(force_model, r / relax_s) * dd / r
            end
            moves[loc] = relax_s * relax_α * f
        end
        mx = 0.0
        for (loc, gi) in enumerate(band)
            all_pts[gi] = all_pts[gi] + moves[loc]
            mx = max(mx, hypot(moves[loc][1], moves[loc][2]) / relax_s)
        end
        mx < relax_tol && break
    end
    for gi in band
        p = all_pts[gi]
        cur_fixed[gi] = inside_hole(p, hole) ? project_out(p, hole) : p
    end
    return cur_fixed
end

# ============================================================================
# iter-0 gradient validation: AD vs central FD on a few hole nodes
# ============================================================================
RUN_FD_CHECK = false   # gradient already validated in normals_check.jl (AD≈FD, cos 1.000)
if RUN_FD_CHECK
println("\n--- Iter-0 gradient check (AD vs FD) ---")
Δ0, _, _ = compliance_grad(hole0)
g_ad = hole_grad(Δ0)
probe = round.(Int, range(1, nθ; length = 6))
hh = 1.0e-6
@printf("  %4s %8s %8s | %12s %12s %12s %12s\n", "node", "x", "y", "AD_x", "FD_x", "AD_y", "FD_y")
maxrel = 0.0
for j in probe
    fdx = let hp = copy(hole0); hp[j] = hole0[j] + SVector(hh, 0)
        hm = copy(hole0); hm[j] = hole0[j] - SVector(hh, 0)
        (compliance(hp) - compliance(hm)) / 2hh
    end
    fdy = let hp = copy(hole0); hp[j] = hole0[j] + SVector(0, hh)
        hm = copy(hole0); hm[j] = hole0[j] - SVector(0, hh)
        (compliance(hp) - compliance(hm)) / 2hh
    end
    rel = hypot(g_ad[j][1] - fdx, g_ad[j][2] - fdy) / max(hypot(fdx, fdy), 1e-30)
    global maxrel = max(maxrel, rel)
    @printf("  %4d %8.3f %8.3f | %12.3e %12.3e %12.3e %12.3e\n",
            j, hole0[j][1], hole0[j][2], g_ad[j][1], fdx, g_ad[j][2], fdy)
end
@printf("  worst AD-vs-FD rel error on probed nodes = %.2e\n", maxrel)
end  # if RUN_FD_CHECK

# ============================================================================
# Optimization loop
# ============================================================================
hole = deepcopy(hole0)
A_target = hole_area(hole0)
hist = (compl = Float64[], gnorm = Float64[], a = Float64[], b = Float64[])
frames = NamedTuple[]

println("\n--- Optimization loop (equibiaxial, ellipse → circle expected) ---")
@printf("%4s %12s %10s %8s %8s %9s\n", "iter", "compliance", "‖g̃‖", "a_est", "b_est", "‖gi‖/‖gb‖")
for it in 0:n_iter
    Δ, u, b = compliance_grad(hole)
    g = hole_grad(Δ)
    # boundary-only gradient: report the interior pullback we drop (Option A). It is
    # a discrete stencil artifact and should stay small while the spacing is clean.
    g_int = sqrt(sum(gi -> Δ[2gi - 1]^2 + Δ[2gi]^2, interior_idx))
    g_bnd = sqrt(sum(x -> x[1]^2 + x[2]^2, g))
    pull_ratio = g_int / max(g_bnd, 1.0e-30)
    g̃ = helmholtz_loop(g, hole, filt_r)
    C = dot(b, u)
    gn = sqrt(sum(x -> x[1]^2 + x[2]^2, g̃))
    c = hole_centroid(hole)
    a_est = maximum(abs(p[1] - c[1]) for p in hole)
    b_est = maximum(abs(p[2] - c[2]) for p in hole)
    push!(hist.compl, C); push!(hist.gnorm, gn)
    push!(hist.a, a_est); push!(hist.b, b_est)
    push!(frames, (hole = deepcopy(hole), pts = make_pts(hole),
                   u = copy(u), g = deepcopy(g̃), C = C))
    @printf("%4d %12.5e %10.3e %8.4f %8.4f %9.2e\n", it, C, gn, a_est, b_est, pull_ratio)
    it == n_iter && break
    # adjoint-only descent step on the hole boundary (normalized to max_move), then
    # restore hole area, re-space the interior with the WTP repel force law so the
    # cloud stays valid as the hole moves, and refresh connectivity. One
    # shape_gradient call per iteration, total.
    gmax = maximum(x -> hypot(x[1], x[2]), g̃)
    global hole = rescale_to_area!([hole[j] - (max_move / max(gmax, 1e-30)) * g̃[j]
                                    for j in eachindex(hole)], A_target)
    relax_interior!(hole)
    global adjl = find_neighbors(make_pts(hole), k)
    global neumann_adjl = adjl[neumann_idx]
end

# ============================================================================
# Visualization
# ============================================================================
disp_mag(f) = sqrt.(f.u[1:N] .^ 2 .+ f.u[(N + 1):2N] .^ 2)

fig = Figure(; size = (1150, 460))
ax1 = Axis(fig[1, 1]; title = "shape + displacement ‖u‖", aspect = DataAspect(),
           xlabel = "x", ylabel = "y")
ax2 = Axis(fig[1, 2]; title = "shape gradient on hole (−g̃ = descent)", aspect = DataAspect(),
           xlabel = "x", ylabel = "y", limits = (-0.8, 0.8, -0.8, 0.8))
ax3 = Axis(fig[1, 3]; title = "compliance history", xlabel = "iter", ylabel = "C")
record(fig, joinpath(@__DIR__, "plate_with_hole_evolution.gif"), 1:length(frames); framerate = 2) do fi
    f = frames[fi]
    empty!(ax1); empty!(ax2)
    dm = disp_mag(f)
    scatter!(ax1, [p[1] for p in f.pts], [p[2] for p in f.pts]; color = dm,
             colormap = :viridis, markersize = 4)
    hx = [p[1] for p in f.hole]; hy = [p[2] for p in f.hole]
    lines!(ax1, vcat(hx, hx[1]), vcat(hy, hy[1]); color = :white, linewidth = 2)
    lines!(ax2, vcat(hx, hx[1]), vcat(hy, hy[1]); color = :black, linewidth = 2)
    arrows!(ax2, hx, hy, [-g[1] for g in f.g], [-g[2] for g in f.g];
            lengthscale = 0.05 / max(maximum(g -> hypot(g...), f.g), 1e-30),
            color = :crimson)
    ax1.title = @sprintf("iter %d   C = %.4e", fi - 1, f.C)
end
empty!(ax3); lines!(ax3, 0:(length(hist.compl) - 1), hist.compl; linewidth = 2)
scatter!(ax3, 0:(length(hist.compl) - 1), hist.compl)
save(joinpath(@__DIR__, "plate_with_hole_opt_summary.png"), fig)

@printf("\nDONE. a:%.3f→%.3f  b:%.3f→%.3f  C:%.4e→%.4e\n",
        hist.a[1], hist.a[end], hist.b[1], hist.b[end],
        hist.compl[1], hist.compl[end])
println("Saved: plate_with_hole_evolution.gif  +  plate_with_hole_opt_summary.png")
