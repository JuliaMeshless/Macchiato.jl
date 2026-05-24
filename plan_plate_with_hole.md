# Plan — Plate-with-Hole Shape Optimization (filter-based / non-parametric)

**Date**: 2026-05-24
**Status**: planning. Supersedes the cantilever as the active optimization
problem (the cantilever is retired — its job, validating the manual adjoint,
is done: `rel_err ~1e-7` + the decisive FD experiment in
`docs/boundary_gradient_noise.md`).

Relates to: `plan_shape_optimization_pipeline.md` (roadmap/gap analysis),
`plan_shape_optimization_AD.md` §plate-with-hole, `plan_manual_adjoint.md`
(adjoint internals), `docs/boundary_gradient_noise.md` (why ℓ² node gradients
are noisy → need a Hilbertian metric).

## Why this problem, why now

The cantilever was a gradient-correctness fixture and was over-solved (corner
`D_act=max` kink, node-collapse at the tapering tip, soft penalty). The
plate-with-hole is the documented driving validation and is *better
conditioned* for showcasing the optimizer:

- design boundary is a **smooth closed loop** — no corners, no clamped-edge
  coupling, no shape-coupled tip load;
- clean geometric constraint (**fixed hole area**);
- **known analytic optimum** (Kirsch: under equibiaxial load the circle
  minimizes peak hoop stress; by symmetry it is also the min-compliance hole at
  fixed area).

## Methodology decision: filter-based, NOT explicit parametrization

We replace the roadmap's "Phase 4 spline parameterization" with **non-parametric
/ filter-based (Vertex Morphing / Sobolev-gradient) shape optimization**. Per
the user's design philosophy ([[feedback-minimal-parametrization]]): every hole-
boundary node is a free design variable; smoothness and well-posedness come from
a single **physical-radius Helmholtz filter**, not from hand-placed control
points. One knob (`r`, a length = minimum feature size). Zero designer
preprocessing. Scales to STL surfaces where explicit parametrization is
hopeless.

**The filter is the parametrization.** Implemented consistently on both sides:
- *Design side:* node motion comes from a filtered control field → geometry
  stays smooth, only smooth shapes are reachable.
- *Sensitivity side:* the raw nodal gradient is filtered through the same
  operator → that is the Sobolev/Riesz gradient, provably a descent direction
  (`dJ[-g_H] = -‖g_H‖²_{H¹} ≤ 0`).

Must use the **physical arc-length** Laplacian `(I − r²Δ_Γ)`, not the cantilever
filter's segment-count stencil — so `r` is mesh-independent (no per-grid retune).

## The design → loss → gradient chain

```
control field s (on hole loop)
   │  F = (I − r²Δ_Γ)^{-1}   (morphing filter, physical r)
   ▼
boundary node motion  x_b = x_b0 + (F s) ⊙ n     (n = polyline normals, Phase D L2)
   │  RBF interpolation of boundary displacement  (∂pts_i/∂pts_b, differentiable)
   ▼
interior nodes x_i follow  →  forward solve (validated)  →  u  →  loss L
   ▲                                                                  │
   └──── Fᵀ · (n ⊙ (∂L/∂x_b + interior-pullback)) ◄── shape_gradient (manual adjoint) ◄─ ∂L/∂u
```

Reuses, essentially unchanged: `shape_gradient`, `TractionLayout`/
`apply_traction!` (outer load), `polyline_normals`/`NormalJacobian` (hole-loop
normals — cleaner here than the cantilever, smooth closed loop). New: morphing-
filter layer, interior RBF deformation + its pullback, projected area
constraint, optimization driver.

## Staged objectives (user choice: "both, staged")

### Stage 1 — Compliance, equibiaxial → circle (optimizer validation)
- Objective: compliance `bᵀu`. Reuses the validated compliance gradient
  wholesale.
- Loading: equibiaxial **traction** (force-controlled) on the outer edges —
  the textbook compliance-minimization setup; reuses the traction machinery.
  (The AD plan wrote "equibiaxial Dirichlet"; that framing belongs to the
  stress/Kirsch objective. Revisit control type at Stage 2.)
- Constraint: fixed hole area.
- Start: ellipse (e.g. 2:1). Expected optimum: **circle**.
- Success: converges to a circle; **mesh-independent** (same shape at two
  resolutions with the same physical `r`); area held exactly; monotone descent.

### Stage 2 — Peak stress, uniaxial → optimal oval (engineering result)
- Objective: min–max von Mises / hoop stress on the hole boundary, via a smooth
  aggregate (KS or p-norm) so it's differentiable; its adjoint feeds the same
  `shape_gradient` (∂L/∂u from the stress recovery).
- Loading: uniaxial tension. Expected optimum: an oval elongated along the load
  (and the equibiaxial case recovers the Kirsch circle as a check).
- New machinery: boundary stress recovery + KS/p-norm aggregation + its ∂/∂u.

## Component checklist (maps to pipeline-plan gaps)

- [x] **Geometry + cloud**: Cartesian grid + carved ellipse + hole ring, outer/
      hole boundaries classified (`plate_with_hole_optimize.jl`). Plus 3D STL
      geometry (`make_plate_with_hole_stl.jl`) + WTP 3D cloud (`*_cloud3d.jl`).
- [x] **Forward-solve sanity**: Kirsch K_t recovered — **DONE** via manufactured-
      solution test (polar annulus + exact Kirsch traction BC). L2 0.62%, K_t=2.966.
- [x] **Hole normals (Phase D L2)**: `polyline_normals` in the forward
      (`build_layout`) AND the gradient (`normal_jacobians`). Replaced the
      radial-about-centroid hack (37° wrong on the ellipse → wrong σ·n=0 →
      suppressed gradient). Confirmed exact (0°) + machine-exact AD≡FD
      (`normals_check.jl`). **This was the "gradients look small" bug.**
- [~] **Morphing-filter / Helmholtz**: `helmholtz_loop` on the closed hole loop
      (arc-length, physical `r=0.10`) — works as the Sobolev smoother. TODO:
      verify mesh-independence (same shape at two resolutions, same physical `r`).
- [ ] **Interior deformation** (gap #2) — **⚠ BLOCKING — DO THIS NEXT.** Interior
      cloud is FIXED ⇒ degrading near-hole stencils + stale `adjl` + a HARD failure
      (engulfing) once the boundary passes the initial margin. Needs an RBF morph
      (∂pts_i/∂pts_b) carried into the gradient. **Larger optimization steps make
      this worse, not better — they are NOT a substitute.**
- [~] **Constraint** (gap #5): hole area held by exact radial rescale each step
      (works). Projected-gradient form still optional.
- [x] **Optimizer** (gap #4): hand-rolled adjoint-only descent (normalized step;
      no FD, no line-search solves — 1 `shape_gradient`/iter). Backtracking path
      also present (toggle in the loop).
- [ ] **Robustness** (gap #7): min node-spacing / non-inversion guard — not yet.
- [ ] **Stage 2**: boundary stress recovery (biased near free surfaces — flagged) +
      KS/p-norm aggregate + its ∂/∂u — not started.

## Defaults (adjustable)
- Plate `[0,W]×[0,H]`, hole centered; start ellipse semi-axes `(a,b)=(0.3,0.15)`
  in plate units (TBD with W,H). Equibiaxial traction magnitude O(1).
- Mesh: start coarse-ish (validate cheap), then a finer cloud for the
  mesh-independence check. `PHS(3; poly_deg=3)`, `k=35` (project convention).
- Filter radius `r` = a physical fraction of the hole radius (e.g. `0.5·a`).

## Open questions (to confirm before/while building)
1. Plate vs hole size ratio and load magnitude (sets stress-concentration regime
   and how far ellipse→circle has to travel).
2. Outer loading control: traction (Stage 1, recommended) vs Dirichlet (the AD
   plan's wording) — confirm for Stage 2's stress framing.
3. Optimizer: hand-rolled projected-gradient first, or jump to NLopt SLSQP?
4. Interior deformation: re-interpolate every iteration (simple) vs incremental.

## Status (2026-05-24, updated after Kirsch investigation)

- **2D forward solve: NOW VALIDATED (manufactured-solution test).**
  `examples/plate_with_hole/plate_with_hole_kirsch.jl` — polar annulus mesh with
  **exact Kirsch traction applied at the outer boundary**. This is a manufactured-
  solution test that eliminates finite-plate effects. Results:
  - θ=90° interior profile (the stress-concentration direction): **worst |rel| = 1.2%**
    at r/R=1.01, rapidly converging to <0.2% for r/R > 1.5
  - Global interior L2 stress error: **0.62%**
  - K_t via interior extrapolation: **2.966** (1.1% below the exact 3.0)
  - Fitted far-field σ∞ = 1.001 (0.1% error)
  
  **The earlier forward scripts (`plate_with_hole_forward.jl`, original O-grid
  `plate_with_hole_kirsch.jl`) were NOT valid.** The claimed "far-field σxx = 1.006"
  was a coincidental cancellation at a single cherry-picked point; the actual
  interior profile had 10-20% errors. Root cause: comparing a finite-plate solution
  (clamped or traction edges at finite distance, free top/bottom edges) against the
  *infinite*-plate Kirsch formula — these are different boundary-value problems.

  **Lesson**: the RBF-FD solver IS accurate, but the validation must use a
  manufactured-solution (or equivalent) approach where the exact BCs are known.

- **3D STL → cloud: VALIDATED.** `examples/plate_with_hole/plate_with_hole_cloud3d.jl`
  — WTP `PointBoundary(stl)` + `discretize(VanDerSandeFornberg())` fills the
  volume and **correctly leaves the hole tunnel empty** (3D octree isinside, not
  the 2D single-loop winding test). `split_surface!(75°)` isolates the **hole
  wall (surface3) = design surface**, top/bottom faces, 4 outer walls. Caveat:
  thin plate (T=0.1) at Δ=0.05 is under-resolved through thickness — use finer Δ
  or thicker plate for a real 3D solve.

## 3D lift — key finding and the fork

**Macchiato's `LinearElasticity` is 2D plane-stress ONLY** (model docstring;
`_ℒ_mixed_partial` is hardcoded `MonomialBasis{2}`; assembly takes 3 second-
derivative operators = 2D; `lame_parameters` is plane-stress). The 3D-native
parts are WTP clouds and the *heat* solver — **not elasticity, and not the manual
adjoint** (both 2D). So a 3D elasticity *solve* is a new build:

- **3D elasticity forward** (moderate, example-prototype first): 3D Navier-Cauchy
  (3 disp components; 6 second-derivative operators d2x/d2y/d2z/d2xy/d2xz/d2yz;
  full-3D λ, not plane-stress λ\*; 3D traction σ·n). RBF `Partial(2,d)`/
  `MixedPartial` already exist. Validate on the STL cloud (thin-plate → ≈ Kirsch).
- **Semi-3D design**: hole-wall nodes move in-plane only (z-constrained,
  extruded) ⇒ the design space stays 2D while the state is 3D.
- **3D shape gradient** (the big piece, deferred): the manual adjoint is 2D.
  Options when we get there — (a) extend the manual adjoint to 3D (6 operators,
  3D pullbacks); (b) FD on the filtered/low-dim design (no new adjoint); (c) a
  differentiable forward (PDESolveIFT/Mooncake) in 3D.

**Note on thin-plate semi-3D physics**: for a thin uniform plate with in-plane
z-invariant design, the 3D answer ≈ 2D plane stress — so the *physics* gain over
2D is small; the payoff is exercising the STL→3D-solve→3D-surface-opt pipeline we
ultimately need. Thicker plates / z-varying design make 3D physics matter.

## CURRENT STATE — where a fresh session resumes (2026-05-24, latest)

The **2D Stage-1 path is the active, working line** (we did NOT take the 3D fork
yet — 3D is deferred, see §"3D lift").

**Done & validated:**
1. **Solver** — Kirsch manufactured-solution test passes (K_t=2.966, interior L2
   stress 0.62%). `plate_with_hole_kirsch.jl`.
2. **Hole normals** — switched radial-about-centroid → `polyline_normals` (Phase D
   L2) in the forward (`build_layout`) AND the gradient (`normal_jacobians`). This
   was the bug behind "gradients look small": radial normals were **37° wrong** on
   the 2:1 ellipse, mis-modeling σ·n=0 and suppressing the sensitivity. After the
   fix: normal error 0° (exact for a θ-sampled ellipse), AD≡FD (cos 1.000),
   gradient ~15× larger. Confirmed in `normals_check.jl`.
3. **Stage-1 optimizer** (`plate_with_hole_optimize.jl`) — equibiaxial compliance
   minimization, **adjoint-only** (no FD, no line-search solves; 1 `shape_gradient`
   per iter). RUNS and MOVES THE RIGHT WAY: ellipse → circle (a 0.40→0.387,
   b 0.20→0.230), compliance ↓2.5% over 20 iters @ dx=0.05. Artifacts:
   `plate_with_hole_evolution.gif`, `plate_with_hole_opt_summary.png`.

**NEXT STEP — NON-NEGOTIABLE, must come before anything else: interior deformation
(mesh motion).** The interior cloud is currently FIXED while the hole boundary
moves. This is the blocking defect, not a polish item:
- it injects *growing* numerical error — as the boundary nears the fixed interior,
  near-hole spacing compresses into anisotropic, ill-conditioned RBF-FD stencils,
  and the once-computed `adjl` becomes stale (wrong neighbours); AND
- it **hard-fails** once the boundary passes the initial interior margin (~0.26 in
  y): interior nodes end up *inside* the hole ⇒ stencils straddle the void ⇒
  garbage solve. The target circle needs b≈0.283 > 0.26, so it WOULD engulf.

⇒ **Do NOT "converge" by bumping `max_move`/`n_iter`/finer `dx` — a larger step
just engulfs sooner. That is the opposite of the fix.** First carry the interior
*with* the boundary: a smooth RBF morph (boundary displacement → interior),
refresh `adjl`, and add the chain-rule term `Jᵀ_morph · Δpts_interior` to the
gradient (the morph is a smooth function of the boundary, so it stays
differentiable). Only *after* this is driving ellipse→circle meaningful.

**Only after the morph:** mesh-independent Helmholtz `r` (same shape at two
resolutions); proper projected area constraint; then Stage 2 (stress objective +
boundary-stress recovery); then the 3D lift.

**Then:** Stage 2 (stress objective + boundary-stress recovery — recovery is
biased near free surfaces, flagged), and the 3D lift (needs a 3D elasticity model;
elasticity is 2D-only — see §"3D lift").

**Perf note:** the adjoint (`shape_gradient`) ≈ 2 forward solves and is
design-dimension-independent; the slow parts are FD validation (24 solves to check
6 nodes) and line-search forward solves. Validate the gradient ONCE (done), then
run adjoint-only (`RUN_FD_CHECK=false`, normalized step) for fast evolution runs.

**Files** (`examples/plate_with_hole/`):
- `make_plate_with_hole_stl.jl` → `plate_with_hole.stl` — 3D thin plate + elliptical hole (watertight, genus-1).
- `plate_with_hole_cloud3d.jl` — WTP 3D cloud from the STL (hole tunnel correctly empty; hole wall = `surface3`).
- `plate_with_hole_kirsch.jl` — manufactured-solution Kirsch verification (the step-0 proof).
- `plate_with_hole_forward.jl` — earlier finite-plate forward; **SUPERSEDED** (its infinite-plate-Kirsch comparison was invalid; kept only as a record).
- `plate_with_hole_optimize.jl` — Stage-1 optimizer (adjoint-only mode; line-search path also present).
- `normals_check.jl` — diagnostic that found + confirmed the normals fix.
