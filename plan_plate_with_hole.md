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

- [ ] **Geometry + cloud**: rectangular plate, elliptical hole; RBF cloud of the
      plate-minus-hole; classify outer (loaded) vs inner (free) boundary loops.
- [ ] **Forward-solve sanity**: reproduce the analytic stress concentration
      (Kirsch K_t = 3 for a circular hole, uniaxial) before any optimization.
- [ ] **Morphing-filter layer** (pipeline gap #1, our substitute for splines):
      physical arc-length `(I − r²Δ_Γ)`; control field → normal motion; same
      operator on sensitivities.
- [ ] **Interior deformation** (gap #2): RBF interpolation of boundary
      displacement to interior nodes; differentiable pullback.
- [ ] **Constraint** (gap #5): exact hole-area via projected gradient
      (`p ← p − (⟨p,∇A⟩/⟨∇A,∇A⟩)∇A`) + a correction step; replaces the
      cantilever's soft penalty.
- [ ] **Optimizer** (gap #4): start with a hand-rolled projected-gradient +
      Armijo (reuse cantilever loop structure); consider NLopt SLSQP later.
- [ ] **Robustness** (gap #7): min node-spacing / non-inversion guard on the
      hole loop; backtrack rather than abort.
- [ ] **Stage 2 only**: boundary stress recovery + KS/p-norm + adjoint.

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

## Immediate next step
Stage 1, step 1: build the plate-with-hole geometry + forward solve and verify
the analytic stress concentration. No optimization until the forward problem is
trustworthy.
