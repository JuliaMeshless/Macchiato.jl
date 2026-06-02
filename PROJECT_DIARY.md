# Project diary — RBF-FD discrete-adjoint shape optimization

Short human index of *what we learned* and *what to do next*. Newest on top.
Full detail: `RBF_FD_shape_optimization.tex` (the story),
`boundary_gradient_noise.md` (the granular probe log).

---

## 2026-06-02 — plate-with-hole closed; two-front optimizer works

**State.** Plate-with-hole (2D plane-stress, biaxial load, elliptical hole →
circle) validated end-to-end. Discrete adjoint exact (FD 1e-6). The two-front
optimizer converges to the true circle. Ready for framework extraction + 3D.

### Lessons (the durable ones)

- **Three failure modes — never conflate them.**
  - **(A) descent-direction noise** = `∂W/∂x`, the geometry-sensitivity of the
    RBF-FD weights. Intrinsic, present on a *fresh* cloud, broadband.
    Cure: a smooth low-dim **design space** (contract the nodal gradient onto a few
    Fourier modes). Not optional.
  - **(B) cloud degradation** — frozen stencils stretch as nodes move.
    Cure: **remesh**.
  - **(C) objective bias** — a cloud fit to a *stale* geometry has its discrete
    optimum off the true one (±0.07 here, shallow-optimum amplified).
    Cure: **re-anchored remesh**. NOT intrinsic — a fresh cloud at the circle has
    `∂C/∂a₂ ≈ 5e-8`.
- **Filtering the gradient is the wrong cure.** The artifact is broadband and
  overlaps the signal; no frequency filter separates them faithfully. Always pair
  a smoothness metric (hi/lo) with a **faithfulness** check (cosine vs the true
  low-mode gradient).
- **Accurate forward solve ≠ smooth shape gradient.** `dT/dr = −K⁻¹(dK/dr)T` runs
  the rough `∂W/∂x` through `K⁻¹`; `T` can be machine-accurate while its
  sensitivity is rough.
- **Remeshing is NOT precluded under the discrete adjoint.** Only differentiating
  *through* the remesh, or demanding one fixed discrete objective, is precluded.
  Morph (differentiable) *within* an interval, re-anchor (re-initialize) *between*.
  Descend on the **gradient**, not on the (remesh-jumping) `C` value.
- **Calibrate from the cloud, never by hand.** mode cap `= ⌊π·r/ρ⌋`
  (stencil-Nyquist), Sobolev length `= ρ/r`. Hand-set knobs overfit — a cull change
  once *flipped the m=2 sign*.
- **The clean-DOF budget is the fundamental limit.** At dx=0.05/k=35 it's m=2.
  More modes need refinement (smaller `ρ/r`) — a cost trade, not a tuning knob.
- **Indicators:** `morph_drift` fires but is conservative; `stencil_growth` is the
  principled trigger (operator validity) but slack (≤1.09); the other four were
  dormant on this gentle deformation — insurance for harsher regimes.
- **Settling step backfires under a stale morph** (lingers in the distorted
  regime). Fixed normalized step marches through. Revisit settling now that
  remeshing caps distortion.
- **Validate on the REAL PDE gradient early.** The smooth-field `∂W/∂x` surrogate
  under-estimated the real noise (no `K⁻¹`) and mis-attributed it.

### Validated
Kirsch `K_t = 2.966` (0.62% stress) · adjoint FD 1e-6 · bias = stale-cloud
(`probe_remesh_unbiased`) · two-front `a₂ 0.099→0.015` (circle), `C 5.18→4.88e-5`.

### Next steps (priority order)
1. **Framework extraction** — `DesignSpace` / `Extension` / `Indicator` registry
   into `src/`, `FourierModes` first, `shape_gradient` unchanged. The
   plate-with-hole code (`plate_with_hole_twofront_opt.jl`) is the seed.
2. **Numerical efficiency** (the 3D blocker) — today weights rebuild every
   iteration and the morph refactors every remesh on N≈6.4k. Need: reuse /
   incremental factorizations, sparse weight updates only where nodes moved.
   Possibly LSQ/oversampled RBF-FD (Neumann robustness, Phase 2).
3. **3D / STL** — surface/spherical-harmonic design space *or* **Biancolini RBF
   control points** (natural for STL); a 3D elasticity model (currently 2D
   plane-stress only); STL→volume cloud via WhatsThePoint. First experiment doubles
   as multi-mode (finer dx → m_cap>2) + efficiency preview.
4. *(Deferred)* membership-staleness indicator (current-kNN); land local stencil
   shift/scale in `RBF.jl-1` for forward-accuracy robustness at higher poly_deg.

### Canonical files (post-cleanup, `examples/plate_with_hole/`)
- `plate_with_hole_twofront_opt.jl` — **THE optimizer** (two-front closed loop)
- `plate_with_hole_kirsch.jl` — solver validation · `probe_remesh_unbiased.jl` — bias test
- `plate_with_hole_fourier_opt.jl` — capped-mode (fixed-cloud, bias vehicle) ·
  `plate_with_hole_ellipse_opt.jl` — 1-DOF proof
- reference probes: `probe_fourier_modes`, `probe_weight_sensitivity`,
  `probe_stencil_conditioning`, `probe_thermal_swap`
- 3D pipeline: `make_plate_with_hole_stl.jl`, `plate_with_hole_cloud3d.jl`,
  `normals_check.jl`

---

## Earlier (condensed from `boundary_gradient_noise.md`)
- **Phases A/B/C/D-L1/D-L2** — discrete adjoint built and FD-validated; dead-load
  traction Jacobians; differentiable polyline normals; Helmholtz boundary filter.
- **Cantilever (Phase 3)** — retired; adjoint validated, 46.6% compliance
  reduction. Job done.
- **Phase D L3 (the noise saga)** — long investigation that diagnosed `∂W/∂x` and
  ruled out every gradient-side cure (combined pullback, Sobolev/Riesz/Tikhonov
  filtering, Hermite/ghost stabilization, smoother kernels, LSQ, ensemble
  averaging). Conclusion that reorganized everything: the cure is structural
  (design space + remeshing), not a gradient filter. → led to the 2026-06-02 work.
