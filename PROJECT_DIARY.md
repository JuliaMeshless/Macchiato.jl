# Project diary — RBF-FD discrete-adjoint shape optimization

Short human index of *what we learned* and *what to do next*. Newest on top.
Full detail: `RBF_FD_shape_optimization.tex` (the story),
`boundary_gradient_noise.md` (the granular probe log).

---

## 2026-06-03 — 3D parametrization-comparison facility: core built + gradient-exact; sphere-recovery optimizer diagnosed (stale-cloud bias)

**Direction (user).** Build a facility to compare surface parametrizations on 3D
shape-opt problems. Three axes: PROBLEM × DESIGN-SPACE × PHYSICS, with a common
optimizer + metrics. **Key unification:** every smooth parametrization (spherical
harmonics, Biancolini RBF, FFD, Bézier) is, at the current surface, a linear
sensitivity matrix `B = ∂(surface pts)/∂l`; the optimizer is identical, only
`apply` (=B) and `contract` (=Bᵀ) differ. Hole→sphere is the GROUND-TRUTH rung
(exact Eshelby optimum) to certify metrics; flagship flexibility benchmarks
(all four chosen): **Vigdergauz hole → fillet → heat-sink → Pironneau drag**.
Build order: **shakedown first** on hole→sphere, then climb.

**Built + validated this session:**
- **3D differentiable normals** (Phase-D-L2 analogue) — `triangle_normals`,
  `NormalJacobian3D`, `update_traction_coeffs_3d!`, `extract_normal_sensitivities_3d!`,
  threaded into `shape_gradient_3d` via `normal_jacobians`. FD-validated:
  Jacobian 2.6e-10, full live-normal gradient median 0.999999
  (`test_shape_gradient_3d_normals.jl`). Frozen-normal tests unchanged (no regression).
- **SH radial design space** (`src/optimization/design_space_3d.jl`):
  `SphericalHarmonicModes` (icosphere template + real `Y_lm` + radial
  `r(θ,φ)=Σ c_lm Y_lm`), `contract_gradient` (=Bᵀ), exact `cavity_volume`/
  `volume_gradient`/`project_volume`, `fit_ellipsoid_sh`, degree-set constructor.
  Self-test (`test_sph_design_space.jl`): orthonormality 7e-5, contract==Bᵀ 3e-10,
  volume-grad 2e-10, ellipsoid fit 1e-2, pure Y₀₀⇒exact sphere. ALL PASS.
- **3D Laplace morph** — `LaplaceExtension` made dimension-generic (2D path
  unchanged; 3D adds the zz term + SVector{3}).
- **Octree geometry** (`cavity_sphere_recovery_sh.jl`) — spherical shell (outer
  icosphere + cavity icosphere) filled by **WTP `Octree`** volume fill.
- **Full-chain shape gradient FD-EXACT** on the real 3D cavity-compliance problem
  (adjoint → `morph_transpose` → `contract`): median AD/FD = 0.99995.

**Hard-won 3D geometry/discretization lessons (durable):**
- **Use WTP `Octree`, NOT `SlakKosec`, for a cavity (multiply-connected) domain.**
  SlakKosec runs away to the point cap. `Octree(mesh; spacing, alpha)` from a
  triangulated `SimpleMesh` (outer outward + cavity flipped inward) is robust.
  `discretize(PointBoundary,…)` **preserves my boundary nodes** and only fills
  the volume, so the cavity nodes = SH template (connectivity known for normals).
- **`max_points` in `Octree` is the TARGET total, not a cap** (`n=round(max·ratio)`).
  Set it to `V_solid/Δ³` for a uniform cloud.
- **Boundary nodes must be ≳ Δ (coarser than / equal to the volume), never denser.**
  A cavity finer than the volume → coplanar surface stencils → singular.
- **✅ RESOLVED (2026-06-04) — flat faces are NOT intrinsically singular; the cube
  crash was boundary-finer-than-volume, cured by `h_boundary ≥ Δ`.**
  `examples/cavity_3d/diagnose_flat_boundary.jl` rebuilds the EXACT octree pipeline
  with a cube outer boundary and checks every stencil's degree-3 3D Vandermonde
  rank (σ_min/σ_max — the quantity `bunchkaufman!` throws on). Verdict depends
  ONLY on face spacing `h` vs volume spacing `Δ`:
    - `h = Δ`   → **0 singular stencils** (boundary nodes get 13–21 off-plane
      volume neighbors; worst σ_min/σ_max = 8.8e-4, comfortable). Flat cube WORKS.
    - `h = Δ/2` → 3362 singular; the worst boundary nodes have **0 volume
      neighbors, max-off-plane = 0.000** — i.e. all 50 k-NN are coplanar face nodes.
    - `h = Δ/3` → 20327 singular.
  Mechanism (the user's intuition, quantified): boundary nodes lie on a 2D sheet,
  so within radius `r` there are `~π(r/h)²` of them (∝ r²); the first interior
  layer is at depth `~Δ`. When `h < Δ` the in-plane count saturates the `k`-budget
  BEFORE `r` reaches the sparser interior layer → stencil collapses onto the plane
  → `P` loses column rank → `SingularException`. When `h ≥ Δ` the nearest interior
  node (depth ≤ h) beats most in-plane neighbors → genuinely 3D "cloud" stencil.
  NOT flatness, NOT duplicate nodes (min NN distance never ~0). This is exactly the
  "boundary ≳ Δ, never denser" rule above — now proven necessary & sufficient for
  the SingularException. **Consequence: the spherical-outer-boundary workaround is
  unnecessary; the cube benchmark is viable with its outer face grid at spacing ≥ Δ.**
  **WIRED IN (2026-06-04):** `cavity_sphere_recovery_sh.jl` now uses a flat-faced
  CUBE outer boundary (`outer_cube(L_OUT, H_OUT)`, `H_OUT=Δ`). Assembles + solves
  with NO SingularException; full-chain gradient FD-check median AD/FD = 0.99972 ✓
  (adjoint machinery unchanged). Hydrostatic σ∞·n on flat faces is the EXACT
  uniform-stress BC, and edge/corner normal averaging is harmless (σ·n = σ∞ n ∀n).
  Optimizer convergence is UNCHANGED from the sphere version (asph 34.3%→35.9%, C
  drops 5× then no-descent, ‖g_c‖ blows up 1.4e4→1.5e7 over 4 iters) — that's the
  separately-tracked stale-cloud-bias / per-node-noise stall, NOT a cube artifact;
  cure remains the two-front re-anchored-remesh loop (next).

**Sphere-recovery optimizer — DIAGNOSED, not yet converging.** Gradient is exact,
but descent stalls: C drops ~10× while shape barely moves (asph 34.3%→34.2%),
then no-descent. Signature = 2D **failure mode C (stale-cloud objective bias)**:
the cloud anchored to the ellipsoid has its *discrete* compliance optimum near
the ellipsoid, not the sphere; plus residual discrete noise / cavity
under-resolution (ρ≈3·Δ, 162 cavity nodes). The 2D cure is the **two-front
re-anchored remesh**: descend on ‖Jᵀg‖, re-anchor a fresh octree cloud between
steps (morph within, re-init between), fixed normalized step — NOT a C-based line
search (C jumps across remeshes). `anchor()` already rebuilds a fresh cloud; the
remaining work is wiring the two-front loop + raising cavity resolution.

**Two-front loop BUILT (2026-06-04) — `cavity_sphere_recovery_twofront.jl`.**
Reuses the cube geometry + adjoint; adds quality indicators (3D measures), fixed
normalized Sobolev step, re-anchor-on-trip. Findings:
- **Independent-cloud remesh THRASHES.** The absolute indicators (min_gap, min_sep,
  cavity_cv) trip on EVERY fresh octree cloud (the jittered fill inherently makes
  ~6e-3 near-duplicate interior↔boundary pairs), so it remeshes every iter; each
  cloud is an independent random discretization ⇒ gradient-DIRECTION noise ⇒ asph
  random-walks 31–38%, ‖g‖ spikes ~5e5. (morph_drift/stencil_growth stay 0/1 — the
  RELATIVE degradation measures never engage because we never morph.)
- **CRN (common random numbers) CURES the thrash.** `Random.seed!(SEED)` at the top
  of `anchor()` ⇒ consecutive remeshed clouds differ only by the smooth design
  change, not independent jitter ⇒ the stochastic gradient becomes a smooth
  deterministic one ⇒ monotone descent. This is the key 3D-specific fix (2D used a
  clean Cartesian cull so didn't need it).
- **Residual = cavity UNDER-RESOLUTION bias, monotone in ρ/r_ref.** With CRN the
  loop cleanly descends the DISCRETE compliance — but when ρ≳r_ref (stencil bigger
  than the cavity) the discrete optimum is ASPHERICAL, so asph rises. Measured:
  ρ/r_ref=1.04 (Δ0.10/k50) → asph drifts UP to 42%; =1.015 (Δ0.08/k50) → wanders
  33–39%; =0.95 (Δ0.08/k40) → DESCENDS to 27%. So ρ<r_ref is the lever.
- **k is NOT the lever (conditioning).** Lowering k to shrink ρ starves the
  poly_deg=3 3D saddle system (20 monomials) — k=40 spiked ‖g‖ to ~4e7 (vs ~1e4 at
  k=50). Shrink ρ via finer Δ ONLY (also improves conditioning). At k=50, ρ/Δ≈5.2,
  so ρ<r_ref needs Δ≲0.06 (~120k DOF) — or NSUB=3 cavity (642 nodes) + Δ≈0.05
  (~190k DOF) for a properly-resolved cavity surface too. Compute-gated from here.
**Node-generation direction (user, 2026-06-04):** WTP **Octree is the SOTA node
generator** and the intended path — do NOT abandon it. The Cartesian-lattice cull in
`cavity_sphere_recovery_twofront.jl` is a TEMPORARY shakedown simplification (clean
uniform stencils cheaply, deterministic so no CRN needed). The octree needs two bits
of shape-opt development to come back as the default: (a) suppress the near-surface
**near-duplicate** volume points (~6e-3 spacing) that trip min_sep/min_gap and forced
remesh-every-iteration; (b) make routine use of its **graded spacing**
(`BoundaryLayerSpacing`), which IS compulsory for the harder geometries (the graded
attempt here only failed because the transition raised interior spacing_cv→0.47 and
made the gradient noisy — an implementation/tuning issue, not a reason to avoid it).

**Simplified node-gen + larger hole (2026-06-04, end of session).** Switched the
twofront example to a DETERMINISTIC Cartesian-lattice-cull interior (no octree/jitter)
+ a LARGER cavity (ax,ay,az=0.62,0.48,0.55, r_ref≈0.55, L_OUT=1.0). This FIXED the loop
MECHANICS: clean uniform stencils (spacing_cv 0.16 vs octree's 0.31–0.47, min_sep≈Δ, no
near-duplicates) ⇒ the absolute indicators no longer trip ⇒ **FRONT-1 morphing finally
engages** (morph_drift climbs 0→0.6 over ~9 iters, then remesh — the proper two-front
regime; deterministic so no CRN needed). ρ_cav=0.26 (0.48·r_ref), well-resolved, ~16k
nodes/48k DOF, fits easily. ALSO learned: my OOM forecast for direct LU was way off — a
78k-DOF graded run used only ~15 GB (free bottomed 3G). And the graded-spacing attempt
fits RAM but its fine/coarse TRANSITION raised interior spacing_cv→0.47 ⇒ noisy gradient
(implementation issue; graded stays compulsory for harder geometries — see Node-gen note).

**BUT asph still does NOT recover the sphere — and it's NOT resolution/noise.** Across
EVERY config (octree & lattice, ρ/r_ref 1.04→0.48, noisy & clean) the optimizer drives
asph UP/wandering, never →0. **Decisive sphere-stationarity test (start AT the sphere,
ax=ay=az):** the discrete gradient at the sphere is NONZERO (‖g_proj‖≈5e4–9e4 after
volume projection) and asph climbs away (0→7→15%). So the sphere is NOT a stationary
point of the DISCRETE compliance on the cube domain. **Pin placement changes the C
landscape:** rpin=0.6 (ON the r_ref≈0.55 cavity) ⇒ sphere→7% INCREASES C (sphere lower,
fixed-step overshoots); rpin=0.9 (out near faces) ⇒ sphere→7% DECREASES C (aspherical
genuinely lower-C). So the 3-2-1 pins + cube discretization are SETTING the optimum, not
the physics. Three live causes, not yet separated:
  (a) **3-2-1 pins break symmetry** (A@+x,B@−x,C@+y is asymmetric) ⇒ spurious l=2
      gradient at the sphere. Moving pins out flipped the C-sign but ‖g‖ stayed ~8e4 —
      necessary fix, not sufficient. Try a SYMMETRIC rigid-mode removal (nullspace
      projection of the 6 rigid modes instead of 3 asymmetric pins).
  (b) **Cube outer boundary has CUBIC symmetry** ⇒ the finite-domain optimum carries a
      cubic harmonic (l=4) the l=2 design can't represent; pins break cubic→lower sym,
      leaking into l=2. **STRONGEST LEAD: for the GROUND-TRUTH sphere-recovery rung,
      revert the OUTER boundary to a SPHERE** (the original design) — spherical symmetry
      makes the sphere provably the optimum. The cube was introduced only to exercise
      flat faces; that flat-boundary singularity question is DONE/validated separately
      (diagnose_flat_boundary.jl), so the cube belongs in a flat-boundary MECHANICS test,
      NOT the exact-optimum recovery benchmark. I (Claude) conflated the two by putting
      the cavity-recovery benchmark on a cube.
  (c) **Fixed normalized step can't settle a shallow optimum** — it always moves
      STEP_FRAC·r_ref regardless of ‖g‖, so near a shallow min it overshoots and jitters
      (2D got away with it because the gradient vanished cleanly at the circle and the
      maxδ<1e-14 check fired; 3D has a nonzero noise floor). Add a backtracking
      line-search WITHIN a morph interval (C IS comparable there — no remesh) and keep
      the fixed step only on the post-remesh iter.

**NEXT (resume here, priority order):** (0a) **revert outer boundary to a SPHERE** for
the recovery benchmark (keep cube as a separate flat-boundary mechanics check) — most
likely to make the sphere the true optimum; (0b) re-run the sphere-stationarity test —
expect ‖g_proj‖→~0 at the sphere; (0c) if still nonzero, swap 3-2-1 pins for a symmetric
nullspace rigid-mode removal and/or add within-interval line search; (0d) THEN the
ellipsoid→sphere recovery should converge. (1) [done — two-front loop wired, mechanics
validated]. (2) add the **Biancolini RBF** design space (second `AbstractDesignSpace`
subtype, same Bᵀ seam) + metrics harness (recovery error, DOF efficiency, cond(BᵀB),
hi/low) to compare SH vs Biancolini; (3) climb to the Vigdergauz hole.

---

## 2026-06-02 (latest) — 3D differentiable normals (Phase-D-L2 analogue) validated

**State.** The 3D traction adjoint no longer needs frozen normals. Added the 3D
analogue of 2D's `polyline_normals`/`NormalJacobian`: **area-weighted triangle
vertex normals** with a closed-form sparse Jacobian, threaded into
`shape_gradient_3d` via a new `normal_jacobians` kwarg (default `nothing` ⇒
frozen-normals path unchanged; the frozen traction test still validates at
median 1.000000 — no regression). FD-validated two ways
(`test_shape_gradient_3d_normals.jl`): (A) the triangle-normal Jacobian vs
central FD, worst `2.6e-10`; (B) the full compliance gradient with normals
*moving* under the design perturbation, median `|AD|/|FD| = 0.999999`, 69/69
within 5%.

### What was added (`src/optimization/manual_adjoint_3d.jl`)
- `NormalJacobian3D` — sparse `∂N_i/∂pts` over the 1-ring (variable-length
  `cols` + 3×3 `blocks`), vs 2D's fixed `(prev,next)` pair.
- `triangle_normals(pts_flat, faces, neumann_ids, vertex_faces)` — vertex normal
  `N_i = g_i/‖g_i‖`, `g_i = Σ_{f∋i}(p_b-p_a)×(p_c-p_a)` (the un-normalized cross
  is `2·area`, so the sum is area-weighted). Per-triangle
  `∂m/∂p_a = skew(p_c-p_b)`, `∂m/∂p_b = skew(p_a-p_c)`, `∂m/∂p_c = skew(p_b-p_a)`
  (sum zero ⇒ translation-invariant); normalization pullback `(I-NNᵀ)/‖g‖`.
- `update_traction_coeffs_3d!` — refresh `layout.coeffs` to live normals
  (3D analogue of `update_traction_coeffs!`).
- `extract_normal_sensitivities_3d!` — the n-side term `-ηᵀ(∂A/∂n·∂n/∂pts)u`:
  `Sₙ = ∂L/∂n_i` contracted via the **same `_traction_coeff_3d`** the forward
  uses (coeff is linear in n ⇒ `∂coeff/∂n_d = _traction_coeff_3d(eq,cb,e_d,…)`,
  never hardcoded), then `Δp_j += (∂N_i/∂p_j)ᵀ Sₙ`.

### Lessons / notes (durable)
- **The boundary triangle mesh is part of the discrete problem, not differentiated
  through.** Connectivity (`faces`, `vertex_faces`) is built ONCE and held fixed;
  only vertex coordinates move. Same contract as the 2D boundary loop and as
  remeshing (morph within an interval, re-anchor between).
- **Edge/corner vertex normals blend incident faces** (e.g. a box edge gives
  `(0,-1,1)/√2`) — the 3D analogue of the 2D length-weighted corner normal.
  Harmless for *gradient validation* (AD matches FD of the same discrete problem
  regardless of physics); for a real optimization the corner-normal-freeze trick
  from 2D (override + zero its Jacobian) carries over if a sharp feature misaligns
  the traction BC.
- **`WTP.discretize` preserves the input boundary points exactly** (verified: same
  set, same count), so a structured per-face grid can be re-triangulated from the
  returned cloud — no need for the volume fill and the surface mesh to agree on
  ordering; match by coordinate.
- **Reused, didn't reimplement:** the normal-derivative coeffs come straight out
  of `_traction_coeff_3d` evaluated at unit normals; `_propagate_weight_gradient!`,
  `_find_nzval`, the layout, and the whole forward/adjoint chain are untouched.

### Next steps (3D, priority order) — unchanged except step 1 now DONE
1. ~~Differentiable 3D normals~~ **DONE (this entry).**
2. **3D design space** — surface/spherical harmonics or Biancolini RBF control
   points as a new `AbstractDesignSpace`; contract the nodal gradient onto it as
   `FourierModes.contract_gradient` does in 2D.
3. **Wire the two-front loop in 3D** — `LaplaceExtension` is dimension-agnostic
   (add the `zz` term); reuse the `Indicator` registry.
4. *(Cleanup)* Meshes `Box`→mesh→`PointBoundary(mesh)` path so the surface mesh
   comes from WTP/Meshes instead of the test's hand-built per-face triangulation.

---

## 2026-06-02 (later) — framework extracted; 3D elasticity adjoint validated

**State.** The 2D two-front framework is extracted into `src/` (`AbstractDesignSpace`
+ `FourierModes`, `AbstractExtension` + `LaplaceExtension`, the `Indicator` registry +
six measures); the plate example now consumes it and `shape_gradient` is unchanged.
3D is underway: `LinearElasticity3D` (Navier–Cauchy, full Lamé), the 6-operator
block assembly, and `shape_gradient_3d` — **adjoint FD-validated for both
interior-only Dirichlet** (`test_shape_gradient_3d.jl`, median 1.0000) **and
mixed Dirichlet + traction with frozen normals** (`test_shape_gradient_3d_traction.jl`,
median 1.0000, the Phase-B analogue). The 3D traction-row family is the 3×3
analogue of 2D's (`TractionLayout3D` / `build_traction_layout_3d` /
`apply_traction_3d!` / `extract_neumann_sensitivities_3d!`).

### Found + fixed: an RBF.jl eval-point pullback bug (3D first-derivatives)
The traction validation initially failed *only* at Neumann-node self-coordinates
(`AD ≫ FD`, FD stable across step sizes ⇒ real bug, not FD noise). Localized with
a normals-free, elasticity-free test (`S = ΔW:W(x)`, FD vs pullback): `Δdata`
exact, `Δeval` wrong. Root cause in **`RadialBasisFunctions.jl-1`
`_backward_partial_poly_3d!`**: it **hardcoded** the monomial ordering
`[1,x,y,z,xy,xz,yz,x²,…]`, but the real 3D `MonomialBasis` order is
`multiexponents(4, deg)` = `x³,x²y,x²z,x²,xy²,xyz,xy,…`. The forward uses the real
order (so it was exact); only this first-order-partial eval-backward indexed the
wrong monomials. (The 2nd-partial/mixed-partial eval-backwards were already
generic via `multiexponents`, which is why the interior test passed.) Fixed by
rewriting it generically, mirroring the 2D version. **Lesson:** never hardcode the
monomial ordering — derive it from the basis. A normals-free `S=ΔW:W(x)` probe is
the fastest way to separate a weight-pullback bug from a physics/normals bug.

Also unified `_propagate_weight_gradient!` to infer `dim` from the points (removed
the duplicate `_propagate_weight_gradient_3d!`).

### Lessons (3D-specific, the durable ones)
- **`poly_deg=3` is mandatory in 3D too — and a regular Cartesian lattice breaks it.**
  A structured grid is polynomial-unisolvency-degenerate at `poly_deg=3` (C(6,3)=20
  monomials go linearly dependent), so the local RBF saddle systems go singular and
  `_build_weights` *throws*. `poly_deg=2` builds but is garbage (O(1) operator errors,
  cond(A)~1e10). Cure: an **irregular cloud**. Use **WhatsThePoint** to fill the volume
  (SlakKosec) — unisolvent by construction, better-conditioned, and it's the same path
  as STL→cloud. (A hand-jitter ≥0.2·dx also works but WTP is the right tool.)
- **Every boundary node needs a real BC.** Applying the interior PDE at an
  unconstrained boundary node (`interior_rows=true` there, no Dirichlet/Neumann) makes
  A near-singular ⇒ `η=A⁻ᵀ(∂L/∂u)` explodes and AD/FD diverges by ~1e10. That was a
  *test* bug, not an adjoint bug: on a well-posed all-Dirichlet problem the adjoint is
  exact. Validate the adjoint only on a well-posed forward problem.
- **The 3D pipeline mirrors 2D exactly.** 6 second-derivative operators (xx,yy,zz,
  xy,xz,yz) instead of 3; `MixedPartial(d1,d2)` and its `_pullback_weights!` are
  dimension-generic and machine-precision in 3D. No new math — just more bookkeeping.

### Next steps (3D, priority order)
1. **Differentiable 3D normals** — `shape_gradient_3d` traction is validated with
   *frozen* normals (Phase-B analogue). For moving boundaries add the 3D L2 analogue:
   triangle-based vertex normals (the 2D polyline-normal analogue) + their closed-form
   Jacobian, threading an `extract_normal_sensitivities_3d!` term. (This is what the
   "normal terms" worry was about — real, but a separate piece from the eval-pullback bug.)
2. **3D design space** — surface/spherical harmonics *or* Biancolini RBF control points
   (natural for STL), as a new `AbstractDesignSpace` subtype; contract the nodal gradient
   onto it exactly as `FourierModes.contract_gradient` does in 2D.
3. **Wire the two-front loop in 3D** — `LaplaceExtension` is already dimension-agnostic
   (just add the `zz` term in the morph Laplacian); reuse the `Indicator` registry.
4. *(Cleanup)* the 3D validation test hand-builds box-face surface points; consider a
   Meshes `Box`→mesh→`PointBoundary(mesh)` path. Low priority — the volume fill (the
   part that matters) already uses WTP `discretize`.

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
