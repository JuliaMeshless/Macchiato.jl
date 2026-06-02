# Boundary Gradient Noise at Higher Grid Resolutions

**Date**: 2026-05-23/24
**Status**: root cause identified, fix proposed

## Symptom

On grids finer than 9×5 (17×9, 33×17), the compliance gradient w.r.t. boundary
point y-coordinates exhibits Nyquist-like noise — alternating sign at every
adjacent point along the boundary. This noise:
- Causes ~10-27% symmetry violation (under y → −y) at 17×9
- Forces heavy Helmholtz filtering, which kills the true descent direction
- Causes the optimizer to stall or diverge at 17×9 and 33×17

The 9×5 grid (45 pts, k=35) works correctly: 46.6% compliance reduction,
smooth shape, FD `rel_err = 1.4e-7`.

## Investigation

### Hypothesis 1 (rejected): insufficient k on finer grids

Increasing k from 35 to 140 on 17×9 does NOT reduce the noise. The noise
ratio (energy in 2nd difference / total gradient energy) stays at ~3.8
regardless of k — nearly pure Nyquist alternation (ratio = 4.0 is perfect
alternation).

### Hypothesis 2 (rejected): ill-conditioned stiffness matrix

`cond(A) = 1e12–1e17` on both 9×5 and 17×9, but the forward solve residuals
are `5-8e-10`. UMFPACK handles these matrices fine. Conditioning is a red
herring.

### Hypothesis 3 (rejected): complex-step AD bug

Complex-step differentiation (`Im[f(x+ih)]/h`) was attempted but the RBF
`_build_weights` contains small linear solves. With `ComplexF64`, LU pivot
choices differ from `Float64`, changing the real part of the weights. The
complex-step derivative is consistent (stable across `h = 1e-20` to `1e-6`)
but of a different function. AD vs FD confirms the AD gradient is correct
(`rel_err = 9.5e-7` at 17×9).

### Root cause (confirmed): destructive cancellation between operator pullbacks

The manual adjoint computes the gradient as:

```
Δpts = Σ_{op ∈ {d2x,d2y,d2xy,dx,dy}} _pullback_weights!(ΔW_op, W_op, cache_op, op)
```

Each operator's pullback produces a `Δpts` contribution independently. On the
17×9 grid (N=153, k=35), the per-operator norms are:

```
op      ‖Δpts_op‖    noise ratio (2nd-deriv energy / total)
d2x     1.63e+04     2.46
d2y     6.79e+04     1.50
d2xy    4.25e+03     2.53
dx      2.97e+04     2.52
dy      7.84e+04     2.20
---
SUM     1.72e+03     2.97
AD      9.19e+02     3.80
```

The individual pullbacks have norms of **1.6–7.8×10⁴**, but their sum is only
**1.7×10³** — a 10-40× cancellation. The physically meaningful gradient
(compliance reduction w.r.t. boundary coordinates) is the small residual
after near-perfect cancellation between the ∂² and ∂ operator contributions.

#### Why cancellation occurs

In the forward pass, the Neumann BC `σ·n = t` couples ∂/∂x, ∂/∂y
(first-derivative) weights with ∂²/∂x², ∂²/∂y², ∂²/∂x∂y (second-derivative)
weights. The traction boundary condition is satisfied when these operator
contributions balance each other.

In the adjoint pass, each ΔW_op is pulled back to point coordinates through a
**different** local RBF adjoint system (`_pullback_weights!` solves an
operator-specific adjoint for the stencil). The 5 pullback solutions have
different numerical character. When summed, the signal (true physical
gradient) cancels, leaving a small residual. But the **noise** in each
pullback (from stencil-level numerics) does not cancel as efficiently,
amplifying the relative noise from ~2.5 per-operator to ~3.8 in the sum.

This is a numerical cancellation problem in the chain rule, analogous to
computing `f'(x) - g'(x)` when `f ≈ g` — finite differences would lose all
precision, AD handles it correctly, but our manual decomposition into
per-operator pullbacks re-creates the cancellation after each pullback has
introduced its own numerical errors.

## Proposed fix: combined pullback per stencil

Instead of 5 independent pullbacks, form **one** combined pullback per
stencil that accounts for all 5 operators simultaneously:

```
# Current (5 separate pullbacks, cancellation after the fact):
for op in [d2x, d2y, d2xy, dx, dy]:
    _propagate_weight_gradient!(Δpts, ΔW_op, W_op, cache_op, ..., op)

# Proposed (1 combined pullback):
_propagate_combined_weight_gradient!(Δpts, ΔW_list, W_list, cache_list, ..., op_list)
```

The combined pullback solves **one** adjoint system per stencil that encodes
all 5 operators. The cancellation between ∂² and ∂ operators happens
**inside the linear solve** (at machine precision), before the result is
accumulated into Δpts. This eliminates the noise amplification from
post-pullback cancellation.

### Implementation

The new function lives in `src/optimization/manual_adjoint.jl` and replaces
the 5 separate `_propagate_weight_gradient!` calls in `shape_gradient`
(`ext/MacchiatoMooncakeExt.jl:108-118`).

The combined pullback needs a companion in RadialBasisFunctions —
`_pullback_weights_combined!` — that takes multiple (W_k, ΔW_k, ℒ_k) tuples
for the same stencil and produces a single (Δdata, Δeval) pair. The local
adjoint system combines the RBF and monomial basis contributions from all
operators, weighted by the ΔW entries.

Cost: one larger linear solve per stencil instead of 5 smaller ones. In
practice this should be **faster** because the RBF kernel matrix and
monomial basis are shared across operators — only the operator-specific
evaluation vectors differ.

### Compatibility

- Phase A (Dirichlet-only): only 3 operators (d2x, d2y, d2xy), no Neumann.
  Combined pullback still applies — 1 solve instead of 3.
- Phase B (mixed BC): 5 operators as described.
- Phase D L1 (∂b/∂pts): unaffected — the b-side contribution is added
  separately via `extract_load_sensitivities!`.
- Phase D L2 (∂n/∂pts): unaffected — the normal sensitivity is a separate
  `extract_normal_sensitivities!` call.

### Validation strategy

1. Verify `rel_err < 1e-6` on 9×5 (Phase A + B) — must match current
   accuracy.
2. Check noise ratio on 17×9 bottom edge — should drop from 3.8 to < 1.0.
3. Check symmetry error — should drop from ~10% to < 0.1%.
4. Run 40-iter optimization on 17×9 — should achieve ≥ 40% compliance
   reduction with monotone convergence.

## Files to modify

| File | Change |
|------|--------|
| `src/optimization/manual_adjoint.jl` | Add `_propagate_combined_weight_gradient!` |
| `ext/MacchiatoMooncakeExt.jl:108-118` | Replace 5 `_propagate_weight_gradient!` calls with 1 combined call |
| `RadialBasisFunctions.jl-1/src/` | Add `_pullback_weights_combined!` (or expose sufficient primitives) |
| `examples/shape_opt/diagnostics/test_noise_decompose.jl` | Diagnostic script (already written) |

## Other fixes applied in this session

| Item | File | Status |
|------|------|--------|
| `D_act = max(\|y_top\|, \|y_bot\|)` traction model | `shape_optimization_phase3_cantilever.jl` | Done |
| `shape_gradient` returns `η` | `MacchiatoMooncakeExt.jl` | Done |
| `diff` → `gdiff` shadowing bug | `shape_optimization_phase3_cantilever.jl` | Done |
| Line search guardrail | `shape_optimization_phase3_cantilever.jl` | Done |
| Fast subsampled FD validation | `shape_optimization_phase3_cantilever.jl` | Done |
| Relaxed `T<:Real` in RBF for complex numbers | `RadialBasisFunctions.jl-1/src/solve/types.jl` | Done |
| Relaxed `Float64` in Macchiato assembly | `manual_adjoint.jl` | Done |

---

# Addendum (2026-05-24): the root cause is contested

**Status**: the "operator-pullback cancellation" root cause above is
**disputed**. A second reviewer (fresh read of this doc, the `.tex` report,
`test_noise_decompose.jl`, and the FD-validation code) argues the noise is
**intrinsic to the discrete gradient**, not a numerical artifact of the
pullback decomposition, and that the proposed combined pullback will therefore
not reduce it. This section states both positions fully so a later session can
run the decisive experiment and settle it before any code is written in
`RadialBasisFunctions.jl`.

The two hypotheses make **opposite, falsifiable predictions**, so this is
resolvable cheaply. Do the experiment in §"Decisive experiment" first.

## Hypothesis A — operator-pullback cancellation (the incumbent)

*This is the theory argued in the body of this document.* In summary:

- The 5 per-operator pullbacks have norms 1.6–7.8×10⁴ but sum to 1.7×10³ — a
  10–40× cancellation.
- The physical gradient is the small residual after near-perfect cancellation
  between the ∂² and ∂ contributions.
- The signal cancels efficiently; the per-pullback **noise** (from stencil-level
  numerics) does not, so relative noise is amplified from ~2.5 per-operator to
  ~3.8 in the sum.
- **Fix**: one combined pullback per stencil, so the cancellation happens
  *inside* the local linear solve at machine precision, before noise is
  accumulated. Predicted to drop noise ratio 3.8 → <1.0.

**Steelman.** The cancellation magnitudes are real and measured
(`test_noise_decompose.jl`). Second-derivative operators scale like 1/h² and
first-derivative like 1/h, so on finer grids the operand norms grow and the
relative size of the physical residual shrinks — which is consistent with the
problem appearing only at 17×9 and worse at 33×17. The local RBF saddle systems
are genuinely ill-conditioned (PHS(3)+poly_deg=3 on a structured grid has
near-collinear stencils), so it is plausible that the *adjoint* of those local
solves amplifies high-frequency components differently per operator. If true,
re-associating the chain rule (combined solve) is a legitimate fix.

## Hypothesis B — intrinsic Nyquist content of the ℓ² shape gradient (dissent)

The noise is the **true gradient** of the discrete compliance objective with
respect to free node positions. It is the classic shape-optimization
sawtooth/checkerboard: the un-regularized ℓ² shape gradient on a free-node
design space carries high-frequency content that grows as the grid refines. The
combined pullback computes the *same mathematical quantity* and will reproduce
the same noise.

**Evidence against Hypothesis A / for Hypothesis B:**

1. **FD agrees to 9.5e-7, and FD does not use the decomposition.** The FD check
   (`examples/shape_opt/shape_optimization_phase3_cantilever.jl:611-619`) perturbs
   coordinates and re-solves the forward problem. It never forms the 5
   per-operator pullbacks. If the noise were an artifact of *summing* those
   pullbacks, FD would not reproduce it and rel_err would be large. It is 1e-6.
   Therefore FD produces the **same noisy gradient** → the noise is in the true
   discrete gradient. A correct combined pullback, computing the same gradient,
   cannot remove it (beyond ~1 digit of rounding).

2. **The validated quantity *is* the noisy one.** In the decomposition table,
   the `AD` row (the `shape_gradient` result, the thing validated against FD)
   has noise ratio 3.80. So "the gradient is correct to 1e-6" and "the gradient
   is ~pure Nyquist" are both true *of the same vector*. Correctness does not
   imply smoothness; here it implies the noise is real.

3. **The FD probe is strided, which sharpens the point.** `fd_probe =
   free_entries[1:div(n_free,n_probe):end][1:n_probe]` (line 608) samples 20
   **non-adjacent** DOFs. The agreement is therefore pointwise AD≈FD at those
   coordinates. Because the noise modes are the *large-magnitude* components
   (ratio 3.8 ⇒ noise dominates the norm), agreement on the norm means FD
   reproduces the large alternating values → FD is noisy too.

4. **The cancellation is not catastrophic.** 10–40× cancellation of float64
   operands loses ~1.5 significant digits (≈14 of 16 retained). Catastrophic
   cancellation — the kind that manufactures O(1)-fraction garbage — requires
   operands agreeing to ~7+ digits before subtraction. A 1.5-digit loss cannot
   create a Nyquist mode that is ~95% of the gradient energy. The arithmetic in
   the body does not reach the regime the body's argument needs.

5. **The `.tex` report already states Hypothesis B** (and disclaims A). Report
   §"Why filter at all?" (`report_phase_D_L2_helmholtz.tex`, the para beginning
   "The raw discrete gradient…"): *"This is not a bug — the FD validation passes
   to 1e-7 — but it is an artefact of the local RBF stencil truncation."* The
   report prescribes filtering as the remedy. This doc's body reverses that into
   a numerical-bug theory. The two documents are inconsistent; the dissent holds
   the report is right.

6. **Grid dependence is the signature of an un-regularized shape gradient.** As
   h→0 the shortest representable wavelength shrinks, so high-frequency energy
   grows with resolution and the Nyquist mode dominates — textbook behavior for
   ℓ² shape gradients, independent of how the chain rule is associated.

**Under Hypothesis B the real fix is regularization / design space, not the
pullback:**

- The Helmholtz filter is *not* a workaround — it is a Sobolev/Hilbertian
  gradient (steepest descent in H¹ rather than ℓ²), the established remedy.
  "It kills the descent direction" is a tuning symptom: (a) rescale the filtered
  direction so its inner product with the raw gradient is preserved; (b) make
  `r` a fixed **physical** length, not a segment count, so the filter is
  mesh-independent (the same physical smoothing at 9×5, 17×9, 33×17).
- Or restrict the design space to a smooth boundary parameterization (a few
  B-spline / Fourier modes), which removes the Nyquist mode by construction.

## Where the two hypotheses agree

- The forward solve, the adjoint solve, and the FD-validated correctness of
  `shape_gradient` are not in question.
- The 9×5 grid works; the problem is resolution-dependent.
- Heavy filtering currently destroys the descent direction.

They disagree only on **whether re-computing the same gradient via a combined
pullback will reduce the noise** (A: yes, 3.8→<1.0; B: no, ≈unchanged).

## Decisive experiment (do this first)

Compute a **full, non-strided** central-difference gradient on the bottom-edge
(and top-edge) y-coordinates at 17×9 — every adjacent node, not a stride — and
measure *its own* `noise_ratio` (the same metric as `test_noise_decompose.jl`,
≈4.0 = pure alternation). FD is ground truth and bypasses the pullback entirely.

| Outcome | Implication | Action |
|---------|-------------|--------|
| FD noise ≈ 3.8 (matches AD) | Noise is intrinsic to the discrete gradient (Hypothesis B). Combined pullback computes the same gradient → cannot help. | **Do not** build `_pullback_weights_combined!`. Invest in mesh-independent Helmholtz (physical `r`, rescaled direction) and/or a smooth boundary design space. |
| FD noise < 1.0 (smooth) while AD = 3.8 | The adjoint is genuinely producing noise the true gradient lacks (Hypothesis A). The strided probe (line 608) hid it. | Investigate the pullback. Combined pullback is justified. Also fix the validation to be adjacency-aware. |

Predicted cost: ~20–40 forward solves on the bottom edge (~a few minutes with
`jlrun`). Cheap relative to authoring a combined-pullback primitive in another
repo. The dissent predicts the first row; the incumbent predicts the second.

Write `examples/shape_opt/diagnostics/test_fd_edge_noise.jl` for this; capture the printed noise
ratios for both edges and append the result + verdict here.

## Note on the strided FD validation

Regardless of which hypothesis wins, the FD validation in the Phase 3 example
(strided 20-point probe, norm-based rel_err) **cannot detect adjacent-node
Nyquist disagreement** between AD and FD. If we ever need to certify smoothness
(not just pointwise correctness), add an adjacency-resolved FD check on a
contiguous run of edge nodes.

---

# RESULT (2026-05-24): Hypothesis B confirmed — noise is intrinsic

**The decisive experiment is done.** `examples/shape_opt/diagnostics/test_fd_edge_noise.jl` reuses the
*exact* validated loss/gradient pair from the Phase 3 example
(`loss_compl_only` / `grad_compl_only_ad`) via a new setup-only `include` mode,
then full-central-differences **every contiguous** bottom- and top-edge
y-coordinate at 17×9 (64 forward solves) and measures the same `noise_ratio`
metric on both the FD (ground truth, no pullback) and AD vectors.

```
noise_ratio (≈4.0 = pure Nyquist, →0 = smooth):
  edge          FD          AD
  bottom    3.6285      3.5567
  top       3.5724      3.4836

Adjacency-resolved FD vs AD on contiguous edge nodes (y-component):
  bottom    ‖AD-FD‖/‖FD‖ = 1.07e-01
  top       ‖AD-FD‖/‖FD‖ = 1.19e-01

Step-robustness (bottom): noise_ratio = 3.6285 at both h=1e-4 and h=5e-5.
```

**Verdict: FD reproduces the AD noise (Hypothesis B).** FD bypasses the
per-operator pullback entirely and still produces ~pure Nyquist alternation
(≈3.6), matching AD (≈3.5). The noise is the **true gradient of the discrete
ℓ² compliance objective**, not an artifact of summing the 5 pullbacks. The
incumbent Hypothesis A is **rejected**: a combined pullback would compute the
same quantity and cannot reduce the noise. This matches the `.tex` report's
original position and the dissent's arithmetic argument (a 10–40× cancellation
loses ~1.5 digits, far too little to manufacture a mode that is ~95% of the
gradient energy).

The per-node walk confirms FD ≈ AD to ~5 digits at **every** edge node — the
two are literally the same noisy vector. The 10–12% edge rel_err is **not**
spread over the edge; it is one node: the bottom-right **corner** (i=145, x=8.0),
FD=+6.07e1 vs AD=+1.20e2. That is the `D_act = max(|y_top|, |y_bot|)` kink: at
the symmetric reference `|y_top| = |y_bot| = D`, so `max` is non-differentiable.
Central FD averages the left/right one-sided derivatives (~half), while
`add_D_actual_load_sensitivity!` assigns the full one-sided value to both corners
(the `>=` tie). This is a subgradient ambiguity at the symmetric start, not a
Nyquist phenomenon, and is orthogonal to this decision — but worth a look if the
corner gradient ever matters off-symmetry.

## Action (supersedes the "Proposed fix" above)

**Do NOT build `_pullback_weights_combined!` in RadialBasisFunctions.** The
combined pullback is abandoned — it would not change the result. The "Proposed
fix: combined pullback per stencil" and "Files to modify → RadialBasisFunctions"
sections above are **superseded by this result**; left in place only as a record
of the rejected hypothesis.

Instead, pursue the Hypothesis-B remedies (a proper Hilbertian/Sobolev gradient,
not a noise patch):

1. **Mesh-independent Helmholtz filter.** Make `r` a fixed *physical* length, not
   a boundary-segment count, so smoothing is identical at 9×5 / 17×9 / 33×17;
   and rescale the filtered direction to preserve its inner product with the raw
   gradient (so "it kills the descent direction" — the current tuning symptom —
   goes away). Today `helmholtz_r` is hand-scaled per grid (`= 2.0` for 48
   vertices, was `1.0` for 9×5) — that is the mesh-dependence to remove.
2. **Or** restrict the design space to a smooth boundary parameterization (a few
   B-spline / Fourier modes), which removes the Nyquist mode by construction.

The CLAUDE.md "Repo state → Phase D L3" entry (which records the combined-pullback
plan as the next step) should be updated to reflect this verdict.

---

# UPDATE (2026-06-01): mechanism localized to ∂W/∂x — generic across physics

Three new fixed-cloud probes in `examples/plate_with_hole/` sharpen the
2026-05-24 verdict and correct one earlier claim.

## The per-mode adjoint is exact; the m=2 sign was misrecorded

`probe_fourier_modes.jl` FD-validates the adjoint contraction
`dC/daₘ = Σⱼ g_rad,ⱼ cos(mθⱼ)` mode-by-mode on the fixed iter-0 ellipse cloud
(N=6438, nθ=48). Every mode matches FD to `ratio = 1.000` (rel.err 1e-6–1e-4),
including the near-Nyquist ones. So the adjoint is not noisy — it is the exact
gradient of the discrete objective at every frequency.

The radial-gradient **spectrum rises toward Nyquist**:

```
amp:  m=2: 7.6e-7   m=4: 8.7e-6   m=8: 6.6e-6   m=12: 3.1e-5   m=20: 3.1e-5
```

The circle-seeking m=2 mode is ~40× *below* the near-Nyquist band. **Correction
to the old record:** the m=2 (cos2θ) gradient is `dC/da₂ = +1.83e-5`, **positive
— the correct circle-seeking sign** (descent reduces a₂ → ellipse→circle), FD-
confirmed. The earlier "−1.1e-6, wrong sign, 400× below floor" came from a
remeshing-based probe setup, not the physics. The m=2 signal is real and
correctly oriented; it is merely small relative to the high-mode gradients.

## The rise is generic across physics (thermal swap)

`probe_thermal_swap.jl` runs the *same* plate-with-hole cloud with a scalar
Laplace PDE (∇²T=0, insulated hole, balanced through-flux, thermal-compliance
objective), shape gradient by FD. The spectrum **rises toward Nyquist exactly
like elasticity**:

```
            ‖modes 2..6‖   ‖top-6 near Nyquist‖   ratio hi/lo
elasticity    8.99e-6           4.01e-5              4.46
thermal       1.04e+3           3.68e+3              3.54
```

⇒ the mechanism is not elasticity-specific.

## The roughness is manufactured by ∂W/∂x (physics-free)

`probe_weight_sensitivity.jl` removes the PDE entirely. With smooth analytic
fields f, η (the RBF-FD Laplacian reproduces ∇²f to 3.5e-3), it differentiates
the surrogate `S = ηᵀ(Wxx+Wyy)f` w.r.t. each hole node's radial position —
i.e. only the `ηᵀ(∂W/∂x)f` weight-sensitivity term that appears in *every*
discrete shape gradient. The result is **broadband**: near-Nyquist modes
(m=22: 3.39) are as large as low modes (m=2: 0.58), ratio hi/lo = 1.10.

So the high-frequency content is the **geometry-sensitivity of the RBF-FD
differentiation weights**, independent of the physics. The forward operator W
is accurate and smooth (`Wf ≈ ∇²f`); it is `∂W/∂x` that is rough. Since in the
continuous limit `∂(Wf)ᵢ/∂xⱼ → 0` for i≠j (moving one node cannot change the
true operator elsewhere), the broadband content is a **finite-h consistency
error** of the weight sensitivity, with scale set by the node spacing h.

## Consequence for a fix (general, auto-calibrated)

The discrete shape gradient = smooth physical part (decays with frequency) +
∂W/∂x artifact (broadband, Nyquist-dominated, vanishes as h→0). Any general,
parameter-free remedy must operate on the gradient + the cloud (not the
physics), with constants derived from the cloud: e.g. a Sobolev/Riesz descent
whose smoothing length is the RBF-FD **stencil radius** (the scale at which the
stencil-membership artifact decorrelates), discretized by an RBF-FD operator on
the cloud, plus smooth node motion (a velocity field) so refinement/topology
never changes discontinuously. This supersedes the per-case knobs (cull radius,
gap threshold, mode count, hand-set Helmholtz length), which were shown to be
overfitting — notably, changing the interior-node cull *flipped the sign* of the
projected m=2 gradient, because the node cloud is itself the discretized domain.

Probes: `probe_fourier_modes.jl`, `probe_thermal_swap.jl`,
`probe_weight_sensitivity.jl` (all standalone, fixed-cloud, no tuning).

## Follow-ups (2026-06-01, same session)

**Adjoint cross-coupling verified.** `probe_adjoint_coupling.jl` FD-validates
`shape_gradient`'s full `Δpts` at interior nodes adjacent to the hole — the path
where a boundary Neumann row's stencil contains interior nodes (9326 such
occurrences here). Adjoint vs FD matches to ~3 digits at the interior-adjacent
nodes; the only large rel-errs are on near-zero components (FD truncation on
~1e-9 values). A missing coupling term would give O(1) disagreement. ⇒ both
stencil cross-couplings (interior-row⇄boundary-node and boundary-row⇄interior-
node) are correctly assembled. No missing pullback path.

**Artifact GROWS with refinement (correction).** `probe_weight_refinement.jl`
runs the ∂W/∂x surrogate at dx=0.10/0.0707/0.05 with nθ∝1/dx. Robust result:
stencil radius / boundary spacing ρ/s ≈ 4.48→4.81 ≈ const (stencil ∝ h, as
expected for fixed k). Correction to the §UPDATE claim that ∂W/∂x is a
*vanishing* consistency error: the 2nd-derivative weights scale ~1/h², so their
node-position sensitivity scales ~1/h³ — the operator becomes MORE sensitive to
node placement as h→0. The artifact does not vanish under refinement; it
sharpens (matches the original "noise worsens at finer grids"). This is the
signature of an ill-posed map. Consequence: the regularization must be h-scaled
to track the 1/h^p growth, which is exactly why the smoothing length must be the
stencil radius (∝h), not a fixed physical length. (Caveat: the unnormalized
surrogate's absolute amplitudes also scale ~1/h^p, so it localizes the mechanism
but does not cleanly pin the exponent — that needs the physically-normalized
compliance gradient refined across dx.)

## Riesz-fix attempt (2026-06-01): frequency smoothing is insufficient

`probe_riesz_invariance.jl` builds a physically-normalized Dirichlet-thermal
shape-gradient density and applies a stencil-radius Laplace–Beltrami Riesz
step (I − ℓ²Δ_Γ)⁻¹, ℓ = c·ρ, c=1 fixed, then tests discretization invariance of
the descent direction across dx=0.10/0.0707/0.05.

Result — partial suppression, NO invariance. The Riesz step lowers hi/low
(raw 1.62/3.64/0.73 → smoothed 0.48/0.66/0.43) and roughly doubles low-mode
dominance, but the descent DIRECTION does not converge (smoothed cosine across
resolutions = +0.64 then ≈0). Confounds: coarse resolutions (nθ=20–40, so the
m2–6 fingerprint is itself partly in the contaminated band); an anomalous
dx=0.0707 cloud (hi/low=3.64) that dominates both comparisons; and the
Dirichlet-flux gradient being only mildly high-frequency (objective-dependent
spectral character, unlike the Neumann-compliance Probe A ratio 3.5).

Key conclusion: the ∂W/∂x artifact is BROADBAND and its frequency profile shifts
with the objective, so it OVERLAPS the signal in frequency. No frequency-domain
filter (Helmholtz/Sobolev/truncation) can fully separate them — the entire
"smooth the gradient in a frequency metric" family is at best a partial fix.
This is consistent with the earlier low-mode sign-flip and the need for
case-tuning. The artifact is fundamentally correlated to NODE PLACEMENT (it is
∂W/∂x), while the physical gradient is placement-invariant — which points at a
placement-ensemble (jitter-and-average) attack rather than spectral filtering.
Open: this needs a cleaner invariance test (finer resolutions, robust objective)
to be conclusive vs merely confounded.

## Test 1 (2026-06-01): the artifact is INTERIOR-row, not boundary-stencil

`probe_artifact_rowdecomp.jl` splits the ∂W/∂x surrogate dS/dr_j by row type
(hole one-sided / interior centered / outer) and sweeps k=35/70/140.

```
k=35:   HOLE  hi/low=0.18   INTERIOR hi/low=0.89   (INTERIOR hi-band ≈3× HOLE)
k=70:   HOLE  hi/low=0.10   INTERIOR hi/low=0.43
k=140:  HOLE  hi/low=0.06   INTERIOR hi/low=0.56
‖g‖ TOTAL: 91 → 36 → 25 (bigger stencils shrink the artifact; hi/low plateaus)
```

REFUTES the one-sided-boundary-stencil hypothesis for the high-frequency
artifact: the one-sided HOLE rows are the SMOOTH/low-frequency part (hi/low≈0.1,
dropping with k); the high-frequency content lives in the CENTERED, well-
conditioned INTERIOR rows. Mechanism: a moving boundary node sits in nearby
interior stencils, and as it sweeps the loop, which interior nodes are affected
jumps in a staircase — the Cartesian interior lattice meeting the curved boundary.
A node-LAYOUT effect, not boundary conditioning.

Consequences: (1) Hermite/ghost boundary stabilization would clean the (already
smooth) hole rows, so it is NOT expected to fix the high-frequency artifact;
(2) bigger stencils help by averaging but plateau. Caveat: this surrogate uses
smooth η,f, so it isolates ∂W/∂x only — it does NOT test whether the adjoint
STATE η,u is noisy from an unstable Neumann solve (a separate channel). Next:
operator-order sweep (0th/1st/2nd) — does the artifact climb with derivative
order (the pullback of a 2nd-order operator probes 3rd-order PHS behavior)?

## Test 2b (2026-06-01): the artifact is driven by DERIVATIVE ORDER

`probe_operator_order.jl` measures the ∂W/∂x surrogate artifact for operators of
increasing derivative order q (the pullback probes order q+1 of the kernel):

```
Identity   q=0 (→1st):  ‖g‖=0.00              (interpolation at nodes is exact ⇒ zero sensitivity)
Partial1   q=1 (→2nd):  ‖g‖=1.97   hi/low=0.37
Laplacian  q=2 (→3rd):  ‖g‖=91.4   hi/low=1.14
```

CONFIRMS the "gradient takes derivatives of RBF terms" hypothesis: the artifact
is born at q≥1, its magnitude jumps ~46× from q=1→q=2, and its high-frequency
fraction climbs (0.37→1.14). Mechanism: PHS r³ is only C², so the Laplacian
pullback's 3rd derivative carries ~1/r singularities at the stencil centre; the
q=1 pullback only needs 2nd derivatives (r³ fine there) ⇒ much milder.

Reconciles with Test 1: the real shape gradient's worst artifact is the q=2
INTERIOR PDE operators (Laplacian/elasticity), not the q=1 Neumann boundary rows
— so the high-freq lives in interior rows and boundary stabilization is not the
main lever. Coherent single story across Tests 1 and 2b.

Sharp parameter-free cure prediction: a SMOOTHER kernel. PHS5=r⁵ is C⁴, so its
3rd derivative is smooth (no 1/r) and should crush the q=2 artifact vs PHS3.
A discrete kernel choice, not a tuning knob. (Higher poly_deg won't help — the
singularity is in the RBF part.) Next: sweep PHS3/PHS5/PHS7 (×poly_deg) on the
q=2 artifact and re-validate forward accuracy.

## Cure-test (2026-06-01): smoother kernel FAILS — it's conditioning, not smoothness

`probe_kernel_smoothness.jl` sweeps PHS3/PHS5/PHS7 (poly_deg=3) on forward
accuracy and the q=2 ∂W/∂x artifact:

```
       forward acc (int / hole)     artifact ‖g‖   hi/low(int)   hi_int
PHS3     5.2e-4 / 3.5e-3              91.4           0.89          3.56
PHS5     2.0e-4 / 2.2e-3             252            2.90          7.61
PHS7     3.9e-5 / 6.4e-4             566            9.39         17.5
```

Prediction (smoother kernel → smaller artifact, from the r³ 3rd-derivative
singularity) is FALSIFIED. Smoother kernels improve forward accuracy but make the
artifact WORSE (‖g‖ 6×, interior hi-band 5× from PHS3→PHS7). Corrected mechanism:
the artifact is a CONDITIONING effect. Higher-order PHS are flatter / grow faster
at large r ⇒ more ill-conditioned local Gram matrix; the weight VALUES stay
accurate (polynomial augmentation), but the SENSITIVITY ∂W/∂x goes through the
inverse of that ill-conditioned system and is amplified. Genuine tension:
everything that improves RBF-FD forward accuracy worsens the shape-gradient noise.

### Consolidated diagnosis (what the artifact is)
∂W/∂x = node-position sensitivity of the differentiation weights. It (a) is zero
at operator order 0 and grows steeply with order (q=2 PDE ≫ q=1); (b) grows with
kernel order (conditioning, fights accuracy); (c) concentrates in interior
stencils at the curved boundary (Cartesian-vs-curved staircase); (d) is broadband
(no clean frequency separation); (e) is only mildly reduced by larger stencils.
No knob WITHIN the discretization removes it. Ruled out as the fix: combined
pullback, frequency-domain Riesz/Sobolev smoothing (partial), one-sided-boundary
(Hermite/ghost) stabilization, smoother kernels. Remaining candidates:
(1) placement-ensemble averaging (the artifact is placement-correlated; average
over jittered clouds — untested, attacks the mechanism); (2) contract the nodal
gradient onto a smooth, low-dim DESIGN space / design-velocity field (proven: the
ellipse optimizer worked; bounds design DOF to what the cloud resolves).

## Node-distribution + ensemble (2026-06-01): artifact is intrinsic & stable

`probe_node_ensemble.jl` (q=2 surrogate, boundary fixed, interior jittered):
```
Cartesian:  ‖g‖=91.4  hi/low=1.14
jittered:   ‖g‖=55–84 hi/low=1.4–2.0   (NOT cleaner; slightly worse hi/low)
ensemble K=6: ‖g‖=57.6   ‖g_ens‖/‖g_single‖=0.816  (pure noise would give ~0.41)
```
Both negative: (1) irregular/quasi-uniform clouds do not beat the Cartesian grid
— the artifact is intrinsic, not a probe-grid artifact; (2) placement-averaging
removes only ~18% — the artifact is a STABLE bias present in every cloud (and
anchored to the FIXED boundary node layout, which interior jitter doesn't move),
not placement-noise. Ensemble averaging is not a cure.

### CONDITIONING gap found (RBF.jl-1)
Reading the assembly: RBF.jl-1 builds the local saddle system with the
polynomial block evaluated at RAW GLOBAL coordinates (_poly_entry! = mon(a,
data[i]); _mono_rhs! = ℒmon(bmono, eval_point)) — no shift to the eval point, no
scaling. For x,y∈[-2,2], dx=0.05 the poly block (~x³≈8) and PHS block (r³≈1e-4)
differ ~1e5 in scale ⇒ ill-conditioned local systems (matches cond≈1e12–1e17).
Standard RBF-FD nondimensionalizes the stencil (shift to eval point, scale by
local h); RBF.jl-1 omits this. Local shift/scale is a similarity transform: the
weights are mathematically unchanged, only conditioning/roundoff improves, and a
FIXED scale adds no position-dependent derivative terms. Whether it cleans the
ARTIFACT depends on how much of ∂W/∂x is roundoff vs kernel-intrinsic true
sensitivity (probe_stencil_conditioning.jl sizes the headroom).

## Stencil conditioning (2026-06-01) + FINAL conclusion

`probe_stencil_conditioning.jl` (local saddle system, PHS3+poly3, k=35):
```
stencil @            cond GLOBAL  cond SHIFTED  cond SH+SCALE  ‖x_e‖
interior_near_hole   2.8e5        2.7e5         1.5e4          0.29
interior_far         1.2e7        3.0e5         1.2e4          2.76
outer_boundary       1.5e7        3.0e5         1.6e4          2.83
hole_boundary        1.1e6        1.0e6         9.1e4          0.40
```
Real conditioning gap confirmed: RBF.jl-1's global-coordinate polynomial block
makes cond scale with ‖x_e‖ (up to 1e7 far from origin); shift-to-eval recovers
~40×, shift+scale ~1e4 uniform. WORTH FIXING in RBF.jl-1 (at higher poly_deg /
larger domains it would reach 1e15 and wreck FORWARD accuracy). But it does NOT
cure the artifact: at cond≈1e7 roundoff is ~1e-9, while the artifact is O(1) of
the gradient energy. The artifact is the kernel-intrinsic TRUE sensitivity (a
similarity transform leaves the weights unchanged), not roundoff. (The doc's
earlier cond≈1e15 was the GLOBAL stiffness matrix, not the local stencils.)

### FINAL conclusion (Phase D L3 closed)
The high-frequency shape-gradient noise is ∂W/∂x — the node-position sensitivity
of the RBF-FD differentiation weights. It is intrinsic, real (FD-validated),
generic across physics, driven by operator derivative order (q=2 PDE ⇒ 3rd-order
kernel content, ~46× the q=1 artifact), worsened by smoother/higher kernels
(conditioning), anchored to the boundary node layout (stable bias), and
broadband. Every removal candidate has been tested and ruled out: combined
pullback, frequency Riesz/Sobolev smoothing, Hermite/ghost boundary
stabilization, smoother kernels, quasi-uniform/irregular clouds, placement-
ensemble averaging, local stencil scaling. None remove it, because the artifact
IS the true gradient of the discrete objective, which is genuinely hypersensitive
to high-frequency node motion.

⇒ There is no transformation of the per-node ℓ² gradient that removes the
artifact while preserving per-node design freedom. The robust, validated path is
to contract the gradient onto a smooth, low-dimensional design space (Fourier /
B-spline / reduced basis), bounded to the modes the cloud resolves above the
artifact floor (here ~m=2 plus the physical m=4 — a few clean DOF). That clean-
DOF budget is the fundamental limit. The ellipse optimizer (1 DOF) is the proof
of concept; multi-mode works only with mode count capped below the floor.
Separately: land local stencil shift/scale in RBF.jl-1 for forward-accuracy
robustness at higher poly_deg / larger domains.

## LSQ (2026-06-01): least-squares differentiation IS a cure — conclusion revised

`probe_lsq.jl` compares the q=2 ∂W/∂x artifact for PHS3+poly3 exact collocation
vs GMLS (poly_deg-3 least-squares, local coords) at increasing oversampling n/M:
```
                       acc(int)   ‖g‖    hi/low
PHS3 colloc k=35        5.2e-4    91.4   1.14   (signal ≈ noise)
GMLS n=12 (n/M=1.2)     3e14       —      —     BROKEN (under-determined, PᵀP singular)
GMLS n=25 (n/M=2.5)     5.9e-4    32.2   0.29
GMLS n=50 (n/M=5.0)     7.5e-4     9.2   0.31   (signal ≫ noise)
```
RESULT: LSQ over-determination drops ‖g‖ ~10× and flips hi/low 1.14→0.30 (high-
freq artifact suppressed to ~⅓ of the smooth signal) with NO loss of forward
accuracy. It is the construction, not stencil size: PHS collocation at k=140
only reached ‖g‖=25, hi/low=0.66; GMLS n=50 beats it at a smaller stencil
(3× ‖g‖, 2× hi/low), and at matched stencil GMLS n=25 vs PHS k=35 is 3×/4×
better. Mechanism confirmed: more constraints per weight ⇒ W(X) smoother in node
position ⇒ ∂W/∂x smoother. Dial is monotone above n/M≳2.5 (n≈M is rank-deficient)
— no sweet-spot tuning. GMLS uses local shifted coords, so it also sidesteps the
global-coordinate conditioning gap.

### REVISED conclusion (supersedes the "no cure" verdict above)
The earlier "no transformation removes the artifact" verdict was based on
collocation-only fixes. Least-squares / oversampled differentiation (GMLS, or
LSQ-RBF-FD à la Tominec–Larsson–Heryudono) DOES substantially reduce it: ~10× in
magnitude, ~4× in SNR, raising the clean-DOF floor. Hyperviscosity remains the
wrong tool here (forward-stability, not weight sensitivity; Probe B showed smooth
state doesn't help; its Δ^k term is higher-order ⇒ worse ∂W/∂x). Caveats: this is
the smooth-surrogate q=2 test (real shape gradient needs the full adjoint on LSQ
weights — a constant-factor cost) and GMLS-poly vs PHS+poly is a method change,
but the oversampling TREND isolates the LSQ effect. Next: build LSQ/oversampled
differentiation into the forward+adjoint shape gradient and re-run a multi-mode
optimizer to confirm the higher clean-DOF budget end-to-end.

## CORRECTION (2026-06-01): the LSQ "cure" was confounded by stencil size

The "REVISED conclusion" above over-claimed. A fairness test (probe_lsq_rbf.jl,
requested to remove the poly-vs-PHS confound) instead exposed a STENCIL-SIZE
confound: the GMLS result (n=50) was compared against PHS collocation at k=35.
PHS collocation at MATCHED k=50 already gives hi/low=0.25 (vs 1.14 at k=35), so
most of the apparent "flip 1.14→0.31" was bigger stencil/averaging, not LSQ. At
matched n=50: PHS-colloc (‖g‖=24.6, hi/low=0.25) vs GMLS (‖g‖=9.2, hi/low=0.31)
— GMLS ~2.7× smaller magnitude but comparable/worse hi/low. My attempt at a
clean PHS+poly LSQ failed (dropped the unisolvency constraint ⇒ square case
acc=10 garbage, LSQ cases worse than collocation). Cross-probe numbers are also
not fully consistent (k=50 vs the earlier k=70/140 are non-monotonic) ⇒ no single
number is trustworthy in isolation.

Reliable takeaways: (a) the artifact falls steeply with stencil/averaging size —
the dominant lever; (b) pure-polynomial LSQ (GMLS) avoids the RBF conditioning
roughness ⇒ smaller magnitude, but not a dramatic hi/low win at matched stencil;
(c) adding raw PHS columns to an LSQ fit REINTRODUCES the roughness. Net: LSQ is
NOT a confirmed silver bullet; "more averaging" (larger stencil and/or LSQ over a
larger support) is the real, modest lever. probe_lsq_fair.jl runs the stencil-
size-controlled comparison to quantify the residual LSQ effect cleanly. The
robust path remains a smooth low-dim design space (ellipse optimizer = proof).

## (2c) SETTLED (2026-06-01): LSQ helps beyond stencil size, modestly

probe_lsq_fair.jl sweeps stencil s for BOTH methods (same cloud/metric):
```
 s   | PHS colloc ‖g‖  hi/low | GMLS LSQ ‖g‖  hi/low | GMLS acc
 25  |   60.8    0.50         |  32.2   0.29        | 5.9e-4
 50  |   24.6    0.25         |   9.2   0.31        | 7.5e-4
 75  |   25.1    0.63         |   5.5   0.20        | 1.2e-3
100  |   25.1    0.54         |   4.5   0.07        | 1.9e-3
```
Verdict (between the over-claim and the deflation): LSQ helps BEYOND averaging
size. Evidence = PHS collocation PLATEAUS (‖g‖~25, hi/low~0.25–0.6 for s≥50, no
improvement) while GMLS keeps dropping (‖g‖ 9.2→4.5, hi/low 0.31→0.07). At
matched s, GMLS is 1.9–5.6× smaller in magnitude and clearly better on hi/low
for s≥75. So collocation cannot push the artifact below ~(25, 0.25) at any
stencil; LSQ can. BUT it's modest+conditional, not a silver bullet, and GMLS
forward accuracy degrades at large s (5.9e-4→1.9e-3) since poly_deg-3 can't fit a
large support. Sweet spot: moderate oversampling (s≈50, osr≈5): acc=7.5e-4,
‖g‖ 2.7× better than collocation, hi/low=0.31. Pushing further needs poly_deg↑
with s (re-invokes operator-order cost). Caveat: GMLS is poly-only; my PHS+poly
LSQ was buggy, so the poly-vs-PHS confound isn't fully removed — but the
collocation-plateau-vs-LSQ-improvement trend is the reliable signal. Next (1):
carry this matched-stencil comparison onto the real PDE shape gradient.

## (1) REAL gradient (2026-06-01): LSQ does NOT transfer; surrogate was not representative

probe_real_gradient_lsq.jl computes the REAL Dirichlet-thermal compliance shape
gradient by FD, forward solve via PHS collocation vs GMLS-LSQ, matched s=50:
```
  PHS3 collocation : ‖d‖=1.29e4   hi/low=3.80
  GMLS LSQ         : ‖d‖=2.16e6   hi/low=2.54
```
Key corrections to the surrogate-based optimism:
1. The ∂W/∂x surrogate (smooth η,f) UNDERESTIMATED the real noise: PHS surrogate
   hi/low≈0.25 at s=50, but the REAL gradient is hi/low=3.80 (matches the early
   thermal Probe A, 3.54). The surrogate is good for LOCALIZING ∂W/∂x but its
   spectrum is NOT representative of the real shape gradient.
2. Why: the real gradient runs through the SOLVE — dJ/dr ~ K⁻¹(dK/dr), and K⁻¹
   AMPLIFIES the weight-sensitivity in the modes K is ill-conditioned in (the
   boundary/Neumann rows). The surrogate has no solve, so it missed this. This
   RE-ELEVATES the Neumann/boundary-conditioning hypothesis for the REAL gradient.
3. LSQ does NOT help the real gradient (hi/low 3.80→2.54, still high; ‖d‖ 167×
   larger). Caveat: GMLS's own Neumann-row solve accuracy likely inflates the
   167×, so GMLS isn't fairly tested as a solver — but the robust fact (real
   hi/low≈3.8 ≫ surrogate, LSQ doesn't bring it down) holds.

NET: the LSQ route looked good on the surrogate but does not survive the real
gradient. The real shape-gradient noise is dominated by the SOLVE amplifying
boundary/Neumann weight-sensitivity through K⁻¹ — not the bare interior ∂W/∂x the
surrogate measured. Robust working path remains the smooth low-dim design space
(ellipse optimizer converged on the REAL problem). Methodological lesson: validate
on the real PDE gradient early; the smooth-field surrogate mis-attributes the
dominant noise.

## DEFINITIVE (2026-06-01): confound resolved — gradient noise is intrinsic

probe_hermite_real.jl: Hermite-stabilized Neumann gives real-gradient hi/low=3.54
(vs collocation 3.80) — barely changed, ‖d‖ 40× larger. probe_forward_accuracy.jl
then resolves the confound on a manufactured harmonic T=x²−y² (∇²T=0):
```
  PHS collocation : interior rel.err = 3.5e-14
  GMLS LSQ        : interior rel.err = 3.8e-10
  Hermite         : interior rel.err = 1.7e-10
```
ALL THREE forward solves are accurate (incl. Neumann boundaries). So the inflated
gradient magnitudes were NOT solver inaccuracy — the gradient verdict was fair:
all three discretizations solve accurately yet all have a high-frequency-
dominated real shape gradient (hi/low 2.54–3.80). Neither LSQ nor Hermite cures
it (LSQ is marginally best at 2.54, still far from usable).

Deep statement: an accurate forward solve does NOT imply a smooth shape gradient.
T≈K⁻¹b is accurate, but dT/dr = −K⁻¹(dK/dr)T runs through dK/dr (the high-freq
∂W/∂x), AMPLIFIED by K⁻¹. Accuracy of T (≈K⁻¹b for smooth b) is decoupled from
the smoothness of K⁻¹(dK/dr). That's why the ∂W/∂x surrogate underestimated (no
K⁻¹) and why no operator/BC variant (collocation/LSQ/Hermite/smoother kernel/
bigger stencil) removes it.

FINAL: the real per-node shape gradient is intrinsically high-frequency-dominated,
invariant to the discretization method, uncurable by any discretization or
boundary-stabilization lever tested. The robust, proven path is the smooth
low-dim design space — contracting the noisy nodal gradient onto a few smooth
modes averages the high-freq noise (ellipse optimizer converged on the REAL
problem). LSQ stays a marginal helper (lowest hi/low) for use atop a smooth
design space, not a standalone cure. Phase D L3 closed.

## Tikhonov-on-λ (2026-06-01): the dominant noise is the AMPLIFIED ADJOINT

probe_tikhonov_adjoint.jl smooths the global adjoint λ (Helmholtz, length c·h)
and recomputes g_l=−λᵀ(dC/dl)u (flux objective, s=50):
```
  c=0 (true λ)      ‖g‖=1.29e4  hi/low=3.80
  c=1 (len=h)       ‖g‖=1.58e4  hi/low=0.29   ‖Δλ‖/‖λ‖=1.41
  c=2 (len=2h)      ‖g‖=1.04e4  hi/low=1.19
  c=4 (len=4h)      ‖g‖=7.78e3  hi/low=3.61
  c=8,16            ‖g‖→1e7     hi/low→12     (smoother blows up)
```
DECOMPOSITION (new): smoothing λ at ONE node spacing collapses hi/low 3.80→0.29
(the stencil floor). So the DOMINANT real-gradient noise is the amplified global
adjoint λ (source a), NOT the stencil sensitivity (source b ≈ 0.29). Mechanism:
the flux observable's ∂J/∂u is a derivative (high-freq) ⇒ C⁻ᵀ amplifies it ⇒ λ is
mostly Nyquist junk (‖Δλ‖=141% at minimal smoothing).

Caveats: (1) the naive smoother (Helmholtz with the ill-conditioned RBF-FD
Laplacian) is FRAGILE — non-monotonic, narrow sweet spot at c=1, blows up for
larger c. (2) ‖Δλ‖=141% is aggressive ⇒ faithfulness of the c=1 descent
direction (low-mode fidelity vs true) is UNVERIFIED.

Implication / clean fix: the noise originates from a high-freq OBSERVABLE
sensitivity (∂J/∂u = pointwise derivative). Formulate the observable VARIATIONALLY
(divergence-theorem / smooth-test-function flux ∫∇T·∇w) ⇒ ∂J/∂u smooth ⇒ λ smooth
by construction ⇒ gradient at the ~0.3 floor with NO smoothing parameter. This is
the principled, parameter-free version of Tikhonov-on-λ. Next: (a) faithfulness
check (low-mode cosine of smoothed vs true gradient); (b) variational-observable
test. NOTE: this revises the earlier "uncurable" framing for the OBSERVABLE-driven
part of the noise — much of the real-gradient noise was the adjoint amplification
of a rough observable, which a smooth observable formulation removes parameter-free.

## CORRECTION (2026-06-01): the Tikhonov-on-λ "decomposition" was an artifact

probe_faithful_variational.jl tested (a) faithfulness of the c=1-smoothed gradient
and (b) a smooth observable. BOTH negative:
(a) cosine(g_smoothed_low, g_true_low)=0.51, low-mode amplitude ratio 3.41, and
    the smoothing INVENTED modes (m=3: a_true≈0 → a_smooth=-747). So the
    3.8→0.29 drop was the ILL-CONDITIONED RBF-FD-Laplacian smoother mangling the
    signal — smooth but UNFAITHFUL. The "dominant noise = amplified λ, removable"
    decomposition in the previous section is NOT established (retracted).
(b) smooth observable J=∫T gave hi/low=2.95 (≈ flux's 3.80), and ‖λ_s‖=2.3e3 vs
    ‖λ_flux‖=47 — because the interior-indicator ∂J/∂u has a sharp boundary jump
    (high-freq) that C⁻ᵀ amplifies. Not actually a smooth observable.

Caveat: both had implementation flaws (ill-conditioned smoother; non-smooth
indicator), so not an airtight refutation of Tikhonov-on-λ / variational-
observable. A clean redo would need a WELL-CONDITIONED smoother (normalized graph
Laplacian / spectral low-pass) and a GENUINELY smooth ∂J/∂u (no boundary jump).
But combined with the whole pattern (LSQ confound, no real-gradient transfer,
Hermite no help, Tikhonov unfaithful), the robust conclusion stands: the real
per-node shape gradient is intrinsically high-frequency-dominated; the proven
working path is the smooth low-dim design space (ellipse optimizer). Methodological
note: a smoothness metric (hi/low) alone is misleading — always pair it with a
FAITHFULNESS check (low-mode cosine vs the true gradient) before claiming a fix.

---

# UPDATE (2026-06-02): capped-mode optimizer built; the objective bias is a STALE-CLOUD artifact, not intrinsic

## Capped-Fourier optimizer, auto-calibrated from cloud geometry

`plate_with_hole_fourier_opt.jl` is the NEXT-STEP-fork-(2) optimizer: capped Fourier
design space + Sobolev metric + area constraint + Laplace interior morph
(transpose-corrected, FD-validated) + fixed-adjacency forward solve. The two design
knobs are no longer hand-set — they are DERIVED from the cloud (parameter-free):

```
ρ      = max neighbor distance (stencil radius)         ≈ 0.30 = 6·dx   here
m_cap  = ⌊π·r/ρ⌋   (stencil-Nyquist: wavelength ≥ 2ρ)   ⇒ m_cap = 2     here
sob_r  = ρ/r       (Sobolev length = stencil radius)    ≈ 1.06          here
```

At dx=0.05 / k=35 the clean-DOF budget is **m=2 only**. The hand-set `m=2:4`,
`sob_r=0.5` was UNSTABLE (m≥3 noise walks up 7×, C non-monotone — the file's own
header feared this). Calibrated `m=2` + morph gives clean monotone descent. (The
global-max ρ is set by outer-corner one-sided stencils; a hole-local ρ would give
cap≈4 — a real calibration sensitivity, resolved on the conservative/data-consistent
side since m≥3 demonstrably behaved as noise.)

## The objective-bias discovery — and its resolution

With the EXACT adjoint (FD 1e-6) and ZERO design-space noise, the binding limit
became **objective discretization bias**: on a single fixed/morphed cloud the
discrete-compliance optimum sits at a₂≈±0.07, NOT the true circle a₂=0, SIGN-BRACKETED:

```
fixed cloud (interior culled around the START ellipse, MORPH=false):  a₂* = +0.078
Laplace morph (MORPH=true, marches monotonically THROUGH the circle): a₂* = −0.065
```

Shallow-optimum amplified (C varies only ~2% over a₂∈[−0.07,+0.08], so the O(hᵖ)
objective error maps to O(0.07) in design space).

**RESOLUTION — it is a STALE-CLOUD artifact, NOT intrinsic.** `probe_remesh_unbiased.jl`
measures the m=2 design gradient at the circle on FRESH, clean, circle-culled clouds:

```
nominal (symmetric, freshly circle-culled) cloud:  dC/da₂ = −4.6e-8  ≈ 0
   (~1000× below the driving-gradient scale ~3e-5 — the circle IS the discrete optimum)

K=12 randomized clouds (interior offset + boundary phase, all high-quality):
   per-cloud ‖m2‖ = 4.7e-6     ‖mean vector‖ = 2.3e-6     ratio = 0.49
   ⇒ via slope dC/da₂≈3e-4·a₂:  residual a₂* ≈ 0.008,  cloud-noise floor ≈ ±0.015
```

So re-anchored remeshing debiases ~10× (a₂*: 0.07 → 0.008 ≈ circle). The SYMMETRIC
nominal cloud is EXACTLY zero, so the residual is a symmetry-breaking artifact of the
random offsets — reducible by remesher quality/symmetry/refinement. The earlier ±0.07
bias came ENTIRELY from optimizing on a cloud fitted to the WRONG (initial-ellipse)
geometry. Re-anchoring each design to a fresh clean cloud removes it.

## Three failure modes — assign the right cure to each (do NOT conflate them)

The plate-with-hole shape opt has THREE distinct obstacles:

| # | failure | stale-cloud artifact? | cure |
|---|---------|----------------------|------|
| A | descent-direction NOISE: per-node ℓ² gradient is Nyquist-dominated (hi/lo 3.5–3.8), m=2 signal 40× below | **NO** — present on a FRESH iter-0 cloud (probe_fourier_modes, test_fd_edge_noise) | design-space contraction (a few smooth modes averages it out) — the parametrization's essential job |
| B | cloud DEGRADATION: moving nodes / morphing distorts the cloud, ill-conditions fixed stencils (run #4 corruption) | yes | re-anchored remesh + quality monitoring |
| C | objective BIAS: a cloud fitted to a stale geometry has its discrete optimum off the true one (±0.07) | yes | re-anchored remesh (proven above) |

**Consequence for the OLD per-node route (design = the nodes):** periodic
re-initialization would have cured (B) and (C) but NOT (A). On a fresh cloud the
per-node gradient is STILL noise-dominated from iteration 1, so the boundary roughens
immediately (~40× faster than m=2 progress). Re-initialization helps the per-node route
ONLY if it SMOOTHS the boundary (low-pass) — which is mathematically the design-space
projection in disguise. So the PARAMETRIZATION does the essential work (A); remeshing is
the complementary cure for (B)+(C). The two are not substitutes.

## Two-front deformation-robust architecture (discrete adjoint + remeshing — NOT precluded)

Remeshing is precluded ONLY if you (i) differentiate THROUGH it, or (ii) demand exact
stationarity of one fixed discrete objective. Drop both:

- **Front 1 — morph (within a remesh interval):** smooth, differentiable boundary→interior
  extension; exact discrete adjoint through it (today's machinery). Quality indicators
  that matter: local stencil condition number (`probe_stencil_conditioning`),
  neighbor-distance CV (spacing uniformity), min boundary gap.
- **Front 2 — remesh (between intervals):** when quality degrades, regenerate a fresh
  high-quality cloud for the current design and RE-ANCHOR (reset reference + re-factorize
  the morph) so each remesh reads as a RE-INITIALIZATION, not a perturbed continuation.
  This is what removes (B)+(C).
- **Descend on the GRADIENT** (‖Jᵀg‖→0), not on the (remesh-jumping) C value.
  Compromises: ignore the remesh value-jump (bounded by discretization accuracy — noise,
  not signal); gradient-based stopping; convergence to the remesh-consistency floor
  (±0.015 in a₂ here, set by remesher quality).

New files (2026-06-02): `plate_with_hole_fourier_opt.jl` (calibrated capped-mode
optimizer, MORPH on, settling-step tried+rejected — it lingers in the morph-distorted
regime), `probe_remesh_unbiased.jl` (the stale-cloud-bias test).
