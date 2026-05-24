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
