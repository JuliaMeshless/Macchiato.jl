# Manual Adjoint for RBF-FD Shape Optimization

**Last updated**: 2026-05-23 (Phase D L2 + Helmholtz sensitivity filter)

## Status Summary

| Phase | Description | Status |
|-------|-------------|--------|
| Phase A | Dirichlet-only manual adjoint | **Validated** — `rel_err = 2.3e-11` |
| Phase B | Mixed Dirichlet + traction BCs | **Validated** — `rel_err = 1.4e-07` |
| Phase C | Clean interface + direct RBF backward | **Complete** — 155x speedup |
| Phase 3 | Cantilever compliance minimization (live loads, FD-validated) | **Done** — `rel_err = 1.4e-07`, 39% reduction with corners frozen |
| Phase D L1 | `∂b/∂pts` for shape-dependent dead loads | **Done** — `extract_load_sensitivities!`, FD-validated |
| Phase D L2 | Differentiable polyline normals (corners unfrozen) | **Done** — `rel_err = 1.4e-07`, 52% reduction with 3-point filter |
| Phase D L2-follower | Live normals as input to t(n) for follower loads | Not started |
| Phase D reg (Helmholtz) | PDE sensitivity filter on the boundary loop | **Done** — 46.6% reduction at r=1, smoother shape; 52% (3-point) was noise exploitation |
| Phase 1 (traced) | Monolithic Mooncake trace | **Abandoned** — 5+ min LLVM compile for 25 pts |
| Phase 2 (traced) | `gradient(sim, loss; wrt=:pts)` API | Implemented but superseded by manual adjoint |

### What's implemented

- **`src/optimization/manual_adjoint.jl`**: `extract_weight_sensitivities_elasticity!`, `allocate_weight_gradients`, `assemble_elasticity_from_weights`, `TractionLayout` + `build_traction_layout`, `apply_traction!`, `extract_neumann_sensitivities!`, **`extract_load_sensitivities!`** (Phase D L1 — ηᵀ·∂b/∂pts for shape-dependent dead loads)
- **`ext/MacchiatoMooncakeExt.jl`**: unified `shape_gradient` with kwargs for Dirichlet-only or mixed BCs; new **`traction_jacobians`** kwarg accepts per-Neumann-point 2×2 Jacobians and adds the ηᵀ·∂b/∂pts term automatically. Backwards-compatible (default `nothing` ⇒ frozen-load behavior).
- **`examples/shape_optimization_manual_adjoint_phaseA.jl`**: Dirichlet-only AD-vs-FD validation
- **`examples/shape_optimization_manual_adjoint_phaseB.jl`**: Mixed-BC AD-vs-FD validation (frozen loads)
- **`examples/shape_optimization_phase3_cantilever.jl`**: end-to-end shape opt loop with live tractions + boundary-loop sensitivity filter. Iter-0 FD validation included (rel_err 1.4e-7).

### Phase C status — COMPLETE

1. **Unify orchestrators** — `shape_gradient_dirichlet` + `shape_gradient_mixed_bc` → single `shape_gradient` with kwargs — **DONE**
2. **Eliminate `build_rrule`** — Step 4 now uses `_pullback_weights!` from RadialBasisFunctions.jl directly. No Mooncake tracing, no LLVM compilation. — **DONE**
3. **Warmup** — updated to cover new `_build_weights_and_cache` + `_pullback_weights!` paths — **DONE**
4. **Validated** — Phase A: `rel_err = 2.3e-11`, Phase B: `rel_err = 1.4e-07` — **DONE**

### Performance (45-pt grid, k=35)

| Metric | Old (build_rrule) | New (direct RBF) | Speedup |
|--------|-------------------|-------------------|---------|
| Phase A cold | 357 s | 2.3 s | **155x** |
| Phase A warm | 25 ms | 6.6 ms | 3.8x |
| Phase B cold | 404 s | 2.6 s | **155x** |
| Phase B warm | 12 ms | 9.6 ms | 1.2x |

Cold time dominated by Julia compilation of new RBF functions + stencil solves.
Warm time is pure computation (sub-10ms per gradient evaluation).

### What was removed

- `_make_weight_closure`, `build_weight_rule`, `apply_weight_rule` — deleted
- `_make_weight_closure_subset`, `build_weight_rule_subset`, `apply_weight_rule_subset` — deleted
- `_pts_from_flat` Mooncake primitive registration — deleted (function kept as plain Julia)
- `weight_rules` parameter from `shape_gradient` — deleted
- ~200 lines of Mooncake glue code removed from ext file

### What was added to RadialBasisFunctions.jl

- `_build_weights_and_cache(ℒ, data, eval_points, adjl, basis)` → `(W, cache)` — forward with cache
- `_pullback_weights!(Δdata, Δeval, ΔW_nzval, W, cache, ...)` — backward pass (pure Julia, no Mooncake)

### Key architectural decisions (settled)

- **Mooncake ONLY at the stencil level**: individual `_build_weights` calls, each a single primitive. The global assembly/solve/adjoint is manual linear algebra.
- **`_pts_from_flat` as Mooncake primitive**: collapses ~3N traced scalar ops into one trace node — the dominant lever behind the 339s → seconds speedup.
- **Single binding for data/eval_points**: closures use `q = _pts_from_flat(p); _build_weights(op, q, q, ...)` matching the tested rrule code path.
- **Per-DOF gating** (not per-point AND): `active[i]` and `active[i+N]` gate independently, supporting mixed DOF-type BCs.
- **`interior_rows` gate**: separates interior extraction (d2x/d2y/d2xy) from Neumann extraction (dx/dy), avoiding spurious ΔW contributions.
- **Level 1 (frozen normals)**: Neumann coefficients use reference-configuration normals. Level 2 (differentiable normals) can be added later via a `∂coeff/∂pts` term in Step 3.
- **`∂b/∂pts` is split in two** — the implicit `ηᵀ·∂b/∂pts` piece (the "η-side", part of the `∂L/∂u · du/dpts` chain) lives inside `shape_gradient`. The explicit `∂L/∂b · ∂b/∂pts` piece (the "u-side", e.g. `uᵀ·∂b/∂pts` for compliance) lives **outside** `shape_gradient` because the function doesn't know the loss's b-dependence structure. The user reuses `extract_load_sensitivities!` with their own `∂L/∂b` coefficient vector. For self-adjoint elasticity these two pieces sum to the textbook `2·uᵀ·∂b/∂pts`; for non-self-adjoint RBF-FD operators they're distinct contributions and BOTH are required for FD agreement.
- **Neumann classification must be index-based, not spatial.** Predicates like `is_right(p) = p[1] ≈ L` (default `isapprox` tol ~1.5e-8) flip false under any FD perturbation larger than that — corrupting both the reference gradient and any live-load update. Cache `[is_right(points0[i]) for i in neumann_ids]` once at init and look up by local Neumann index. The Phase 3 example demonstrates this pattern (`neumann_is_right_idx`).

---

## Motivation

Mooncake's `build_rrule` on the full loss function (pts → weights → assembly →
solve → loss) generates a monolithic trace whose IR is too large for LLVM to
compile in reasonable time. Even with only 25 points, `build_rrule` times out
after 3+ minutes of JIT compilation. Eliminating sparse-sparse matrix operations
(direct CSC construction) fixes correctness but not compilation time — the
element-wise vector ops and fill loops still generate too many trace nodes.

**The fundamental issue**: Mooncake traces every scalar operation between
primitives. The CSC assembly alone involves iterating over ~2000 nzvals, each
generating multiple trace nodes. This does not scale.

**The solution**: Decompose the gradient computation into independent pieces,
each fast to compile, and combine them manually. Use Mooncake rrules **only**
at the stencil level (where they are efficient and already compiled), and
handle the global composition with manual linear algebra.

**Verified unusable (2026-05-18)**: brute-force trace on the 25-pt Neumann
test took 5min 51s of LLVM compile *and* failed on a `BoundsError` in the
IFT pullback (sparsity mismatch when `batch_overwrite_sparse_rows!` adds
structural nonzeros to A while `ΔA.nzval` keeps its pre-Neumann size). Both
the compile time and the structural-tangent mismatch are fundamental to the
monolithic-trace approach — settled rationale for this plan.

---

## Scope: What This Adjoint Computes

This adjoint computes `∂L/∂pts` — the gradient of the loss with respect to
**every point coordinate** in the cloud (boundary and interior). It is the
discrete-AD analog of the classical Hadamard shape derivative; the two differ
in framing:

| Framework                                  | Output                                |
|--------------------------------------------|---------------------------------------|
| Continuous shape calculus (Hadamard)       | Boundary integral; interior never appears |
| Discretize-then-differentiate (this plan)  | Full `∂L/∂pts` vector; boundary is a subset |

In shape optimization the design-relevant slice is `∂L/∂pts_b`. The interior
slice `∂L/∂pts_i` is **not** discarded by this adjoint: it is the input to
the next chain-rule stage (mesh-morphing adjoint), which projects it back to
the boundary via `(∂pts_i/∂pts_b)ᵀ · ∂L/∂pts_i`. With a differentiable
morphing map (RBF interpolation as proposed in
`plan_shape_optimization_pipeline.md` §2), both slices contribute to the
final design gradient. With non-differentiable cloud regeneration, the
interior slice is dropped downstream — but this adjoint still produces it.

What this means at the implementation level:
- Steps 1–5 below always produce the full `Δpts` vector. There is no cheap
  path to "boundary entries only" in the discrete framework — the global
  adjoint solve and stencil pullbacks couple all DOFs by construction. The
  cost is fixed at ~2× forward solve.
- Downstream layers (mesh-morphing adjoint, boundary parameterization) select
  / combine entries. Those are out of scope for this document.
- Phase A validation must compare AD vs FD on the **full** `Δpts` vector
  (not just the boundary entries) — FD operates on all coordinates too.

---

## Architecture Overview

The full gradient `∂L/∂pts` is computed in five steps:

```
Step 1: Forward pass           assemble A, solve A u = b
Step 2: Adjoint solve          solve Aᵀ η = ∂L/∂u   →  ΔA = -η uᵀ
Step 3: Extract ΔW             ΔA → ΔW_k for each weight matrix (manual)
Step 4: Local sensitivity      each ΔW_k → Δpts_k via _build_weights rrule
Step 5: Accumulate             Δpts = Σ_k Δpts_k + direct ∂L/∂pts terms
```

Steps 1-3 and 5 are manual linear algebra (no AD). Step 4 uses Mooncake rrules
on **individual** `_build_weights` calls — each call is a single primitive with
a pre-compiled rrule, so `build_rrule` on it completes in seconds, not minutes.

### Diagram

```
pts ─┬─ _build_weights(∂²/∂x²) → W_d2x ─┐
     ├─ _build_weights(∂²/∂y²) → W_d2y ─┤
     ├─ _build_weights(∂²/∂x∂y)→ W_d2xy ┤  assemble  → A ─ solve → u → L
     ├─ _build_weights(∂/∂x)  → W_dx  ──┤              ↑            │
     └─ _build_weights(∂/∂y)  → W_dy  ──┘              η            │
        ↑   ↑   ↑   ↑   ↑                             │            │
        │   │   │   │   │                             │     ∂L/∂u = 2u
        │   │   │   │   │   (Step 4: local rrules)     │            │
        │   │   │   │   │                             │            │
        └───┴───┴───┴───┘                              │            │
        ΔW_d2x, ΔW_d2y, ...   ←────────────────────────┘            │
        (Step 3: extract from ΔA)       (Step 2: η = A⁻ᵀ ∂L/∂u)    │
                                                                     │
                              ΔA = -η uᵀ  ←──────────────────────────┘
```

The critical difference from the global-trace approach: Mooncake never sees the
assembly step or the global solve. It only sees individual `_build_weights`
calls, each of which is a single primitive with a known rrule.

---

## Step-by-Step Details

### Step 1 — Forward Pass

Unchanged from current code. Assemble A via `make_system_differentiable`
(the direct-CSC version), apply BCs, solve for u.

```julia
A = make_system_differentiable(model, pts_flat, N, adjl, basis, λstar, μ)
b = zeros(2N)
apply_dirichlet!(A, b, dirichlet_dofs, dirichlet_vals)
apply_neumann!(A, b, ...)  # if applicable
u = A \ b
L = loss(u, pts_flat)
```

### Step 2 — Adjoint Solve (IFT)

Given `∂L/∂u` (computed analytically or via a tiny AD call on the scalar loss
function), solve the adjoint system and form ΔA:

```julia
η = A' \ (∂L/∂u)             # single sparse linear solve
ΔA = -η * u'                 # rank-1 update, but we only need entries at
                             # positions matching the weight-matrix sparsity pattern
```

For `active_dofs` (Dirichlet rows excluded), zero out ΔA at inactive rows
(same logic as `PDESolveIFT` rrule).

**Cost**: one sparse LU factorization (already computed in forward pass — reuse
the factors) plus one triangular solve. O(nnz).

### Step 3 — Extract ΔW from ΔA

The system matrix A is assembled from weight matrices W_k with known
coefficient structure. For 2D elasticity:

```
A = [ A₁₁  A₁₂ ]    where  A₁₁ = c₁ W_d2x + c₂ W_d2y
    [ A₂₁  A₂₂ ]          A₁₂ = c₃ W_d2xy
                           A₂₁ = A₁₂
                           A₂₂ = c₂ W_d2x + c₁ W_d2y
    c₁ = λstar + 2μ,  c₂ = μ,  c₃ = λstar + μ
```

By the chain rule, for any weight-matrix entry `(i, j)`:

```
∂L/∂W_d2x[i,j] = c₁ · ΔA₁₁[i,j] + c₂ · ΔA₂₂[i,j]
∂L/∂W_d2y[i,j] = c₂ · ΔA₁₁[i,j] + c₁ · ΔA₂₂[i,j]
∂L/∂W_d2xy[i,j] = c₃ · ΔA₁₂[i,j] + c₃ · ΔA₂₁[i,j]
```

For first-order operators (Neumann BCs):

```
∂L/∂W_dx[i,j] = contribution from Neumann row assembly  (see Step 3b)
∂L/∂W_dy[i,j] = contribution from Neumann row assembly
```

This step is **pure linear algebra** — no AD needed. It extracts entries from
ΔA and scales by known coefficients. The sparsity pattern of each ΔW_k matches
the corresponding weight matrix.

**For Neumann rows** (overwritten by `batch_overwrite_sparse_rows!`):
The row of A at a Neumann point is:

```
A[row, a_col[p]] = Σ_m coeffs[p,m] * W_m[weight_row, w_col[p]]
```

So the gradient contribution to W_m is:

```
ΔW_m[weight_row, w_col[p]] += coeffs[p,m] * ΔA[row, a_col[p]]
```

This is exactly what the `batch_overwrite_sparse_rows!` rrule's pullback
computes. We can either:
- (a) Call that rrule independently, or
- (b) Implement the same coefficient-transpose logic manually (trivial)

### Step 4 — Local Sensitivity via `_build_weights` rrule

For each weight matrix W_k with nonzero ΔW_k, propagate to point coordinates.
Each call is independent:

```julia
# Build a rule for each operator individually (fast — single primitive).
# FIX-3: the closure builds `pts_from_flat(p)` twice. The `_build_weights`
# rrule was tested with `data === eval_points`. Two distinct SVector arrays
# both depending on p should sum correctly through Mooncake, but if Phase A
# AD/FD shows a 2× error, use a single binding: `q = pts_from_flat(p);
# _build_weights(op, q, q, adjl, basis)`. See Corrections §3.
rule_d2x  = Mooncake.build_rrule(
    p -> _build_weights(Partial(2,1), pts_from_flat(p), pts_from_flat(p), adjl, basis),
    pts_flat)
rule_d2y  = Mooncake.build_rrule(
    p -> _build_weights(Partial(2,2), pts_from_flat(p), pts_from_flat(p), adjl, basis),
    pts_flat)
# ... etc.

# Forward + backward for each weight matrix (pseudocode — actual Mooncake API
# requires seeding the output tangent via CoDual manipulation, not a bare
# `pullback(rule, Δoutput)` call).
Δpts_d2x  = pullback(rule_d2x, ΔW_d2x)
Δpts_d2y  = pullback(rule_d2y, ΔW_d2y)
# ... etc.
```

**Why this is fast**: each `build_rrule` traces through a single primitive call
(plus the `pts_from_flat` conversion — 25 SVector constructions). The trace is
shallow (~50 scalar ops + one primitive). LLVM compiles this in seconds.

**Why this is correct**: the `_build_weights` rrule in RadialBasisFunctions.jl
already implements the exact stencil-level adjoint from the differentiation
document (solving local interpolation systems and propagating through kernel
derivatives). It is already tested and validated.

**Caching**: these rules can be built once and reused across gradient
evaluations. The sysimage warmup already exercises all five operators, so
the rules are pre-compiled after the first build.

### Step 5 — Accumulate and Reshape

```julia
Δpts_flat = Δpts_d2x + Δpts_d2y + Δpts_d2xy + Δpts_dx + Δpts_dy
# Reshape to Vector{SVector{2, Float64}} for the optimizer
Δpts = [SVector{2}(Δpts_flat[2i-1], Δpts_flat[2i]) for i in 1:N]
```

If the loss depends directly on pts (e.g., perimeter regularization), add
`∂L/∂pts` directly (computed analytically or via a tiny ForwardDiff call).

---

## Generalization to Other Physics

The five-step structure is **PDE-agnostic**. Only Step 3 (coefficient extraction)
changes with the physics. Here is how different model classes map to this
framework:

### Scalar PDE (heat equation, Poisson)

- **Operator**: L = ∇² (Laplacian)
- **Weight matrices**: 1 (W_d2 = ∂²/∂x² + ∂²/∂y², or separate W_d2x, W_d2y)
- **Variables**: 1 (temperature T)
- **Assembly**: A = W_d2x + W_d2y (N×N, single block)
- **Step 3 extraction**: ΔW_d2x = ΔA, ΔW_d2y = ΔA (trivial)

### Stokes Flow (velocity-pressure)

- **Operator**: μ∇²u - ∇p = f, ∇·u = 0
- **Weight matrices**: W_d2x, W_d2y (viscous), W_dx, W_dy (pressure gradient)
- **Variables**: 3 in 2D (v_x, v_y, p)
- **Assembly**: Block 3×3 system with different operators per block
- **Step 3 extraction**: Each block of ΔA maps to its contributing W_k with
  appropriate coefficients

### Linear Elasticity (3D)

- **Same structure as 2D** but with 3 displacement variables (u_x, u_y, u_z)
- More weight matrices (∂²/∂x², ∂²/∂y², ∂²/∂z², ∂²/∂x∂y, ∂²/∂x∂z, ∂²/∂y∂z)
- Block 3×3 system, same coefficient structure as 2D
- **Step 3 extraction**: identical logic, just more blocks

### Nonlinear PDEs (e.g., hyperelasticity, Navier-Stokes)

- The system is nonlinear: F(u) = 0
- Newton's method linearizes: J(u_k) Δu = -F(u_k)
- The adjoint uses J^T (the Jacobian transpose)
- **Key difference**: J depends on u, so ∂J/∂pts includes ∂J/∂u · ∂u/∂pts terms
- The stencil-level logic is unchanged — only the coefficient extraction in
  Step 3 needs to account for state-dependent coefficients

### Time-Dependent PDEs

- Discretize in time: u^{n+1} = G(u^n, pts)
- Adjoint runs backward in time: λ^n = (∂G/∂u)^T λ^{n+1} + ∂L/∂u^n
- Each time step's ∂G/∂u is assembled from weight matrices like the steady case
- **Key difference**: need to store intermediate states (checkpointing) or
  recompute them during the backward pass

### What Stays the Same

Across all these cases:

| Component | Dependency | Changes with physics? |
|-----------|-----------|----------------------|
| Step 1 (forward solve) | PDE model | Yes — different operators, BCs |
| Step 2 (adjoint solve) | System matrix A | No — always Aᵀ η = ∂L/∂u |
| Step 3 (extract ΔW) | Assembly coefficients | **Yes — only this step** |
| Step 4 (local rrule) | RBF kernel + operator | No — `_build_weights` handles this |
| Step 5 (accumulate) | Nothing | No |

**Only Step 3 is physics-dependent.** It can be implemented as a dispatchable
function `extract_weight_sensitivities!(ΔW_list, ΔA, model, u, ...)` that each
model type implements.

---

## Efficient Implementation of Step 3

The naive approach — compute ΔA = -η uᵀ as a full sparse matrix — is wasteful:
ΔA has the same sparsity pattern as A, which is 4× the size of a single weight
matrix. Most entries are never used.

Instead, compute ΔW_k **directly** from η and u without forming ΔA:

```julia
# For elasticity, ΔW_d2x[i,j] = c₁·ΔA₁₁[i,j] + c₂·ΔA₂₂[i,j]
# where ΔA₁₁[i,j] = -η[i] * u[j]  (for active row i)
# So ΔW_d2x[i,j] = -(c₁·η[i]·u[j] + c₂·η[i+N]·u[j+N])

function extract_weight_sensitivities!(
    ΔW_list::Vector{SparseMatrixCSC{Float64,Int}},
    η::Vector{Float64},
    u::Vector{Float64},
    active::BitVector,
    model::LinearElasticity,
    λstar::Float64,
    μ::Float64,
)
    N = length(u) ÷ 2
    c1, c2, c3 = λstar + 2μ, μ, λstar + μ

    ΔW = ΔW_list  # [ΔW_d2x, ΔW_d2y, ΔW_d2xy]

    for j in 1:N, idx in nzrange(W_template, j)
        i = W_template.rowval[idx]

        # FIX-1, FIX-2: see "Corrections (review pass)" at end of document.
        # The AND below is too strict (per-DOF mask), and this loop must
        # exclude Neumann rows (which were assembled from W_dx/W_dy, not
        # W_d2x/y/xy). Phase A: both issues are moot (full clamping, no
        # Neumann). Phase B: gate each term independently and skip Neumann
        # rows here.

        # Only active rows contribute (Dirichlet rows are identity — zero gradient)
        if active[i] && active[i + N]
            ηi_x, ηi_y = η[i], η[i + N]
            uj_x, uj_y = u[j], u[j + N]

            ΔW[1].nzval[idx] = -(c1 * ηi_x * uj_x + c2 * ηi_y * uj_y)
            ΔW[2].nzval[idx] = -(c2 * ηi_x * uj_x + c1 * ηi_y * uj_y)
            ΔW[3].nzval[idx] = -(c3 * ηi_x * uj_y + c3 * ηi_y * uj_x)
        end
    end

    return nothing
end
```

This computes ΔW_k in **O(nnz)** time with a single pass over the sparsity
pattern. No sparse-sparse operations, no temporary matrices.

The `W_template` is any of W_d2x/W_d2y/W_d2xy (they all share the same pattern).
We can store it from the forward pass.

### Neumann Contribution

For Neumann rows, the contribution to ΔW comes from the `batch_overwrite_sparse_rows!`
coefficient structure. This can be computed with a second pass:

```julia
for k in 1:n_neumann_rows
    i_global = neumann_rows[k]
    wr = weight_rows[k]
    for p in col_ptr[k]:(col_ptr[k+1] - 1)
        a_col = a_cols[p]
        w_col = w_cols[p]
        Δa = -η[i_global] * u[a_col]   # ΔA[i_global, a_col]
        for m in 1:M
            c = coeffs[(p - 1) * M + m]
            idx_w = _find_nzval(weights[m], wr, w_col)
            if idx_w > 0
                ΔW_list[pde_m + m].nzval[idx_w] += c * Δa
            end
        end
    end
end
```

---

## Implementation Plan

### Phase A: Dirichlet-only gradient — **DONE** (`6c64cb5`)

1. **`extract_weight_sensitivities_elasticity!`** — computes ΔW_k from η, u, active
   via closed-form coefficient map. Per-DOF gated. O(nnz).
2. **`allocate_weight_gradients`** — pre-allocate ΔW matrices sharing W_template's pattern
3. **`assemble_elasticity_from_weights`** — forward assembly outside Mooncake trace
4. **`build_weight_rule` / `apply_weight_rule`** — per-operator rule build + pullback
5. **`shape_gradient_dirichlet`** — full Steps 1-5 orchestrator
6. **`_pts_from_flat`** Mooncake primitive — collapses ~3N traced scalar ops
7. **Validation script**: `examples/shape_optimization_manual_adjoint_phaseA.jl`
   — **not yet run**; expected rel_err ~1e-10, compile < 10s

### Phase B: Neumann (traction) support — **DONE** (`3db5380`)

1. **`TractionLayout` + `build_traction_layout`** — pre-computed coefficient bundle
   for traction BC assembly and its adjoint
2. **`apply_traction!`** — forward pass, non-traced
3. **`extract_neumann_sensitivities!`** — coefficient-transpose backward pass
4. **`allocate_neumann_weight_gradients`** — ΔW_dx, ΔW_dy allocation
5. **`build_weight_rule_subset` / `apply_weight_rule_subset`** — for operators
   evaluated on a point subset (Neumann rows)
6. **`shape_gradient_mixed_bc`** — full Phase B orchestrator (Steps 1-5 with
   both interior and Neumann extraction loops)
7. **Validation script**: `examples/shape_optimization_manual_adjoint_phaseB.jl`
   — **not yet run**; expected rel_err < 1e-3

### Phase C: Clean interface + sysimage — **IN PROGRESS**

**C1. Reduce duplication** — The two orchestrators (`shape_gradient_dirichlet`,
    `shape_gradient_mixed_bc`) share ~80% of their code. Unify into a single
    `shape_gradient` that inspects the BC configuration and dispatches accordingly.
    The split between interior extraction (d2x/d2y/d2xy) and Neumann extraction
    (dx/dy) can be driven by whether a `TractionLayout` is provided (or is empty).

**C2. Warmup augmentation** — Add manual adjoint paths to `warmup.jl`:
    - `_pts_from_flat` forward + backward
    - `build_weight_rule` for all 5 operators (full + subset)
    - `apply_weight_rule` / `apply_weight_rule_subset` for each
    - `shape_gradient_dirichlet` full call (25-pt problem)
    - `shape_gradient_mixed_bc` full call (25-pt problem)
    Rebuild sysimage after: `julia ~/julia_sysimages/shape_opt/build.jl` (~10 min).

**C3. Validate Phase A and Phase B** — Run the two validation scripts under `jlrun`.
    Confirm `‖grad_AD - grad_FD‖ / ‖grad_FD‖ < 1e-3` for both. Fix any failures.

**C4. Dispatchable `extract_weight_sensitivities(model, ...)`** — Currently the
    elasticity extraction is hardcoded. Add a dispatchable function that each model
    type implements, keeping the five-step framework PDE-agnostic.

**C5. Scale test** — Run on 100+ point problems, measure compile time and gradient
    accuracy.

### Phase 3: Cantilever Compliance Minimization — **DONE**

First end-to-end shape-opt loop on the pipeline. Lives in
`examples/shape_optimization_phase3_cantilever.jl` (~500 lines, self-contained).

#### Setup

```
Domain:  [0, L] × [-D, D]   (rectangle, N = 9×5 = 45 points)
BCs:     left edge clamped  (Dirichlet: u = v = 0)
         right edge loaded  (Neumann: parabolic shear, t_y(y) = P·(D²-y²)/(2I))
         top/bottom free    (Neumann: zero traction)
Loss:    L = compliance + ρ·(V - V₀)²   (two-sided area penalty, ρ = 1e3)
Design:  y-coords of top, bottom, right boundary points
         minus the two right corners (frozen — see "design-mask choices" below)
```

#### Result (40 iters, k=35, ρ_pen=1e3)

| Quantity | Reference | Optimized | Δ |
|----------|-----------|-----------|---|
| Compliance bᵀu | 71.80 | 43.74 | **−39.1%** |
| Area | 16.000 | 15.998 | −0.02% |
| Warm gradient time | — | 9 ms | — |
| AD/FD gradient rel_err (iter 0) | — | **1.4e-7** | — |

Success criteria pass: ≥5 iters with no NaN, monotone compliance (Armijo-enforced
to floating-point tolerance), area within 2%, ≥20% compliance reduction, warm grad
<20 ms.

#### What this required beyond Phase C

Three things, none of which were obvious from the original Phase 3 sketch:

1. **`∂b/∂pts` in the gradient pipeline** — the original sketch assumed
   `∂L/∂u = b` was sufficient and that `b` could be frozen. For shape-dependent
   tractions (parabolic shear depends on the y-coord of where it's applied),
   that's the **dead-load** approximation and gives the wrong gradient. Fixed
   by `extract_load_sensitivities!` + the new `traction_jacobians` kwarg on
   `shape_gradient`. See "η-side vs u-side" decision above.
2. **Sensitivity filtering** — unfiltered, the discrete loss landscape has
   a Nyquist-mode checkerboard that the optimizer exploits within ~14 iters,
   producing physically absurd shapes (alternating outward/inward boundary
   perturbations). Suppressed by a single α=0.25 3-point average along the
   CCW boundary loop, with `free_mask` respect (frozen DOFs neither donate
   nor receive averaged values). Lives inline in the Phase 3 example —
   simple enough that promoting it to source isn't worth the API surface yet.
3. **Index-based Neumann classification.** See key decision above. Spatial
   predicates broke the FD reference until fixed.

#### Architecture of the loop

```julia
# Init (once)
adjl            = find_neighbors(points0, k)
traction_layout = build_traction_layout(...)   # frozen normals, frozen connectivity
neumann_is_right_idx = [is_right(points0[i]) for i in neumann_idx]   # cached classification

# Iteration body
for iter in 1:max_iter
    # 1. Update b_vals from current shape (dead-load values follow the geometry)
    update_traction_loads!(traction_layout, pts_flat, neumann_idx, traction_at)

    # 2. Build per-Neumann 2×2 Jacobians at current shape
    jacobians = build_traction_jacobians(pts_flat, neumann_idx, traction_jacobian_at)

    # 3. Adjoint gradient — η-side ∂b/∂pts is added automatically because
    #    we passed traction_jacobians.
    b_now  = build_b_from_layout(traction_layout, dirichlet_dofs, dirichlet_vals, N)
    result = shape_gradient(pts_flat, ...; ∂L_∂u = _ -> b_now,
                                          traction_jacobians = jacobians, ...)

    # 4. u-side ∂b/∂pts (∂L/∂b = u for compliance) — extract_load_sensitivities!
    #    reused with u in place of η.
    extract_load_sensitivities!(grad, traction_layout, result.u, neumann_idx, jacobians)

    # 5. Add volume penalty gradient
    grad += 2 * ρ_pen * (V - V_target) * dV/dpts

    # 6. Sensitivity filter along boundary loop, then mask to free DOFs
    filter_along_boundary!(grad_smooth, grad, boundary_loop, free_mask; α=0.25)
    grad_smooth .*= free_mask

    # 7. Armijo backtracking line search; update pts_flat
end
```

The full path takes ~9 ms warm per gradient evaluation. FD reference at iter 0
takes ~7.5 s (90 design vars × 4 evals × ~20 ms each).

#### Design-mask choices (load-bearing decisions)

- **x-coords frozen everywhere.** Only y-coords of boundary points can move.
  Keeps the discretization x-locations fixed so the parabolic profile
  interpretation stays clean.
- **Right corners (L, ±D) frozen.** Their reference normals `(1, 0)` are
  baked into `traction_layout`. The moment a corner moves outward into the
  geometrically "top-right" region, that normal misrepresents the surface.
  Live normals are Phase D L2. Freezing the corners is the honest workaround
  until L2 lands.
- **Interior + left edge frozen.** Standard.

#### Known limitations of the current result

- The optimized shape is **not** a clean Michell taper. It shows real material
  redistribution (middle dimples inward, corners stay frozen) plus some
  residual bulging near x=7 (one vertex inboard of the frozen right corner).
  That residual is a true minimum of the discrete loss landscape — the
  gradient is correct (FD-validated) and the filter kills the Nyquist mode,
  but longer-wavelength modes near a frozen-DOF transition aren't suppressed
  by a 3-point average.
- The volume penalty oscillates the area at ~±0.05 around target, causing
  alternating sign in `2ρ(V-V₀)·∇V` and a sawtooth in `‖∇L‖` after iter ~13.
  Cosmetic; doesn't block convergence.

#### Promotion candidates (deferred — judgment call)

- `update_traction_loads!`, `build_traction_jacobians`, and the index-based
  classification pattern are likely to be reused for any dead-load problem.
  Currently live in the Phase 3 example; promote to `manual_adjoint.jl` once
  a second user appears.
- `filter_along_boundary!` is currently inline. Promote once a multi-pass /
  Helmholtz-PDE variant is needed.

### Phase D: Generalize beyond linear elasticity / dead loads

- **L1 (done):** `extract_load_sensitivities!` + `traction_jacobians` kwarg.
  Supports any dead-load problem where each Neumann row's `b` depends on its
  own point coordinates via a closed-form Jacobian.
- **L2 (done):** differentiable polyline normals via `polyline_normals`,
  `NormalJacobian`, `extract_normal_sensitivities!`, `update_traction_coeffs!`,
  and the `normal_jacobians` kwarg on `shape_gradient`. The chord-based
  formula `n_i = R(-π/2)·normalize(p_{next} - p_{prev})` gives smooth normals
  at edge midpoints (equal to the canonical edge normal) and length-weighted
  normals at corners. For the cantilever, corners use the chord-formula
  normal, which biases the bending BC and flips compliance sign — so the
  Phase 3 example freezes the corner *normal* at the canonical `(1, 0)` while
  leaving the corner *coordinates* free; L2 sensitivity flows through the
  corner's boundary-loop neighbors (edge midpoints whose chord touches the
  corner). FD-validated `rel_err = 1.4e-7`; 40-iter loop achieves **52%
  compliance reduction**, up from the corner-frozen 39%. See §Phase 3 L2
  results below.
- **L2 follower-load extension:** the current L2 only differentiates the
  bilinear assembly. Genuine follower loads (`t = t(n)` — e.g. pressure
  `t = -p·n`) would also need `∂t/∂n · ∂n/∂pts` in the b-side. Not started.
- **Regularization (done):** Helmholtz-PDE sensitivity filter on the boundary
  loop, `helmholtz_filter_along_boundary!` in the Phase 3 example. See
  §Helmholtz sensitivity filter below.
- **Body forces:** `b` with a true `∂b/∂pts` contribution from per-point body
  loads (e.g. gravity on a varying-density mesh). Same machinery as L1, just
  populated on interior rows.
- **Scalar PDE / SolidEnergy:** dispatchable `extract_weight_sensitivities`
  per model type (Phase C4 was the original framing). Still open.
- **Navier-Stokes extension:** see dedicated section below.

### Phase 3 L2 results (corners unfrozen)

```
Domain:        same as Phase 3 (9×5 cantilever, parabolic shear)
Design vars:   19 y-coords (top, bottom, right boundary — corners now included)
Corner normal: frozen at (1, 0) (beam-natural) — see §"Why freeze corners"
L2 path:       chord normals + ∂n/∂pts at non-corner Neumann pts
Iterations:    40 (Armijo-monotone)
```

| Quantity | Phase 3 (corners frozen) | Phase D L2 (corners free) |
|----------|--------------------------|---------------------------|
| Compliance C₀ → C₄₀ | 71.80 → 43.74 (−39.1%) | 71.80 → 34.40 (**−52.1%**) |
| Area drift | −0.02% | −0.04% |
| iter-0 AD/FD rel_err | 1.4e-7 | 1.4e-7 |
| Warm gradient time | ~9 ms | ~9 ms |

#### Why freeze the corner *normal* but let the corner *coordinate* move?

For a 90° polygon corner, the chord-tangent formula gives a length-weighted
normal `n = R(-π/2)·(p_next - p_prev)/|·|`. For a rectangle with Δx ≠ Δy this
is NOT the angular bisector — at the cantilever's top-right corner it's
`(1, 2)/√5 ≈ (0.447, 0.894)`, biased toward the longer (top) edge. Imposing
`σ·n = 0` with that normal mixes σ_xx and σ_xy in a way that's poorly
aligned with beam bending physics (which expects σ_xx = 0 at the free end).
With the chord-formula corner, the cantilever's compliance sign flips
(C₀ = -182.7 instead of +71.8), tripping the physical guardrail at iter 1.

Fix: freeze corner normals at `(1, 0)` (canonical right-edge normal, same as
Phase 3 used everywhere). The corner *coordinates* are still free design
variables, so the optimizer can move them. L2 still captures the geometry
sensitivity at the corner because the corner appears as a `(prev, next)`
neighbor of the boundary-loop vertices on either side — when the corner
moves in y, those neighbors' chord normals rotate, and the gradient picks up
their `∂n/∂(corner_y)` contributions. The `NormalJacobian` for corners
themselves is zeroed so `extract_normal_sensitivities!` is a no-op at corner
rows.

#### Implementation surface (Phase D L2)

- `src/optimization/manual_adjoint.jl`:
  - `NormalJacobian` struct (i_prev, i_next + four `SVector{2}` blocks).
  - `polyline_normals(pts_flat, boundary_loop, neumann_loop_pos)`.
  - `update_traction_coeffs!(layout, normals, λstar, μ)`.
  - `extract_normal_sensitivities!(...)`.
- `ext/MacchiatoMooncakeExt.jl`: new `normal_jacobians` kwarg on
  `shape_gradient`. Default `nothing` ⇒ frozen-normal behavior, backwards-
  compatible with Phase A/B/3 (Phase B regression rel_err unchanged).
- `examples/test_polyline_normals.jl`: unit test, analytic Jacobians vs
  `central_fdm(5,1)`, worst case 3.9e-12.
- `examples/shape_optimization_phase3_cantilever.jl`: L2 wired in,
  `is_corner` removed from free_mask, `build_live_normals` helper overrides
  corner entries at the canonical `(1, 0)` with zero Jacobians.

### Helmholtz sensitivity filter (Phase D reg)

The single-pass 3-point average has transfer function `H₃(ω) = cos²(ω/2)`:
nullifies ω=π (Nyquist) but only mildly attenuates intermediate
frequencies. With L2 unfreezing the corners, the optimizer found a 52%
compliance reduction whose final shape exhibits visible long-wavelength
undulations on the top and bottom edges. **Iter-0 gradient diagnostics
ruled out a bug**: top/bottom symmetry holds to 2e-8, global Nyquist
projection ratio is 1.5e-12 (filtered cleanly), the raw gradient is
correctly oscillatory at Nyquist within each edge (RBF stencil noise) and
the 3-point filter removes it. The undulations are therefore a
discrete-loss-landscape feature surfaced once L2 made the corner DOFs
available — exactly the case that calls for stronger low-pass smoothing.

The Helmholtz-PDE filter solves
`(I − r² ∇²_loop) f = g`
along the closed CCW boundary loop, with `∇²_loop` the discrete 1D
periodic Laplacian (stencil `[-1, 2, -1]`). Transfer function:

```
H_H(ω; r) = 1 / (1 + 4 r² sin²(ω/2))
```

| ω | H₃ (3-point) | H_H, r=1 | H_H, r=2 |
|---|--------------|----------|----------|
| 0 (DC) | 1 | 1 | 1 |
| π/4 (period 8) | 0.85 | 0.62 | 0.29 |
| π/2 (period 4) | 0.50 | 0.33 | 0.09 |
| π (Nyquist) | 0 | 0.20 | 0.06 |

Frozen entries (per `free_mask`) act as Dirichlet boundary values inside
the filter system: their row is the identity `f_k = g_k`, anchoring the
clamped left edge so the gradient is attenuated as it approaches that
boundary (physical). The cost is one 24×24 dense solve per iteration —
sub-millisecond at the cantilever scale.

#### Result comparison (40 iterations each)

| Configuration | Compliance reduction | Boundary shape | Notes |
|---|---|---|---|
| L2 + 3-point, corners frozen | 47.6% | mild waviness | "frozen-corner baseline with L2" |
| L2 + 3-point, corners free | **52.1%** | visible undulations | optimizer exploits unfiltered modes |
| L2 + Helmholtz r=1, corners free | 46.6% | mostly smooth | sweet spot |
| L2 + Helmholtz r=2, corners free | 38.5% | very smooth | over-regularized |

The 52.1% with the 3-point filter was the *apparent* minimum on an
under-regularized discrete loss; the Helmholtz r=1 result at 46.6% is
the *honest* minimum on the smooth design subspace. The compliance gap
(5.5 pp) is the cost of restricting the design to physically meaningful
modes.

#### Implementation surface (Helmholtz)

- `examples/shape_optimization_phase3_cantilever.jl`:
  - `helmholtz_filter_along_boundary!(out, in_, loop, free_mask; r)`.
  - `sensitivity_filter::Symbol = :helmholtz` and `helmholtz_r = 1.0`
    constants — toggle `:helmholtz` ↔ `:nyquist` to switch filters.
- Not yet promoted to `manual_adjoint.jl` source — same threshold as
  `filter_along_boundary!` (await second user / sysimage benefit).

#### Open shortcomings flagged by this comparison

- The Helmholtz `r` is hand-tuned. A principled choice would scale `r`
  with the boundary segment length or the expected feature size of the
  optimum. Adaptive `r` is open.
- The penalty oscillation in `‖∇L‖` (sawtooth from `2ρ(V − V₀)∇V` sign
  flipping between line-search trials) is still present. Augmented
  Lagrangian or a hard area constraint via projection would fix it
  cleanly.
- The 9×5 grid is coarse. A finer discretization (matching the planned
  Phase C5 scale test) would reveal whether the smooth Helmholtz result
  converges to a clean Michell taper as `h → 0`.
- Boundary-segment lengths differ between top/bottom edges (Δx=1.0) and
  the right edge (Δy=0.5). The Helmholtz filter treats them as a
  uniform 1D mesh, which slightly over-smooths the longer edges relative
  to the right edge. A geometric Laplacian with arc-length weights would
  be more rigorous but the visual difference at this scale is marginal.

---

## Corrections (review pass) — ALL APPLIED in implementation

Three issues flagged on re-read. All three are addressed in the committed code.
Step 2's math (`Aᵀ η = ∂L/∂u`, `ΔA = -η uᵀ`), the elasticity coefficient extraction
signs, and the W_template-shares-pattern assumption all check out.

### §1 Per-DOF `active` mask, not per-point AND — **APPLIED**

The implementation uses per-term gating as specified:

```julia
t11 = active[i]    ? -η[i]   * u[j]   : 0.0
t22 = active[i+N]  ? -η[i+N] * u[j+N] : 0.0
t12 = active[i]    ? -η[i]   * u[j+N] : 0.0
t21 = active[i+N]  ? -η[i+N] * u[j]   : 0.0
```

See `extract_weight_sensitivities_elasticity!` in `src/optimization/manual_adjoint.jl:74-91`.

### §2 Step 3's main loop must skip Neumann rows — **APPLIED**

The implementation uses `interior_rows::BitVector` (length N) threaded through
`extract_weight_sensitivities_elasticity!`. When set, Neumann rows are skipped in
the main d2x/d2y/d2xy extraction loop, and `extract_neumann_sensitivities!` handles
the dx/dy contribution separately.

See the `interior_rows` keyword in `manual_adjoint.jl:49-93` and the dual-loop
structure in `shape_gradient_mixed_bc` (ext/MacchiatoMooncakeExt.jl:532-549).

### §3 `_build_weights` closure binds `pts_from_flat(p)` twice — **APPLIED**

The implementation uses a single binding via `_make_weight_closure` /
`_make_weight_closure_subset`:

```julia
function _make_weight_closure(op, adjl, basis)
    let op = op, adjl = adjl, basis = basis
        function (p::Vector{Float64})
            q = Macchiato._pts_from_flat(p)
            return _build_weights(op, q, q, adjl, basis)  # single binding
        end
    end
end
```

See `ext/MacchiatoMooncakeExt.jl:299-306`.

### §4 Cosmetic notes — **RESOLVED**

- "Neumann Contribution" index: resolved by using `_find_nzval` lookups instead of direct
  list indexing in `extract_neumann_sensitivities!`.
- Step 5 accumulation: five operators in the implemented `shape_gradient_mixed_bc`,
  three in `shape_gradient_dirichlet`.
- `pullback` API: the implementation uses `Mooncake.value_and_pullback!!` with
  `friendly_tangents=true` — the cleanest available API. See `apply_weight_rule`
  in `ext/MacchiatoMooncakeExt.jl:337-346`.

---

## Neumann BCs in the Manual Approach

Neumann BCs fit naturally into the manual adjoint framework. They are handled
by the same `batch_overwrite_sparse_rows!` coefficient-transpose logic already
designed, just executed manually instead of inside a Mooncake trace.

### How Neumann rows contribute to the gradient

A Neumann row `r` of the system matrix is a linear combination of weight-matrix
entries:

```
A[r, a_col[p]] = Σ_m coeff[p, m] * W_m[weight_row, w_col[p]]
```

where `W_m` are first-derivative weight matrices (W_dx, W_dy), `coeff` encodes
the PDE physics (e.g., for traction: σ·n expanded into ∂u/∂x, ∂u/∂y, ∂v/∂x,
∂v/∂y with Lamé parameters and normal components), and `a_col`/`w_col` handle
the variable offset for vector-valued problems.

In the adjoint pass, the gradient flows backward through this assignment:

```
ΔW_m[weight_row, w_col[p]] += coeff[p, m] * ΔA[r, a_col[p]]
```

This is a **coefficient transpose** — the same coefficients that combine W_m
into A now distribute ΔA back to ΔW_m. This is implemented as a simple loop
over Neumann rows and their column entries (O(n_neumann × k) time, where k is
the stencil size). No AD needed.

### Level 1 vs Level 2

- **Level 1 (frozen normals)**: The normal vectors `n` at boundary points are
  computed once at the reference configuration and treated as constants. The
  coefficients (which contain `n_x`, `n_y`) are fixed. Only `∂W/∂pts` matters.
  This is what Phase B implements.

- **Level 2 (differentiable normals)**: The normals depend on the boundary
  geometry, which depends on `pts`. The coefficients contain `∂n/∂pts` terms
  that must be included. This requires differentiating through WhatsThePoint's
  normal computation — a separate concern from the PDE differentiation. It can
  be added later without changing the architecture: add a `∂coeff/∂pts` term in
  Step 3, computed via ForwardDiff on the normal computation.

### Why Neumann doesn't break the framework

Neumann BCs only affect **which rows of A get overwritten** and **which weight
matrices contribute**. The fundamental structure — weight matrices are built
locally, assembled globally with known coefficients — is unchanged. The only
addition to Step 3 is a second loop over Neumann rows to accumulate their ΔW
contributions, which is O(n_neumann × k) and uses the same coefficient-transpose
pattern.

---

## Extension to Navier-Stokes

### Problem statement

Steady incompressible Navier-Stokes in 2D:

```
(u·∇)u = -∇p/ρ + ν∇²u + f_x
(u·∇)v = -∇p/ρ + ν∇²v + f_y
∇·u = 0
```

State variables: velocity (u, v) and pressure p. The system is nonlinear
(convection term) and has a saddle-point structure (no pressure in the
continuity equation).

### Forward pass (Newton's method)

The nonlinear residual F(u, v, p) = 0 is solved via Newton iteration:

```
J(u_k) · Δu = -F(u_k)
u_{k+1} = u_k + Δu
```

where J is the Jacobian of F w.r.t. the state variables, evaluated at the
current iterate. At convergence (k → ∞, u_k → u*):

```
J(u*) = ∂F/∂(u,v,p) evaluated at the converged solution
```

The Jacobian has block structure:

```
    [ J_uu   J_uv   G_x ]   [Δu]     [F_x]
J = [ J_vu   J_vv   G_y ] · [Δv] = - [F_y]
    [ D_x    D_y    0   ]   [Δp]     [F_p]
```

Where each block is assembled from weight matrices with solution-dependent
coefficients. For example, the x-momentum diagonal block `J_uu` at row i,
column k:

```
J_uu[i,k] = u_i · W_dx[i,k] + v_i · W_dy[i,k]           ← linearized convection
          + δ_{ik} · (W_dx u + W_dy v)_i                 ← convection diagonal
          - ν · (W_d2x + W_d2y)[i,k]                     ← viscous term
```

The pressure gradient block: `G_x = W_dx / ρ`, `G_y = W_dy / ρ`.
The continuity block: `D_x = W_dx`, `D_y = W_dy`.

**Key observation**: every block of J is a linear combination of the same five
weight matrices (W_dx, W_dy, W_d2x, W_d2y, and W_d2xy if using mixed
derivatives). The coefficients are **solution-dependent** — they involve u_i,
v_i, and the reconstructed gradients (W_dx u)_i, etc. — but the weight matrices
themselves are the same set used for linear elasticity.

### Adjoint pass

The adjoint equation for a nonlinear forward problem F(u*, pts) = 0 is:

```
J(u*)ᵀ · η = ∂L/∂u
```

where J is the Jacobian at the **converged** solution, and ∂L/∂u is the
gradient of the loss w.r.t. the state (easy to compute — e.g., for L = ∫u²,
∂L/∂u = 2u). This is a single linear solve, same structure as Step 2 in the
linear case.

The gradient of the loss w.r.t. point coordinates is:

```
dL/dpts = ∂L/∂pts - ηᵀ · (∂F/∂pts)
```

where ∂F/∂pts is evaluated **holding the state (u,v,p) fixed**. For the x-momentum
residual at row i:

```
F_x,i = u_i (W_dx u)_i + v_i (W_dy u)_i - (1/ρ)(W_dx p)_i - ν(W_d2x u + W_d2y u)_i - f_x,i
```

Taking ∂/∂pts (state held fixed):

```
∂F_x,i/∂pts = u_i · Σ_j (∂W_dx[i,j]/∂pts) u_j      ← convection, x-velocity
            + v_i · Σ_j (∂W_dy[i,j]/∂pts) u_j      ← convection, y-velocity
            - (1/ρ) · Σ_j (∂W_dx[i,j]/∂pts) p_j    ← pressure gradient
            - ν · Σ_j (∂W_d2x[i,j]/∂pts + ∂W_d2y[i,j]/∂pts) u_j   ← viscous
```

Every term has the form: **Σ_j (∂W[i,j]/∂pts) × (coefficient × state_j)**.
This is structurally identical to the linear elasticity case — the only
difference is that the coefficients now depend on the converged solution (u, v,
p) rather than being material constants (λ, μ).

### Implication for Step 3 (extract ΔW)

The coefficient extraction for Navier-Stokes follows the same pattern as linear
elasticity but with more terms. For each sparsity entry (i,j) shared by all
weight matrices:

```
ΔW_dx[i,j]  = -(u_i η_x,i u_j + v_i η_y,i u_j - η_x,i p_j/ρ + η_p,i u_j)
               + ... (from y-momentum and other blocks)

ΔW_dy[i,j]  = -(u_i η_x,i v_j + ... )
               + ...

ΔW_d2x[i,j] = -(-ν η_x,i u_j - ν η_y,i v_j)
               + ...

ΔW_d2y[i,j] = -(-ν η_x,i u_j - ν η_y,i v_j)
               + ...
```

Each ΔW_k entry is a **linear combination** of η-weighted state products. The
coefficients are assembled from the PDE structure (convection velocities,
viscosity, density). This is a straightforward extension of the elasticity
extraction function — more terms, but the same principle.

### What changes vs. linear elasticity

| Aspect | Linear Elasticity | Navier-Stokes |
|--------|-------------------|---------------|
| Forward pass | Single linear solve | Newton iterations |
| Adjoint matrix | A (forward operator) | J (Jacobian at convergence) |
| Variables | 2 (u_x, u_y) | 3 (u, v, p) |
| Block structure | 2×2 symmetric | 3×3 saddle-point |
| Coefficients in Step 3 | Constants (λ, μ) | Depend on u, v, p |
| Number of terms per ΔW_k | 2-4 | 6-12 |
| Weight matrix set | Same (W_d2x, W_d2y, W_d2xy) | Same + W_dx, W_dy |
| Step 3 implementation | ~20 lines | ~50 lines |

### What stays the same

| Aspect | Unchanged |
|--------|-----------|
| Step 2 (adjoint solve) | Same structure: Jᵀ η = ∂L/∂u |
| Step 3 (coefficient extraction) | Same pattern: η-weighted state products, O(nnz) |
| Step 4 (local rrule) | Identical — `_build_weights` rrules are operator-agnostic |
| Step 5 (accumulation) | Identical |
| Neumann BCs | Same coefficient-transpose logic |
| Cost scaling | ~2 forward solves regardless of #parameters |

### The saddle-point issue

The zero pressure-pressure block in J (the (3,3) block) means J is indefinite,
not positive-definite. This affects the linear solver choice (UMFPACK handles
indefinite systems; iterative solvers need preconditioners like SIMPLE or
augmented Lagrangian). But it does **not** affect the adjoint architecture —
the adjoint equation Jᵀ η = ∂L/∂u is still a single linear solve with the same
sparsity pattern.

### Nonlinear iteration count

The adjoint cost is one additional linear solve (on Jᵀ) regardless of how many
Newton iterations the forward pass required. The Jacobian at convergence must
be stored (or its factors reused). This is standard practice in PDE-constrained
optimization — the "discrete adjoint" approach always uses the converged
Jacobian.

### Summary: Navier-Stokes is a straightforward extension

The manual adjoint framework extends to Navier-Stokes with **no architectural
changes**. The only new work is implementing
`extract_weight_sensitivities(::IncompressibleNavierStokes, ...)` — a ~50-line
function that encodes the PDE's coefficient structure. Steps 1, 2, 4, and 5 are
unchanged. The cost remains ~2 forward-equivalent solves per gradient,
independent of the number of design parameters.

---

## Why This Architecture Generalizes

The key architectural insight:

> **The local weight computation is the only part that needs AD.**
> Everything else — assembly coefficients, global solve, loss function — is
> explicit linear algebra.

The RBF-FD method reduces any PDE to: (1) compute stencil weights locally,
(2) assemble them into a global sparse matrix with PDE-specific coefficients.
Step (1) involves solving small dense systems (RBF kernel evaluations) — this
is where AD is valuable and efficient. Step (2) is just bookkeeping — linear
combinations with coefficients that are either constants (linear PDEs) or
functions of the converged solution (nonlinear PDEs).

In the backward pass, the chain rule splits cleanly:

```
∂L/∂pts = ∂L/∂u · ∂u/∂pts                    (state sensitivity)
        + Σ_k ∂L/∂W_k · ∂W_k/∂pts             (weight sensitivity)
        + direct ∂L/∂pts terms                 (e.g., perimeter regularization)
```

- `∂L/∂u · ∂u/∂pts` = adjoint solve + coefficient extraction (Steps 2-3, manual)
- `∂L/∂W_k · ∂W_k/∂pts` = local rrule (Step 4, Mooncake)
- Direct terms (Step 5, manual or ForwardDiff)

**Only the coefficient extraction in Step 3 is PDE-specific.** It maps the PDE
operator's continuous form to the discrete combination of weight matrices.
Everything else — adjoint solve, local rrules, accumulation — is generic
infrastructure.

This means adding a new PDE requires implementing **one function** (~20-50
lines) that encodes how the PDE's differential operator combines RBF-FD weight
matrices into the global system. The rest of the gradient computation is
reused.

For reference: the `Differentiation_of_Meshless_Solver.txt` document describes
this same two-level structure (global adjoint + stencil-level accumulation) and
arrives at the same conclusion — the cost is ~2 forward solves regardless of
the number of parameters, and the stencil-level work is embarrassingly parallel.
