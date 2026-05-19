# Manual Adjoint for RBF-FD Shape Optimization

**Last updated**: 2026-05-19 (Phase 3 planned)

## Status Summary

| Phase | Description | Status |
|-------|-------------|--------|
| Phase A | Dirichlet-only manual adjoint | **Validated** — `rel_err = 2.3e-11` |
| Phase B | Mixed Dirichlet + traction BCs | **Validated** — `rel_err = 1.4e-07` |
| Phase C | Clean interface + direct RBF backward | **Complete** — 155x speedup |
| Phase 3 | Practical application: cantilever compliance minimization | **Planned** — see §Phase 3 below |
| Phase D | Generalize to other physics | Not started |
| Phase 1 (traced) | Monolithic Mooncake trace | **Abandoned** — 5+ min LLVM compile for 25 pts |
| Phase 2 (traced) | `gradient(sim, loss; wrt=:pts)` API | Implemented but superseded by manual adjoint |

### What's implemented

- **`src/optimization/manual_adjoint.jl`** (434 lines): `extract_weight_sensitivities_elasticity!`, `allocate_weight_gradients`, `assemble_elasticity_from_weights`, `TractionLayout` + `build_traction_layout`, `apply_traction!`, `extract_neumann_sensitivities!`, `allocate_neumann_weight_gradients`
- **`ext/MacchiatoMooncakeExt.jl`** (557 lines): `_pts_from_flat` primitive + rrule, per-operator `build_weight_rule` / `apply_weight_rule`, `build_weight_rule_subset` / `apply_weight_rule_subset`, `shape_gradient_dirichlet` orchestrator, `shape_gradient_mixed_bc` orchestrator
- **`examples/shape_optimization_manual_adjoint_phaseA.jl`** (200 lines): Dirichlet-only AD-vs-FD validation
- **`examples/shape_optimization_manual_adjoint_phaseB.jl`** (293 lines): Mixed-BC AD-vs-FD validation

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

### Phase 3: Practical Application — Cantilever Compliance Minimization — **PLANNED**

This is the first end-to-end shape optimization example. Unlike Phase A/B
(gradient validation: compute ∂L/∂pts once, compare against FD), Phase 3 runs
a full optimization loop — gradient → optimizer step → point update → neighbor
recompute → repeat — producing a visibly improved design. This is the
minimum-viable demonstration that the manual adjoint pipeline works for actual
shape optimization, not just gradient checking.

#### Problem statement

Minimize the compliance (external work) of a 2D cantilever beam subject to a
volume constraint by moving boundary points.

```
Domain:  [0, L] × [-D, D]   (rectangle, N = 9×5 = 45 points)
BCs:     left edge clamped  (Dirichlet: u = v = 0)
         right edge loaded  (Neumann: parabolic shear traction, P = 1000)
         top/bottom free    (Neumann: zero traction)
Loss:    compliance = bᵀu   (with ∂L/∂u = b for linear elasticity)
Constraint: volume ≤ volume₀   (or area ≤ A₀ in 2D)
```

**Known optimal shape**: a tapered beam — thicker near the clamped end,
thinner near the loaded tip — approximating a parabolic thickness profile.
This is the classical Michell / Bernoulli optimum for a tip-loaded cantilever.

#### Why this problem (and not the plate-with-hole yet)

1. **All BC machinery exercised** — mixed Dirichlet + traction, exactly the
   Phase B setup. No new BC code needed.
2. **Simple loss** — compliance `bᵀu` has the trivial adjoint source
   `∂L/∂u = b` (because `b` is independent of `u` for linear elasticity with
   fixed external loads). No complex functional to derive.
3. **Known optimum** — the beam should taper following a parabolic-like
   profile. Qualitatively verifiable: if the optimizer thickens the clamped
   end and thins the tip, it's working.
4. **Same scale as validation** — 45 points is enough to see meaningful shape
   change. No new scale challenges.
5. **Compelling PR demo** — starting from a rectangle, after ~20 iterations
   the design visibly bends toward the optimal taper. Side-by-side initial vs
   optimized plots tell the story immediately.

The plate-with-hole (plan's "driving application") is a better second example —
it needs a more complex geometry setup and is more visually striking once the
pipeline is proven.

#### Design variables

All point coordinates are design variables, but only boundary points can move
freely. Interior points must follow via mesh morphing to maintain point-cloud
quality. The design-relevant gradient is `∂L/∂pts_b` — the boundary slice of
the full Δpts vector.

**Mesh morphing** (simple approach for this first example): freeze interior
points and move only free boundary points. The top, bottom, and right edges
are free to move; the left edge (Dirichlet) stays fixed. The interior points
are held fixed — this limits the number of iterations before remeshing is
needed but avoids the complexity of a differentiable mesh morphing step for
now.

#### Optimization loop

```julia
pts = reference_rectangle(L, D, nx, ny)
adjl = find_neighbors(pts, k)
traction_layout = build_traction_layout(...)  # frozen normals (Level 1)

for iter in 1:max_iter
    pts_flat = vcat([collect(p) for p in pts]...)

    result = shape_gradient(
        pts_flat, model, N, adjl, basis, active,
        dirichlet_dofs, dirichlet_vals, ∂L_∂u;
        interior_rows = interior_rows,
        traction_layout = traction_layout,
        neumann_ids = neumann_idx,
        neumann_adjl = neumann_adjl,
    )

    # Gradient descent with line search (or use NLopt L-BFGS)
    α = linesearch(pts_flat, result.Δpts, loss_fn)
    pts_flat .-= α .* result.Δpts

    # Project volume constraint (move boundary outward if volume < target)
    if current_volume < target_volume
        # scale boundary coordinates to maintain constraint
    end

    # Rebuild after geometry update
    pts = _pts_from_flat(pts_flat)
    adjl = find_neighbors(pts, k)
    # traction_layout is kept frozen (Level 1 normals at reference config)
end
```

#### What this adds to the codebase

1. **`examples/shape_optimization_phase3_cantilever.jl`** (~200 lines):
   - Sets up the cantilever geometry, BCs, and loss
   - Runs the optimization loop
   - Plots initial vs optimized shape + convergence history
2. **No new source code in src/** — everything needed is already implemented
   (Phase A/B/C). This is purely an example script.
3. **Optional: simple line search** — could be inlined in the script or use
   NLopt's `LD_LBFGS` with inequality constraint for volume.

#### Expected results

| Quantity | Initial (rectangle) | Optimized |
|----------|--------------------|-----------|
| Compliance bᵀu | baseline | lower (20-40% reduction) |
| Volume | V₀ | ≤ V₀ (active constraint) |
| Shape | rectangular | tapered (thicker at clamp, thinner at tip) |
| Gradient norm ‖Δpts‖ | large | decreasing, approaching KKT |

#### Success criteria

- [ ] Optimization runs for ≥ 15 iterations without NaN / divergence
- [ ] Compliance decreases monotonically (or nearly so)
- [ ] Final shape is visibly tapered (qualitative check)
- [ ] Volume constraint is satisfied at final iteration
- [ ] Gradient norm decreases by at least 1 order of magnitude
- [ ] Warm gradient evaluation time < 20 ms per iteration

#### Implementation notes

- **Frozen normals**: Neumann normals are computed once at the reference
  rectangle and kept constant (Level 1). The boundary points move but the
  normal vectors in `traction_layout` are not updated. This is valid for
  small shape changes; for large deformations of the loaded edge, normals
  should be recomputed.
- **Neighbor recomputation**: `find_neighbors` must be called each iteration
  after point coordinates change. This is O(N log N) via KD-tree and
  negligible compared to the PDE solve.
- **Volume constraint**: simplest approach is a penalty term
  `L += ρ * (V - V₀)²` with increasing penalty parameter ρ. NLopt
  inequality constraint is cleaner but adds a dependency. Start with the
  penalty approach for simplicity.
- **Interior point freezing**: only boundary points on the top, bottom, and
  right edges are design variables. A simple mask zeros out the interior and
  left-edge components of Δpts. This avoids the need for mesh morphing.

### Phase D (future): Generalize

- Implement `extract_weight_sensitivities` for SolidEnergy (scalar PDE)
- Add support for body forces (∂b/∂pts term)
- Level 2: differentiable normals for Neumann BCs
- Navier-Stokes extension (see dedicated section below)

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
