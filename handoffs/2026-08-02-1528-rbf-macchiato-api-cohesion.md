---
slug: rbf-macchiato-api-cohesion
created: 2026-08-02-1528
status: implemented
---

> **Implementation note (2026-08-02):** steps 1–4 plus the step-5 minimum are implemented
> on `feat/rbf-api-cohesion` (this repo) and `feat/operator-api-polish` (RadialBasisFunctions).
> The collision was resolved by un-exporting `Dirichlet`/`Neumann`/`Robin` from RBF (not
> Macchiato), per review. The step-1 equivalence check uncovered an upstream RBF bug —
> mixed partials were wrong at the default `poly_deg = 2` (fixed upstream, `cebf9a87`).
> Step-5's ambitious path is scoped in `bc_unification_design.md`. Macchiato's branch
> requires an RBF 0.7.0 release before it can merge (compat is already set).

# Handoff: Make Macchiato consume RadialBasisFunctions.jl's modern API (operator macro, mixed_partial, Hermite BCs)

## Goal / why this matters

RadialBasisFunctions.jl (RBF) has grown a modern, documented API — the `@operator` macro,
operator algebra, `mixed_partial`, and a Hermite boundary-condition path — but Macchiato
still consumes RBF the way it did two API generations ago: bare convenience constructors,
manual sparse-matrix combination of private `.weights` fields, a hand-rolled mixed partial,
and a completely separate BC system. The docs (especially `docs/src/custom_pdes.md`) teach
users the old pattern too. The task: bring Macchiato's source and docs onto RBF's
recommended API surface, and file the RBF-side API gaps this exercise exposes.

## Background & current state

**RBF today** (v0.6 + unreleased Enzyme work, `~/dev/RadialBasisFunctions`):

- Constructors `laplacian`, `partial`, `mixed_partial`, `gradient`, `hessian`, `custom`, …
  are keyword-only for `eval_points` and `hermite` since 0.6 (see CHANGELOG).
- **Operator algebra + `@operator` macro** (`src/operators/operator_macro.jl`,
  `operator_algebra.jl`): `(@operator κx*∂²(1) + κy*∂²(2))(x)` fuses the whole expression
  into a **single weights build** (SumKernel/ScaledKernel applied per stencil). RBF's own
  docs (`docs/src/guides/pde_operators.md`) call this "the recommended way to build
  operators the built-ins don't cover" — Helmholtz, advection-diffusion, ∇⋅(κ∇).
- **Hermite BC path** (`docs/src/guides/boundary_conditions.md`): every constructor takes
  `hermite = (is_boundary, bc, normals)` with `Dirichlet()/Neumann()/Robin(α,β)/Internal()`
  types, embedding BCs into the operator weights. Composed (macro) operators support it
  too — `SumKernel`/`ScaledKernel` implement the `(x, xᵢ, normal)` form.
- Limitation: Hermite normal-form only exists for PHS bases with ∂, ∇, ∂², ∇²
  (RBF issue #136) — relevant to any BC-unification plan below.

**How Macchiato actually uses RBF** — only the old surface:

| Site | Pattern |
|---|---|
| `src/models/energy.jl:73-75` | `laplacian(...)` then `A = α * ∇².weights` |
| `src/models/mechanics.jl:141-168` | 3 ops (`partial` ×2, `custom`) then manual sparse combos `A₁₁ = (λ*+2μ)*W_∂²x + μ*W_∂²y` |
| `src/models/mechanics.jl:51-123` | `_ℒ_mixed_partial`: ~70-line hand-rolled mixed partial incl. a log2-probe hack for monomial ordering |
| `src/models/fluids.jl:110-112` | `laplacian(...; k = 40)` hardcoded; `make_f` body references undefined `k`, `cₚ` (unfinished stub, throws if called) |
| `src/upwinding.jl` | `partial` per dim — fine, plus `autoselect_k` |
| `src/boundary_conditions/numerical/ghost.jl:98`, `derivatives.jl:6,30`, `mechanics.jl:137-138` | more `.weights` reaches |
| `docs/src/custom_pdes.md:64-75` | teaches `A = ∇².weights` and never mentions `@operator` |

Nothing in Macchiato uses `@operator`, operator algebra, `mixed_partial`, or `hermite`.

## Key files / locations

- Macchiato: `src/models/mechanics.jl`, `src/models/energy.jl`, `src/models/fluids.jl`,
  `src/boundary_conditions/core/bc_hierarchy.jl`, `src/Macchiato.jl:42-70` (BC exports),
  `docs/src/custom_pdes.md`, `docs/src/design.md:60-70` (row-replacement BC description)
- RBF: `src/operators/operator_macro.jl`, `src/operators/operator_algebra.jl`,
  `src/operators/mixed_partial.jl`, `src/solve/api.jl` (Hermite validation),
  `docs/src/guides/pde_operators.md`, `docs/src/guides/boundary_conditions.md`

## Decisions & conclusions (findings)

1. **`_ℒ_mixed_partial` is dead weight.** RBF's `mixed_partial(data, 1, 2; ...)`
   (`MixedPartial` operator) covers both the radial and the monomial basis actions —
   including the monomial-ordering issue the Macchiato docstring warns about (RBF's
   `MixedPartial` calls `∂mixed` on the `MonomialBasis` evaluator directly). Replace
   `custom(coords, _ℒ_mixed_partial; ...)` with `mixed_partial(coords, 1, 2; ...)` and
   delete ~70 lines. Verify equivalence with a quick weights comparison first.

2. **Macchiato re-implements the anisotropic-diffusion example by hand.** The elasticity
   blocks `A₁₁ = (λ*+2μ)*W_∂²x + μ*W_∂²y` are literally the `@operator κx*∂²(1) + κy*∂²(2)`
   example in RBF's PDE guide; energy's `α * ∇².weights` is `@operator α * ∇²` (or
   `∇ ⋅ (α∇)`). Migrating gets one fused weights build per block and removes manual sparse
   algebra. Note mechanics currently *reuses* `W_∂²x`/`W_∂²y` across two blocks (3 builds
   total); per-block macros would also be 3 builds (A₁₁, A₂₂, A₁₂) — a wash on cost, a
   clear win on readability.

3. **`.weights` is private API and everyone touches it** (11 sites + the tutorial). RBF
   exports no accessor. Downstream assembly genuinely needs the matrix, so this is an
   RBF-side gap, not just Macchiato sloppiness.

4. **Scalar `*` is missing on *built* operators in RBF.** `operator_algebra.jl` defines
   `+`/`-` for `RadialBasisOperator` (combines weights) but `*`/`/` only for symbolic
   `AbstractOperator`s. So `α * laplacian(x)` fails, which is exactly what pushes users to
   `α * op.weights`. Adding `Base.:*(::Number, ::RadialBasisOperator)` closes the loophole
   that forces `.weights` access.

5. **Two parallel BC universes with colliding exported names.** Macchiato defines and
   exports abstract `Dirichlet`, `Neumann`, `Robin` (`src/Macchiato.jl:46`); RBF exports
   concrete structs with the same three names. `using Macchiato, RadialBasisFunctions` —
   which `custom_pdes.md` instructs — makes those names ambiguous at the REPL. Beyond the
   collision, the stack has two BC application mechanisms: Macchiato's row replacement +
   ghost nodes (`design.md:66-67`, `numerical/ghost.jl`) vs RBF's Hermite embedding, with
   zero sharing.

6. **`custom_pdes.md` teaches private API and has an ordering landmine.** It tells users to
   define `Macchiato._num_vars` and call `Macchiato._coords` / `Macchiato._ustrip`
   (underscored = private; `_num_vars` is even *exported*, an API smell on its own). And
   Step 1's `make_system` calls `laplacian` although `using RadialBasisFunctions` only
   appears in Step 2 — works in the Documenter sandbox via late binding, `UndefVarError`
   for anyone copying Step 1 alone. Macchiato deliberately does not re-export RBF
   operators, but the docs never say "you need `using RadialBasisFunctions`" up front.

7. **Stencil-size inconsistency**: `k = 40` hardcoded in `fluids.jl:110` and
   `custom_pdes.md:66` vs `DEFAULT_STENCIL_SIZE` in energy/mechanics vs `autoselect_k` in
   upwinding. Three conventions for one knob.

## What's left / next steps

Ordered; 1–4 are independent of the bigger BC question and can ship as one PR each.

1. **Macchiato: adopt `mixed_partial`** — replace `_ℒ_mixed_partial` + `custom` in
   `src/models/mechanics.jl`; add a weights-equivalence test against the old path before
   deleting it.
2. **Macchiato: adopt operator algebra / `@operator`** in `energy.jl` and `mechanics.jl`
   `make_system`; standardize the stencil-size default while there (kill the hardcoded 40s).
3. **Docs: rewrite `custom_pdes.md`** — move `using RadialBasisFunctions` into Step 1 with
   a sentence saying operators come from RBF and are not re-exported (or decide to
   re-export a curated operator set from Macchiato — a design decision to make first);
   show the `@operator` macro (a Helmholtz or advection-diffusion custom PDE would
   showcase it far better than plain Poisson); stop teaching `Macchiato._coords`/`_ustrip`
   (add/export public accessors for "stripped coordinates of the domain cloud"); rename or
   un-underscore `_num_vars` in the extension interface.
4. **RBF: file/implement the two API gaps** — (a) public `weights(op)` accessor (or
   `SparseArrays.sparse(op)` / `Matrix(op)` conversions), (b) scalar `*`/`/` on built
   `RadialBasisOperator`. Then migrate Macchiato's 11 `.weights` sites onto it.
5. **BC unification (design discussion before code)** — minimum: resolve the
   `Dirichlet`/`Neumann`/`Robin` export collision. Ambitious: lower Macchiato's semantic
   BCs (`Temperature`, `VelocityInlet`, …) onto RBF's Hermite path so BCs are embedded in
   operator weights instead of row surgery — but RBF issue #136 (PHS-only, ∂/∇/∂²/∇²-only
   normal forms) means row replacement must survive as the fallback. Worth a small design
   doc weighing accuracy gains vs the ghost-node machinery it would replace.
6. **Optional RBF enhancement surfaced here**: a multi-operator build that shares stencil
   factorizations (`operators(data, (ℒ₁, ℒ₂, ℒ₃); ...)`) — elasticity builds 3 operators
   over an identical adjacency list and re-factorizes each stencil 3×.

## Gotchas / constraints

- **RBF 0.6 removed positional `eval_points`** — any new call sites must use keywords
  (`laplacian(data; eval_points = ..., k = ...)`). Pre-0.6 snippets in old examples will
  not run.
- `fluids.jl` `make_f` is an unfinished stub (undefined `k`, `cₚ` at line 111) — don't
  treat it as a working migration target; migrating energy/mechanics is the real work.
- Include order in `src/Macchiato.jl` is load-bearing (see CLAUDE.md) — BC-related moves
  must respect it.
- Operator/point-cloud API changes ripple across RBF ↔ WhatsThePoint ↔ Macchiato; check
  all three before assuming a break is local (per project CLAUDE.md).
- The equivalence test in step 1 matters: `_ℒ_mixed_partial`'s docstring claims the
  generic monomial pipeline had an ordering bug; confirm RBF's `MixedPartial` matches the
  hand-rolled weights on a 2D cloud before deleting.
- RBF `main` has unreleased changes (Enzyme AD work); Macchiato's compat is `0.6` — the
  suggestions above only rely on released 0.6 API.
