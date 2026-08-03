# Design note: unifying Macchiato BCs onto RBF's Hermite path

*Status: proposal, not a spec. Companion to `handoffs/2026-08-02-1528-rbf-macchiato-api-cohesion.md` (step 5).*

## The two mechanisms today

**Macchiato (row surgery + ghost nodes).** `make_system` assembles the interior operator
over all nodes; BC application then *replaces rows*: Dirichlet rows become `eᵢ`, Neumann
rows become normal-derivative weights (`boundary_conditions/numerical/derivatives.jl`,
optionally via shadow points), Robin rows a linear combination (`design.md:60-70`). The
transient path adds ghost nodes for Neumann/Robin (`numerical/ghost.jl`) so flux BCs
survive explicit time stepping.

**RBF (Hermite embedding).** Every operator constructor accepts
`hermite = (is_boundary, bc, normals)` with `Dirichlet()/Neumann()/Robin(α,β)/Internal()`.
Boundary information enters the *local collocation systems*, so the returned weights
already satisfy the BCs — no row replacement, and the local systems stay symmetric near
boundaries (the ill-conditioning that motivates Hermite in the first place; see RBF's
theory reference).

## Why unify

1. **Accuracy.** Row replacement uses one-sided stencils whose consistency near the
   boundary is the weakest link; Hermite stencils are the principled fix. This is the
   whole reason the Hermite machinery exists upstream.
2. **Less machinery.** The ghost-node subsystem (~200 lines incl. `build_neumann_diffusion`)
   and the shadow-point derivative code exist to work around what Hermite embeds for free.
3. **One vocabulary.** Macchiato's `Dirichlet`/`Neumann`/`Robin` hierarchy would *lower*
   to RBF's BC structs mechanically: `PrescribedValue → RBF.Dirichlet()`,
   `PrescribedFlux/ZeroFlux → RBF.Neumann()`, `Convection → RBF.Robin(α, β)` (α, β from
   the film coefficient), with the BC *value* still applied by Macchiato on the RHS.

## Constraints (why this is staged, not a rewrite)

- **RBF #136:** Hermite normal forms exist only for PHS bases with ∂, ∇, ∂², ∇². IMQ and
  Gaussian bases, and exotic operators, cannot lower yet. Row replacement must remain the
  fallback until upstream closes this.
- **Value handling:** Hermite embeds the *type* of BC in the weights; the boundary *values*
  `g(x, t)` still land on the RHS. Macchiato's time-dependent BC values (`bc(x, t)`) fit,
  but the RHS assembly changes shape — Dirichlet rows disappear from the solve entirely
  (RBF's Hermite gives Dirichlet points a single-entry stencil).
- **Vector systems:** elasticity's 2N×2N block system applies different BC types per
  component (e.g. Traction couples ∂x and ∂y of both components). Hermite handles scalar
  operators per block; traction-style coupled BCs likely stay on row surgery in phase 1.
- **Eval-points asymmetry:** the transient path evaluates operators on interior points only
  (`eval_points = vol`) with data on all points — Hermite supports this shape (it is the
  standard use), but Macchiato's index bookkeeping (`vol_ids`, surface ranges) must adapt.

## Proposed stages

1. **Plumb, don't switch.** Teach `Domain` to produce the `(is_boundary, bc, normals)`
   triple from its `boundaries` dict (surface ranges + normals already exist in
   WhatsThePoint). Expose as `hermite_data(domain)`; no behavior change.
2. **Steady scalar pilot.** `SolidEnergy.make_system` gains a `hermite = true` keyword:
   when set (and basis is PHS), build `laplacian(x; hermite = hermite_data(domain))` and
   skip row replacement for Dirichlet/Neumann/Robin surfaces. Compare against the MoMS
   suite — this is a pure accuracy experiment with a kill switch.
3. **Transient scalar.** Replace `build_neumann_diffusion`'s ghost-node construction with a
   Hermite-built `W` for the PHS case; keep ghosts as the non-PHS fallback.
4. **Decide on vector systems** only after 2–3 hold up; traction BCs may justify upstream
   work on component-coupled Hermite stencils instead.

## Open questions

- Does Hermite's Dirichlet single-entry stencil interact cleanly with Macchiato's
  convention of keeping boundary DOFs in the global vector?
- `Convection` lowers to `Robin(α, β)` — but Macchiato's α/β accessors (energy BCs) and
  RBF's exported `α`/`β` (Robin accessors) shadow each other inside Macchiato. Rename one
  (RBF un-exporting single-char Greek exports is probably right, same rationale as the
  Dirichlet/Neumann/Robin un-export).
- Benchmarks: Hermite weight builds solve larger local systems; measure assembly cost on
  the 2d_square fixtures before committing the default.
