---
slug: navier-stokes-solver
created: 2026-07-06-1702
status: open
---

# Handoff: Implement the incompressible Navier-Stokes solver in Macchiato

## Goal / why this matters
`IncompressibleNavierStokes` is exported and documented but **non-functional** — its
`make_f` references undefined variables and its body is comment-only pseudocode. It's the
last major gap from the pre-conference review. The eventual target (user-chosen) is a
**full nonlinear NS solver validated** against an analytical/benchmark case (Kovasznay flow
or lid-driven cavity vs. literature). This is genuine numerical implementation, not cleanup,
and there is **no in-house template** (the Niederer benchmark is reaction-diffusion, not NS)
— treat it as greenfield and expect real velocity–pressure-coupling difficulty.

## Background & current state
- Macchiato is an RBF-FD meshless PDE solver. A systematic pre-conference review fixed the
  heat model's transient boundary conditions (see below) and cleaned up the API; NS was
  deliberately deferred to this handoff.
- `IncompressibleNavierStokes{M<:AbstractViscosity, P}` has fields `μ` (a viscosity model:
  `NewtonianViscosity` or `CarreauYasudaViscosity`) and `ρ`. `_num_vars(::INS, dim) = dim + 1`
  (velocity components + pressure).
- NS is **transient-only**: there is a `make_f` but no `make_system` (no steady path).
- `make_f(::IncompressibleNavierStokes, domain; kwargs...)` is **broken**: it destructures
  only `(; μ, ρ)` then computes `α = k / (cₚ * ρ)` (undefined `k`, `cₚ` — copy-pasted from the
  heat model), and the closure body (steps as comments: intermediate velocity, pressure
  correction, correct velocity) just does `mul!(view(du, vol_ids), w, u)`. Guaranteed
  `UndefVarError`. The docstring already warns "under active development and not yet fully
  functional."

## Key files / locations
- `src/models/fluids.jl:104` — the broken `make_f(::IncompressibleNavierStokes, …)`; also the
  viscosity models (`NewtonianViscosity`, `CarreauYasudaViscosity`) and the INS struct. Line
  `~110` `laplacian(all_points, vol; k = 40)` still uses a literal 40 (elsewhere replaced by
  `DEFAULT_STENCIL_SIZE`).
- `src/boundary_conditions/fluids.jl` — fluid BCs: `VelocityInlet` (Dirichlet velocity),
  `PressureOutlet` (Dirichlet pressure), `VelocityOutlet` (`ZeroFlux`/Neumann zero-gradient).
- `src/boundary_conditions/walls.jl` — `Wall` (a `PrescribedValue(0)` — no-slip velocity).
- `src/set.jl:130` — `_field_index_in_model(::IncompressibleNavierStokes, …)`: field layout is
  **variable-major**, `u=1, v=2, w=3 (3D), p=dim+1`, i.e. state = `[u₁..uN, v₁..vN, (w₁..wN,)
  p₁..pN]`, each block length N. Accessors `velocity(sim)` → `(u,v[,w])`, `pressure(sim)`.
- `src/simulation.jl:58` — `_run!(sim, ::Transient)` builds an `ODEProblem` from `make_f` and
  calls `OrdinaryDiffEq.solve`. A dedicated NS solve path would be a new `_run!` method
  dispatching on the fluid model (see architecture fork below).
- `src/boundary_conditions/numerical/ghost.jl` — `build_neumann_diffusion` builds a
  ghost-node diffusion operator with Neumann/Robin flux folded in (from the heat fix). The
  same ghost machinery is directly reusable for velocity Neumann (`VelocityOutlet`
  zero-gradient) and for the pressure-Poisson wall Neumann BC.
- `src/models/energy.jl:31` — `make_f(::SolidEnergy, …)`: the reference for how a transient
  model assembles an RBF-FD operator + applies BCs. Mirror its structure.

## Decisions & conclusions
- **Method: Chorin / fractional-step projection** (avoids the saddle-point velocity–pressure
  system):
  1. Intermediate velocity (explicit): `u* = uⁿ + Δt(−(uⁿ·∇)uⁿ + ν∇²uⁿ)`, apply velocity BCs.
  2. Pressure Poisson: `∇²p = (ρ/Δt) ∇·u*`, with `∂p/∂n = 0` at walls (Neumann) and `p` pinned
     at a `PressureOutlet` (Dirichlet); pressure is otherwise defined up to a constant.
  3. Correction: `uⁿ⁺¹ = u* − (Δt/ρ)∇p` → (approximately) divergence-free.
- **RBF 0.5 operator API is sufficient** (installed, matches `[compat] RadialBasisFunctions =
  "0.5"`): `gradient` (∇p), `laplacian` (ν∇²u), `partial`/`jacobian` (for ∇·u = Σ ∂uᵢ/∂xᵢ),
  `custom` + `@operator(∇ ⋅ (…))` (Niederer-style), and a **native Hermite-BC facility**
  (`BoundaryCondition`, `Dirichlet/Neumann/Robin`, `classify_stencil`, `HermiteStencil`,
  `custom(…; hermite=(is_boundary, bc, normals))`) — worth evaluating vs. the hand-rolled
  ghost approach.
- **Architecture fork — UNRESOLVED, needs a decision.** The projection method does a linear
  pressure-Poisson solve every step, which does not fit the plain `make_f → ODEProblem →
  OrdinaryDiffEq.solve` pipeline. Options (recommended first):
  1. **Dedicated fractional-step loop** — a specialized `_run!(sim, ::Transient)` dispatching
     on the fluid model that marches the 3 steps directly with its own pressure-Poisson
     assembly/solve. Cleanest, most standard, most likely to validate. *Recommended.*
  2. **Projection-as-ODE** — `make_f` velocity RHS solves the pressure Poisson each eval to
     project onto divergence-free; reuses the pipeline but wastes solves per RK step and
     threading pressure through the state is awkward.
  3. **Mass-matrix DAE** — velocity dynamic + pressure algebraic (`∇·u=0`), reusing the
     stiff-solver machinery from the heat fix; elegant but exposes inf-sup/checkerboard
     instability most directly. Highest numerical risk.
- **Fallback (if validation stalls):** mark `IncompressibleNavierStokes` clearly experimental
  and de-export it, rather than ship broken code to the conference.

## What's left / next steps
1. **Decide the architecture** (fork above; recommend the dedicated fractional-step loop).
2. **Fix the scaffolding** so `make_f` (or the new solve path) is correct and non-crashing:
   remove the `k`/`cₚ` reference, use `ν = μ(0)/ρ` from the viscosity model (`bc.μ` is callable;
   Newtonian returns constant), `DEFAULT_STENCIL_SIZE` for `k`.
3. **Assemble the operators** over the cloud: vector Laplacian (per velocity component),
   gradient (∇p), divergence (∇·u*). Reuse `build_neumann_diffusion`/ghost machinery for the
   velocity `VelocityOutlet` zero-gradient Neumann and for the pressure-Poisson `∂p/∂n=0` wall
   condition. Pin pressure at `PressureOutlet` (or one reference node if no outlet).
4. **Wire BCs:** `VelocityInlet` → Dirichlet on velocity; `Wall` → no-slip (Dirichlet u=0);
   `VelocityOutlet` → zero-gradient Neumann on velocity; `PressureOutlet` → Dirichlet pressure.
5. **Validate incrementally:** first unsteady **Stokes** (drop the nonlinear advection — linear,
   isolates diffusion + projection) against **Taylor–Green decay** (analytical); then full NS
   with advection against **Kovasznay** (steady) or **lid-driven cavity** vs. Ghia et al.
6. **Add tests** mirroring the heat transient tests (bounded, divergence-reduced, matches the
   analytical case to expected order). Un-mark experimental once validated.

## Gotchas / constraints
- **Velocity–pressure coupling is the hard part.** Collocation RBF-FD for incompressible flow
  is prone to spurious/checkerboard pressure modes (inf-sup / LBB). The projection split helps;
  if pressure oscillations appear, consider oversampled/staggered pressure nodes, PSPG-style
  stabilization, or the RBF Hermite-BC facility. Budget time for this.
- **Pressure is defined up to a constant** — the Poisson system is singular under all-Neumann
  BCs; pin one node (or use a `PressureOutlet`) and/or enforce zero-mean.
- **Pressure-Poisson Neumann compatibility:** `∂p/∂n` at walls comes from the wall-normal
  momentum balance; a plain `∂p/∂n=0` is the usual first-order choice but check the
  compatibility (solvability) condition of the Neumann Poisson problem.
- **Field layout is variable-major** (`[all-u; all-v; (all-w;) all-p]`, each length N) — index
  carefully; see `_field_indices` in `src/set.jl`.
- **No `make_system` for NS** (transient-only); don't try to route it through the steady
  `LinearSolve.LinearProblem(domain)` path (which also now throws a clear single-model error).
- **No secrets involved.** No redactions needed.
- **Session constraint that applied during the review:** the user asked not to touch git state
  (no commits / no gitignore edits / leave untracked files alone) — confirm current wishes.
- Related still-open review item: **Phase F test coverage** (upwinding, io, utils, domain
  add!/delete!) and the `[sources]` git-pin on WhatsThePoint (registration blocker, external).
