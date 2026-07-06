---
slug: pre-conference-review-remaining
created: 2026-07-06-1709
status: open
---

# Handoff: Remaining work from the Macchiato pre-conference review

## Goal / why this matters
A systematic pre-conference review of Macchiato is largely complete (correctness bugs,
transient boundary conditions, a performance footgun, and API/professionalism cleanup are all
done and verified green at **1228/1228 tests**). This doc tracks the **remaining phases** so a
fresh agent (or a later session) can finish the polish and pick up the deferred pieces. None of
the review's changes are committed yet — the working tree holds everything.

## Background & current state — what's DONE (verified green)
- **Phase A — correctness bugs (done):** transient flux-BC no-op; broken transient energy source
  term; mistyped `Domain` no-BC constructor + `add!`; `show` writing to stdout; unguarded Robin
  stencil; generic exception in `set.jl`. No-BC `Domain` constructor removed (BCs now required);
  `add!(domain, bc, name)` given replace-preserving-range semantics.
- **Phase A.1 — transient Neumann/Robin (headline, done):** the penalty form blew up (±80k on
  adiabatic); replaced with a **ghost-node + tying-matrix** diffusion operator matching the
  Niederer benchmark pattern. New file `src/boundary_conditions/numerical/ghost.jl`
  (`build_neumann_diffusion`); `src/models/energy.jl` transient `make_f` reworked;
  `src/simulation.jl` `_create_ode_problem` now Dirichlet-only BC closures + `Tuple`-ized;
  default `Transient` solver changed `Tsit5` → `FBDF` (`src/models/time.jl`). All four BC families
  (Dirichlet / Adiabatic / HeatFlux / Convection) verified stable and physical.
- **Phase B — perf footgun (done):** RHS model/BC callables `Tuple`-ized (no per-timestep dynamic
  dispatch); ghost source loop tidied to a per-surface function barrier.
- **Phase C+E — API/professionalism (done):** removed dead/duplicate exports (`node_drop`, `cov`,
  `make_memory_contiguous`, `ranges_from_permutation`, `permute!`, duplicate `findmin_turbo`) and
  unused abstract types (`AbstractOperator`, `AbstractProblem`); deduped imports; added
  `const DEFAULT_STENCIL_SIZE = 40`; non-side-effecting `Convection` show; gated `solve.jl`
  `println`/`@time` behind `verbose=false` + clear single-model `ArgumentError`; LICENSE year
  → 2022-2026; README `rectangle` note; `getting_started.md` stale-solver comment; CI action pins
  bumped (checkout@v4, setup-julia@v2, cache@v2, codecov@v4) + docs-deploy deduplicated (docs job
  removed from `CI.yml`, kept `documenter.yml`); removed unused `Random`/`Adapt` deps; added stdlib
  compat `LinearAlgebra="1"`, `SparseArrays="1"`.

## Key files / locations
- Working plan (private, not in repo): `~/.claude/plans/perform-a-systematic-review-memoized-catmull.md`.
- NS handoff (deferred big piece): `handoffs/2026-07-06-1702-navier-stokes-solver.md`.
- Untested modules: `src/upwinding.jl`, `src/io.jl`, `src/utils.jl`, `src/domain.jl` (`add!`/`delete!`).
- Test harness: `test/runtests.jl` (imports `TestItemRunner` + `@run_package_tests` but there are
  **zero `@testitem` blocks** — it's a no-op; real tests run via a manual `include` loop).
- Docs nit: `docs/src/custom_pdes.md:122` tells users to read the private `sim._solution`.

## Decisions & conclusions (so the next agent doesn't relitigate)
- **BC-signature unification** (`make_bc` vs `make_bc!` arg order) was **deliberately deferred** —
  it's cosmetic and risks the working, tested steady path right before the conference.
- **BC named-constructor macro** (deduping the ~5 `X(v::Number)/X(f::Function)` pairs) was
  **deliberately skipped** — YAGNI; the explicit constructors are clearer and greppable than a macro.
- **`Statistics` dep kept** (not removed) — it's used by the docs env and has compat; removing it
  would entangle `docs/Project.toml`. Minor stale-dep; Aqua would flag it.
- **ShadowPoints relocation** (move WTP's inward `ShadowPoints` into Macchiato beside the new ghost
  module) was **deferred post-conference** — cross-repo change; not needed by the transient fix.

## What's left / next steps
1. **Phase F — test coverage** for the untested modules: `upwinding`, `io` (`exportvtk`/`savevtk!`),
   key `utils`, and `domain` `add!`/`delete!`. Mirror the existing testset style; use `@test_throws`
   for the error paths (e.g. `add!` on a nonexistent surface, single-model `ArgumentError`).
2. **TestItemRunner decision:** either convert the suite to `@testitem` blocks (making
   `@run_package_tests` real) or drop `TestItemRunner` from `test/Project.toml` and the
   `@run_package_tests` call. Currently it's dead weight.
3. **Aqua.jl quality gate:** add Aqua as a test dep + an `Aqua.test_all(Macchiato)` testset. Expect
   it to flag the `Statistics` stale dep and possibly method ambiguities / unbound type params —
   triage and fix or document exceptions.
4. **Doc nit:** stop documenting the private `sim._solution` in `custom_pdes.md`. Cleanest fix is a
   thin public `solution(sim)` accessor (wraps `_get_solution_vector`) + update the tutorial.
5. **Deprecated warning:** a `ustrip(::AbstractArray)` deprecation shows in test logs (pattern in
   `derivatives.jl` and elsewhere) — update to the non-deprecated call.
6. **Navier-Stokes:** see `handoffs/2026-07-06-1702-navier-stokes-solver.md`.

## Gotchas / constraints
- **Nothing is committed.** The user asked (during the review) not to touch git state — no commits,
  no `.gitignore` edits, no delete/move of untracked files. Confirm current wishes before staging.
- **External registration blocker (cannot fix from inside Macchiato):** `Project.toml` `[sources]`
  git-pins WhatsThePoint to `main`; General registration requires WhatsThePoint registered first.
- **Uncommitted `Project.toml` deltas predating the review** (RBF `0.3`→`0.5`, examples deps) look
  deliberate and should be committed when the user is ready; the RBF 0.5 bump is what fixed the old
  transient `classify_stencil` bug, so keep it.
- Untracked working notes in the repo root (`comparison_kernelinterpolation.md`,
  `symbolic_pde_plan.md`) and `.claude/` are the user's git call — leave them alone.
- No secrets involved; no redactions needed.
