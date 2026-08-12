---
slug: maxwell-julia-logo-example
created: 2026-08-11-2335
status: done
---

# Handoff: implement the Julia-logo Maxwell showcase example (replaces double slit)

## Goal / why this matters

Implement `examples/maxwell_julia_logo.jl` per the approved spec at `docs/superpowers/specs/2026-08-11-maxwell-julia-logo-design.md`: a pulse scattering off the three Julia logo dots as dielectric disks, animated to `examples/maxwell_julia_logo.gif`, replacing the double-slit showcase. The spec is the design authority — this doc is the session brief around it.

## Background & current state

This session built a Maxwell TMz example suite from scratch (2026-08-11). Everything below exists in the working tree, **uncommitted**:

- `examples/maxwell_cavity_2d.jl` — the validated solver core (exact TM₁₁ cavity mode; rel-L2 ≈ 1.7e-3 on all three fields, energy drift +6.7e-3, wall pinning 1.5e-18). Do not modify; it is the numerical foundation and the style template.
- `examples/maxwell_double_slit.jl` + `examples/maxwell_double_slit.gif` — the showcase being replaced. **Delete both** as part of this task (Kyle approved). Before deleting, copy its `sponge_at` profile, soft-source pattern, and energy-plateau assert style — the spec references them.
- `docs/superpowers/specs/2026-08-11-maxwell-julia-logo-design.md` — the approved design: scene layout, all numeric parameters, model equations, assert list, animation layout, file changes. Read it in full first.

The double slit validated the open-domain machinery quantitatively: measured fringe maxima [0.721, 1.183, 1.678] vs two-slit theory [0.656, 1.2, 1.744] — side lobes ~0.065 inward (finite-slit-width bias, expected), symmetry 0.001.

## Key files / locations

- Spec (authority): `docs/superpowers/specs/2026-08-11-maxwell-julia-logo-design.md`
- Solver/style template: `examples/maxwell_cavity_2d.jl`
- Sponge + soft source + steady-energy patterns: `examples/maxwell_double_slit.jl` (delete after mining)
- Custom-PDE contract: `docs/src/custom_pdes.md`; transient path internals: `src/simulation.jl` (`_create_ode_problem`), `src/boundary_conditions/boundary_conditions.jl:22-31` (Dirichlet closure pins variable block 1 = Ez only)
- Examples env: `examples/Project.toml` (has everything needed; stdlibs load via `@stdlib`)

## Decisions & conclusions (do not relitigate)

- Dielectric dots (ε = 4, tanh-smoothed over 1.5h), NOT PEC — chosen for internal refraction/whispering-gallery physics and because waves visible inside the brand-colored dots render better. No interior boundary loops needed; the cloud is a plain rectangle.
- Point burst below-left at (0.75, 0.95) so dot arrivals stagger red → green → purple; the causal peak-order assert is the headline diagnostic.
- Replace (delete) the double slit rather than keep three Maxwell examples.
- GIF output (not MP4) — established preference this session; render every 2nd frame at 24 fps (full cadence hit 80 MB on the double slit; halved landed at 33 MB).
- Assert thresholds follow the house convention: run once, then set to ~2.5× observed, recording the observations in comments (see `examples/helmholtz_cylinder.jl:191`).

## What's left / next steps

1. Read the spec end-to-end.
2. Write `examples/maxwell_julia_logo.jl` (structure mirrors `maxwell_cavity_2d.jl`; model/recorder/render per spec).
3. Calibration run: `julia --project=examples examples/maxwell_julia_logo.jl` — check the log text, not the exit code (piping through `tee` masks failures). Verify causal peak order prints, tune `A_src` if peak |Ez| far from 1–2, calibrate the marked thresholds.
4. Extract 2–3 frames (`ffmpeg -i ... -vf "select=..."`) and visually check: expanding ring, refraction inside dots, staggered lighting, ring-down, no interface ringing (if ringing: widen `w_ε` to 2–3h).
5. Final clean end-to-end run with tightened asserts.
6. Delete `examples/maxwell_double_slit.jl` and `examples/maxwell_double_slit.gif`.
7. Open the final GIF for Kyle (`open examples/maxwell_julia_logo.gif`). Do not commit anything unless he asks.

## Gotchas / constraints

- **WhatsThePoint `main` (pinned in `examples/Manifest.toml`): each named surface must be ONE closed loop at discretize time.** `split_surface!` before `discretize` throws "zero signed area". A single rectangle loop (`:surface1`) with one `PrescribedValue(0.0)` entry is correct here. (Related known issue, out of scope: `test/end_2_end/2d_square.jl` uses split-before-discretize and will break under WTP main.)
- Transient path: `make_f` receives NO kwargs — hardcode `k = 40`, `PHS(3; poly_deg = 4)`, shared `adjl`. `run!` keeps only the final state — all frames/diagnostics must be captured in a `DiscreteCallback` with `save_positions = (false, false)` (load-bearing). `run!(sim; adaptive = false, ...)` is load-bearing for RK4.
- `set!` has no field map for custom models — assign `sim.u0 = zeros(3N)` directly, and push the t = 0 frame from it before `run!`.
- Apply `inv_ε` by scaling dEz AFTER the two curl `mul!`s, BEFORE adding hyperviscosity/source/sponge terms (spec has the exact term order).
- Total energy now needs the ε weight: `½Σ(ε·Ez² + Hx² + Hy²)`.
- iCloud stalls multi-MB writes into the synced repo — render GIF to `tempname()*".gif"` then `mv` (pattern in both existing Maxwell examples).
- `seed!(42)` before discretize; Bridson density lands ~2900 pts/unit² at h = 0.02, so expect N ≈ 22k for the 3.2 × 2.4 domain.
- Runtime expectation: ~17–20 s solve, a few minutes total including the GIF render — run it in the background and read the log.
- No secrets involved anywhere in this task.
