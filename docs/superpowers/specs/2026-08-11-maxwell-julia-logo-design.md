# Maxwell showcase example: pulse scattering off the Julia logo (dielectric dots)

**Status:** approved design, not yet implemented. Replaces the double-slit showcase.

## Context

The examples suite has a validated time-domain Maxwell TMz solver (`examples/maxwell_cavity_2d.jl` — exact TM₁₁ cavity mode, rel-L2 ≈ 0.17%) and had a double-slit diffraction showcase (`examples/maxwell_double_slit.jl`). Kyle wants the showcase replaced with something more organic/fun: **the three Julia logo dots as dielectric "glass" scatterers, with a pulse fired near them**. The double-slit example and its GIF are to be deleted (they remain in git history and were validated: fringe maxima matched two-slit theory within 0.066, symmetric to 0.001).

This example demonstrates a genuinely new capability for the suite: **inhomogeneous permittivity ε(x)** (the cavity example is vacuum, the double slit used PEC walls + conductivity sponges only).

## Scene

- Domain: 3.2 × 2.4, plain rectangle — **no interior holes** (dielectric dots have nodes inside them). Boundary is one closed CCW loop (`:surface1` by default), PEC (`PrescribedValue(0.0)`). Same sponge-lined chamber pattern as the double slit had.
- Three dielectric disks, radius `r_dot = 0.32`, centers on a point-up equilateral triangle with side `s = 0.72` (edge gaps 0.08, like the real logo), centroid `(2.0, 1.25)`:
  - green top: `(2.0, 1.25 + s/√3)` ≈ (2.0, 1.666)
  - red bottom-left: `(2.0 − s/2, 1.25 − s/(2√3))` ≈ (1.64, 1.042)
  - purple bottom-right: `(2.0 + s/2, 1.25 − s/(2√3))` ≈ (2.36, 1.042)
- Pulse source at `(0.75, 0.95)` — below-left, so dot arrival distances stagger: red ≈ 0.89, green ≈ 1.44, purple ≈ 1.61.
- Sponge: `L_sp = 0.3` band on all four walls, `σ_max = 40`, quadratic ramp — reuse `sponge_at` from the double-slit script verbatim.

## Physics model

TMz Maxwell with inhomogeneous ε, conductivity sponge, hyperviscosity, and a soft source:

- ∂Ez/∂t = (1/ε(x))·(∂Hy/∂x − ∂Hx/∂y) + γ∇²Ez − σEz + src·s(t)
- ∂Hx/∂t = −∂Ez/∂y + γ∇²Hx − σHx
- ∂Hy/∂t = ∂Ez/∂x + γ∇²Hy − σHy

Permittivity: `ε(x) = 1 + (ε_d − 1)·Σ_dots 0.5·(1 − tanh((‖x − c_d‖ − r_dot)/w_ε))` with `ε_d = 4.0` (index 2) and `w_ε = 1.5h` — the smooth transition avoids interface ringing with centered RBF-FD. Precompute per-node `inv_ε = 1/ε` and apply after the curl `mul!`s: `@. dEz = inv_ε * dEz` then add hyperviscosity/source/sponge terms. H updates are unchanged (μ = 1 everywhere).

Wave speed inside dots is c/2, so internal wavelength is λ₀/2 = 0.25 = 12.5h — resolved at the standard h.

## Numerics (all inherited from the validated examples)

- h = 0.02; expect N ≈ 22k (assert range 16k–34k, calibrate). `seed!(42)` before discretize (Bridson fill uses global RNG).
- Operators in `make_f` (which receives NO kwargs — hardcode): `PHS(3; poly_deg = 4)`, `k = 40`, one shared `adjl = find_neighbors(x, k)` for Dx, Dy, ∇². Keep the three operator-sanity asserts (Dx·x ≈ 1 etc.).
- γ = 0.05h²; RK4 fixed step via `run!(sim; adaptive = false, callback = cb)`; dt = 0.006 (CFL 0.3), `t_end = 6.0` (1000 steps).
- Model struct `MaxwellTMzLogo{T, V <: AbstractVector{T}} <: Macchiato.AbstractModel` with fields γ, inv_ε, σ, src, ω0, t0, τ_p (type-parameterized). `Macchiato.num_vars(...) = 3`; u = [Ez; Hx; Hy] blocks of N. `sim.u0 = zeros(3N)` explicitly (recorder pushes the t=0 frame from it).
- `coords = node_coordinates(cloud)` (boundary-first ordering, matches u blocks) for building ε/σ/src vectors before `Domain` construction.

## Excitation

Soft additive source on dEz: spatial Gaussian `A_src·exp(−(‖x − x_src‖/w_src)²)` with `w_src = 0.04` (2h); temporal ~2-cycle burst `s(t) = sin(ω0·t)·exp(−((t − t0)/τ_p)²)` with λ₀ = 0.5 (ω0 = 2π/λ₀), `t0 = 0.9`, `τ_p = 0.25`. Start `A_src = 28` and calibrate so peak |Ez| ≈ 1–2.

Timeline: burst ~t ∈ [0.4, 1.4]; red dot lights ~1.5; green/purple ~2.4–2.6; whispering-gallery ring-down and re-radiation to ~4.5; sponges absorb by t_end = 6.

## Recorder & diagnostics

DiscreteCallback (always-true condition, `save_positions = (false, false)`) with `next_frame::Ref` cadence `Δt_frame = 5·dt = 0.03` → 200 frames + t=0. Each frame: Ez copy, t, total energy `½Σ(ε·Ez² + Hx² + Hy²)` (note the ε weight), and three per-dot stored energies over precomputed node masks (`‖x − c_d‖ < r_dot`).

Asserts (thresholds marked *calibrate* are set to ~2.5× observed after the first run, the convention used by both existing Maxwell examples):

1. Point count range; operator sanity in `make_f`.
2. All finite; outer-wall Ez < 1e-10 (Dirichlet pins block 1 exactly).
3. **Causal arrival order:** `t_peak(red) < t_peak(green) < t_peak(purple)` for the per-dot energy curves.
4. **Penetration:** each dot's peak stored energy exceeds a floor (*calibrate*; a PEC dot would store ~0) — proves the dielectric physics is engaged.
5. **Ring-down:** each dot's final energy < 0.5 × its peak (*calibrate*).
6. **Absorption/stability:** total energy at t_end < 0.1 × its maximum (*calibrate*) — sponges work, nothing grows late.
7. GIF exists, filesize > 100 KB.

## Animation

Two panels, `Figure(size = (1100, 480))`, `colsize!` ~0.72 for the field:

- Left: Ez scatter (`:RdBu`, fixed symmetric colorrange ≈ 0.5×max|Ez| over frames, `markerspace = :data`, `markersize = 1.4h`), each dot drawn with a translucent fill (alpha ≈ 0.12) plus outline (linewidth ≈ 4) in the official Julia colors — red `#CB3C33`, green `#389826`, purple `#9558B2` — source position marked, sponge extent dashed, title Observable with t.
- Right: the three per-dot energy curves in matching colors, growing frame-by-frame (Observable point vectors, like the cavity probe panel), vertical time cursor, normalized y-axis.
- `record(fig, tmpgif, 1:2:length(frames); framerate = 24)` (~100 rendered frames; dense wave fields palettize badly — the double slit hit 80 MB at full cadence, 33 MB halved). Write to `tempname()*".gif"` then `mv` into `examples/maxwell_julia_logo.gif` (iCloud stall workaround). Target ≤ 35 MB.

## File changes

- **New:** `examples/maxwell_julia_logo.jl` → `examples/maxwell_julia_logo.gif` (gitignored output).
- **Delete:** `examples/maxwell_double_slit.jl`, `examples/maxwell_double_slit.gif`.
- No `src/`, `Project.toml`, or cavity-example changes. All imports already available in `examples/Project.toml` (stdlibs like Random/Statistics load via `@stdlib`).

## Verification

1. `julia --project=examples examples/maxwell_julia_logo.jl` — all asserts pass, per-dot peak times print in causal order, GIF saved. Check the log text, not the exit code (piping through `tee` masks failures).
2. Calibrate the *calibrate* thresholds from the first run's printed values; record observations in comments (house convention).
3. Extract 2–3 GIF frames (ffmpeg) and visually confirm: expanding ring, refraction/focusing inside dots, staggered lighting, ring-down; no interface ringing artifacts (if present, widen `w_ε` to 2–3h).

## Session-learned gotchas a fresh implementer must know

- WhatsThePoint `main` (pinned by `examples/Manifest.toml`): each named surface must be ONE closed loop at discretize time — do NOT `split_surface!` before `discretize` (zero-area loop error). A plain rectangle loop like the cavity example is fine.
- Transient path: `make_f` gets no kwargs; Dirichlet closures pin variable block 1 only (= Ez — exactly PEC); `run!` discards all but the final state, so frames/diagnostics must be captured in the callback; `set!` has no field map for custom models — assign `sim.u0` directly.
- `mul!` 5-arg accumulate pattern for the curl RHS; hyperviscosity is 3 extra `mul!`s with the shared-adjl Laplacian.
- Reference implementations: `examples/maxwell_cavity_2d.jl` (validated solver core, recorder, GIF save) and the deleted-but-in-git `examples/maxwell_double_slit.jl` (sponge profile, soft source, energy plateau assert — the file at this commit still exists in the working tree until this spec is implemented).
