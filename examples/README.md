# examples/

Run everything via `jlrun` **from this directory** (it resolves `--project=@.`
to `examples/Project.toml` and attaches the shape-opt sysimage):

```bash
cd ~/gitRepos/Macchiato.jl/examples
jlrun plate_with_hole/make_plate_with_hole_stl.jl
jlrun shape_opt/shape_optimization_phase3_cantilever.jl
```

`Project.toml` / `Manifest.toml` live here (the only instantiated env). Scripts
in subfolders activate it via `Pkg.activate(joinpath(@__DIR__, ".."))`, so keep
the one-level-deep layout below — don't nest scripts deeper without fixing that.

## Layout

| Folder | What's in it |
|--------|--------------|
| `forward/` | Baseline PDE forward-solve demos (elasticity, heat) — predate shape opt. `heat_eq_2d.jl` includes `visualize_results.jl` (co-located). |
| `shape_opt/` | Cantilever adjoint shape-optimization: `*_phaseA/B` (Dirichlet / mixed-BC gradient validation), `phase3_cantilever` (full optimization loop), `test_polyline_normals` (Phase D L2 unit test). **The cantilever is retired** — its job (validating the manual adjoint) is done; see `docs/boundary_gradient_noise.md`. |
| `shape_opt/diagnostics/` | One-shot investigations into the boundary-gradient Nyquist noise: `test_fd_edge_noise` (the decisive experiment), `test_noise_decompose`, `test_stencil_noise`, `test_complex_step`. `test_fd_edge_noise` includes `../shape_optimization_phase3_cantilever.jl` in setup-only mode. |
| `plate_with_hole/` | **ACTIVE problem.** `make_plate_with_hole_stl.jl` generates `plate_with_hole.stl` (3D thin plate, elliptical through-hole) — the Stage-1 geometry for the filter-based shape opt. See `plan_plate_with_hole.md`. |

## Active work

The current problem is the **plate-with-hole** (`plan_plate_with_hole.md`), not
the cantilever. Method: non-parametric / Vertex-Morphing shape optimization
(every hole-boundary node free + one physical-radius Helmholtz filter).
