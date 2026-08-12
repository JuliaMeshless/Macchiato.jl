# Benchmarks vs KernelInterpolation.jl

Head-to-head benchmarks of the Macchiato stack (Macchiato + [RadialBasisFunctions.jl](https://github.com/JuliaMeshless/RadialBasisFunctions.jl)) against [KernelInterpolation.jl](https://github.com/JoshuaLampert/KernelInterpolation.jl)'s own benchmark suite ([`benchmark/benchmarks.jl` at commit `e36034c`](https://github.com/JoshuaLampert/KernelInterpolation.jl/blob/e36034c3f1e0bf1e9393c8e63740c058629c6aca/benchmark/benchmarks.jl)), with settings held identical across both solvers.

Only benchmarks with **full feature parity** are included. For each one, `benchmarks.jl` defines the two implementations back to back (`ki_*` then `ours_*`) under a comment block stating the shared settings, so parity can be verified by reading. All node sets are generated once with KernelInterpolation's own generators and fed coordinate-for-coordinate to both sides. `verify.jl` asserts before any timing that the two sides produce matching numbers (interpolant values, Poisson solutions, rhs! outputs) — if the settings ever drift apart, the run aborts instead of producing a misleading comparison.

## How to run

```bash
julia benchmark-vs-kernelinterpolation/setup.jl   # one-time environment setup
julia --project=benchmark-vs-kernelinterpolation benchmark-vs-kernelinterpolation/run.jl
```

`run.jl` verifies parity, tunes and runs the suite, writes `results.json`, and prints a comparison table.

## Settings map

| Setting | KernelInterpolation.jl | Macchiato stack |
|---|---|---|
| Gaussian kernel | `GaussKernel{d}(shape_parameter = ε)`: φ(r) = exp(−(εr)²) | `Gaussian(ε)`: identical formula and convention |
| PHS kernel | `PolyharmonicSplineKernel{2}(3)`: φ(r) = r³ | `PHS(3)`: identical |
| Polynomial augmentation | `m` ⇒ monomials of total degree ≤ m−1 | `poly_deg` ⇒ total degree ≤ poly_deg, so `poly_deg = m − 1` |
| Stencil selection | `KNearestNeighbors(25)`, NearestNeighbors.jl KDTree | `k = 25` kwarg, same KDTree machinery |
| Dense interpolation solve | `Symmetric(A) \ b` → LAPACK Bunch-Kaufman | hardcoded `bunchkaufman!(A, true) \ b` — same factorization |
| Sparse stationary solve | `A \ b` on `SparseMatrixCSC` → UMFPACK LU | `solve(prob, UMFPACKFactorization())`, passed explicitly |
| Time-dependent rhs! | fill b = (f; g) in place, then `mul!(du, A, u, −1, true)` | identical formulation, built from RBF.jl operator weights |

## Included benchmarks

| Benchmark | KI entries | Ours | Notes |
|---|---|---|---|
| interpolation 1D | `interpolate` (Gauss ε = 0.5, `m = 3`) | `Interpolator` (Gauss ε = 0.5, `poly_deg = 2`) | **Deviation from KI's stock suite**: KI's Gauss kernel is order 0 (no polynomial). Our `Interpolator` cannot disable augmentation (`MonomialBasis` rejects degree −1), so both sides run degree-2 augmentation instead. |
| RBF-FD operator build 2D | `RBFFDBasis` ×2 (Lagrange / standard local basis) | `laplacian(nodes; basis, k)` | **Seam caveat**: KI's build stops after stencil selection + local weights; ours also assembles the sparse operator matrix in the same call. 436 stencil solves on each side. |
| RBF-FD Poisson 2D | `solve_stationary` ×2 (prebuilt basis) | assemble from prebuilt weights + UMFPACK | Both sides: prebuilt stencil weights outside the benchmark; benchmark = sparse assembly + b evaluation + UMFPACK solve of the same 436×436 system. |
| RBF-FD Poisson 2D end-to-end | basis + `SpatialDiscretization` + `solve_stationary` | `SolidEnergy` → `Domain` → `LinearProblem` → solve | Not in KI's stock suite — added because the task is "benchmark this package": full package-level pipeline from raw coordinates to solution, exercising Macchiato proper. `SolidEnergy(k=ρ=cₚ=1, source=−f)` solves −ΔT = f, matching `PoissonEquation`. Solutions agree with KI to ~7e−5 rather than machine precision: Macchiato orders nodes boundary-first while KI merges interior-first, and on this symmetric grid the k-NN cutoff has distance ties, so 8 of 436 stencils pick a different equidistant neighbor — identical settings, benign ordering effect. |
| RBF-FD Advection 2D rhs! | `KernelInterpolation.rhs!` | hand-built `ours_rhs!` from `directional` weights | Macchiato ships no advection model, so the ours side is user-authored from RBF.jl operators following Macchiato's custom-model pattern (`docs/src/custom_pdes.md`) — not a shipped Macchiato solver. Same rhs formulation, same Rodas5P state. |

## Skipped benchmarks (no feature parity)

| KI benchmark | Reason skipped |
|---|---|
| interpolation 2D, interpolation evaluation 2D, least squares 2D | ThinPlateSpline kernel (r²·log r) not implemented in RadialBasisFunctions.jl |
| computing Lagrange basis, interpolation 2D Lagrange basis | ThinPlateSpline kernel + no `LagrangeBasis` equivalent |
| interpolation 5D, interpolation 5D cholesky, interpolation 5D KrylovJL_GMRES | Wendland kernel not implemented |
| multiscale interpolation 1D, multiscale evaluation 1D | Wendland kernel + no multiscale/multilevel interpolation |
| Poisson 2D, Advection 1D, Heat 2D (global collocation) | Wendland kernel + no global (dense) collocation PDE solver — our stack is RBF-FD only |
| RBF-FD evaluation Lagrange/standard basis 2D | our off-node evaluation (`regrid`) builds new local stencils — a different algorithm than evaluating the existing RBF-FD basis expansion, so no setting parity |
| RBF-FD Poisson 2D least squares | overdetermined RBF-FD (centers ≠ evaluation nodes) not implemented |

## Environment

| Package | Version |
|---|---|
| KernelInterpolation.jl | 0.3.17-DEV, pinned to commit `e36034c` (the RBF-FD machinery is unreleased) |
| RadialBasisFunctions.jl | 0.7.0 + `fix/phs-laplacian-dimension` branch (commit `1e26422f`), `Pkg.develop` — see below |
| Macchiato.jl | 0.1.0 (this repo, `Pkg.develop`) |
| WhatsThePoint.jl | 0.2.0, `main` branch (mirrors Macchiato's `[sources]` entry) |

Both sides run in the same Julia process with the same BLAS configuration.

### Bug found by this benchmark's parity verification

Registered RadialBasisFunctions.jl 0.7.0 hardcodes the **3D** constants of Δrᵏ = k(k+d−2)·r^(k−2) in the fused `∇²` functors for all PHS kernels (e.g. 12r instead of 9r for PHS3 in 2D; the per-dimension `∂²` functors and the Gaussian/IMQ Laplacians are dimension-aware and correct). Every 2D `laplacian()` operator — including Macchiato's `SolidEnergy` steady solve — used a wrong operator: the initial Poisson parity check produced weights exactly 4/3 × KernelInterpolation's while the advection (first-derivative) weights matched at 1e−13, which pinpointed the fused Laplacian. The benchmarks therefore run against the fix branch, where the parity checks pass at machine precision.

## Results

Run 2026-08-04 on an Apple M5 Pro (Julia 1.12, 13 threads). Medians from BenchmarkTools' tuned sampling; full distributions in `results.json`.

| Benchmark | KernelInterpolation.jl (median) | Macchiato stack (median) | Speedup |
|---|---|---|---|
| interpolation 1D | 22.04 μs | 1.33 μs | 16.5× |
| RBF-FD operator build 2D | 2.45 ms (standard) / 2.59 ms (Lagrange) | 704 μs | 3.5× |
| RBF-FD Poisson 2D | 1.32 ms (standard) / 6.96 ms (Lagrange) | 470 μs | 2.8× |
| RBF-FD Poisson 2D end-to-end | 12.85 ms | 1.55 ms | 8.3× |
| RBF-FD Advection 2D rhs! | 4.65 μs | 3.98 μs | 1.2× |

Speedups are against the faster KI variant. Allocation counts follow the same pattern — e.g. Poisson end-to-end allocates 111.3 MiB / 1.41 M allocations in KI vs 10.0 MiB / 18.7 k for us, and our advection rhs! is fully allocation-free (0 B vs 32 B).

### Threading control

Both sides run in one Julia process (same thread pool, same BLAS/UMFPACK), and both packages multithread the benchmarked hot paths: KernelInterpolation uses `Threads.@threads` in its kernel-matrix/RBF-FD assembly code, and RadialBasisFunctions builds weights through a KernelAbstractions CPU kernel. Since thread-scaling efficiency differs between implementations, the suite was also re-run with `JULIA_NUM_THREADS=1` (`results-1thread.json`) to isolate pure algorithmic cost:

| Benchmark | KI (1 thread) | Ours (1 thread) | Speedup (1 thread) | Speedup (13 threads) |
|---|---|---|---|---|
| interpolation 1D | 20.21 μs | 1.32 μs | 15.3× | 16.5× |
| RBF-FD operator build 2D | 15.45 ms | 1.76 ms | 8.8× | 3.5× |
| RBF-FD Poisson 2D | 2.33 ms | 445 μs | 5.2× | 2.8× |
| RBF-FD Poisson 2D end-to-end | 32.72 ms | 2.48 ms | 13.2× | 8.3× |
| RBF-FD Advection 2D rhs! | 4.84 μs | 4.16 μs | 1.2× | 1.2× |

Threads help KernelInterpolation more than us at this problem size (e.g. their operator build gains 6.3× from 13 threads vs our 2.5×), so the multithreaded headline numbers *understate* our single-threaded algorithmic advantage.

All parity checks passed before timing: interpolant values and the RBF.jl-level Poisson solutions match KI to ≤ 1e−8 (advection operator weights to ~4e−13), and both packages land the identical discretization error vs the exact Poisson solution (2.450e−2 vs 2.451e−2). The Macchiato end-to-end path agrees with KI to ~7e−5 due to the documented stencil tie-breaking ordering effect. Note that the operator-build comparison carries the seam caveat above (our build includes sparse assembly, KI's excludes it) — despite the extra work it remains 3.5× faster.
