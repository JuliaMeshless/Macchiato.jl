# Macchiato ecosystem vs. KernelInterpolation.jl

A comparison of two Julia approaches to kernel/RBF-based numerical methods and PDE solving.

## Context

Both stacks solve similar surface-level problems (scattered data interpolation, meshless PDE collocation with kernels) but occupy very different parts of the design space — different numerical methods, different scope boundaries, different target users. This document positions Macchiato relative to KernelInterpolation.jl and highlights where each is the better choice.

Packages under comparison:

- **Macchiato.jl** — physics-model layer (heat, elasticity, incompressible Navier–Stokes)
- **RadialBasisFunctions.jl** — RBF interpolation and differential operators
- **WhatsThePoint.jl** — point cloud generation and geometry tooling
- **KernelInterpolation.jl** — https://github.com/JoshuaLampert/KernelInterpolation.jl

---

## One-line positioning

- **Macchiato + RadialBasisFunctions.jl + WhatsThePoint.jl**: A 3-package ecosystem splitting geometry / RBF operators / physics. Local RBF-FD collocation. Targets engineering-scale PDE simulation on complex 3D geometries (STL import → point cloud → sparse solve → VTK).
- **KernelInterpolation.jl**: A single unified package focused on the mathematics of kernel interpolation — classical scattered-data interpolation plus generalized (Hermite–Birkhoff) symmetric collocation for PDEs, in arbitrary dimension. Global (dense) collocation. Research/pedagogical orientation.

---

## Side-by-side

| Axis | Macchiato ecosystem | KernelInterpolation.jl |
|---|---|---|
| **Architecture** | 3 packages: WTP (geometry) → RBF (operators) → Macchiato (physics) | Single unified package |
| **Maturity** | Macchiato v0.1 (alpha, unregistered); RBF v0.5 (stable); WTP v0.2 (beta) | v0.3.10, registered, published (Zenodo DOI) |
| **Numerical method** | RBF-FD with local k-NN stencils (default k=40) → sparse system | Global (symmetric) collocation → dense system |
| **Scaling** | O(N·k²) assembly, O(N·k) solve (sparse) — scales to ~10⁵–10⁶ nodes | O(N³) solve — practical up to ~10³–10⁴ nodes |
| **Dimensions** | 1D/2D/3D (3D is the real target) | Arbitrary dimension (explicit design goal) |
| **Kernel families** | PHS (r, r³, r⁵, r⁷), IMQ, Gaussian | Gaussian confirmed in README; `src/kernels/` subdirectory and "(conditionally) positive definite kernels" framing imply a broader catalog (likely Matérn, Wendland, PHS) |
| **Polynomial augmentation** | Degree 0–2, matched to PHS order | Yes, automatic for conditionally PD kernels |
| **Geometry pipeline** | First-class: WhatsThePoint handles STL/OBJ import, PCA normals, sharp-edge splitting, octree, node repulsion, boundary-layer spacing | Built-in hypercube / hypersphere / random node generators; optional Meshes.jl / QuasiMonteCarlo.jl for more |
| **BC model** | Typed hierarchy: Dirichlet/Neumann/Robin + physics-specific aliases (`Temperature`, `Displacement`, `Convection`, …) | Hermite–Birkhoff symmetric collocation — BCs expressed as differential-operator conditions at nodes |
| **PDE framing** | Concrete physics models: `SolidEnergy`, `LinearElasticity`, `IncompressibleNavierStokes`. Closed-form system assembly per model. | Generic `Equation` / differential-operator API. Builds the collocation system from a user-specified PDE. |
| **Time-dependent PDEs** | `make_f` returns an ODE function; delegated to OrdinaryDiffEq.jl | Supported; delegates to OrdinaryDiffEq.jl (+ OrdinaryDiffEqRosenbrock recommended) |
| **Linear algebra** | Sparse via SparseArrays + LinearSolve.jl | Dense (implied — no sparse/iterative hooks mentioned) |
| **GPU** | Partial — RBF.jl uses KernelAbstractions + Adapt; Macchiato has CUDA in Project.toml | None mentioned |
| **Autodiff** | Native Enzyme and Mooncake rules in RBF.jl (shape params, weights) | None mentioned |
| **Units** | Unitful.jl required throughout WTP and Macchiato | Not mentioned |
| **Output / viz** | WriteVTK, Makie (via extension), JLD2 snapshots | Plots.jl, VTK export/import |
| **Design ethos** | Engineering-simulation stack; physics-first user; opinionated defaults | Math-of-kernels package; expose the full kernel/collocation machinery; dimension-agnostic |

---

## Where they differ substantively

### 1. Local (RBF-FD) vs. global collocation

This is the deepest architectural split.

RBF-FD (Macchiato/RBF.jl) approximates differential operators using **local stencils of k nearest neighbors** and assembles a **sparse** weight matrix. Per-row cost is O(k³) (tiny local solve), total assembly O(N·k³), and the global solve is sparse. This is what makes ~10⁵–10⁶-node problems tractable on a workstation. Trade-off: accuracy is tied to stencil size, polynomial augmentation degree, and node quality, and conditioning of local systems can bite on bad stencils.

Global symmetric collocation (KernelInterpolation.jl) solves a **single dense N×N kernel system** for all unknowns at once. This gives spectral / exponential convergence for smooth kernels (Gaussian, Matérn) on smooth solutions — genuinely the gold standard when you can afford it — but the dense matrix becomes ill-conditioned quickly and N is capped in practice.

**Consequence:** these tools are not really competitors for the same problem. Macchiato is aimed at 3D engineering problems where N is large and geometry is complex. KernelInterpolation.jl is aimed at problems where spectral accuracy matters more than scale, or at arbitrary-dimensional problems where stencil construction becomes awkward.

### 2. Scope discipline

Macchiato deliberately splits concerns across three packages. WhatsThePoint doesn't know about RBFs. RadialBasisFunctions doesn't know about physics. Macchiato is thin — it wires models and BCs to operator assembly. This layering is a real design commitment: the numerical core and the geometry core can be reused without the physics layer, and someone else can build a different physics layer on top.

KernelInterpolation.jl keeps it all in one package: kernels, nodes, equations, differential operators, interpolation, I/O, visualization, and callbacks all co-located. For a research/teaching package this is a feature, not a flaw — it's self-contained and easy to reason about.

### 3. Geometry

This is possibly the largest practical gap. WhatsThePoint ships a serious geometry pipeline: STL/OBJ import, PCA normals with MST+DFS orientation (Hoppe 1992), dihedral-angle surface splitting for sharp edges, octree-accelerated point-in-volume tests, multiple volume-fill algorithms (Slak–Kosec, Van-der-Sande–Fornberg, Fornberg–Flyer, octree-guided), boundary-layer refinement, and node repulsion (Miotti 2023). It is designed to take a CAD-ish surface mesh and produce a usable meshless discretization.

KernelInterpolation.jl's built-in node generators cover simple domains (hypercubes, hyperspheres, grids, random) and delegate anything more to Meshes.jl / QuasiMonteCarlo.jl. That's the right trade-off for a dimension-agnostic research package — but it's not aimed at "I have a turbine-blade STL."

### 4. Physics models vs. generic PDE

Macchiato offers closed-form, typed physics models (`SolidEnergy`, `LinearElasticity`, `IncompressibleNavierStokes`) and domain-specific BC types (`Temperature`, `Traction`, `Convection`, `VelocityInlet`, …). A user writes "here is my material, here are my BCs, run it." The symbolic PDE roadmap (`symbolic_pde_plan.md`) aims to add a generic Symbolics.jl front-end for users who want to write custom linear steady-state PDEs declaratively.

KernelInterpolation.jl is generic from the ground up: you hand it a `NodeSet`, an equation, kernels, and BCs. There are no physics models in the sense of Macchiato's `SolidEnergy`. This is appropriate for its audience — people doing research on kernel methods or teaching the subject.

### 5. Hermite–Birkhoff generalized interpolation

KernelInterpolation.jl explicitly leans on **generalized (Hermite–Birkhoff) interpolation** for PDEs. In symmetric collocation, BCs are expressed as differential-operator conditions at boundary nodes, and the system matrix is built from kernel evaluations under those operators. This is mathematically clean and gives well-posed, provably-convergent methods.

RadialBasisFunctions.jl also supports **Hermite stencils** for boundary conditions (so BCs like Neumann can be enforced exactly in the local stencil), but the framing is different: stencils are local, BCs are applied as row-replacements in the sparse system, and the abstraction is "this stencil enforces this derivative" rather than "this node satisfies this operator equation." Same underlying math, different packaging.

### 6. Modern Julia ecosystem integration

RBF.jl has a noticeably more ambitious integration with the modern Julia SciML/ML world:

- Enzyme and Mooncake autodiff rules (shape-parameter gradients, training workflows)
- KernelAbstractions + Adapt for GPU
- LuxCore extension (RBF layers in neural nets)

KernelInterpolation.jl, based on the README, does not advertise AD or GPU. It is a well-engineered CPU-dense package with clean docs and SciML-aligned time integration, but it isn't reaching into the differentiable-programming or GPU worlds. Given its single-maintainer research context, that's a reasonable scope.

---

## When to recommend which

Pick **KernelInterpolation.jl** when:

- You need spectral accuracy on a smooth problem with modest N.
- You are working in 4D+ or arbitrary dimension.
- You are teaching or researching kernel methods and want the math to be legible.
- You want a single registered package with no ecosystem commitment.
- You want the full symmetric-collocation framing (Hermite–Birkhoff) out of the box.

Pick **Macchiato + RBF.jl + WhatsThePoint.jl** when:

- Your problem is 2D/3D engineering-scale (N in the 10⁴–10⁶ range).
- You have real geometry (STL, sharp edges, internal surfaces, boundary layers).
- You want physics-typed APIs (materials, BCs by name, VTK output).
- You need AD or GPU at the RBF layer.
- You want the operator algebra (`Δ + ∂x + custom`) with sparse assembly.

---

## Honest weaknesses worth naming

**Macchiato side:**

- Version 0.1, unregistered. KernelInterpolation.jl has a 0.3.10 registered release and a Zenodo DOI — easier to cite today.
- Narrower kernel catalog than KernelInterpolation.jl appears to have (no Matérn, no Wendland at present).
- Three-package split raises the onboarding cost for new users.
- No symbolic PDE front-end yet (roadmap in `symbolic_pde_plan.md`).

**KernelInterpolation.jl side:**

- Dense solve caps practical problem size.
- Minimal geometry tooling for complex domains.
- No AD / GPU / sparse-iterative path.
- Smaller community footprint (17 stars, single maintainer) — fine for a research package, worth naming.

---

## Open questions worth verifying before publishing externally

- Full kernel catalog in KernelInterpolation.jl (`src/kernels/` contents not directly inspected — very likely includes Matérn, Wendland, polyharmonic; worth confirming).
- Whether KernelInterpolation.jl has any sparse or iterative solver path (README implies dense; not independently verified).
- Whether `src/differential_operators.jl` in KernelInterpolation.jl hides a local-operator mode that would narrow the RBF-FD gap described above.
