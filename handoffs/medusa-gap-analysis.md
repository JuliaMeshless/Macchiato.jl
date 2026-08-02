# Medusa → JuliaMeshless Gap Analysis

**What this is.** A feature-by-feature recon of [Medusa](https://gitlab.com/e62Lab/medusa) (e62Lab's MIT-licensed C++ strong-form meshless / RBF-FD library) against our JuliaMeshless ecosystem — RadialBasisFunctions.jl (numerics), WhatsThePoint.jl (geometry / point clouds), and Macchiato.jl (physics orchestration). The goal is a porting backlog: what Medusa has that we don't, where its code lives, how hard each is to bring over, and how much it would matter.

**Method.** Nine recon agents read a fresh shallow clone of Medusa's default branch (latest commit 2026‑05‑16) module by module. Each feature was then gap-analyzed against our three repos, and **every** resulting claim was handed to an independent adversarial verifier that grepped our code (or the Medusa clone, for "we do better" claims) trying to *refute* it. Of 43 candidate gaps, 1 was killed as a false gap, leaving **42 verified gaps** (16 fully confirmed, 26 partial where we have some but not all of the capability). All Medusa file paths below were checked to exist in the clone. The highest-impact claims were additionally spot-checked by reading the Medusa source directly.

**Headline.** We are *not* behind on core numerics — our RBF-FD stencil engine, kernels, node generation, and (uniquely) our Hermite symmetric-collocation BCs, autodiff, GPU path, and the whole OrdinaryDiffEq/LinearSolve backbone are at parity or ahead. Where Medusa is ahead is **breadth and architecture built on top of the numerics**: a generic implicit operator-assembly DSL, dimension-generic vector physics (3D elasticity, coupled domains, complex-valued PDEs, working incompressible flow), solution-adaptive refinement, and a richer geometry layer (analytic shapes, CSG, NURBS, parametric fills). Almost none of it requires new low-level math — most is orchestration and plumbing we can build on the parts we already have.

---

## Scoring

**Difficulty** (to implement in our ecosystem): 1 = trivial port (<1 day) · 2 = a few days · 3 = new module, ~1–2 weeks · 4 = multi-week, cross-package architecture · 5 = research-grade / ecosystem redesign.

**Impact** (value to Macchiato's RBF-FD multiphysics mission): 1 = cosmetic/niche · 2 = nice-to-have · 3 = meaningful capability broadening · 4 = unlocks new problem classes or major robustness · 5 = transformative.

Scores below are the adversarially-*amended* values (verifiers were allowed to revise the analyst's first pass). Where several dimensions surfaced the same capability, they're merged into one row here.

---

## Priority-ranked backlog

Sorted by impact, then by ascending difficulty (best leverage first within a tier). "Target" is the package the work mostly lands in.

| # | Gap | Target | Diff | Impact | Key Medusa files |
|---|-----|--------|:----:|:------:|------------------|
| 1 | **Implicit operator-assembly DSL** (scalar rows + coupled-vector blocks) | RBF + Macchiato | 4 | **5** | `operators/ImplicitOperators.hpp`, `operators/ImplicitVectorOperators.hpp` |
| 2 | **Working incompressible flow solver** (Chorin projection + ACM; non-Newtonian) | Macchiato | 4 | **4** | `examples/thermo_fluid/*.cpp`, `examples/nonNewtonian_fluid/` |
| 3 | **3D & plane-strain linear elasticity** (dimension-generic vector PDE) | Macchiato | 3 | **4** | `operators/ImplicitVectorOperators.hpp`, `examples/linear_elasticity/point_contact3d.cpp` |
| 4 | **Coupled multi-domain / material-interface assembly** | Macchiato | 3 | **4** | `examples/coupled_domains/poisson_coupled_domains.cpp` |
| 5 | **Complex-valued PDE support** (EM scattering, Schrödinger, Helmholtz) | cross-cutting | 3 | **4** | `examples/electromagnetic_scattering/`, `examples/quantum_mechanics/` |
| 6 | **A-posteriori error indicator** (re-apply operator at order p+2) | Macchiato + RBF | 3 | **4** | `examples/imex_indicator/imex.cpp` |
| 7 | **Solution-adaptive refinement loop** (indicator → refine → re-solve) | cross-cutting | 4 | **4** | `examples/imex_indicator/`, `domains/HalfLinksRefine.hpp` |
| 8 | **Analytic domain primitives** (Box/Ball/… self-discretizing, tagged boundaries) | WhatsThePoint | 2 | **3** | `domains/BoxShape.hpp`, `domains/BallShape.hpp`, `domains/DomainShape.hpp` |
| 9 | **Directionally-balanced support selection** (FindBalancedSupport) | RBF | 2 | **3** | `domains/FindBalancedSupport.hpp`, `.cpp` |
| 10 | **Per-stencil coordinate scaling/normalization** (ScaleToClosest/Farthest) | RBF | 2 | **3** | `approximations/ScaleFunction.hpp` |
| 11 | **Wave equation / 2nd-order-in-time transient** | Macchiato | 2 | **3** | `examples/wave_equation/wave_equation_2D.cpp` |
| 12 | **Visibility-criterion stencil selection** (cracks / non-convex) | WhatsThePoint | 2 | **3** | `examples/customization/custom_stencils_visibility.cpp` |
| 13 | **Periodic boundary conditions** | cross-cutting | 2 | **3** | `examples/cahnHilliard_equation/cahnHilliard_*.cpp` |
| 14 | **Discretization-level CSG** (boolean subtract/union, internal-boundary labeling) | WhatsThePoint | 3 | **3** | `domains/DomainDiscretization.hpp`, `domains/ShapeDifference.hpp` |
| 15 | **Local h-refinement of an existing cloud** (HalfLinksRefine) | WhatsThePoint | 3 | **3** | `domains/HalfLinksRefine.hpp`, `.cpp` |
| 16 | **Biharmonic & high-order (3rd/4th) analytic RBF operators** | RBF | 3 | **3** | `examples/customization/custom_operators_biharmonic.cpp`, `examples/cahnHilliard_equation/BiharmonicOp.hpp` |
| 17 | **Parametric-domain / surface node generation** (GeneralSurfaceFill + compute_normal) | WhatsThePoint | 3 | **3** | `domains/GeneralSurfaceFill.hpp`, `domains/compute_normal.hpp` |
| 18 | **Per-component/per-equation BC surgery + row equilibration** | Macchiato | 3 | **3** | `operators/ImplicitVectorOperators.hpp`, `examples/linear_elasticity/fretting_fatigue.cpp` |
| 19 | **WLS / GFDM weighted-least-squares engine** | RBF | 3 | **3** | `approximations/WLS.hpp`, `approximations/WeightFunction.hpp` |
| 20 | **NURBS / CAD-native multi-patch geometry** | new package | 4 | **3** | `domains/NURBSPatch.hpp`, `domains/NURBSShape.hpp`, `domains/cad_helpers.hpp` |
| 21 | **Systematic numerical verification suite** (convergence + scale/shift invariance + cross-path) | cross-cutting | 2 | **3** | `test/end2end/poisson_implicit.cpp`, `test/end2end/diffusion_explicit.cpp` |
| 22 | Standalone scattered interpolants (PU blend + modified Shepard IDW) | RBF | 2 | 2 | `interpolants/PUApproximant.hpp`, `interpolants/Sheppard.hpp` |
| 23 | Rigid transforms on discretizations (translate/rotate points + normals) | WhatsThePoint | 2 | 2 | `domains/TranslatedShape.hpp`, `domains/RotatedShape.hpp` |
| 24 | Multiquadric kernel + streamlined custom-RBF-from-φ(r²) | RBF | 2 | 2 | `approximations/Multiquadric.hpp`, `approximations/RBFBasis.hpp` |
| 25 | Rank-revealing (SVD) fallback local solver | RBF | 3 | 2 | `approximations/JacobiSVDWrapper.hpp` |
| 26 | Public multi-operator shared-factorization + operator-set API | RBF | 2 | 2 | `operators/computeShapes.hpp`, `operators/shape_flags.hpp` |
| 27 | Operator eval on a fitted global Interpolator at query points | RBF | 1 | 2 | `approximations/RBFInterpolant.hpp` |
| 28 | Public signed-distance & boundary-projection API | WhatsThePoint | 1 | 2 | `domains/DomainShape.hpp` |
| 29 | KDTreeMutable / incremental spatial index | WhatsThePoint | 3 | 2 | `spatial_search/KDTreeMutable.hpp` |
| 30 | Pure-Neumann (all-flux) singular-system regularization | Macchiato | 1 | 2 | `examples/poisson_equation/poisson_neumann_2D.cpp` |
| 31 | Run provenance / reproducible archiving (HDF5 + XML config) | Macchiato | 2 | 2 | `io/HDF.hpp`, `io/XML.hpp` |
| 32 | Examples-in-CI regression + doctests + symbolic stencil-weight tests | cross-cutting | 2 | 2 | `test/approximations/RBFFD_test.cpp`, `examples/CMakeLists.txt` |

---

## Detailed writeups (impact ≥ 3)

### 1. Implicit operator-assembly DSL — the one architectural gap that matters most
**Target: RadialBasisFunctions + Macchiato · Difficulty 4 · Impact 5 · Partial**

Medusa lets you write a discretized PDE as the equation itself. From `examples/linear_elasticity/cantilever_beam.cpp:60`:

```cpp
(lam+mu)*op.graddiv(i) + mu*op.lap(i) = 0.0;   // governing equation, row i
op.value(i) = {ux, uy};                         // Dirichlet
op.traction(i, lam, mu, {-1,0}) = {0, tval};    // σ(u)·n = t, assembled from Lamé constants
```

Each `op.<operator>(i)` returns a lazy row object that additively scatters stencil weights into the global sparse matrix and RHS. Two features make this powerful: (a) every operator takes an *optional equation-row argument* decoupling the equation row from the unknown column — which is exactly how ghost nodes and interface conditions are expressed — and (b) **row/column offsets** (`setRowOffset`/`setColOffset`) pack multiple discretizations or coupled vector components into one block matrix. The vector version (`ImplicitVectorOperators.hpp`) adds block-diagonal scatter, `graddiv` (∇∇·) off-diagonal coupling, the Hooke's-law `traction(node,λ,μ,n)` operator, and component-level addressing `eq(k).c(j)`.

**Where we stand.** The verifier confirmed a genuine *partial* equivalent on the scalar side: RBF's `@operator` macro + operator algebra (`Identity`/`ScaledOperator`/`SumOperator`) already parses `∇²`, `∇·(κ∇)`, etc. into composable operators, and every operator is already a global `SparseMatrixCSC`. What's missing is the **coupled-vector block DSL** — today Macchiato hand-assembles the `[A₁₁ A₁₂; A₁₂ A₂₂]` blocks in `src/models/mechanics.jl` and the traction rows in `src/boundary_conditions/mechanics.jl` bespoke per model.

**Port notes.** A `RowOp` builder accumulating `(row,col,val)` COO triplets with `row_offset`/`col_offset`, then one `sparse()` call — no need to re-derive shapes since RBF weights are already sparse matrices. Julia's operator overloading maps cleanly (`op.lap(i)` → lazy struct with `+`, scalar `*`, unary `-`, and `=`(rhs)). Port `graddiv` and `traction` as the vector couplings; `src/set.jl` already computes the `var_offset` block indices to reuse. This is the substrate that gaps #3, #4, #18 below all build on, which is why it's the highest-leverage item despite difficulty 4.
*Reference: Medusa technical report, matrix-row DSL + block offsets.*

### 2. Working incompressible flow solver
**Target: Macchiato · Difficulty 4 · Impact 4 · Partial**

Macchiato's `IncompressibleNavierStokes` is a documented broken stub — `src/models/fluids.jl:104` references undefined `k`/`cₚ` copy-pasted from the heat model and would throw `UndefVarError`. Medusa ships two working schemes: a **Chorin projection** (implicit/explicit momentum, a pressure-Poisson step regularized against its constant null space by a Lagrange-multiplier row prefactored once, then velocity correction) in `examples/thermo_fluid/dvd_imp_phs_2D.cpp`, and a matrix-free **artificial-compressibility (ACM)** variant (`p -= c²Δt·ρ·div(u)`) in `examples/thermo_fluid/lid_driven_acm_2D.cpp`, both with Boussinesq buoyancy for natural convection. Non-Newtonian (`examples/nonNewtonian_fluid/`) is nearly free once the base exists.

**Port notes.** Building blocks already exist: `upwind` advection (`src/upwinding.jl`), RBF `StrainRate` (for the shear-rate invariant driving μ_eff), and the `NewtonianViscosity`/`CarreauYasudaViscosity` types — so power-law μ_eff = μ·‖E‖^((n-1)/2) is close to free. Ghost nodes (`src/boundary_conditions/numerical/ghost.jl`) already give the wall-flux treatment. Coupling momentum to the existing `SolidEnergy` model via Boussinesq gives natural convection. Largest single physics deliverable; there is a parked handoff for exactly this.

### 3. 3D & plane-strain linear elasticity
**Target: Macchiato · Difficulty 3 · Impact 4 · Partial**

Our `LinearElasticity` is hardcoded 2D plane-stress: `make_system` builds a fixed 2N×2N block from ∂²x/∂²y/∂²xy, `lame_parameters` computes only the plane-stress λ*, and the traction BC is 2 hardcoded rows. Medusa's `ImplicitVectorOperators` assemble dim·N blocks for *any* dimension (vector Laplacian, `graddiv`, Hooke traction), validated on 3D Boussinesq point contact (`examples/linear_elasticity/point_contact3d.cpp`).

**Port notes.** Rewrite `src/models/mechanics.jl` `make_system` to loop over component blocks built from second partials rather than 3 hand-written blocks — RBF already has dimension-generic `Hessian`, `MixedPartial`, `StrainRate`, `Divergence`, so **no new numerics needed**. Add a `plane_strain` flag to `lame_parameters`; generalize the traction BC to N rows via the full Hooke tensor. Lands much more cleanly *after* gap #1.

### 4. Coupled multi-domain / material-interface assembly
**Target: Macchiato · Difficulty 3 · Impact 4 · Partial**

The canonical pattern for stitching two discretizations across a shared interface into one system: each domain keeps its own stencils (no cross-material smearing), blocks placed with offsets, and per interface node-pair one row enforces value continuity, its partner flux continuity. Backbone for conjugate heat transfer, two-material conduction, FSI, EM interface jumps. Macchiato's `Domain` is single-cloud and steady solving is capped at one model per domain (`src/solve.jl:19` throws).

**Port notes.** From `examples/coupled_domains/poisson_coupled_domains.cpp`: interface nodes of the outer domain are re-added to the inner with negated normal; two operators write into one (N_in+N_out) matrix via offsets; per pair, `u_in − u_out = jump` and `λ₂·neumann_in + λ₁·neumann_out = 0`. We have the pieces — RBF `normal_derivative` for flux rows, `Regrid`/`Interpolator` for interface transfer — the single-model restriction to lift lives in `src/solve.jl` + `src/domain.jl`. Exchange is monolithic (one solve), not iterative.

### 5. Complex-valued PDE support
**Target: cross-cutting (mostly Macchiato) · Difficulty 3 · Impact 4 · Partial**

End-to-end `ComplexF64` physics: frequency-domain Helmholtz scattering with a stretched-coordinate PML and Sommerfeld/anisotropic-conormal interface conditions (`examples/electromagnetic_scattering/dielectric_cylinder.cpp`, `anisotropic_cylinder.cpp`), and the time-dependent Schrödinger equation (`examples/quantum_mechanics/`). Opens electromagnetics, photonics, quantum, and frequency-domain acoustics.

**Port notes.** The verifier confirmed our numerical substrate is *already complex-ready*: `Interpolator` uses `promote_type` over the RHS and operators apply real geometry-only weights to complex fields fine. The block is at the Macchiato layer — `src/models/energy.jl` `make_system` hardcodes real (`A = α·weights`, `b = zeros(eltype(A))`). Make the assembled matrix/unknown/BC values scalar-type-generic so a complex coefficient (Helmholtz k², −iΔt for Schrödinger) flows through. Add a PML BC region and a Sommerfeld ABC (our Robin machinery already handles the complex `ik₀ + 1/(2r)` coefficient). Schrödinger then drops onto the existing transient path since OrdinaryDiffEq integrates complex state.

### 6–7. A-posteriori error indicator & solution-adaptive refinement
**Target: Macchiato + RBF (indicator, d3/i4) then cross-cutting (full loop, d4/i4) · Confirmed**

This is the standard route to efficient meshless accuracy on peaked/multiscale problems, and it's a listed absence on our side (WhatsThePoint's octree self-flags as a-priori-only, "not solution-adaptive"). Two staged deliverables:

- **Indicator (do first, standalone).** Solve at RBF-FD order p, then re-apply the *same* governing operators to the computed solution *explicitly* at order p+2 (larger stencils, support ≈ 2·C(order+d,d)); the per-node residual (re-evaluated RHS − original RHS) is the error field — interior nodes use the operator difference, boundary nodes the value/Neumann-operator difference. From `examples/imex_indicator/imex.cpp` (despite the "IMEX" name, *no time stepping is involved*; ref IEEE 2022, doi 10.1109/… document 9854342). Cheap because RBF operators are already parametrized by `poly_deg`/`k`, so the higher-order set is just a second `RadialBasisOperator`; Macchiato holds the solution/RHS/BC classification, so the indicator sits there. **Gotcha:** boundary nodes must diff the value/normal-derivative operator, not the Laplacian.
- **The loop.** Feed indicator-flagged regions into local node insertion (gap #15), re-discretize, re-solve. Closes the AMR story.

### 8. Analytic domain primitives that self-discretize
**Target: WhatsThePoint · Difficulty 2 · Impact 3 · Partial**

Medusa ships a `DomainShape` hierarchy (`BoxShape`, `BallShape`, `PolygonShape`, `PolyhedronShape`) where each shape knows its own `contains()`/`bbox()` and discretizes its own boundary and interior at uniform or varying density, tagging distinct boundary-type IDs per face. WhatsThePoint has *no* analytic shapes — even a unit square or annulus must come from a mesh file or be hand-assembled. This is the single biggest **ergonomic** gap: analytic test domains (square, disk, annulus, L-shape) dominate RBF-FD convergence studies.

**Port notes.** The verifier found the containment/bbox half is *already delivered by Meshes.jl* (`p ∈ geom`, `boundingbox`, plus Box/Ball/Sphere/Disk/Ngon/… primitives) — so this is a thin ergonomic wrapper, not new geometry. Use `sample(geom, MinDistanceSampling(h))` for variable density; **gotcha:** `discretize(Box)` in 2D triangulates the *filled* box, so pull `boundary(box)` for the outline and tag segments yourself, then thread through `PointBoundary`/`PointSurface`. Also port Medusa's generic bisection `projectPointToBoundary` so analytic shapes without a mesh still support projection.

### 9. Directionally-balanced support selection (FindBalancedSupport)
**Target: RadialBasisFunctions · Difficulty 2 · Impact 3 · Confirmed**

Grow a stencil until every axis-aligned quadrant around the center is occupied (identity frame for interior nodes; an SVD normal-orthogonal tangent frame plus a minimum boundary-neighbor quota for boundary nodes) instead of blindly taking the k nearest. Directly attacks the classic RBF-FD failure mode where plain kNN picks a lopsided one-sided neighborhood near boundaries or in anisotropic clouds, giving ill-conditioned local solves. We have *only* plain kNN today (`find_neighbors` → `NearestNeighbors.knn`).

**Port notes.** From `src/domains/FindBalancedSupport.cpp`: quadrant occupancy = sign bits of (xⱼ − xᵢ) projected onto the frame; boundary frame = null space of the outward normal via `svd`. Reuse `KDTree.knn` with growing k. **Gotchas:** boundary nodes need tangent-space balance *and* ≥ 2(d−1)+1 boundary neighbors; use a ~1e-6 tolerance so near-axis nodes don't falsely mark a quadrant; cap at max_support and return the unbalanced max rather than looping forever. Node type (interior/boundary) lives in WhatsThePoint's `PointCloud`, so thread the boundary flag through.

### 10. Per-stencil coordinate scaling/normalization
**Target: RadialBasisFunctions · Difficulty 2 · Impact 3 · Confirmed**

Medusa's engines carry a scaling policy (`ScaleToClosest`/`ScaleToFarthest`) mapping support nodes into a normalized local frame (pᵢ − p)/s before assembly, de-scaling derivative weights by 1/sᵒʳᵈᵉʳ. Controls conditioning and decouples the shape parameter from stencil size/spacing. We assemble in **raw physical coordinates** everywhere (verified across `assembly.jl`, `execution.jl`, and every basis evaluator), so a fixed Gaussian/IMQ ε does not adapt to local density and PHS+polynomial conditioning degrades for large stencils / non-uniform clouds. (This visibly surfaces the `scale` parameter present in Medusa's biharmonic closed form — see #16.)

**Port notes.** `NearestNeighbors` already returns neighbor distances from `find_neighbors`, so s is nearly free. Recenter to the eval point, scale by mean/max neighbor distance, and correct weights by 1/sᵐ using the existing `derivative_order` trait (which already exists and is exactly the needed exponent). **Gotcha:** must stay consistent with the cached Bunch-Kaufman factorization and the Enzyme/Mooncake AD rules. Cheap, high-leverage, and a prerequisite for stable 4th-order operators.

### 11. Wave equation / 2nd-order-in-time transient
**Target: Macchiato · Difficulty 2 · Impact 3 · Partial**

Our transient path is strictly first-order: `_create_ode_problem` hardcodes `ODEProblem`. Medusa solves the 2D wave equation with an implicit two-level central scheme (`examples/wave_equation/wave_equation_2D.cpp`). Acoustics, elastodynamics, and EM waves are currently inexpressible.

**Port notes.** OrdinaryDiffEq already provides `SecondOrderODEProblem`/`DynamicalODEProblem`, so the numerics are a dependency away. Add a model whose `make_f` produces ü = c²∇²u and dispatch `Transient` to `SecondOrderODEProblem`; the existing ghost/shadow Neumann and Dirichlet-pinning closures reuse directly. Mostly plumbing in `src/models/time.jl` + a new `src/models/wave.jl`.

### 12. Visibility-criterion stencil selection
**Target: WhatsThePoint · Difficulty 2 · Impact 3 · Partial**

Line-of-sight-filtered stencils so supports never reach across a crack, notch, or reentrant corner: take k-nearest candidates, keep only those with unobstructed straight-line visibility (sample along the segment, test domain containment). Essential for fracture and non-convex geometry where naive kNN smears the operator across a gap.

**Port notes.** We have two of three ingredients — the containment test (`isinside`/`TriangleOctree`) and kNN topology. Build a visibility-filtered adjacency and inject it: RBF operators already accept a precomputed `adjl` kwarg, so **no core operator API change is needed**. Sample ~100 points per candidate segment, reject if any leaves the domain. Medusa's own example (`custom_stencils_visibility.cpp`) is labeled experimental/inefficient — treat it as an algorithm sketch.

### 13. Periodic boundary conditions
**Target: cross-cutting · Difficulty 2 · Impact 3 · Partial**

For spinodal decomposition, box turbulence, crystal growth. Neither library has first-class periodic infra; Medusa hand-rolls it in the Cahn-Hilliard examples by cloning interior nodes within a band of each border to the opposite side with a clone→original map, then either copying clone values (explicit) or tying with identity rows (implicit). Notably, Macchiato's `build_neumann_diffusion` (`ghost.jl`) *already implements exactly this clone/tie pattern* for ghost nodes — so the machinery exists, it just needs generalizing.

**Port notes.** WhatsThePoint owns the node cloning; Macchiato owns the tie. Add a first-class `PeriodicBoundary` BC type so it composes with the existing hierarchy rather than living only in examples. Corners/edges need several images.

### 14. Discretization-level CSG (boolean ops on point clouds)
**Target: WhatsThePoint · Difficulty 3 · Impact 3 · Partial**

`DomainDiscretization::add`/`subtract` chop interior nodes falling inside a subtracted hole, keep/append the correct boundary nodes, and reverse normals on subtracted boundaries — the meshless recipe for `annulus = disk − disk` or a channel with an embedded obstacle, or electrodes/cavities with distinct internal BCs (`examples/poisson_equation/quadrupole.cpp`). WhatsThePoint's `combine_surfaces!` only concatenates labels — no inside-test chopping, reclassification, or normal reversal.

**Port notes.** Port the point-set-level `subtract!(cloud, other; name)` first — it's the portable, high-value piece and needs no analytic-shape layer. Reuse `isinside`/`TriangleOctree` for the inside predicate and `filter(f, ::PointVolume)` (already exists) to cull; append the carved boundary as a new named internal surface so Macchiato BC assignment just works. Watch Medusa's negative-margin trick for boundary classification and the normal reversal. Depends on rigid transforms (#23) for clean operand placement.

### 15. Local h-refinement of an existing cloud (HalfLinksRefine)
**Target: WhatsThePoint · Difficulty 3 · Impact 3 · Confirmed**

Refine a chosen region of an already-filled domain *in place*: for each region node insert new nodes at the midpoints of the "half links" to its support, projecting boundary-boundary midpoints back onto the boundary along averaged normals and filtering by a min-distance fraction via a mutable KD-tree (`domains/HalfLinksRefine.hpp/.cpp`, confirmed to take a `region` argument defaulting to all). The missing primitive for locally increasing resolution without regenerating the whole cloud — and the natural substrate for the adaptive loop (#7). Our pipeline can only re-fill from scratch at a new spacing.

**Port notes.** For each region node, for each support neighbor, propose a midpoint, reject if a KDTree query finds an accepted node within frac·h. **Gotcha:** `NearestNeighbors.KDTree` is immutable — either rebuild per pass (Medusa does 3 passes) or maintain a background grid for the incremental min-distance test (this is where KDTreeMutable, #29, helps). Reuse `_project_to_boundary` from `repel`'s octree variant for boundary projection; skip links touching ghost nodes; return the added indices so refinement can iterate.

### 16. Biharmonic & high-order (3rd/4th) analytic RBF operators
**Target: RadialBasisFunctions · Difficulty 3 · Impact 3 · Confirmed** *(merged from 3 dimensions)*

Our analytic RBF derivatives cap at 2nd order — `partial` explicitly throws for order > 2 ("Use the custom operator for higher orders"), and no basis provides ∂³/∂⁴ or ∇⁴. This blocks compact discretization of 4th-order PDEs: Kirchhoff plate/beam bending, Cahn-Hilliard phase-field, biharmonic stream-function. Medusa ships a closed-form `Biharmonic` operator; verified in `examples/customization/custom_operators_biharmonic.cpp:27`, the PHS action is exactly:

```
k·(k−2)·(dim+k−2)·(dim+k−4)·r^(k−4) / scale⁴
```

with a matching `Monomials` overload summing 4th-order and mixed 2+2 derivatives.

**Port notes.** Add ∂³/∂⁴ (and a `Biharmonic`) basis-action methods in `src/basis/polyharmonic_spline.jl` (mirror the existing ∂²/∇²/H blocks per PHS order) and extend `src/operators/monomial/partial.jl` beyond order 2, then wrap through the existing `Custom`/`@operator` machinery — **no new solve machinery**, the weight kernel is order-agnostic. `Combinatorics.multiexponents` (already used in `monomial.jl`) generates the multi-index. **Gotchas:** need PHS7 and poly_deg ≥ 4 for enough smoothness; 4th-order weights are conditioning-sensitive, so pair with the scale-function gap (#10). Once it exists, Cahn-Hilliard is reachable *for free* via OrdinaryDiffEq's `SplitODEProblem` (stiff linear biharmonic part prefactored, nonlinear `∇²(c³)` explicit).

### 17. Parametric-domain / surface node generation
**Target: WhatsThePoint · Difficulty 3 · Impact 3 · Partial**

`GeneralSurfaceFill` places variable-density nodes in the *parameter space* of any user parametrization r(t) with a Jacobian metric correction (α = h/‖J·u‖), deriving outward normals from the Jacobian via an SVD null-space (`compute_normal`) — discretizing polar curves, tori, and general manifolds without a mesh (ref arXiv:2005.08767, Duh–Kosec–Slak). Every WhatsThePoint generator starts from an existing boundary point set; there's no "parametrization in, nodes out" engine. Our `sample_surface` only handles imported triangle meshes, losing exact-geometry fidelity.

**Port notes.** `Meshes.jl` already has `ParametrizedCurve(fun, range)` and `MinDistanceSampling` — reuse those plus WhatsThePoint's `_BridsonGrid` for target-space proximity. The new piece is candidate generation in parameter space scaled by α, with proximity checked by the *mapped* distance. Normal = last left-singular vector of J, sign-fixed so det[J n] < 0 (**Medusa warns inward normals are the usual cause of a downstream volume-fill failure**). This is the shared engine the NURBS gap (#20) builds on.

### 18. Per-component BC surgery + row equilibration
**Target: Macchiato · Difficulty 3 · Impact 3 · Confirmed**

Medusa's `op.eq(i).c(j)` sets a single scalar BC on one component of one equation (e.g. symmetry plane: zero normal displacement on one component, zero shear via a derivative on another), and `examples/linear_elasticity/fretting_fatigue.cpp` row-equilibrates the matrix (scale by inverse row 1-norms) before solving to cope with mixed traction/displacement magnitudes. Our mechanics BCs are whole-node vector objects — no way to pin one component or mix a value BC on one DOF with a derivative BC on another at the same node.

**Port notes.** Add a `ComponentBC` wrapper (component index + inner Dirichlet/Neumann) and teach `write_bc_dirichlet!`/traction `make_bc!` to target row `global_i + (c−1)·N` selectively. Row equilibration is a diagonal left-scaling before `dropzeros` in `src/solve.jl` — trivial, or delegate to a LinearSolve preconditioner. **Note:** the contact benchmarks are inhomogeneous *analytic* tractions (closed-form Flamant/Hertz), **not** true unilateral contact — no KKT complementarity exists in either codebase, so this is not "contact mechanics," it's mixed-BC surgery. *(ref doi 10.1002/nme.6067)*

### 19. WLS / GFDM weighted-least-squares engine
**Target: RadialBasisFunctions · Difficulty 3 · Impact 3 · Confirmed (with a caveat)**

An alternative local engine solving an *overdetermined* weighted least-squares fit (χ = W(WB)⁺Lb) instead of exact saddle-point collocation — robust to noise, poor distributions, and rank-deficient stencils, and it returns a fit residual as a built-in quality indicator. This is the entire GFDM/MLS branch of the literature, which we cannot express because every stencil is a square, exactly-solved system.

**Caveat worth flagging:** one verifier noted that *for Macchiato's PHS+polynomial RBF-FD mission specifically*, monomial-augmented RBF-FD largely supersedes classical WLS, so the highest-value slice of this gap is really the **scale policy (#10)** and the **SVD fallback (#25)** it shares, not a full parallel engine. Build the full WLS engine only if courting the GFDM/MLS community or noisy-data regression use cases.

**Port notes.** Tall-matrix least squares via `\`, `qr`/`svd`/`pinv`; weight functions reuse existing RBF value evaluators. **Gotcha:** rank-deficient stencils need an SVD min-norm solve, not normal equations; keep WB assembly type-stable for AD/GPU parity.

### 20. NURBS / CAD-native geometry
**Target: new package · Difficulty 4 · Impact 3 · Confirmed**

Medusa evaluates tensor-product NURBS curves/surfaces (De Boor in homogeneous coordinates, rational Jacobian, precomputed derivative patches) and discretizes multi-patch NURBS domains directly with a parametric advancing-front fill — a CAD-native meshless pipeline straight from B-rep geometry, *no meshing step*. We have nothing (`reconstruction.jl` is an empty stub). The single most "research-differentiating" geometry capability, but the highest effort.

**Port notes.** Read `NURBSPatch.hpp` (De Boor on homogeneous control points, hodograph derivatives, quotient-rule Jacobian) and `NURBSShape.hpp`. Evaluate reusing an existing Julia NURBS/spline package for the math rather than porting De Boor. The parametric fill is the shared engine with #17. Best as its own package layered on WhatsThePoint.

### 21. Systematic numerical verification suite
**Target: cross-cutting (tests) · Difficulty 2 · Impact 3 · Partial**

Medusa's end2end tests assert normalized L∞ error below per-resolution thresholds at 2–3 refinements, cross-check four independent code paths agree to 1e-5, and — most valuably — re-run the same Poisson solve under domain translate (−123, 235), scale ×1000, and scale ×0.001 to probe stencil scale-robustness (`test/end2end/poisson_implicit.cpp`, `diffusion_explicit.cpp`). We have one multi-resolution convergence test (elasticity) and single-resolution MMS Poisson with loose thresholds (L2 < 5e-2), but **zero invariance-under-transform tests**. Convergence order and scale-invariance are the defining correctness contract of an RBF-FD library.

**Port notes.** Generalize the existing MMS pattern into a resolution loop asserting both an error threshold and an estimated order p = log(eᵢ/eᵢ₊₁)/log(hᵢ/hᵢ₊₁); reuse `create_2d_square_domain`. **This test will likely *expose* gap #10** — shape-parameter bases (Gaussian/IMQ) will fail scale-invariance without local coordinate normalization while PHS+poly should pass, so the invariance test doubles as a driver for the scale-normalization fix. Pure test code, no new deps.

---

## Smaller gaps (impact 2)

| Gap | Target | Diff | Note |
|-----|--------|:----:|------|
| Standalone scattered interpolants (PU blend + modified Shepard IDW) | RBF | 2 | Smooth post-processing fields where the global `Interpolator` (dense O(n³)) is infeasible; `regrid` already covers discrete transfer. `interpolants/PUApproximant.hpp`, `Sheppard.hpp` (ref arXiv:1610.07050) |
| Rigid transforms on discretizations (translate/rotate points + normals) | WTP | 2 | Enabler for CSG (#14); `domains/TranslatedShape.hpp`, `RotatedShape.hpp` |
| Multiquadric kernel + streamlined custom-RBF-from-φ(r²) | RBF | 2 | We have Gaussian/IMQ/PHS but not plain MQ; a `φ(r²)`→operators concept eases custom kernels. `approximations/Multiquadric.hpp`, `RBFBasis.hpp` |
| Rank-revealing (SVD) fallback local solver | RBF | 3 | Graceful degradation on degenerate stencils instead of a failed Bunch-Kaufman; `approximations/JacobiSVDWrapper.hpp` |
| Public multi-operator shared-factorization + operator-set API | RBF | 2 | Compute several operators (∇, ∇², …) from one local factorization per node; `operators/computeShapes.hpp` |
| Operator eval on a fitted global `Interpolator` at query points | RBF | 1 | Differentiate the interpolant off-node; `approximations/RBFInterpolant.hpp` |
| Public signed-distance & boundary-projection API | WTP | 1 | Expose `isinside`/projection as first-class; `domains/DomainShape.hpp` |
| KDTreeMutable / incremental spatial index | WTP | 3 | Backs efficient in-place refinement (#15); `spatial_search/KDTreeMutable.hpp` |
| Pure-Neumann (all-flux) singular-system regularization | Macchiato | 1 | Pin one node or add a Lagrange row for all-Neumann problems; `examples/poisson_equation/poisson_neumann_2D.cpp` |
| Run provenance / reproducible archiving | Macchiato | 2 | Medusa serializes domain + solution + config (HDF5 + XML); we have JLD2/VTK but no config-capture. `io/HDF.hpp`, `io/XML.hpp` |
| Examples-in-CI + doctests + symbolic stencil-weight tests | cross-cutting | 2 | Run examples as regression tests; assert closed-form stencil weights. `test/approximations/RBFFD_test.cpp` |

---

## What we do better

The adversarial verifiers (grepping the Medusa clone to *refute* each claim) confirmed the following as genuine advantages we hold:

- **Hermite / symmetric collocation with BCs baked into the stencil** — our flagship. `classify_stencil` sorts each stencil into Interior/Dirichlet/Hermite and assembles symmetric-collocation weights embedding Dirichlet/Neumann/Robin/Internal conditions directly into the local RBF system, preserving matrix symmetry and giving higher accuracy than one-sided or fictitious-node treatments. Medusa uses one-sided/ghost treatments only.
- **Autodiff of the weight computation** — the collocation solve is differentiable end-to-end via Enzyme *and* Mooncake, including derivatives w.r.t. node positions **and the RBF shape parameter**, with a cached Bunch-Kaufman factorization giving O(n²) adjoints. Medusa has no AD.
- **GPU weight evaluation** via KernelAbstractions/Adapt. Medusa is CPU/OpenMP only.
- **The entire OrdinaryDiffEq time-integration backbone** — adaptive step-size with embedded error control, implicit (FBDF default), IMEX/SSP/Rosenbrock/exponential families, stiffness detection. Medusa ships *only* fixed-step explicit RK (embedded Fehlberg pair discarded — no adaptivity) and Adams-Bashforth AB1–5, with no implicit or IMEX class at all. Tellingly, Medusa's own PDE examples hand-roll Euler rather than use these.
- **LinearSolve.jl backbone** — any algorithm/preconditioner incl. GPU CUSPARSE, selectable per-run. Medusa hardcodes its Eigen solver choice.
- **Gradient-limited (Lipschitz) spacing** with octree-accelerated fill (`max_growth`), so prescribed spacing fields transition smoothly. 
- **PVD transient time-series output** — self-indexing `.pvd`/`.vtm` series consumable directly in ParaView. Medusa has no timestep-series abstraction.
- **`upwind` advection operator** for convection-dominated stability. Medusa's advection-diffusion example uses centered stencils only and explicitly does no upwinding.
- **Named-surface boundary abstraction with automatic normal-angle splitting** (`split_surface!`) — mixed Dirichlet/Neumann on distinct edges is a per-name assignment rather than manual index bookkeeping.
- **Package-health CI + Aqua.jl auditing** — Codecov coverage, downgrade/compat testing, benchmark tracking (AirspeedVelocity), Runic auto-format, and Aqua's method-ambiguity/type-piracy/stale-dep checks. Medusa's 4-job GitLab pipeline has only a cpplint formatting gate.

## Parity (neither ahead)

Core numerics are largely equivalent: RBF kernels with polynomial augmentation (both Gaussian/IMQ/PHS + analytic derivatives to 2nd order + monomial augmentation); the augmented RBF-FD saddle-point stencil solve; local RBF interpolants; point-set-to-point-set transfer (our `Regrid` ≈ their PUApproximant for the discrete case); the Slak–Kosec advancing-front variable-density fill (arXiv:1812.03160, both implement it); 2D point-in-polygon (winding number); PCA/Jacobian normal estimation; graded surface sampling; kNN stencil selection over a static KD-tree; the node container abstraction; explicit operator application (their `ExplicitOperators` ≈ our `W*u` matvec); parallel shape computation (their OpenMP `computeShapes` ≈ our KernelAbstractions kernel); the custom-operator plug-in mechanism (their CRTP `applyAt0` ≈ our `Custom{N}` + `@operator`); ghost fictitious nodes for flux BCs; CSV I/O; profiling timers; and the per-module test-suite structure. Our composable operator algebra (`@operator`, `Identity`/`Scaled`/`Sum`) matches the *ergonomics* of Medusa's row-assembly DSL for scalar problems — the gap is only the coupled-vector block side (#1).

---

## Recommended implementation order

Sequenced for leverage and dependencies, not raw score:

**Tier 1 — foundational, unblocks the rest**
1. **Implicit operator-assembly DSL** (#1) — the substrate for 3D elasticity, coupled domains, and per-component BCs. Highest impact; do it first even at difficulty 4.
2. **Per-stencil scale normalization** (#10) — cheap, fixes conditioning, prerequisite for 4th-order operators, and driven out by the invariance suite.
3. **Analytic domain primitives + rigid transforms + CSG** (#8, #23, #14) — cheap ergonomic wins that make every test, example, and convergence study easier to author.
4. **Systematic verification suite** (#21) — lock in convergence-order and scale-invariance contracts before building more physics on top; it will surface #10.

**Tier 2 — high-impact physics & robustness (once #1 lands)**
5. **3D / plane-strain elasticity** (#3) — near-free after the DSL.
6. **A-posteriori error indicator** (#6) — standalone, cheap, opens the adaptivity story.
7. **Biharmonic / high-order operators** (#16) — unlocks Cahn-Hilliard and plate bending, nearly free once #10 is in.
8. **FindBalancedSupport** (#9) — stencil robustness across the board.

**Tier 3 — major new capabilities**
9. **Working incompressible flow** (#2) — fixes the broken stub; largest single physics deliverable.
10. **Complex-valued PDEs** (#5) and **coupled multi-domain** (#4) — both mostly Macchiato-layer generalizations of parts we already have.
11. **HalfLinksRefine** (#15) → **full adaptive loop** (#7) — closes AMR on top of #6.

**Tier 4 — breadth & polish**
12. Wave/2nd-order-in-time (#11), periodic BCs (#13), per-component BC surgery (#18), visibility stencils (#12), parametric fills (#17) → NURBS (#20), standalone interpolants (#22), and the remaining infra/dev items.

---

*Generated from a multi-agent recon of the Medusa default branch (commit as of 2026‑05‑16) against JuliaMeshless dev checkouts. Every gap was adversarially verified against our source; Medusa file paths were checked to exist and the top claims read directly. Difficulty/impact are engineering estimates, not commitments.*
