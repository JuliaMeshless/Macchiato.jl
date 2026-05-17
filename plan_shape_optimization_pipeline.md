# Shape Optimization Pipeline — Gap Analysis

## What we have (or are close to having)

| Capability | Status |
|-----------|--------|
| Forward solvers (elasticity, NS, heat) | Done |
| Point cloud generation (WhatsThePoint) | Done |
| BC specification on named surfaces | Done |
| `_build_weights` with Mooncake rrules (5 operators) | Done |
| Gradient ∂L/∂pts (manual adjoint) | Designed, not implemented |
| Neumann BC differentiation (Level 1) | Designed, not implemented |

## The full gradient chain

```
design params l  →  boundary pts  →  interior pts  →  PDE solve  →  state u  →  loss L
      ↑                ↑                 ↑                ↑             ↑           ↑
   geometry        mesh/point        RBF-FD          forward       loss          ∂L/∂u
   kernel          cloud gen         assembly        solve         function
```

The backward chain:

```
∂L/∂l  ←  ∂pts_b/∂l  ←  ∂pts_i/∂pts_b  ←  ∂L/∂pts  ←  ∂L/∂u  ←  ∂L/∂L = 1
  ↑            ↑              ↑               ↑            ↑
design     geometry       point cloud     manual       analytic
params     kernel         deformation     adjoint      or tiny AD
```

## What's missing (in dependency order)

### 1. Geometry kernel: ∂pts_b/∂l

The boundary is defined by design parameters `l` (spline control points, hole
radii, etc.). We need the Jacobian of boundary point positions w.r.t. these
parameters.

**Options**:
- **Explicit parameterization** (Phase 3): parameters ARE boundary point
  coordinates. Trivial Jacobian (identity). Good for validation, not for
  production (too many DOFs, jagged shapes).
- **Spline/FDD parameterization** (Phase 4): design parameters are a small set
  of control variables. Jacobian via ForwardDiff on the spline evaluation or
  analytical derivatives. Smooth shapes, few DOFs.
- **WhatsThePoint-native** (future): if WTP surfaces are defined
  parametrically, the Jacobian could come from WTP itself (consistent with
  Level 2 normal differentiation).

**Effort**: 1-2 weeks for spline-based parameterization; trivial for explicit.

### 2. Interior point deformation: ∂pts_i/∂pts_b

When boundary points move, interior points must follow to maintain point cloud
quality. The deformation must be smooth (for gradient continuity) and maintain
adequate spacing (for RBF-FD accuracy).

**Options**:
- **RBF interpolation** (recommended): use the same `_build_weights` machinery
  to interpolate boundary displacements to interior points. The deformation is
  smooth, differentiable (via existing rrules), and respects the point cloud
  structure. Cost: one extra interpolation per design iteration.
- **Regenerate from scratch**: run WTP's point generator on the new boundary.
  Gives optimal point quality but introduces discontinuities (different point
  count, different adjacency). Only viable with an optimizer that tolerates
  stochastic gradients or after many small steps.
- **Spring/Laplacian smoothing**: treat edges as springs, solve for
  equilibrium. Simple but can cause point clustering.

**Effort**: 1 week for RBF interpolation approach.

### 3. Point cloud quality management

After several shape updates, the point cloud degrades (stretched cells,
clustered points). Need a strategy:

- **Quality metric**: minimum distance between points, local anisotropy ratio
- **Trigger**: regenerate when quality falls below threshold
- **Continuation**: after regeneration, restart optimizer with new point cloud
- **Alternative**: use a very robust point generator that produces consistent
  point counts and topology (requires WTP work)

**Effort**: ongoing tuning; 1 week for basic quality checks + regeneration.

### 4. Optimizer integration

Connect the gradient to an optimization algorithm.

**Required**:
- Design variable bounds (points must stay inside domain, hole can't invert)
- Line search or trust-region globalization
- Constraint handling (volume, stress)

**Options**:
- **NLopt.jl**: standard choice. L-BFGS for unconstrained, MMA or SLSQP for
  constrained. Simple API: provide `(L, ∂L/∂l)` callback.
- **Optim.jl**: native Julia, more control, fewer constrained algorithms.
- **IPOPT**: state-of-the-art for constrained nonlinear optimization. Requires
  Julia wrapper (Ipopt.jl). Best for large-scale constrained problems.

**Effort**: 1 week for NLopt integration with constraints. Ipopt adds 1 more
week for interface setup.

### 5. Constraint formulation

**Volume/area constraint**: typically `∫_Ω dV = V_target` or `Area(hole) =
A_target`. For a point cloud, approximate via quadrature or boundary integral.
Gradient: ∂(volume)/∂pts is geometric — can use ForwardDiff or analytical
formulas.

**Stress constraint**: `σ_vm < σ_yield` everywhere. This is a
pointwise constraint (one per point), which is too many. Standard approaches:
- **Aggregation**: p-norm or KS function: `max_i σ_i ≈ (Σ σ_i^p)^(1/p)`. A
  single scalar constraint. Gradient flows through the PDE solve → already
  handled by the manual adjoint if the loss includes stress.
- **Active-set**: only enforce at points where stress exceeds threshold.

**Manufacturing constraints**: minimum hole radius, minimum wall thickness,
symmetry. These are geometric — easy to implement analytically.

**Effort**: 1 week for volume + aggregated stress constraints.

### 6. Multi-physics interface

The pipeline should work for elasticity, Navier-Stokes, and heat transfer with
the same API. The manual adjoint architecture already supports this — only
`extract_weight_sensitivities` changes with the physics. Need:

- **Unified `shape_gradient(model, domain, loss_fn, params)`** that dispatches
  on model type
- **Unified constraint interface**: `volume_constraint(domain)`,
  `stress_constraint(domain, model)`
- **Unified forward solve**: already exists via `make_system` + `make_bc!`

**Effort**: 1 week for clean dispatch + testing across models.

### 7. Robustness and recovery

- **Failed forward solves**: if the point cloud becomes too distorted, the PDE
  solve may fail (singular stencils). Need fallback: regenerate, reduce step,
  or backtrack.
- **NaN gradients**: detect and recover (typically a point cloud quality issue).
- **Checkpointing**: save optimization state periodically (JLD2 already
  available in Macchiato).
- **Convergence detection**: gradient norm, step size, objective change.

**Effort**: 1 week for basic robustness; ongoing hardening.

## Phased roadmap

### Phase 3: Explicit-parameter optimization (3-4 weeks)

The simplest end-to-end pipeline: boundary point coordinates ARE the design
parameters.

- [ ] Implement manual adjoint (Phases A-B from plan_manual_adjoint.md)
- [ ] RBF-based interior point deformation
- [ ] NLopt integration (L-BFGS)
- [ ] Volume constraint
- [ ] Validate: optimize a cantilever beam shape to minimize compliance
- [ ] Validate: optimize a plate-with-hole to minimize peak stress

**Delivers**: working shape optimization, validates the full gradient chain.
Limited to small problems (point count = DOF count) and needs smoothing.

### Phase 4: Design parameterization (3-4 weeks)

Introduce a reduced design space for smooth, manufacturable shapes.

- [ ] Spline/Bezier boundary parameterization with analytical Jacobian
- [ ] Free-Form Deformation (FFD) lattice as alternative
- [ ] Multi-level: coarse design params → fine boundary resolution
- [ ] Validate: same problems as Phase 3 with smooth, production-quality shapes
- [ ] Validate: parameter count reduction (1000s of points → 10s of parameters)

**Delivers**: production-ready shape parameterization.

### Phase 5: Constrained + multi-physics (3-4 weeks)

- [ ] Stress constraints (aggregated)
- [ ] Ipopt integration for constrained optimization
- [ ] Navier-Stokes shape optimization (e.g., minimize pressure drop in a duct)
- [ ] Heat transfer shape optimization (e.g., maximize heat flux through a surface)
- [ ] Multi-objective: Pareto front for compliance vs. volume

**Delivers**: multi-physics constrained optimization.

### Phase 6: Advanced features (ongoing)

- [ ] Level 2: differentiable normals (Neumann BCs with moving boundaries)
- [ ] Topology changes (hole merging/splitting)
- [ ] Adaptive point insertion/removal during optimization
- [ ] Time-dependent adjoint (for transient problems)
- [ ] Parallel gradient computation (stencil-level parallelism)
- [ ] GPU-accelerated forward/adjoint solves

## What makes this realistic

The key enabling factor is that the manual adjoint architecture doesn't need to
change as we add features. Each phase adds new capabilities by extending Step 3
(coefficient extraction) or adding new parameterization layers, but the core
gradient computation (adjoint solve + local rrules + accumulation) is the same.

The stencil-level nature of the computation also means:
- **Memory**: O(N × k) not O(N²) — you never form dense global matrices
- **Parallelism**: each stencil in Step 3 and Step 4 is independent
- **Scaling**: cost per gradient is ~2× forward solve, independent of the
  number of design parameters (the key adjoint property)

## Immediate next step

Implement Phase A from `plan_manual_adjoint.md` — Dirichlet-only gradient
validation. This is the foundation everything else builds on. Estimated: 1-2
days of implementation + testing.
