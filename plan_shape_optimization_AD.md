# Shape Optimization via AD: Implementation Roadmap

This document describes the phased implementation of differentiable PDE solving in
Macchiato.jl, with shape optimization as the driving application. The goal is a
user-facing API — `gradient(sim, loss; wrt=:pts)` — backed by a correct, efficient
reverse-mode AD chain through the full RBF-FD pipeline.

**Before any work in this repo begins**, RadialBasisFunctions.jl must be updated (Phase
0 below). Phase 1 cannot start until the Phase 0 checklist is green.

paths:
RadialBasisFunctions.jl: /home/dmiotti/gitRepos/RadialBasisFunctions.jl-1

---

## Phase 0: Prerequisites in RadialBasisFunctions.jl

The AD chain for elasticity assembly calls `_build_weights` with five operators:
`Partial(2,1)`, `Partial(2,2)`, `MixedPartial(1,2)`, `Partial(1,1)`, `Partial(1,2)`.
`Partial` and `Laplacian` already have Mooncake rrule!!s. Two are missing:

| Operator | Issue | Fix |
|----------|-------|-----|
| `MixedPartial` | No `@is_primitive` or rrule!!. Mooncake falls through to `bunchkaufman!` (LAPACK), crashes. | Add backward math + rrule!! following the `Partial` pattern. |
| `Directional` | Routes through `Jacobian` which also has no rrule!!. Gradient silently terminates at any Neumann BC row. | Rewrite `_build_weights(Directional, ...)` to compose `Partial` calls — no new rrule!! needed. |

The full specification is in
`RadialBasisFunctions.jl/plan_ad_extension.md`, which includes:
- Mathematical derivation of `grad_applied_mixed_partial_wrt_x` for all basis types
  (PHS1/3/5/7, IMQ, Gaussian)
- Exact files to modify with code snippets
- Design rationale for the `Directional` rewrite
- Three test sets that must pass before this phase is complete

### Phase 0 checklist — must be green before proceeding to Phase 1

- [ ] `MixedPartial`: `grad_applied_mixed_partial_wrt_x` implemented in
  `src/solve/operator_second_derivatives.jl` for all basis types
- [ ] `MixedPartial`: `_optype`, `_get_grad_funcs`, `_get_rhs_closures` dispatch added
  to `src/solve/ad_shared.jl`
- [ ] `MixedPartial`: `@is_primitive` + `rrule!!` added to
  `ext/RadialBasisFunctionsMooncakeExt/RadialBasisFunctionsMooncakeExt.jl`
- [ ] `Directional`: `_build_weights(Directional, ...)` rewritten to compose `Partial`
  calls in `src/operators/directional.jl`
- [ ] All existing RBF tests still pass (`julia test/runtests.jl`)
- [ ] New test: `@testset "MixedPartial _build_weights"` passes (all basis types, 2D+3D)
- [ ] New test: `@testset "Directional _build_weights"` passes (constant direction,
  spatially varying, gradient w.r.t. direction vector)
- [ ] New test: `@testset "Elasticity assembly gradient"` passes — all five operators
  chained in one loss, FD/AD agreement at `rtol=1e-3`

---

## Framework Architecture

The full AD chain for shape optimization is:

```
pts → make_system_differentiable(model, pts, adjl, basis) → A(pts)   [assembly]
    → apply_dirichlet!(A, b, dirichlet_dofs)              → A_bc      [BC application]
    → PDESolveIFT(active_dofs)                            → u(pts)    [IFT solve]
    → compute_stress_field(model, u, pts, adjl, basis)    → σ(pts)    [post-process]
    → L                                                               [scalar loss]
```

### Responsibility split across packages

| Concern | Package | Status |
|---------|---------|--------|
| `_build_weights` rrule!! for `Partial`/`Laplacian` | RadialBasisFunctions.jl | Done |
| `_build_weights` rrule!! for `MixedPartial` | RadialBasisFunctions.jl | **Phase 0** |
| `_build_weights` for `Directional` via `Partial` composition | RadialBasisFunctions.jl | **Phase 0** |
| `PDESolveIFT` rrule!! (IFT through `A\b`) | Macchiato.jl | **Phase 1** |
| `make_system_differentiable` (per model) | Macchiato.jl | **Phase 1** |
| `Domain` exposing `active_dofs` mask | Macchiato.jl | Phase 2 |
| `b(pts)` differentiability (Neumann normals) | Macchiato.jl | Phase 2 |
| `gradient(sim, loss; wrt=:pts)` user API | Macchiato.jl | Phase 2 |
| Neighbor recomputation for optimization loop | Macchiato.jl | Phase 3 |

### The two IFTs

There are two distinct IFT applications in this stack — do not confuse them:

- **Stencil IFT** (RadialBasisFunctions.jl): differentiates through each stencil solve
  `A_stencil λ = b_stencil` to get `∂L/∂W → ∂L/∂pts`. Implemented for `Partial` and
  `Laplacian`; extended to `MixedPartial` in Phase 0.
- **PDE IFT** (Macchiato.jl, Phase 1): differentiates through the full-system solve
  `A(pts) u = b` to get `∂L/∂u → ∂L/∂A`. `PDESolveIFT` is this primitive. The two
  chain together: `PDESolveIFT` feeds `∂L/∂A` upstream, then the stencil IFT converts
  it to `∂L/∂pts`.

---

## Motivation

Shape optimization requires gradients `∂L/∂pts` where `L` is a functional of the PDE
solution (e.g., von Mises stress at a probe location) and `pts` are node coordinates.
With RBF-FD, moving a point changes the stencil weights, which changes the system matrix
`A(pts)`, which changes the solution `u(pts)`.

Cost via the adjoint method: 2 linear solves (forward + adjoint), independent of the
number of design variables. This is the only scalable approach for O(N) design variables.

---

## What Already Exists

### In RadialBasisFunctions.jl (nothing to add)

- `_build_weights(op, pts, pts, adjl, basis)` for all needed operators.
- Mooncake `rrule!!` for `_build_weights`: propagates `∂L/∂W.nzval → ∂L/∂pts`
  correctly for any operator.
- `find_neighbors(pts, k)` for adjacency list construction.

### In Macchiato.jl

- `LinearElasticity` model with `make_system`: assembles the 2N×2N plane-stress Navier
  system from ∂²x, ∂²y, ∂²xy operators (`src/models/mechanics.jl`).
- `lame_parameters(model)` → `(μ, λstar)`.
- `cantilever_beam_2d.jl` example: validated against the Timoshenko analytic solution.

---

## Mathematical Formulation

### Navier-Cauchy equations (2D plane stress)

```
(λ*+2μ) ∂²u_x/∂x² + μ ∂²u_x/∂y² + (λ*+μ) ∂²u_y/∂x∂y = -f_x
(λ*+μ) ∂²u_x/∂x∂y + μ ∂²u_y/∂x²  + (λ*+2μ) ∂²u_y/∂y² = -f_y
```

System matrix (2N×2N):

```
A = [ A₁₁  A₁₂ ]    A₁₁ = (λ*+2μ) W_∂²x + μ W_∂²y
    [ A₁₂  A₂₂ ]    A₁₂ = (λ*+μ)  W_∂²xy
                     A₂₂ = μ W_∂²x + (λ*+2μ) W_∂²y
```

### Stress computation

```
σ_xx = (λ*+2μ) W_∂x u_x + λ*     W_∂y u_y
σ_yy = λ*      W_∂x u_x + (λ*+2μ) W_∂y u_y
σ_xy = μ (W_∂y u_x + W_∂x u_y)
```

Von Mises stress (plane stress):

```
σ_vm[i] = sqrt(σ_xx[i]² - σ_xx[i]·σ_yy[i] + σ_yy[i]² + 3·σ_xy[i]²)
```

### Implicit Function Theorem (PDE level)

Since `A u = b` holds exactly, the total derivative is:

```
∂u/∂pts = -A⁻¹ (∂A/∂pts · u)
```

For loss `L(u)`, define adjoint `η = Aᵀ \ (∂L/∂u)`. Then:

```
∂L/∂b      = η
∂L/∂A[i,j] = -η[i] · u[j]     (rows i where A[i,:] is a PDE equation, not a BC row)
```

The gradient `∂L/∂A → ∂L/∂pts` flows through the existing `_build_weights` rrule!!.

---

## Test Geometry: Cantilever Beam

Reuse the existing `cantilever_beam_2d.jl` setup.

```
Geometry:  L × 2D rectangle, L=8, D=1
Material:  E=1e7, ν=0.3
BCs:       ALL Dirichlet (prescribed displacement from Timoshenko on all four sides)
Source:    b_rhs FIXED at the original point configuration (Phase 1 simplification)
Loss:      von Mises stress at a probe point near the fixed end (x≈0, y≈0)
```

**Phase 1 simplification**: fixing `b_rhs` means only `A(pts)` varies with geometry. The
gradient `∂L/∂pts` is genuine and non-trivial, but it captures only the stiffness
sensitivity, not the boundary traction sensitivity. This is sufficient to validate the
full AD chain. Phase 2 lifts this by making `b(pts)` also differentiable (required when
Neumann BCs carry normals that depend on point positions).

---

## Phase 1: Implementation

**Prerequisite**: Phase 0 checklist in RadialBasisFunctions.jl must be complete. Verify
by running the three new test sets described in `plan_ad_extension.md` before writing
any code here.

### Target file: `src/optimization/solve_ift.jl`

This file is the deliverable of Phase 1. It introduces `PDESolveIFT`, a generic
Mooncake primitive for differentiating through the assembled PDE linear solve. It knows
nothing about elasticity — only about which rows of `A` are live PDE equations vs
identity BC rows.

### Step 1: `make_system_differentiable`

A thin differentiable adapter that takes a flat coordinate vector and returns the raw
assembled system matrix (without BC modification). It dispatches on model type, so
`LinearElasticity` gives a 2N×2N block system.

This function intentionally operates below `Domain`/`PointCloud` — it takes stripped
scalars and plain arrays so that Mooncake can trace through it via the existing
`_build_weights` rrule!!. In Phase 2 this adapter is replaced by a differentiable path
through `make_system` proper.

```julia
import RadialBasisFunctions: _build_weights, Partial, MixedPartial
using StaticArrays, SparseArrays

function make_system_differentiable(
    ::LinearElasticity,
    pts_flat::Vector{Float64}, N::Int,
    adjl, basis, λstar::Float64, μ::Float64
)
    pts = [SVector{2}(pts_flat[2i-1], pts_flat[2i]) for i in 1:N]

    W_∂²x  = _build_weights(Partial(2, 1),    pts, pts, adjl, basis)
    W_∂²y  = _build_weights(Partial(2, 2),    pts, pts, adjl, basis)
    W_∂²xy = _build_weights(MixedPartial(1, 2), pts, pts, adjl, basis)

    A₁₁ = (λstar + 2μ) * W_∂²x + μ * W_∂²y
    A₁₂ = (λstar + μ)  * W_∂²xy
    A₂₂ = μ * W_∂²x + (λstar + 2μ) * W_∂²y

    return [A₁₁ A₁₂; A₁₂ A₂₂]
end
```

Mooncake traces through this automatically. No custom rrule needed.

### Step 2: `PDESolveIFT`

The core reusable primitive. Parameterized only on `active_dofs::BitVector` — a mask of
length equal to the total number of DOFs where `true` means "this row of A contains a
real PDE equation." Dirichlet BC rows (identity rows) are `false`. This mask is the same
for `LinearElasticity` and `SolidEnergy` — the struct knows nothing about either.

The forward solve uses a sparse LU factorization that is reused for the adjoint solve at
no extra cost.

```julia
using Mooncake, SparseArrays, LinearAlgebra

struct PDESolveIFT
    active_dofs::BitVector   # true = PDE row, false = identity BC row
end

function (s::PDESolveIFT)(A::SparseMatrixCSC{Float64,Int}, b::Vector{Float64})
    return lu(A) \ b
end

Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{PDESolveIFT, SparseMatrixCSC{Float64,Int}, Vector{Float64}}

function Mooncake.rrule!!(
    solver_cd::Mooncake.CoDual{PDESolveIFT},
    A_cd::Mooncake.CoDual{SparseMatrixCSC{Float64,Int}},
    b_cd::Mooncake.CoDual{Vector{Float64}},
)
    solver = Mooncake.primal(solver_cd)
    A      = Mooncake.primal(A_cd)
    b      = Mooncake.primal(b_cd)

    F = lu(A)
    u = F \ b
    u_cd = Mooncake.zero_fcodual(u)

    function pde_solve_pb!!(::Mooncake.NoRData)
        Δu = u_cd.dx
        η  = F' \ Δu      # adjoint solve — reuses the LU factorization
        b_cd.dx .+= η
        ΔA_nzval = A_cd.dx.data.nzval
        rows = rowvals(A)
        for j in 1:size(A, 2)
            for idx in nzrange(A, j)
                i = rows[idx]
                if solver.active_dofs[i]
                    ΔA_nzval[idx] -= η[i] * u[j]
                end
            end
        end
        return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData()
    end

    return u_cd, pde_solve_pb!!
end
```

**Constructing `active_dofs` in Phase 1** (manually, from known point indices):

```julia
function make_active_dofs_elasticity(interior_idx::Vector{Int}, N::Int)
    active = falses(2N)
    active[interior_idx]       .= true   # u_x DOFs
    active[interior_idx .+ N]  .= true   # u_y DOFs
    return active
end
```

In Phase 2 this is derived directly from `Domain` by inspecting the BC types: any
surface with a `Dirichlet` BC contributes identity rows; interior points and Neumann
surfaces contribute active rows.

**BC application** (outside the differentiable path):

Dirichlet rows are set to identity before passing `A` to `PDESolveIFT`. This step is
intentionally outside the rrule!! — those rows do not depend on `pts`, so their gradient
is always zero.

```julia
function apply_dirichlet!(A::SparseMatrixCSC, b::Vector, dirichlet_dofs::Vector{Int}, vals::Vector{Float64})
    for (k, i) in enumerate(dirichlet_dofs)
        A[i, :] .= 0.0
        A[i, i]  = 1.0
        b[i]     = vals[k]
    end
end
```

### Step 3: Differentiable stress computation

No custom rrule needed — Mooncake traces through `_build_weights` automatically:

```julia
function compute_von_mises(
    u_flat::Vector{Float64}, N::Int,
    pts_flat::Vector{Float64}, adjl, basis,
    λstar::Float64, μ::Float64
)
    pts = [SVector{2}(pts_flat[2i-1], pts_flat[2i]) for i in 1:N]
    u_x = u_flat[1:N]
    u_y = u_flat[N+1:2N]

    W_∂x = _build_weights(Partial(1, 1), pts, pts, adjl, basis)
    W_∂y = _build_weights(Partial(1, 2), pts, pts, adjl, basis)

    ∂ux_∂x = W_∂x * u_x;  ∂ux_∂y = W_∂y * u_x
    ∂uy_∂x = W_∂x * u_y;  ∂uy_∂y = W_∂y * u_y

    σ_xx = (λstar + 2μ) .* ∂ux_∂x .+ λstar .* ∂uy_∂y
    σ_yy = λstar .* ∂ux_∂x .+ (λstar + 2μ) .* ∂uy_∂y
    σ_xy = μ .* (∂ux_∂y .+ ∂uy_∂x)

    return sqrt.(σ_xx.^2 .- σ_xx .* σ_yy .+ σ_yy.^2 .+ 3 .* σ_xy.^2)
end
```

### Step 4: Loss function and example script

```julia
# --- Setup (outside loss, fixed across calls) ---
adjl   = find_neighbors(pts_original, k)          # fixed sparsity pattern
model  = LinearElasticity(E=E_val, ν=ν_val)
μ, λstar = lame_parameters(model)
basis  = PHS(3; poly_deg=3)
active = make_active_dofs_elasticity(interior_idx, N)
solver = PDESolveIFT(active)
b_rhs  = build_rhs(pts_original, dirichlet_vals)  # fixed at reference config

# --- Differentiable loss ---
function loss(pts_flat)
    A = make_system_differentiable(model, pts_flat, N, adjl, basis, λstar, μ)
    apply_dirichlet!(A, b_rhs, dirichlet_dofs, dirichlet_vals)  # not differentiated
    u = solver(A, b_rhs)
    σ_vm = compute_von_mises(u, N, pts_flat, adjl, basis, λstar, μ)
    return sum(σ_vm[probe_idx].^2)
end
```

**Fixed adjacency list**: `adjl` is computed once at the reference configuration. This is
both a correctness requirement (the sparsity pattern of `W` must not change within a
single AD trace) and a Phase 1 simplification. A full optimization loop must recompute
`adjl` between gradient steps when points move beyond a safe radius. This is a Phase 3
concern — Phase 1 only evaluates the gradient at the reference configuration.

### Step 5: Validation

```julia
using DifferentiationInterface, FiniteDifferences
import DifferentiationInterface as DI

backend = DI.AutoMooncake(; config = nothing)
grad_ad = DI.gradient(loss, backend, pts_flat)
grad_fd = FiniteDifferences.grad(FiniteDifferences.central_fdm(5, 1), loss, pts_flat)[1]

rel_err = norm(grad_ad - grad_fd) / norm(grad_fd)
@assert rel_err < 1e-3 "AD/FD mismatch: rel_err = $rel_err"
```

---

## Sanity Check at Reference Configuration

Before running AD, verify that the solver recovers the Timoshenko solution to high accuracy:

```julia
A_nom = make_system_differentiable(model, pts_flat_0, N, adjl, basis, λstar, μ)
apply_dirichlet!(A_nom, b_rhs, dirichlet_dofs, dirichlet_vals)
u_nom   = solver(A_nom, b_rhs)
ux_nom  = u_nom[1:N];  uy_nom = u_nom[N+1:2N]

err_ux = norm(ux_nom - ux_exact) / norm(ux_exact)
err_uy = norm(uy_nom - uy_exact) / norm(uy_exact)
@assert err_ux < 1e-10 && err_uy < 1e-10
```

The Timoshenko solution is a degree-3 polynomial. With `PHS(3; poly_deg=3)` the RBF
scheme reproduces it exactly — the residual is floating-point rounding noise. If the
assertion fails, the most likely cause is that `poly_deg=2` was used somewhere in the
assembly.

---

## Implementation Notes

### Operator types (RadialBasisFunctions.jl public API)

| Operator | Type | Call |
|----------|------|------|
| ∂²/∂x²  | `Partial(2, 1)` | `_build_weights(Partial(2,1), pts, pts, adjl, basis)` |
| ∂²/∂y²  | `Partial(2, 2)` | `_build_weights(Partial(2,2), pts, pts, adjl, basis)` |
| ∂²/∂x∂y | `MixedPartial(1, 2)` | `_build_weights(MixedPartial(1,2), pts, pts, adjl, basis)` |
| ∂/∂x    | `Partial(1, 1)` | `_build_weights(Partial(1,1), pts, pts, adjl, basis)` |
| ∂/∂y    | `Partial(1, 2)` | `_build_weights(Partial(1,2), pts, pts, adjl, basis)` |

Import with: `import RadialBasisFunctions: _build_weights, Partial, MixedPartial`

### Point format

Points must be `Vector{SVector{2, Float64}}` for `_build_weights`. Reconstruct from the
flat vector inside the loss closure:

```julia
pts = [SVector{2}(pts_flat[2i-1], pts_flat[2i]) for i in 1:N]
```

### Mooncake tangent for SparseMatrixCSC

The tangent of a `SparseMatrixCSC` in Mooncake is accessed as `W.dx.data.nzval` — a
`Vector{Float64}` of tangents for the nonzero entries only. The buffer is shared between
the `PDESolveIFT` rrule!! and the upstream `_build_weights` rrule!!, so accumulation
into `ΔA_nzval` is all that is needed.

### Multiple `_build_weights` calls in one loss

The loss calls `_build_weights` five times (∂²x, ∂²y, ∂²xy, ∂x, ∂y). Mooncake traces
each call independently and accumulates gradients correctly — no special handling required.

---

## Basis: Why poly_deg=3 is mandatory for this test

**Always use `PHS(3; poly_deg=3)` with `k=35` neighbors.**

Expanding the Timoshenko displacement field:

```
u(x,y) = -P/(6EI) · [6Lxy  -  3x²y  +  (2+ν)y³  -  (2+ν)D²y]
                       deg2     deg3       deg3         deg1

v(x,y) =  P/(6EI) · [3νLy²  -  3νxy²  +  (4+5ν)D²x  +  3Lx²  -  x³]
                      deg2       deg3         deg1        deg2    deg3
```

Both components contain degree-3 monomials. An RBF scheme with `poly_deg=d` reproduces
polynomials of degree ≤ d exactly:

| Basis | Reproduces Timoshenko? | Sanity check tolerance |
|-------|------------------------|------------------------|
| `PHS(3; poly_deg=2)` | No — O(h²)–O(h⁴) error | ~1e-4 (mesh-dependent) |
| `PHS(3; poly_deg=3)` | **Yes — machine precision** | < 1e-10 |

The `poly_deg=2` default silently gives an inaccurate solve:

```julia
# WRONG — silently uses poly_deg=2
W = _build_weights(Partial(2, 1), pts, pts, adjl, PHS(3))

# CORRECT
basis = PHS(3; poly_deg=3)
W = _build_weights(Partial(2, 1), pts, pts, adjl, basis)
```

Stencil size: the degree-3 monomial basis in 2D has `binomial(2+3, 2) = 10` terms. Use
`k=35` (rule of thumb: k > 3 × number of monomials).

---

## Roadmap

### Phase 2: Integration with `Domain` / `Simulation`

**Goal**: users call `gradient(sim, loss; wrt=:pts)` without touching flat arrays.

Key work:
1. **`Domain` exposes `active_dofs`**: inspect the BC type of each surface —
   `Dirichlet` → false, interior and Neumann → true. No manual index construction.
2. **`make_system` becomes differentiable**: thread `pts` through the existing
   `make_system(model, domain; basis_kw...)` call rather than using
   `make_system_differentiable`. Requires stripping Unitful coordinates at the domain
   boundary before passing to `_build_weights`.
3. **`b(pts)` differentiability**: Neumann BCs impose `A[i,:] = W_∂n[i,:]` where the
   normal `n` depends on `pts`. These rows must also flow through AD. `active_dofs`
   already marks them as true — the missing piece is making `apply_neumann!` traceable.
4. **`gradient(sim, loss; wrt=:pts)` API**: thin wrapper that builds the flat
   `pts_flat`, calls the AD backend, and returns a gradient in the same shape as the
   point cloud.

### Phase 3: Optimization loop

**Goal**: run a gradient-based optimizer to convergence.

Key work:
1. **Neighbor recomputation**: `adjl` is fixed within one gradient evaluation but must
   be updated between optimizer steps when points have moved. Introduce a
   `ShapeOptimizer` iterator that calls `find_neighbors` before each forward solve.
2. **Hole geometry with Neumann BCs**: 2D plate with an elliptical hole. Inner boundary
   = traction-free (σ·n = 0). Outer boundary = equibiaxial Dirichlet. The analytic
   optimum (Kirsch: circle minimizes maximum hoop stress) provides a convergence target.
3. **Multi-model**: the same `PDESolveIFT` primitive is valid for `SolidEnergy` (scalar,
   N×N system) — Phase 3 can demonstrate shape optimization for a heat problem to confirm
   generality.
