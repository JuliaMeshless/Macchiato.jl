# Session Recap: AD Extension Phase 0 & Phase 1

## What Was Done

### Phase 0 — RadialBasisFunctions.jl

The AD chain for linear elasticity assembly requires differentiating `_build_weights`
with five operators: `Partial(2,1)`, `Partial(2,2)`, `MixedPartial(1,2)`, `Partial(1,1)`,
`Partial(1,2)`. All five now have correct Mooncake rrule!!s.

#### `MixedPartial` — new operator, full AD support

**`src/solve/operator_second_derivatives.jl`**
- Added `grad_applied_mixed_partial_wrt_x` for PHS1/3/5/7, IMQ, Gaussian
- Added `grad_applied_mixed_partial_wrt_xi` (negation of wrt_x)
- Added dispatch function `grad_applied_mixed_partial_wrt_x(basis, d1, d2)`

**`src/solve/backward.jl`**
- Added `_backward_mixed_partial_polynomial_section!` using falling-factorial coefficients
- Added `_third_derivative_constant_coeff(e, c)` helper
- Added `_infer_poly_degree_from_nmon(D, nmon)` helper

**`src/solve/ad_shared.jl`**
- Added `_optype(::MixedPartial) = MixedPartial`
- Added `_get_grad_funcs(::Type{<:MixedPartial}, basis, ℒ)`
- Added `_get_rhs_closures(::Type{<:MixedPartial}, ℒ, basis)`

**`ext/RadialBasisFunctionsMooncakeExt/RadialBasisFunctionsMooncakeExt.jl`**
- Added `@is_primitive` for `_build_weights` with `MixedPartial`
- Extended existing rrule!!s to cover `Union{Laplacian, Partial, MixedPartial}`

#### `Partial(2, dim)` — second-order partial, AD support added

The existing `Partial` dispatch only handled `order=1`. For elasticity we need `order=2`.

**`src/solve/operator_second_derivatives.jl`**
- Added `grad_applied_second_partial_wrt_x` for PHS1/3/5/7, IMQ, Gaussian
- Added dispatch `grad_applied_second_partial_wrt_x(basis, dim)` / `_wrt_xi`

**`src/solve/backward.jl`**
- Added `_backward_second_partial_polynomial_section!`
- Added shape-parameter derivative `∂SecondPartial_φ_∂ε`

**`src/solve/ad_shared.jl`**
- Updated `_get_grad_funcs(::Type{<:Partial}, ...)` to dispatch on `ℒ.order`
- Updated `_get_rhs_closures(::Type{<:Partial}, ...)` to dispatch on `ℒ.order`

#### `Directional` — rewrite for automatic differentiability

`_build_weights(Directional, ...)` previously routed through `Jacobian` which had no
rrule!!. Rewritten in `src/operators/directional.jl` to compose `Partial(1, d)` calls,
making it automatically differentiable via the existing `Partial` rrule!!:

```julia
function _combine_partial_weights(v, data, eval_points, adjl, basis, Dim)
    if length(v) == Dim      # constant direction
        return sum(1:Dim) do d
            v[d] * _build_weights(Partial(1, d), data, eval_points, adjl, basis)
        end
    else                     # spatially-varying direction
        return sum(1:Dim) do d
            Diagonal(getindex.(v, d)) * _build_weights(Partial(1, d), ...)
        end
    end
end
```

Hermite path (with Neumann BCs) is unchanged.

#### New tests in `test/extensions/mooncake_ext.jl`

- `"MixedPartial _build_weights"` — all PHS/IMQ/Gaussian basis types, 2D and 3D
- `"Directional _build_weights"` — constant direction, spatially varying, gradient w.r.t. direction vector
- `"Elasticity assembly gradient (all 5 operators)"` — all five operators chained in one loss

---

### Phase 1 — Macchiato.jl

**Goal**: differentiable PDE solve for linear elasticity via the implicit function theorem.

#### `src/optimization/solve_ift.jl` (new file)

Four exported functions:

| Function | Role |
|----------|------|
| `make_system_differentiable(::LinearElasticity, pts_flat, N, adjl, basis, λstar, μ)` | Assemble raw 2N×2N Navier system; Mooncake traces through the five `_build_weights` calls |
| `PDESolveIFT(active_dofs)` + `(A, b) -> u` | Callable struct for `A\b`; has custom rrule!! via Mooncake extension |
| `make_active_dofs_elasticity(interior_idx, N)` | Build `active_dofs` BitVector for 2D elasticity |
| `apply_dirichlet!(A, b, dofs, vals)` | Set BC rows to identity; has Mooncake primitive (zero gradient) |
| `compute_von_mises(u_flat, N, pts_flat, adjl, basis, λstar, μ)` | Von Mises stress field; fully differentiable via `_build_weights` rrule!! |

#### `ext/MacchiatoMooncakeExt.jl` (new file)

Two Mooncake primitives:

**`apply_dirichlet!`** — in-place modification of BC rows. Mooncake primitive with a
pullback that zeros out gradient contributions at BC rows (avoiding sparse-matrix
`setindex!` tracing issues). Gradient through these rows is zero by construction.

**`PDESolveIFT`** — IFT through `A\b`. Pullback implements:
```
η = Fᵀ \ Δu                          (adjoint solve, reuses LU factorization)
Δb  .+= η
ΔA.nzval[idx] -= η[i] * u[j]         (only for active_dofs[i] == true)
```
The `active_dofs` mask ensures Dirichlet identity rows never contribute to `ΔA`.

#### `Project.toml` — Mooncake added as weak dependency

```toml
[weakdeps]
Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"

[extensions]
MacchiatoMooncakeExt = "Mooncake"
```

#### `examples/shape_optimization_ad_phase1.jl` (new file)

End-to-end validation script for the full AD chain:
- Regular 9×5 grid on [0,8]×[-1,1]
- Boundary points: Dirichlet with Timoshenko exact solution
- Interior points: active PDE equations
- Sanity check: `PHS(3; poly_deg=3)` recovers Timoshenko to machine precision
- Validation: `‖grad_AD - grad_FD‖ / ‖grad_FD‖ < 1e-3`

---

## Architecture Summary

```
pts_flat
  │
  ▼
make_system_differentiable(...)        [5× _build_weights rrule!!]
  │  A = [(λ*+2μ)W_∂²x + μW_∂²y,   (λ*+μ)W_∂²xy]
  │      [(λ*+μ)W_∂²xy,   μW_∂²x + (λ*+2μ)W_∂²y]
  ▼
apply_dirichlet!(A, b, ...)            [Mooncake primitive, zero gradient]
  ▼
PDESolveIFT(active_dofs)(A, b)         [IFT rrule!!]
  │  u = A\b
  ▼
compute_von_mises(u, pts_flat, ...)    [2× _build_weights rrule!!]
  │  σ_vm[i] = √(σ_xx² - σ_xx·σ_yy + σ_yy² + 3σ_xy²)
  ▼
L = loss(σ_vm)
```

Backward: Mooncake unwinds the tape, the three Mooncake primitives
(`PDESolveIFT`, `apply_dirichlet!`, `_build_weights`) handle the reverse passes.
The stencil IFT in `_build_weights` converts `∂L/∂W.nzval → ∂L/∂pts`.

---

## Pending (Phase 2+)

- Neumann BC differentiability: the Directional rewrite makes `_build_weights` for
  normal derivatives differentiable. Making `make_bc!` for `TractionFree`/`Traction`
  work inside Mooncake's trace is the remaining Phase 2 work.
- `Domain`-aware `active_dofs` construction (currently manual).
- `gradient(sim, loss; wrt=:pts)` user-facing API.
- Optimization loop with neighbor recomputation between steps (Phase 3).
