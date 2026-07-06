# Symbolic PDE Definition API

## Context

The current custom PDE API (`docs/src/custom_pdes.md`) requires users to manually build RBF operators, assemble sparse matrices, and handle coordinate extraction. This is the main pain point — users must know the RadialBasisFunctions.jl API (`laplacian`, `partial`, `custom`, `find_neighbors`, etc.) to define any new PDE.

**Goal**: Let users write `Dx(Dx(u)) + Dy(Dy(u)) ~ f(x,y)` using Symbolics.jl and have Macchiato assemble the system automatically. BCs stay with existing types (`PrescribedValue`, `ZeroFlux`, etc.) — they're already clean.

**Scope**: Scalar linear steady-state PDEs. Constant coefficients only. Coupled systems, nonlinear, transient, and function-valued coefficients follow later.

## Target API

```julia
using Macchiato, Symbolics, WhatsThePoint, Unitful: m, °

@variables x y
@variables u(x, y)
Dx = Differential(x); Dy = Differential(y)

f(x, y) = -2π^2 * sin(π*x) * sin(π*y)
eq = Dx(Dx(u(x,y))) + Dy(Dy(u(x,y))) ~ f(x, y)
model = SymbolicPDE(eq)

# BCs use existing types — no symbolic BCs in v1
bcs = Dict(
    :surface1 => PrescribedValue(0.0),
    :surface2 => PrescribedValue(0.0),
    :surface3 => ZeroFlux(),
    :surface4 => PrescribedValue((x,t) -> sin(π*x[1])),
)

cloud = discretize(PointBoundary(rectangle(1m, 1m)...), ConstantSpacing(1/33*m))
domain = Domain(cloud, bcs, model)
sim = Simulation(domain)
run!(sim)
```

## Architecture

Two layers:
1. **Core IR** (`src/models/symbolic_pde.jl`): `DiffOpTerm`, `SymbolicPDE <: AbstractModel`, `make_system` — no Symbolics.jl dependency. Uses RBF operator algebra (`Partial`, `Laplacian`, `ScaledOperator`, `Custom{0}`) to compose a single `AbstractOperator` from the IR, then builds one `RadialBasisOperator` via `custom()` or `laplacian()`.
2. **Package extension** (`ext/MacchiatoSymbolicsExt.jl`): parses `Symbolics.Equation` → IR via `SymbolicPDE(eq)` constructor

---

## Step 0: Remove dead `AbstractOperator` — `src/Macchiato.jl`

`Macchiato.jl:119-120` defines and exports `abstract type AbstractOperator end` which is never subtyped or used anywhere in the codebase. It shadows `RadialBasisFunctions.AbstractOperator{N}` (imported via `using RadialBasisFunctions` on line 16), which the new `make_system` code needs for operator algebra dispatch.

**Action**: Remove both lines (the `abstract type` definition and its `export`). This is technically a breaking change to the public API, but the type has no subtypes and no known downstream usage.

After removal, `Partial`, `Laplacian`, `ScaledOperator`, `Custom`, `Identity`, and `RadialBasisFunctions.AbstractOperator{N}` are all available unqualified in all `src/` files.

## Step 1: Core IR types — `src/models/symbolic_pde.jl` (new file)

```julia
struct DiffOpTerm
    coeff::Float64
    orders::Vector{Int}   # orders[i] = derivative order w.r.t. dimension i
end

struct SymbolicPDE{S} <: AbstractModel
    terms::Vector{DiffOpTerm}
    source::S              # nothing, or (x, t) -> Float64
    field_name::Symbol     # default :u, extracted from Symbolics dep var
end
```

Keyword constructor for manual IR construction (validates non-empty terms, consistent dimensionality, no zero-order terms):
```julia
function SymbolicPDE(terms::Vector{DiffOpTerm}; source = nothing, field_name = :u)
```

Required model interface methods:
- `_num_vars(::SymbolicPDE, _) = 1`
- `_field_index_in_model(m::SymbolicPDE, field, dim)` — returns `1` if `field === m.field_name`, else `nothing`
- `make_f(::SymbolicPDE, ...) -> throw ArgumentError` (steady-state guard)
- `Base.show(io, ::SymbolicPDE)`

### Design decision: no `AbstractOperator` constructor on `SymbolicPDE`

The `@operator` macro could enable `SymbolicPDE(@operator(∇²); source=f)`. **Skip this in v1** because:
- `DiffOpTerm` IR is introspectable (validation, display, Laplacian detection); composed `Custom{0}` is an opaque closure
- Dual representation (terms OR operator) doubles `make_system` code paths
- Users wanting `@operator` can use `custom()` directly (existing "Custom PDEs" workflow)

## Step 2: `make_system` with operator algebra — same file

**Sign convention**: `L[u] ~ g` assembles as `Au = b` where `b = g` (matches `SolidEnergy` convention). Note: `LinearElasticity` uses the opposite convention (`L[u] + f = 0` → `b = -f`); `SymbolicPDE` follows `SolidEnergy` since the user writes the equation as `L[u] ~ g` directly.

### Overview

Instead of manually building sparse weight matrices per-term and summing `coeff * op.weights`, the updated approach:
1. Converts the `DiffOpTerm` IR to a single composed `AbstractOperator` using RBF operator algebra (`+`, `*`)
2. Builds one `RadialBasisOperator` via `custom()` (general) or `laplacian()` (optimized path)
3. Extracts `.weights` once

```julia
function make_system(model::SymbolicPDE, domain; kwargs...)
    coords = _ustrip(_coords(domain.cloud))
    n = length(coords)
    k = get(kwargs, :k, 40)
    adjl = find_neighbors(coords, k)

    dim = length(first(coords))
    op = _terms_to_operator(model.terms, dim)
    rbf_op = _build_rbf_operator(op, coords, k, adjl; kwargs...)
    A = rbf_op.weights

    b = if model.source === nothing
        zeros(eltype(A), n)
    else
        map(c -> model.source(c, 0.0), coords)
    end

    return A, b
end
```

### Helper: `_terms_to_operator(terms, dim)` → `AbstractOperator`

Converts the full `DiffOpTerm` vector into a single composed RBF operator:

1. **Group terms** by `orders` vector — combine coefficients for identical derivative patterns (same as before)
2. **Detect Laplacian** via `_detect_laplacian(groups, dim)` — returns the shared coefficient, or `nothing`
   - If `coeff == 1.0` → return `Laplacian()`
   - If `coeff != 1.0` → return `coeff * Laplacian()` (produces `ScaledOperator`)
3. **General case** — convert each group to `coeff * _orders_to_operator(orders, dim)`, then `reduce(+, ops)`

### Helper: `_orders_to_operator(orders, dim)` → `AbstractOperator`

Converts a single derivative pattern to an operator type:

- **1 nonzero entry** → `Partial(order, dim_index)` (e.g., `[0, 2, 0]` → `Partial(2, 2)`)
- **2 nonzero entries, both == 1** (mixed partial ∂²/∂xᵢ∂xⱼ) → guard `dim == 2` (3D mixed partials not yet supported; throw `ArgumentError`), then `Custom{0}(_ℒ_mixed_partial)`
- **Otherwise** → `ArgumentError`

Reuses `_ℒ_mixed_partial` from `src/models/mechanics.jl:57-123` (already in scope since both files are `include`d into the `Macchiato` module — mechanics.jl is included first at line 82 of `Macchiato.jl`).

### Helper: `_build_rbf_operator(op, coords, k, adjl; kwargs...)` → `RadialBasisOperator`

Dispatches on the composed operator type:

```julia
function _build_rbf_operator(::Laplacian, coords, k, adjl; kwargs...)
    return laplacian(coords; k = k, adjl = adjl, kwargs...)
end

function _build_rbf_operator(op, coords, k, adjl; kwargs...)
    return custom(coords, op; k = k, adjl = adjl, kwargs...)
end
```

The pure `Laplacian()` case uses the dedicated `laplacian()` path (which constructs `RadialBasisOperator(Laplacian(), data)` with Laplacian-specific weight computation). Everything else — including `ScaledOperator(c, Laplacian())` — goes through `custom()`, which still evaluates the Laplacian-specific RBF formulas under the hood via `ScaledOperator(basis) → Laplacian()(basis) → ∇²(basis)`. The only difference is the top-level dispatch in `RadialBasisOperator` construction, which is negligible for one-shot assembly.

### Helper: `_detect_laplacian(groups, dim)` → `Float64` or `nothing`

Returns the shared coefficient when `length(groups) == dim` AND each group is a pure 2nd-order partial in a different dimension AND all coefficients are equal. Returns `nothing` otherwise.

## Step 3: Register in module — `src/Macchiato.jl`

After line 83 (`export LinearElasticity, lame_parameters`), add:
```julia
include("models/symbolic_pde.jl")
export SymbolicPDE, DiffOpTerm
```

Add `_field_index_in_model` for `SymbolicPDE` in `src/set.jl` (after line 141, following the `IncompressibleNavierStokes` method):
```julia
function _field_index_in_model(model::SymbolicPDE, field::Symbol, dim)
    return field === model.field_name ? 1 : nothing
end
```

Add a generic `solution` extractor in `src/set.jl` (after existing accessors like `temperature`, `displacement`, etc.) since `SymbolicPDE` has no dedicated accessor:
```julia
"""
    solution(sim, field::Symbol) -> Vector{Float64}

Extract a named field from simulation results. Works with any model that
registers the field via `_field_index_in_model`.
"""
function solution(sim, field::Symbol)
    _has_field(sim, field) || throw(ArgumentError("Simulation does not have field :$field"))
    indices = _field_indices(sim, field)
    return _get_solution_vector(sim)[indices]
end
```
Export `solution` from `src/Macchiato.jl`.

## Step 4: Project.toml — add Symbolics weak dependency

Add after the `[compat]` section in `Project.toml`:
```toml
[weakdeps]
Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"

[extensions]
MacchiatoSymbolicsExt = "Symbolics"
```

Add to `[compat]`:
```toml
Symbolics = "5, 6"
```

Add `Symbolics` to `test/Project.toml` deps (needed for end-to-end tests that use the extension):
```toml
Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"
```

## Step 5: Package extension — `ext/MacchiatoSymbolicsExt.jl` (new file)

Create `ext/` directory. Single responsibility: parse `Symbolics.Equation` into `SymbolicPDE`.

```julia
module MacchiatoSymbolicsExt

using Macchiato: Macchiato, SymbolicPDE, DiffOpTerm
using Symbolics
using Symbolics: Differential
using SymbolicUtils: iscall, operation, arguments, Symbolic

function Macchiato.SymbolicPDE(eq::Symbolics.Equation)
    lhs, rhs = eq.lhs, eq.rhs
    dep_var, indep_vars = _extract_variables(lhs)
    _validate_rhs(rhs, dep_var)
    terms = _parse_lhs(lhs, dep_var, indep_vars)
    source = _build_source(rhs, indep_vars)
    field_name = Symbol(dep_var)
    return SymbolicPDE(terms; source = source, field_name = field_name)
end

end
```

### Parsing algorithm

**`_extract_variables(lhs)`**: Walk the expression tree to find the dependent variable application `u(x, y)`. The function symbol is the dep var, its arguments are the indep vars. Validate exactly one dependent variable found.

**`_parse_lhs(expr, dep_var, indep_vars)`**:
1. Flatten `Add` nodes → list of additive terms
2. For each term, call `_parse_term`:
   - `_peel_coefficient(term)` — separate numeric factor from derivative chain (handle `Mul` nodes, default coeff = 1.0, handle negation via `Mul(-1, ...)`)
   - `_peel_derivatives!(orders, expr, indep_vars)` — recursively unwrap `Differential` applications, increment `orders[dim]` for each
   - Validate the inner expression is `dep_var(indep_vars...)`
   - Return `DiffOpTerm(coeff, orders)`

**`_validate_rhs(rhs, dep_var)`**: Recursively check `rhs` doesn't contain `dep_var`. Throw `ArgumentError` if it does.

**`_build_source(rhs, indep_vars)`**:
- If `rhs` is a `Number`: return `nothing` for 0, or `(x, t) -> Float64(rhs)` for constants
- Otherwise: compile via `Symbolics.build_function(rhs, indep_vars...; expression = Val{false})` which produces `f(x_val, y_val, ...)`. Wrap as `(x, t) -> f(x...)` to match Macchiato's `(coords, t) -> value` convention.

### Linearity validation

During `_peel_coefficient`: if a `Mul` node contains more than one non-numeric symbolic factor, it's nonlinear (e.g., `u * Dx(u)`). Throw `ArgumentError("Non-linear term detected")`.

## Step 6: Tests — `test/models/test_symbolic_pde.jl` (new file)

Add `"models/test_symbolic_pde.jl"` to `test/runtests.jl` testfiles array.

### Test group 1: Core IR (no Symbolics dependency)
- Construct `SymbolicPDE` from `DiffOpTerm` vectors manually
- `_num_vars` returns 1
- `_field_index_in_model` returns 1 for `:u`, nothing for `:T`
- `make_f` throws `ArgumentError`
- Constructor rejects: empty terms, mismatched dimensionality, zero-order terms

### Test group 2: Laplacian detection
- `[DiffOpTerm(1.0, [2,0]), DiffOpTerm(1.0, [0,2])]` → returns `1.0`
- `[DiffOpTerm(3.0, [2,0]), DiffOpTerm(3.0, [0,2])]` → returns `3.0`
- `[DiffOpTerm(1.0, [2,0]), DiffOpTerm(2.0, [0,2])]` → returns `nothing` (different coefficients)
- `[DiffOpTerm(1.0, [2,0])]` → returns `nothing` (only 1 term for 2D)

### Test group 3: Operator conversion
- `_orders_to_operator([2, 0], 2)` returns `Partial(2, 1)`
- `_orders_to_operator([0, 2], 2)` returns `Partial(2, 2)`
- `_orders_to_operator([1, 1], 2)` returns a `Custom{0}` (mixed partial)
- `_orders_to_operator([1, 1], 3)` throws `ArgumentError` (3D mixed unsupported)
- `_terms_to_operator` with Laplacian terms returns `Laplacian()` or `ScaledOperator`
- `_terms_to_operator` with non-Laplacian terms returns a composed operator

### Test group 4: End-to-end Poisson MoMS (requires Symbolics)
- Test file setup: `include("../end_2_end/2d_square.jl")` and `using Symbolics` at the top
- Mirror existing `test/end_2_end/2d_Laplacian_MoMS.jl` pattern
- Use `SymbolicPDE(Dx(Dx(u)) + Dy(Dy(u)) ~ f(x,y))` with polynomial manufactured solution `u = x(1-x) + y(1-y)`, source `f = -4`
- BCs: `PrescribedValue` with analytical solution on all sides
- Extract solution via `solution(sim, :u)` (the generic accessor)
- Verify: boundary error < 1e-10, L2 error < 5e-2, L∞ error < 1e-1
- Reuse `create_2d_square_domain` helper from `test/end_2_end/2d_square.jl`

### Test group 5: Parsing edge cases (requires Symbolics)
- Equation with constant coefficient: `3*Dx(Dx(u)) + Dy(Dy(u)) ~ 0`
- First-order terms: `Dx(u) + Dy(u) ~ 1.0` (advection)
- Mixed partial: `Dx(Dy(u)) ~ 0` in 2D
- RHS containing dependent variable throws `ArgumentError`
- Zero-order term (no derivatives) throws `ArgumentError`

---

## Files to create
- `src/models/symbolic_pde.jl` — core IR types, operator conversion helpers, `make_system`
- `ext/MacchiatoSymbolicsExt.jl` — Symbolics expression parsing
- `test/models/test_symbolic_pde.jl` — unit + integration tests

## Files to modify
- `src/Macchiato.jl:119-120` — **remove** dead `abstract type AbstractOperator end` + its `export` (resolves name collision with `RadialBasisFunctions.AbstractOperator{N}`)
- `src/Macchiato.jl:83` — add `include` + `export` after mechanics.jl; export `solution`
- `src/set.jl:141` — add `_field_index_in_model(::SymbolicPDE, ...)` after IncompressibleNavierStokes method
- `src/set.jl` — add generic `solution(sim, field)` extractor after existing accessors
- `Project.toml` — add `[weakdeps]`, `[extensions]`, and Symbolics compat entry
- `test/Project.toml` — add `Symbolics` to `[deps]`
- `test/runtests.jl` — add test file to testfiles array

## Key code reused

| What | Where | Purpose |
|---|---|---|
| `Partial`, `Laplacian`, `ScaledOperator`, `Custom{0}` | RBF exports | Operator types for algebra composition |
| `+`, `*` on `AbstractOperator` | `RBF/operator_algebra.jl` | Compose DiffOpTerms into single operator |
| `custom(data, op::AbstractOperator; kw...)` | `RBF/custom.jl:66-68` | Build `RadialBasisOperator` from composed op |
| `laplacian(data; kw...)` | `RBF/laplacian.jl` | Optimized Laplacian-specific weight path |
| `find_neighbors(data, k)` | `RBF/utils.jl` | Shared adjacency list |
| `_ℒ_mixed_partial` | `Macchiato/mechanics.jl:57-123` | 2D mixed partial custom op |
| `_coords`, `_ustrip` | `Macchiato/utils.jl` | Coordinate extraction + unit stripping |
| `PrescribedValue`, `ZeroFlux` | Macchiato BC types | Unchanged |
| `create_2d_square_domain` | `test/end_2_end/2d_square.jl` | Test geometry helper |

## Verification

1. `using Macchiato` without `using Symbolics` — no load-time impact, no errors
2. Solve Poisson on unit square with MoMS — match existing test thresholds
3. Verify `SymbolicPDE` with manually constructed IR (no Symbolics) produces correct A matrix by comparing against `laplacian().weights` directly
4. Run Runic on all new files before committing (CI enforces formatting)
