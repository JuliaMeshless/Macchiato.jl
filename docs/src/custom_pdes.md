# Custom PDEs

Macchiato.jl is not limited to the built-in physics models — you can define and solve **any PDE** using the same infrastructure. This tutorial walks through solving the Poisson equation on a unit square with a manufactured solution.

Differential operators come from [RadialBasisFunctions.jl](https://github.com/JuliaMeshless/RadialBasisFunctions.jl), and Macchiato re-exports the operator-building surface (`laplacian`, `partial`, `mixed_partial`, `gradient`, the [`@operator`](https://juliameshless.github.io/RadialBasisFunctions.jl/stable/guides/pde_operators/) macro, the `PHS`/`IMQ`/`Gaussian` bases, and the `weights` accessor) — so `using Macchiato` is all you need.

## The Problem

We solve the 2D Poisson equation:

```math
\nabla^2 u = f \quad \text{on} \quad \Omega = [0, 1]^2
```

with Dirichlet boundary conditions ``u = g`` on ``\partial\Omega``.

We use a **manufactured solution** to verify correctness. Choose an exact solution and derive the source term and BCs from it:

```math
u_{\text{exact}}(x, y) = \sin(\pi x) \sin(\pi y)
```
```math
f(x, y) = \nabla^2 u = -2\pi^2 \sin(\pi x) \sin(\pi y)
```

Since ``\sin(\pi x) \sin(\pi y) = 0`` on all edges of the unit square, the Dirichlet BCs are ``u = 0`` everywhere on the boundary.

## Step 1: Define the Model

Create a model struct that subtypes `AbstractModel` and implement two required methods:

```@setup poisson
import WhatsThePoint as WTP
using Unitful: m
function rectangle(Lx, Ly; n=100)
    dx, dy = Lx / n, Ly / n
    rx, ry = (dx:dx:Lx-dx), (dy:dy:Ly-dy)
    pts = vcat(
        [WTP.Point(x, zero(Ly)) for x in rx],
        [WTP.Point(Lx, y) for y in ry],
        [WTP.Point(x, Ly) for x in reverse(rx)],
        [WTP.Point(zero(Lx), y) for y in reverse(ry)]
    )
    nrms = vcat(
        fill(WTP.Vec(0.0, -1.0), length(rx)),
        fill(WTP.Vec(1.0, 0.0), length(ry)),
        fill(WTP.Vec(0.0, 1.0), length(rx)),
        fill(WTP.Vec(-1.0, 0.0), length(ry))
    )
    areas = fill(dx, length(pts))
    return pts, nrms, areas
end
```

```@example poisson
using Macchiato

struct PoissonModel{F} <: AbstractModel
    source::F  # source term f(x, t) -> value
end

# Number of solution variables (1 for scalar PDE)
Macchiato.num_vars(::PoissonModel, _) = 1

# Assemble the linear system for steady-state
function Macchiato.make_system(model::PoissonModel, domain; kwargs...)
    x = node_coordinates(domain)
    ∇² = laplacian(x; k = 40, kwargs...)
    A = weights(∇²)

    # Evaluate source term at each point
    b = [model.source(xᵢ, 0.0) for xᵢ in x]

    return A, b
end
```

That's it — just a struct and two methods. The key points:
- [`num_vars`](@ref) returns the number of unknowns per point (1 for scalar, `dim` for vector)
- `make_system` builds the system matrix `A` and right-hand side `b`; Macchiato handles BC application and solving
- [`node_coordinates`](@ref) returns the cloud's coordinates unit-stripped, ready for the operator constructors
- `weights` is the supported accessor for an operator's sparse weight matrix
- `k` is the stencil size. If omitted, RadialBasisFunctions.jl picks the minimum the basis needs (12 here), which can produce singular stencils near boundaries where neighbors are nearly collinear — pass a larger value; the built-in Macchiato models use 40

## Step 2: Solve and Verify

Boundary conditions use the generic constructors directly — no aliases or trait definitions needed:

```@example poisson
using WhatsThePoint
using Unitful: m, °, ustrip

# Manufactured solution and source term
u_exact(x) = sin(π * x[1]) * sin(π * x[2])
f_source(x, t) = -2π^2 * sin(π * x[1]) * sin(π * x[2])

# Geometry: unit square point cloud
part = PointBoundary(rectangle(1m, 1m)...)
split_surface!(part, 75°)
dx = 1/33 * m
cloud = discretize(part, ConstantSpacing(dx))

# Model
model = PoissonModel(f_source)

# BCs: use generic constructors directly
bcs = Dict(
    :surface1 => PrescribedValue(0.0),
    :surface2 => PrescribedValue(0.0),
    :surface3 => PrescribedValue(0.0),
    :surface4 => PrescribedValue(0.0),
)

# Solve
domain = Domain(cloud, bcs, model)
sim = Simulation(domain)
run!(sim)

# Verify against exact solution
# For custom models, access the solution vector via the public accessor
u_numerical = solution(sim)
u_exact_vals = u_exact.(node_coordinates(domain))
error = maximum(abs.(u_numerical .- u_exact_vals))
println("Max error: $error")
```

With 33 points per side you should see a max error on the order of ``10^{-4}`` or better.

## Multi-Term Operators: the `@operator` Macro

The Poisson example needs only a bare Laplacian, but most PDEs combine several terms. Rather than building each operator and summing weight matrices by hand, write the operator in mathematical notation with `@operator` — the whole expression is fused into a **single weight matrix**:

```julia
struct AdvectionDiffusion{T, V, S} <: AbstractModel
    ν::T          # diffusivity
    c::V          # advection velocity vector
    source::S     # source term f(x, t) -> value
end

Macchiato.num_vars(::AdvectionDiffusion, _) = 1

# ν∇²u − c·∇u = f, assembled in one fused weights build
function Macchiato.make_system(model::AdvectionDiffusion, domain; kwargs...)
    (; ν, c) = model
    x = node_coordinates(domain)
    op = (@operator ν * ∇² - c ⋅ ∇)(x; k = 40, kwargs...)
    b = [model.source(xᵢ, 0.0) for xᵢ in x]
    return weights(op), b
end
```

Recognized symbols include `∇²`/`Δ`, `∂(dim)`, `∂²(dim)`, `∂(i, j)` (mixed partials), `∇ ⋅ (κ * ∇)` (diffusion), `c ⋅ ∇` (advection), and `f`/`I` (identity), plus scalar coefficients. Built operators also compose directly — `α * op` and `op₁ + op₂` combine the existing weight matrices without re-collocation, which is how the built-in `LinearElasticity` model assembles its blocks. See the [Building PDE Operators guide](https://juliameshless.github.io/RadialBasisFunctions.jl/stable/guides/pde_operators/) in RadialBasisFunctions.jl for the full vocabulary.

## Optional: Named BC Aliases

For readability, you can define named aliases that wrap the generic constructors:

```@example poisson
PoissonValue(value) = PrescribedValue(value)
PoissonFlux(flux) = PrescribedFlux(flux)
PoissonZeroFlux() = ZeroFlux()
```

These are purely syntactic sugar — they construct the same generic types ([`PrescribedValue`](@ref), [`PrescribedFlux`](@ref), [`ZeroFlux`](@ref)) that power all built-in BCs.

## Key Takeaways

1. **Any PDE works.** Define an `AbstractModel` subtype and implement `num_vars` and `make_system`. That's all Macchiato needs.

2. **No boilerplate traits required.** Generic BC types (`PrescribedValue`, `PrescribedFlux`, `ZeroFlux`) dispatch on the mathematical hierarchy (`Dirichlet`/`Neumann`/`Robin`), so they work with any model — no equation-type trait needed.

3. **Operators come from RadialBasisFunctions.jl and are re-exported.** Use `laplacian`, `partial`, `mixed_partial`, `gradient`, or compose multi-term operators with `@operator` — `using Macchiato` brings them all in. Access an operator's sparse matrix with `weights(op)`, never the internal field.

4. **Generic BC types work directly.** `PrescribedValue(value)`, `PrescribedFlux(flux)`, and `ZeroFlux()` work out of the box for custom PDEs. Named aliases like `Temperature` are just convenience wrappers used by the built-in models.

5. **Transient support.** For time-dependent PDEs, implement `make_f(model, domain)` instead of `make_system` to return an ODE right-hand side `f(du, u, p, t)`. Macchiato integrates it with OrdinaryDiffEq.jl automatically. See the [Package Design](@ref) page for details on the transient path. One caveat: in the transient path only Dirichlet surfaces are enforced by the framework — Neumann/Robin flux surfaces must be folded into your operator inside `make_f` via `Macchiato.build_neumann_diffusion(domain; k, operator = ...)`, or they are silently ignored. The `operator` keyword accepts a builder closure for non-Laplacian operators; `examples/niederer_benchmark/niederer_benchmark.jl` is the worked anisotropic example.
