# Custom PDEs

Macchiato.jl is not limited to the built-in physics models — you can define and solve **any PDE** using the same infrastructure. This tutorial walks through solving the Poisson equation on a unit square with a manufactured solution.

## The Problem

We solve the 2D Poisson equation:

```
∇²u = f    on Ω = [0, 1]²
```

with Dirichlet boundary conditions ``u = g`` on ``∂Ω``.

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
Macchiato._num_vars(::PoissonModel, _) = 1

# Assemble the linear system for steady-state
function Macchiato.make_system(model::PoissonModel, domain; kwargs...)
    coords = Macchiato._coords(domain.cloud)
    ∇² = laplacian(Macchiato._ustrip(coords); k=40, kwargs...)
    A = ∇².weights

    # Evaluate source term at each point
    b = map(coords) do pt
        model.source(ustrip.(pt), 0.0)
    end

    return A, b
end
```

That's it — just a struct and two methods. The key points:
- `_num_vars` returns the number of unknowns per point (1 for scalar, `dim` for vector)
- `make_system` builds the system matrix `A` and right-hand side `b`; Macchiato handles BC application and solving

The `laplacian` function comes from [RadialBasisFunctions.jl](https://github.com/JuliaMeshless/RadialBasisFunctions.jl) — it builds a sparse meshless Laplacian operator from the point cloud. You can use any operator from that package: `partial`, `gradient`, or even custom differential operators.

## Step 2: Solve and Verify

Boundary conditions use the generic constructors directly — no aliases or trait definitions needed:

```@example poisson
using WhatsThePoint
using WhatsThePoint: coords
using RadialBasisFunctions
using Unitful: m, °, ustrip

# Manufactured solution and source term
u_exact(x) = sin(π * x[1]) * sin(π * x[2])
f_source(x, t) = -2π^2 * sin(π * x[1]) * sin(π * x[2])

# Geometry: unit square point cloud
part = PointBoundary(rectangle(1m, 1m)...)
split_surface!(part, 75°)
dx = 1/33 * m
cloud = discretize(part, ConstantSpacing(dx), alg=VanDerSandeFornberg())

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
# For custom models, access the raw solution vector directly
u_numerical = sim._solution
pts = points(cloud)
u_exact_vals = [u_exact(ustrip.([coords(pt).x, coords(pt).y])) for pt in pts]
error = maximum(abs.(u_numerical .- u_exact_vals))
println("Max error: $error")
```

With 33 points per side you should see a max error on the order of ``10^{-4}`` or better.

## Optional: Named BC Aliases

For readability, you can define named aliases that wrap the generic constructors:

```@example poisson
PoissonValue(value) = PrescribedValue(value)
PoissonFlux(flux) = PrescribedFlux(flux)
PoissonZeroFlux() = ZeroFlux()
```

These are purely syntactic sugar — they construct the same generic types ([`PrescribedValue`](@ref), [`PrescribedFlux`](@ref), [`ZeroFlux`](@ref)) that power all built-in BCs.

## Key Takeaways

1. **Any PDE works.** Define an `AbstractModel` subtype and implement `_num_vars` and `make_system`. That's all Macchiato needs.

2. **No boilerplate traits required.** Generic BC types (`PrescribedValue`, `PrescribedFlux`, `ZeroFlux`) dispatch on the mathematical hierarchy (`Dirichlet`/`Neumann`/`Robin`), so they work with any model — no equation-type trait needed.

3. **Operators come from RadialBasisFunctions.jl.** Use `laplacian`, `partial`, `gradient`, or build custom differential operators — they all return sparse weight matrices ready for assembly.

4. **Generic BC types work directly.** `PrescribedValue(value)`, `PrescribedFlux(flux)`, and `ZeroFlux()` work out of the box for custom PDEs. Named aliases like `Temperature` are just convenience wrappers used by the built-in models.

5. **Transient support.** For time-dependent PDEs, implement `make_f(model, domain)` instead of `make_system` to return an ODE right-hand side `f(du, u, p, t)`. Macchiato integrates it with OrdinaryDiffEq.jl automatically. See the [Package Design](@ref) page for details on the transient path.
