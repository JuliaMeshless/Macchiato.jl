```@meta
CurrentModule = Macchiato
```

# Macchiato.jl

Solve partial differential equations on scattered point clouds — no mesh required.

Macchiato.jl is a **general-purpose meshless PDE framework**. Define any PDE by implementing a small model interface, and Macchiato handles operator assembly, boundary condition application, and time integration. The package ships with ready-to-use models for heat transfer, linear elasticity, and fluid dynamics — but these are convenience built-ins, not the whole story. See [Custom PDEs](@ref) to learn how to solve your own equations.

## Quick Start

```julia
using WhatsThePoint, Macchiato
using Unitful: m, °

# 1. Geometry: 1m × 1m rectangle point cloud
part = PointBoundary(rectangle(1m, 1m)...)
split_surface!(part, 75°)
cloud = discretize(part, ConstantSpacing(1/33 * m), alg=VanDerSandeFornberg())

# 2. Boundary conditions
bcs = Dict(
    :surface1 => Temperature(0.0),    # bottom
    :surface2 => Temperature(0.0),    # right
    :surface3 => Temperature(100.0),  # top
    :surface4 => Temperature(0.0),    # left
)

# 3. Solve
domain = Domain(cloud, bcs, SolidEnergy(k=1.0, ρ=1.0, cₚ=1.0))
sim = Simulation(domain)
run!(sim)

# 4. Extract results
T = temperature(sim)
```

## Gallery

| Heat Conduction | Cantilever Beam |
|:---:|:---:|
| ![2D temperature field](assets/heat_2d.png) | ![2D beam displacement](assets/cantilever_beam_2d.png) |
| Steady-state temperature on a unit square | Displacement magnitude under end shear |

See the [Examples](@ref) page for complete worked examples with visualization code.

## Why Meshless Methods?

**Meshless methods** operate on scattered point clouds with no connectivity requirements:

- **Simple geometry handling** — drop points on the boundary and fill the interior; no element quality concerns
- **Easy refinement** — add more points where you need accuracy; no remeshing required
- **Natural for moving boundaries** — points move freely without topological constraints

Macchiato.jl uses **radial basis function (RBF)** collocation, where differential operators are approximated at each point using its local neighborhood of nearest neighbors.

## The JuliaMeshless Ecosystem

Macchiato.jl is part of the [JuliaMeshless](https://github.com/JuliaMeshless) organization — three composable packages that form a complete simulation pipeline:

```
┌─────────────────────┐     ┌──────────────────────────┐     ┌─────────────────────────────┐
│   WhatsThePoint.jl  │     │ RadialBasisFunctions.jl  │     │  Macchiato.jl               │
│                     │     │                          │     │                             │
│  Boundary creation  │────▶│  RBF interpolation       │────▶│  PDE model interface        │
│  Surface splitting  │     │  Differential operators   │     │  Boundary conditions        │
│  Interior fill      │     │  (∇², ∂/∂x, custom)      │     │  Simulation & time stepping │
│  Point repulsion    │     │  KNN stencil selection    │     │  Field extraction & VTK I/O │
└─────────────────────┘     └──────────────────────────┘     └─────────────────────────────┘
        Geometry                   Numerics                      PDE Framework
```

- [**WhatsThePoint.jl**](https://github.com/JuliaMeshless/WhatsThePoint.jl) — Point cloud generation from geometric primitives, surface splitting, interior fill, and point repulsion.
- [**RadialBasisFunctions.jl**](https://github.com/JuliaMeshless/RadialBasisFunctions.jl) — RBF interpolation and meshless differential operators (Laplacian, partial, gradient, custom) with automatic stencil selection.
- [**Macchiato.jl**](https://github.com/JuliaMeshless/Macchiato.jl) — General-purpose PDE framework: define custom models or use built-in physics (heat, elasticity, fluids), with boundary conditions, steady-state and transient solvers, field extraction, and VTK export.

## Built-in Models

Macchiato ships with ready-to-use models for common physics. You can also [define your own](@ref "Custom PDEs") for any PDE.

| Physics | Model | Status |
|---------|-------|--------|
| Heat transfer | [`SolidEnergy`](@ref) | Steady-state and transient |
| Linear elasticity | [`LinearElasticity`](@ref) | Steady-state (2D plane stress) |
| Incompressible fluids | [`IncompressibleNavierStokes`](@ref) | In development |

## Next Steps

- [Getting Started](@ref) — step-by-step tutorial from geometry to results
- [Custom PDEs](@ref) — define and solve your own equations
- [Examples](@ref) — complete worked examples with visualization
- [Package Design](@ref) — architecture and extension guide
- [API Reference](@ref) — full type and function documentation
