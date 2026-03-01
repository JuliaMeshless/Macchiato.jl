# Macchiato.jl

[![Build Status](https://github.com/JuliaMeshless/Macchiato.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaMeshless/Macchiato.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaMeshless.github.io/Macchiato.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaMeshless.github.io/Macchiato.jl/dev)
[![License File](https://img.shields.io/badge/license-MIT-blue)](https://github.com/JuliaMeshless/Macchiato.jl/blob/main/LICENSE)

A Julia framework for solving partial differential equations using radial basis function (RBF) meshless methods. Instead of generating and managing a mesh, Macchiato.jl operates directly on scattered point clouds — simplifying geometry handling, avoiding mesh quality issues, and enabling straightforward refinement by simply adding points.

## The JuliaMeshless Ecosystem

Macchiato.jl is part of the [JuliaMeshless](https://github.com/JuliaMeshless) organization, which provides three composable packages:

| Package | Role |
|---------|------|
| [**WhatsThePoint.jl**](https://github.com/JuliaMeshless/WhatsThePoint.jl) | Point cloud generation — boundary discretization, surface splitting, interior fill algorithms |
| [**RadialBasisFunctions.jl**](https://github.com/JuliaMeshless/RadialBasisFunctions.jl) | RBF interpolation and differential operators — Laplacian, gradient, partial derivatives, custom operators |
| [**Macchiato.jl**](https://github.com/JuliaMeshless/Macchiato.jl) | Physics simulation framework — models, boundary conditions, solvers, and post-processing |

WhatsThePoint.jl creates the geometry, RadialBasisFunctions.jl provides the numerical building blocks, and Macchiato.jl ties them together into complete simulations.

## Quick Example

Steady-state heat conduction on a unit square with prescribed temperatures:

```julia
using WhatsThePoint, Macchiato, Unitful: m, °

# Geometry: unit square point cloud
part = PointBoundary(rectangle(1m, 1m)...)
split_surface!(part, 75°)
cloud = discretize(part, ConstantSpacing(1/33 * m), alg=VanDerSandeFornberg())

# Physics + BCs
model = SolidEnergy(k=1.0, ρ=1.0, cₚ=1.0)
bcs = Dict(
    :surface1 => Temperature(0.0),    # bottom
    :surface2 => Temperature(0.0),    # right
    :surface3 => Temperature(100.0),  # top
    :surface4 => Temperature(0.0)     # left
)
domain = Domain(cloud, bcs, model)

# Solve and extract results
sim = Simulation(domain)
run!(sim)
T = temperature(sim)
```

## Supported Physics

- **Heat Transfer** — `SolidEnergy`: steady-state and transient conduction with optional source terms, Dirichlet (`Temperature`), Neumann (`HeatFlux`, `Adiabatic`), and Robin (`Convection`) boundary conditions
- **Linear Elasticity** — `LinearElasticity`: 2D plane stress with `Displacement` and `Traction` boundary conditions, body force support
- **Incompressible Fluids** *(in development)* — `IncompressibleNavierStokes` with Newtonian and Carreau-Yasuda viscosity models

## Boundary Condition System

BCs are organized by mathematical type and physics domain:

| Math Type | Generic Type | Energy | Mechanics | Fluids |
|-----------|-------------|--------|-----------|--------|
| Dirichlet | `PrescribedValue` | `Temperature` | `Displacement` | `VelocityInlet` / `PressureOutlet` |
| Neumann | `PrescribedFlux` / `ZeroFlux` | `HeatFlux` / `Adiabatic` | `Traction` / `TractionFree` | `VelocityOutlet` |
| Robin | — | `Convection` | — | — |

All BCs accept constant values or functions `(x, t) -> value` for spatially/temporally varying conditions.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/JuliaMeshless/Macchiato.jl")
```

Or for development:

```julia
Pkg.develop(url="https://github.com/JuliaMeshless/Macchiato.jl")
```

## Documentation

Full documentation is available at [juliameshless.github.io/Macchiato.jl](https://JuliaMeshless.github.io/Macchiato.jl/dev).

## Contributing

Contributions are welcome! The package uses a dispatch-based design that makes it straightforward to add new physics domains. See the [Package Design](https://JuliaMeshless.github.io/Macchiato.jl/dev/design/) section of the docs for the architecture overview and extension guide.
