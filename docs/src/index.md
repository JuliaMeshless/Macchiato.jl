```@meta
CurrentModule = MeshlessMultiphysics
```

# MeshlessMultiphysics.jl

A Julia framework for solving partial differential equations using radial basis function (RBF) meshless methods.

## Why Meshless Methods?

Traditional numerical PDE solvers (finite elements, finite volumes) require a mesh — a connected grid that partitions the domain into elements. Mesh generation is often the most time-consuming step in a simulation workflow, especially for complex 3D geometries. Mesh quality directly affects solution accuracy, and adaptive refinement requires expensive remeshing.

**Meshless methods** bypass this entirely. They operate on scattered point clouds with no connectivity requirements. This means:

- **Simple geometry handling** — drop points on the boundary and fill the interior; no need to worry about element quality, tangled meshes, or hanging nodes
- **Easy refinement** — add more points where you need accuracy; no remeshing required
- **Natural for moving boundaries** — points move freely without topological constraints

MeshlessMultiphysics.jl uses **radial basis function (RBF)** collocation, where differential operators are approximated at each point using its local neighborhood of nearest neighbors.

## The JuliaMeshless Ecosystem

MeshlessMultiphysics.jl is part of the [JuliaMeshless](https://github.com/JuliaMeshless) organization, which provides three composable packages that together form a complete simulation pipeline:

```
┌─────────────────────┐     ┌──────────────────────────┐     ┌─────────────────────────────┐
│   WhatsThePoint.jl  │     │ RadialBasisFunctions.jl  │     │  MeshlessMultiphysics.jl    │
│                     │     │                          │     │                             │
│  Boundary creation  │────▶│  RBF interpolation       │────▶│  Physics models             │
│  Surface splitting  │     │  Differential operators   │     │  Boundary conditions        │
│  Interior fill      │     │  (∇², ∂/∂x, custom)      │     │  Simulation & time stepping │
│  Point repulsion    │     │  KNN stencil selection    │     │  Field extraction & VTK I/O │
└─────────────────────┘     └──────────────────────────┘     └─────────────────────────────┘
        Geometry                   Numerics                        Physics
```

### WhatsThePoint.jl

Handles point cloud generation from geometric descriptions. Key capabilities:
- Create `PointBoundary` objects from primitives (rectangles, circles, etc.)
- Split boundaries into named surfaces via `split_surface!` for BC assignment
- Fill the interior with `discretize` using spacing algorithms (`ConstantSpacing`)
- Refine distributions with `repel` for quasi-uniform spacing

### RadialBasisFunctions.jl

Provides RBF interpolation and meshless differential operators:
- Build `laplacian`, `partial`, `gradient`, and `custom` operators from point data
- Automatic nearest-neighbor stencil selection
- Polyharmonic spline (PHS) bases with polynomial augmentation
- Sparse weight matrices for efficient linear algebra

### MeshlessMultiphysics.jl

Orchestrates the simulation — this package:
- Defines physics models (`SolidEnergy`, `LinearElasticity`, `IncompressibleNavierStokes`)
- Provides a trait-based boundary condition system with physics-specific aliases
- Assembles the discrete system (matrix or ODE) from operators + BCs
- Runs steady-state (direct linear solve) or transient (ODE integration) simulations
- Extracts solution fields and exports to VTK for visualization

## Supported Physics

| Physics | Model | Status |
|---------|-------|--------|
| Heat transfer | [`SolidEnergy`](@ref) | Steady-state and transient |
| Linear elasticity | [`LinearElasticity`](@ref) | Steady-state (2D plane stress) |
| Incompressible fluids | [`IncompressibleNavierStokes`](@ref) | In development |

## Getting Started

Head to the [Getting Started](getting_started.md) tutorial for a step-by-step walkthrough of a complete heat transfer simulation.
