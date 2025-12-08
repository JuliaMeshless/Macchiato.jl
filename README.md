# MeshlessMultiphysics

[![Build Status](https://github.com/JuliaMeshless/MeshlessMultiphysics.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaMeshless/MeshlessMultiphysics.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaMeshless.github.io/MeshlessMultiphysics.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaMeshless.github.io/MeshlessMultiphysics.jl/dev)
[![License File](https://img.shields.io/badge/license-MIT-blue)](https://github.com/JuliaMeshless/MeshlessMultiphysics.jl/blob/master/LICENSE)

Julia package for solving PDEs via meshless methods.

> [!CAUTION]
> This is under heavy development and does not have robust testing for all methods
yet. Use at your own risk.

## Contributing

We welcome contributions! The package is designed to be easily extensible, especially for adding new physics domains and boundary conditions.

### Adding New Physics Domains

The boundary condition system uses a trait-based design that separates mathematical BC types (Dirichlet, Neumann, Robin) from physics domains (Energy, Fluids, Mechanics, etc.). This makes it easy to add new physics without code duplication.

#### Boundary Condition System Structure

```
src/boundary_conditions/
├── core/                          # Core infrastructure (reusable)
│   ├── physics_traits.jl          # Physics domain trait system
│   ├── bc_hierarchy.jl            # Mathematical BC type hierarchy
│   └── generic_types.jl           # Generic BC implementations
├── numerical/                     # Numerical methods
│   └── derivatives.jl             # Derivative computation
├── energy.jl                      # Energy/thermal BCs
├── fluids.jl                      # Fluid dynamics BCs
├── walls.jl                       # Wall BCs
└── boundary_conditions.jl         # Main orchestrator
```

#### Example: Adding Structural Mechanics

To add a new physics domain (e.g., Structural Mechanics), follow these steps:

**1. Define the physics domain in `core/physics_traits.jl`:**

```julia
"""
    MechanicsPhysics <: PhysicsDomain

Physics domain for structural mechanics models and boundary conditions.
"""
struct MechanicsPhysics <: PhysicsDomain end

# Define compatibility rules
is_compatible(::MechanicsPhysics, ::MechanicsPhysics) = true
```

**2. Create a new file `src/boundary_conditions/mechanics.jl`:**

```julia
"""
    Displacement{T} <: Dirichlet

Prescribed displacement boundary condition.
"""
const Displacement{T} = FixedValue{MechanicsPhysics, T}

# Constructor for convenience
Displacement(value::T) where {T} = FixedValue{MechanicsPhysics, T}(value)

Base.show(io::IO, bc::Displacement) = print(io, "Displacement: $(bc.value)")

"""
    Traction{Q} <: Neumann

Prescribed traction (force per unit area) boundary condition.
"""
const Traction{Q} = Flux{MechanicsPhysics, Q}

Traction(flux::Q) where {Q} = Flux{MechanicsPhysics, Q}(flux)

Base.show(io::IO, bc::Traction) = print(io, "Traction: $(bc.flux)")

"""
    ZeroStress <: Neumann

Stress-free boundary: ∂u/∂n = 0
"""
const ZeroStress = ZeroGradient{MechanicsPhysics}

Base.show(io::IO, ::ZeroStress) = print(io, "ZeroStress")
```

**3. Add the physics domain trait to your model in `src/models/mechanics.jl`:**

```julia
struct ElasticSolid{E, NU} <: AbstractModel
    E::E    # Young's modulus
    ν::NU   # Poisson's ratio
end

# Physics domain trait
physics_domain(::Type{<:ElasticSolid}) = MechanicsPhysics()
```

**4. Update exports in `src/MeshlessMultiphysics.jl`:**

```julia
# Add to physics domain exports
export MechanicsPhysics

# Add mechanics BCs
include("boundary_conditions/mechanics.jl")
export Displacement, Traction, ZeroStress
```

**That's it!** The generic implementations (`FixedValue`, `Flux`, `ZeroGradient`) provide all the mathematical machinery. You just define type aliases with your physics domain and implement the display methods.

For more examples, see the existing implementations in `src/boundary_conditions/energy.jl` and `src/boundary_conditions/fluids.jl`.
