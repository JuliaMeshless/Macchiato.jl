# ============================================================================
# Physics Domain Trait System
# ============================================================================

"""
    PhysicsDomain

Abstract type for physics domain traits. Used to categorize boundary conditions
and models by their physical domain, enabling validation and dispatching.

# Purpose
The physics domain trait system enables:
- Type-safe validation of BC-model compatibility
- Clear separation of physics domains (energy, fluids, mechanics, etc.)
- Extensibility: new physics domains can be added without modifying existing code

# Usage
Define a new physics domain by creating a concrete subtype and defining compatibility rules.

# Example
```julia
# Add a new physics domain
struct MechanicsPhysics <: PhysicsDomain end

# Define compatibility
is_compatible(::MechanicsPhysics, ::MechanicsPhysics) = true
```
"""
abstract type PhysicsDomain end

"""
    EnergyPhysics <: PhysicsDomain

Physics domain for thermal/energy models and boundary conditions.

Compatible boundary conditions: `Temperature`, `HeatFlux`, `Adiabatic`, `Convection`
Compatible models: `SolidEnergy`
"""
struct EnergyPhysics <: PhysicsDomain end

"""
    FluidPhysics <: PhysicsDomain

Physics domain for fluid dynamics models and boundary conditions.

Compatible boundary conditions: `VelocityInlet`, `VelocityOutlet`, `PressureOutlet`
Compatible models: `IncompressibleNavierStokes`
"""
struct FluidPhysics <: PhysicsDomain end

"""
    WallPhysics <: PhysicsDomain

Physics domain for wall boundary conditions. Compatible with both fluid and energy models.

This special domain allows wall BCs to be used in both thermal and fluid simulations.

Compatible boundary conditions: `Wall`
Compatible models: Any model with `FluidPhysics` or `EnergyPhysics`
"""
struct WallPhysics <: PhysicsDomain end

# ============================================================================
# Trait Accessors
# ============================================================================

"""
    physics_domain(T::Type{<:AbstractBoundaryCondition})
    physics_domain(T::Type{<:AbstractModel})

Returns the physics domain for a given boundary condition or model type.

This is a trait function that must be defined for all boundary conditions and models.

# Examples
```julia
physics_domain(Temperature{Float64}) # Returns EnergyPhysics()
physics_domain(SolidEnergy)          # Returns EnergyPhysics()
physics_domain(VelocityInlet{Float64}) # Returns FluidPhysics()
```
"""
function physics_domain end

# ============================================================================
# Compatibility Rules
# ============================================================================

"""
    is_compatible(bc_domain::PhysicsDomain, model_domain::PhysicsDomain)

Check if a boundary condition's physics domain is compatible with a model's physics domain.

# Default Rules
- Domains are only compatible with themselves
- Exception: `WallPhysics` is compatible with both `FluidPhysics` and `EnergyPhysics`

# Extending
To add new compatibility rules for custom physics domains, define additional methods:

```julia
is_compatible(::CustomPhysics, ::EnergyPhysics) = true
```

# Examples
```julia
is_compatible(EnergyPhysics(), EnergyPhysics())   # true
is_compatible(WallPhysics(), FluidPhysics())      # true
is_compatible(EnergyPhysics(), FluidPhysics())    # false
```
"""
is_compatible(::T, ::T) where {T<:PhysicsDomain} = true
is_compatible(::WallPhysics, ::FluidPhysics) = true
is_compatible(::WallPhysics, ::EnergyPhysics) = true
is_compatible(::PhysicsDomain, ::PhysicsDomain) = false
