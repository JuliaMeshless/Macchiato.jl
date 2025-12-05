# ============================================================================
# Physics Domain Traits
# ============================================================================
# These traits identify which physics domain a boundary condition or model
# belongs to, enabling validation and shared parametric types.

"""
    PhysicsDomain

Abstract type for physics domain markers. Used as type parameters for
generic boundary conditions like `FixedValue{P}` and `ZeroGradient{P}`.
"""
abstract type PhysicsDomain end

"""
    EnergyPhysics <: PhysicsDomain

Physics domain for thermal/energy problems (heat conduction, etc.).
"""
struct EnergyPhysics <: PhysicsDomain end

"""
    FluidPhysics <: PhysicsDomain

Physics domain for fluid mechanics problems (Navier-Stokes, etc.).
"""
struct FluidPhysics <: PhysicsDomain end

"""
    WallPhysics <: PhysicsDomain

Physics domain for wall boundary conditions. Compatible with both
energy and fluid physics domains.
"""
struct WallPhysics <: PhysicsDomain end

"""
    GenericPhysics <: PhysicsDomain

Default physics domain for boundary conditions that work with any model.
Skips physics validation.
"""
struct GenericPhysics <: PhysicsDomain end

# ============================================================================
# Physics Domain Trait Accessor
# ============================================================================

"""
    physics_domain(::Type{T}) -> PhysicsDomain

Returns the physics domain for a boundary condition or model type.
Default returns `GenericPhysics()` which is compatible with everything.

Specialize this for your BC/model types:
```julia
physics_domain(::Type{<:MyBC}) = EnergyPhysics()
physics_domain(::Type{<:MyModel}) = FluidPhysics()
```
"""
physics_domain(::Type) = GenericPhysics()

# ============================================================================
# Compatibility Matrix
# ============================================================================

"""
    is_compatible(bc_physics, model_physics) -> Bool

Check if a boundary condition's physics domain is compatible with a model's
physics domain. Used for validation in `Domain` constructor.

Rules:
- Same physics domains are always compatible
- `GenericPhysics` is compatible with everything
- `WallPhysics` is compatible with both energy and fluids
- Different physics domains are incompatible
"""
is_compatible(::P, ::P) where {P <: PhysicsDomain} = true
is_compatible(::GenericPhysics, ::PhysicsDomain) = true
is_compatible(::PhysicsDomain, ::GenericPhysics) = true
is_compatible(::WallPhysics, ::FluidPhysics) = true
is_compatible(::WallPhysics, ::EnergyPhysics) = true
is_compatible(::PhysicsDomain, ::PhysicsDomain) = false
