"""
    PhysicsDomain

Abstract type for physics domain traits. Used to categorize boundary conditions
and models by their physical domain, enabling validation and dispatching.
"""
abstract type PhysicsDomain end

"""
    EnergyPhysics <: PhysicsDomain

Physics domain for thermal/energy models and boundary conditions.
"""
struct EnergyPhysics <: PhysicsDomain end

"""
    FluidPhysics <: PhysicsDomain

Physics domain for fluid dynamics models and boundary conditions.
"""
struct FluidPhysics <: PhysicsDomain end

"""
    WallPhysics <: PhysicsDomain

Physics domain for wall boundary conditions. Compatible with both fluid and energy models.
"""
struct WallPhysics <: PhysicsDomain end

# ============================================================================
# Trait Accessors
# ============================================================================

"""
    physics_domain(T::Type{<:AbstractBoundaryCondition})
    physics_domain(T::Type{<:AbstractModel})

Returns the physics domain for a given boundary condition or model type.
"""
function physics_domain end

# ============================================================================
# Compatibility Rules
# ============================================================================

"""
    is_compatible(bc_domain::PhysicsDomain, model_domain::PhysicsDomain)

Check if a boundary condition's physics domain is compatible with a model's physics domain.
"""
is_compatible(::T, ::T) where {T <: PhysicsDomain} = true
is_compatible(::WallPhysics, ::FluidPhysics) = true
is_compatible(::WallPhysics, ::EnergyPhysics) = true
is_compatible(::PhysicsDomain, ::PhysicsDomain) = false
