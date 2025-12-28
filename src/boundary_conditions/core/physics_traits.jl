# ============================================================================
# Physics Domain Trait System
# ============================================================================

"""
    PhysicsDomain

Abstract type for physics domain traits. Categorizes boundary conditions and models
by their physical domain for type-safe validation and dispatching.
"""
abstract type PhysicsDomain end

"""
    EnergyPhysics <: PhysicsDomain

Physics domain for thermal/energy models and BCs.
"""
struct EnergyPhysics <: PhysicsDomain end

"""
    FluidPhysics <: PhysicsDomain

Physics domain for fluid dynamics models and BCs.
"""
struct FluidPhysics <: PhysicsDomain end

"""
    WallPhysics <: PhysicsDomain

Physics domain for wall BCs. Compatible with both fluid and energy models.
"""
struct WallPhysics <: PhysicsDomain end

# ============================================================================
# Trait Accessors
# ============================================================================

"""
    physics_domain(T::Type{<:AbstractBoundaryCondition})
    physics_domain(T::Type{<:AbstractModel})

Returns the physics domain for a given BC or model type.
Must be defined for all boundary conditions and models.
"""
function physics_domain end

# ============================================================================
# Compatibility Rules
# ============================================================================

"""
    is_compatible(bc_domain::PhysicsDomain, model_domain::PhysicsDomain)

Check if a BC's physics domain is compatible with a model's physics domain.
Domains are compatible with themselves; `WallPhysics` is also compatible with fluid and energy domains.
"""
is_compatible(::T, ::T) where {T<:PhysicsDomain} = true
is_compatible(::WallPhysics, ::FluidPhysics) = true
is_compatible(::WallPhysics, ::EnergyPhysics) = true
is_compatible(::PhysicsDomain, ::PhysicsDomain) = false
