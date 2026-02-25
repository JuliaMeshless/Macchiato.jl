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

"""
    MechanicsPhysics <: PhysicsDomain

Physics domain for solid mechanics models and BCs (displacement, traction).
"""
struct MechanicsPhysics <: PhysicsDomain end

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
is_compatible(::T, ::T) where {T <: PhysicsDomain} = true # same domain

#different models with WallPhysics
is_compatible(::WallPhysics, ::FluidPhysics) = true
is_compatible(::WallPhysics, ::EnergyPhysics) = true
is_compatible(::WallPhysics, ::MechanicsPhysics) = true

#Fallback rule: different domains are compatible but give warning
function is_compatible(bc::PhysicsDomain, model::PhysicsDomain)
    @warn """
    UNDEFINED PHYSICS DOMAIN COMPATIBILITY:
    BC domain: $(typeof(bc))
    Model domain: $(typeof(model))

    There is no compatibility rule defined for these domains.
        This BC will be allowed, but consider defining explicit compatibility rules in `physics_traits.jl`.
        """
    return true
end
