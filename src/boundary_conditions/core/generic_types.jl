# ============================================================================
# Generic Boundary Condition Types
# ============================================================================

"""
    FixedValue{P<:PhysicsDomain, T} <: Dirichlet

Generic Dirichlet BC that prescribes a fixed value across physics domains.

# Type Parameters
- `P`: Physics domain (EnergyPhysics, FluidPhysics, etc.)
- `T`: Value type (Number, Vector, or Function)

Use physics-specific aliases instead of this directly (e.g., `Temperature`, `VelocityInlet`).
"""
struct FixedValue{P<:PhysicsDomain, T} <: Dirichlet
    value::T
end

# BC value accessor
(bc::FixedValue)() = bc.value

# Physics domain trait
physics_domain(::Type{<:FixedValue{P}}) where {P} = P()

"""
    Flux{P<:PhysicsDomain, Q} <: Neumann

Generic Neumann BC that prescribes a flux (∂u/∂n = q).

# Type Parameters
- `P`: Physics domain (EnergyPhysics, FluidPhysics, etc.)
- `Q`: Flux type (Number, Vector, or Function)

Use physics-specific aliases instead of this directly (e.g., `HeatFlux`, `Traction`).
"""
struct Flux{P<:PhysicsDomain, Q} <: Neumann
    flux::Q
end

# BC value accessor
(bc::Flux)() = bc.flux

# Physics domain trait
physics_domain(::Type{<:Flux{P}}) where {P} = P()

"""
    ZeroGradient{P<:PhysicsDomain} <: Neumann

Generic Neumann BC with zero flux (∂u/∂n = 0). Represents symmetry, insulation, or fully-developed flow.

# Type Parameters
- `P`: Physics domain (EnergyPhysics, FluidPhysics, etc.)

Use physics-specific aliases instead of this directly (e.g., `Adiabatic`, `VelocityOutlet`).
"""
struct ZeroGradient{P<:PhysicsDomain} <: Neumann end

# BC value accessor
(bc::ZeroGradient)() = 0.0

# Physics domain trait
physics_domain(::Type{<:ZeroGradient{P}}) where {P} = P()
