# ============================================================================
# Generic Boundary Condition Types
# ============================================================================

"""
    PrescribedValue{P<:PhysicsDomain, F<:Function} <: Dirichlet

Generic Dirichlet BC that prescribes a value via a function.

The function has signature `f(x, t) -> value` where:
- `x`: spatial coordinate of the boundary point
- `t`: time
- Returns the prescribed value at that location and time

# Type Parameters
- `P`: Physics domain (EnergyPhysics, FluidPhysics, etc.)
- `F`: Function type with signature (x, t) -> value

Use physics-specific aliases instead of this directly (e.g., `Temperature`, `VelocityInlet`).
"""
struct PrescribedValue{P <: PhysicsDomain, F <: Function} <: Dirichlet
    f::F
end

# Convenience constructors
PrescribedValue{P}(value::Number) where {P} = PrescribedValue{P}((x, t) -> value)
PrescribedValue{P}(f::Function) where {P} = PrescribedValue{P, typeof(f)}(f)

# BC evaluation: bc(x, t) returns the prescribed value at (x, t)
(bc::PrescribedValue)(x, t) = bc.f(x, t)

# Physics domain trait
physics_domain(::Type{<:PrescribedValue{P}}) where {P} = P()

# ============================================================================

"""
    PrescribedFlux{P<:PhysicsDomain, F<:Function} <: Neumann

Generic Neumann BC that prescribes a flux (normal derivative) via a function.

The flux condition is: ∂u/∂n = f(x, t)

The function has signature `f(x, t) -> flux_value` where:
- `x`: spatial coordinate of the boundary point
- `t`: time
- Returns the prescribed flux value at that location and time

# Type Parameters
- `P`: Physics domain (EnergyPhysics, FluidPhysics, etc.)
- `F`: Function type with signature (x, t) -> flux

Use physics-specific aliases instead of this directly (e.g., `HeatFlux`, `Traction`).
"""
struct PrescribedFlux{P <: PhysicsDomain, F <: Function} <: Neumann
    f::F
end

# Convenience constructors
PrescribedFlux{P}(flux::Number) where {P} = PrescribedFlux{P}((x, t) -> flux)
PrescribedFlux{P}(f::Function) where {P} = PrescribedFlux{P, typeof(f)}(f)

# BC evaluation: bc(x, t) returns the prescribed flux at (x, t)
(bc::PrescribedFlux)(x, t) = bc.f(x, t)

# Physics domain trait
physics_domain(::Type{<:PrescribedFlux{P}}) where {P} = P()

# ============================================================================

"""
    ZeroFlux{P<:PhysicsDomain} <: Neumann

Generic Neumann BC with zero flux: ∂u/∂n = 0.

Represents symmetry, insulation, or fully-developed flow depending on physics domain.

# Type Parameters
- `P`: Physics domain (EnergyPhysics, FluidPhysics, etc.)

Use physics-specific aliases instead of this directly (e.g., `Adiabatic`, `VelocityOutlet`).
"""
struct ZeroFlux{P <: PhysicsDomain} <: Neumann end

# BC evaluation: always returns 0
(bc::ZeroFlux)(x::AbstractVector{T}, t::T) where {T} = zero(T)
#TODO: Consider (bc::ZeroFlux)(x::AbstractVector{A}, t::B) where {A, B} = zero(promote_type(A, B))

# Physics domain trait
physics_domain(::Type{<:ZeroFlux{P}}) where {P} = P()
