# ============================================================================
# Generic Parametric Boundary Conditions
# ============================================================================
# These types are parameterized by physics domain `P`, allowing the same
# mathematical BC to be used across different physics (energy, fluids, etc.).

# ============================================================================
# FixedValue (Dirichlet)
# ============================================================================

"""
    FixedValue{P<:PhysicsDomain, T} <: Dirichlet

Fixed value (Dirichlet) boundary condition parameterized by physics domain.

Mathematically: u = value

# Type Parameters
- `P`: Physics domain (`EnergyPhysics`, `FluidPhysics`, etc.)
- `T`: Value type (scalar, vector, or function)

# Examples
```julia
# Energy: fixed temperature
FixedValue{EnergyPhysics}(100.0)

# Fluids: fixed velocity
FixedValue{FluidPhysics}(1.0)

# Using the friendly aliases (defined in energy.jl, fluids.jl):
Temperature(100.0)      # Same as FixedValue{EnergyPhysics}(100.0)
VelocityInlet(1.0)      # Same as FixedValue{FluidPhysics}(1.0)
```
"""
struct FixedValue{P <: PhysicsDomain, T} <: Dirichlet
    value::T

    function FixedValue{P}(value::T) where {P <: PhysicsDomain, T}
        return new{P, T}(value)
    end
end

physics_domain(::Type{<:FixedValue{P}}) where {P} = P()
(bc::FixedValue)() = bc.value

function Base.show(io::IO, bc::FixedValue{P}) where {P}
    return print(io, "FixedValue{$P}: $(bc.value)")
end

# ============================================================================
# ZeroGradient (Homogeneous Neumann)
# ============================================================================

"""
    ZeroGradient{P<:PhysicsDomain} <: Neumann

Zero-gradient (homogeneous Neumann) boundary condition parameterized by physics domain.

Mathematically: ∂u/∂n = 0

# Type Parameters
- `P`: Physics domain (`EnergyPhysics`, `FluidPhysics`, etc.)

# Examples
```julia
# Energy: adiabatic (insulated) wall
ZeroGradient{EnergyPhysics}()

# Fluids: zero-gradient outflow
ZeroGradient{FluidPhysics}()

# Using the friendly aliases:
Adiabatic()        # Same as ZeroGradient{EnergyPhysics}()
VelocityOutlet()   # Same as ZeroGradient{FluidPhysics}()
```
"""
struct ZeroGradient{P <: PhysicsDomain} <: Neumann
    function ZeroGradient{P}() where {P <: PhysicsDomain}
        return new{P}()
    end
end

physics_domain(::Type{<:ZeroGradient{P}}) where {P} = P()
(bc::ZeroGradient)() = 0.0

function Base.show(io::IO, ::ZeroGradient{P}) where {P}
    return print(io, "ZeroGradient{$P}")
end

# ============================================================================
# FixedGradient (Inhomogeneous Neumann)
# ============================================================================

"""
    FixedGradient{P<:PhysicsDomain, T} <: Neumann

Fixed gradient (inhomogeneous Neumann) boundary condition parameterized by physics domain.

Mathematically: ∂u/∂n = gradient

# Type Parameters
- `P`: Physics domain (`EnergyPhysics`, `FluidPhysics`, etc.)
- `T`: Gradient value type (scalar, vector, or function)

# Examples
```julia
# Energy: prescribed heat flux
FixedGradient{EnergyPhysics}(500.0)

# Using the friendly alias:
HeatFlux(500.0)    # Same as FixedGradient{EnergyPhysics}(500.0)
```
"""
struct FixedGradient{P <: PhysicsDomain, T} <: Neumann
    gradient::T

    function FixedGradient{P}(gradient::T) where {P <: PhysicsDomain, T}
        return new{P, T}(gradient)
    end
end

physics_domain(::Type{<:FixedGradient{P}}) where {P} = P()
(bc::FixedGradient)() = bc.gradient

function Base.show(io::IO, bc::FixedGradient{P}) where {P}
    return print(io, "FixedGradient{$P}: $(bc.gradient)")
end

# ============================================================================
# MixedBC (Robin)
# ============================================================================

"""
    MixedBC{P<:PhysicsDomain, A, B, T} <: Robin

Mixed (Robin) boundary condition parameterized by physics domain.

Mathematically: α·u + β·∂u/∂n = value

# Type Parameters
- `P`: Physics domain (`EnergyPhysics`, `FluidPhysics`, etc.)
- `A`: Type of α coefficient
- `B`: Type of β coefficient
- `T`: Type of RHS value

# Note
For physics-specific Robin BCs with semantic meaning (like `Convection` for
Newton's law of cooling), use the dedicated types instead.

# Examples
```julia
# Generic Robin condition
MixedBC{EnergyPhysics}(1.0, 0.5, 100.0)  # 1.0·u + 0.5·∂u/∂n = 100.0
```
"""
struct MixedBC{P <: PhysicsDomain, A, B, T} <: Robin
    alpha::A
    beta::B
    value::T

    function MixedBC{P}(alpha::A, beta::B, value::T) where {P <: PhysicsDomain, A, B, T}
        return new{P, A, B, T}(alpha, beta, value)
    end
end

physics_domain(::Type{<:MixedBC{P}}) where {P} = P()
α(bc::MixedBC) = bc.alpha
β(bc::MixedBC) = bc.beta
(bc::MixedBC)() = bc.value

function Base.show(io::IO, bc::MixedBC{P}) where {P}
    return print(io, "MixedBC{$P}: $(bc.alpha)·u + $(bc.beta)·∂u/∂n = $(bc.value)")
end
