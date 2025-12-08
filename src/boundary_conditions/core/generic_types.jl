# ============================================================================
# Generic Boundary Condition Types
# ============================================================================

"""
    FixedValue{P<:PhysicsDomain, T} <: Dirichlet

Generic Dirichlet boundary condition that prescribes a fixed value.
Type parameters: `P` (PhysicsDomain), `T` (Value type: Number, Vector, or Function).
"""
struct FixedValue{P <: PhysicsDomain, T} <: Dirichlet
    value::T
end

# BC value accessor
(bc::FixedValue)() = bc.value

# Physics domain trait
physics_domain(::Type{<:FixedValue{P}}) where {P} = P()

"""
    Flux{P<:PhysicsDomain, Q} <: Neumann

Generic Neumann boundary condition that prescribes a flux (normal derivative).
Type parameters: `P` (PhysicsDomain), `Q` (Flux type: Number, Vector, or Function).
"""

# Extending for New Physics
"""
To add Flux support for a new physics domain:

```julia
# Define physics domain (if not already defined)
struct MechanicsPhysics <: PhysicsDomain end

# Create type alias in your physics file (e.g., mechanics.jl)
const Traction{Q} = Flux{MechanicsPhysics, Q}
Traction(flux::Q) where {Q} = Flux{MechanicsPhysics, Q}(flux)
```

# See Also
- `FixedValue`: Generic Dirichlet BC with prescribed value
- `ZeroGradient`: Generic Neumann BC with zero flux
"""
struct Flux{P <: PhysicsDomain, Q} <: Neumann
    flux::Q
end

# BC value accessor
(bc::Flux)() = bc.flux

# Physics domain trait
physics_domain(::Type{<:Flux{P}}) where {P} = P()

"""
    ZeroGradient{P<:PhysicsDomain} <: Neumann

Generic Neumann boundary condition with zero flux (∂u/∂n = 0).

This type provides a reusable implementation for Neumann BCs where the normal
derivative is zero. This represents symmetry, insulation, or fully-developed flow.

# Type Parameters
- `P`: Physics domain (EnergyPhysics, FluidPhysics, MechanicsPhysics, etc.)

# Physical Interpretations by Domain
- **Energy**: Adiabatic/insulated boundary (no heat flux)
- **Fluids**: Zero-gradient velocity outlet (fully developed flow)
- **Mechanics**: Stress-free boundary (no applied forces)

# Direct Usage (Advanced)
```julia
# Create a generic zero-gradient BC for energy physics
bc = ZeroGradient{EnergyPhysics}()
```

# Typical Usage (Recommended)
Instead of using `ZeroGradient` directly, use physics-specific type aliases
that convey physical meaning:

```julia
# Energy physics: thermal insulation
bc = Adiabatic()              # Alias for ZeroGradient{EnergyPhysics}

# Fluid physics: zero-gradient outlet
bc = VelocityOutlet()         # Alias for ZeroGradient{FluidPhysics}

# Future: Structural mechanics
bc = ZeroStress()             # Would be: ZeroGradient{MechanicsPhysics}
```

# Extending for New Physics
To add ZeroGradient support for a new physics domain:

```julia
# Define physics domain (if not already defined)
struct MechanicsPhysics <: PhysicsDomain end

# Create type alias in your physics file (e.g., mechanics.jl)
const ZeroStress = ZeroGradient{MechanicsPhysics}
```

# Implementation Notes
- The BC value is always 0.0 (zero flux)
- Derivative weights are computed based on the specified numerical scheme
- Shadow point methods can be used for higher-order accuracy

# See Also
- `FixedValue`: Generic Dirichlet BC with prescribed value
- `Flux`: Generic Neumann BC with non-zero flux
"""
struct ZeroGradient{P <: PhysicsDomain} <: Neumann end

# BC value accessor
(bc::ZeroGradient)() = 0.0

# Physics domain trait
physics_domain(::Type{<:ZeroGradient{P}}) where {P} = P()
