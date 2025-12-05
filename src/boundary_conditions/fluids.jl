# ============================================================================
# Fluid Boundary Conditions
# ============================================================================
# User-friendly aliases for generic parametric BC types in fluid problems.

# ============================================================================
# VelocityInlet (Dirichlet) - alias for FixedValue{FluidPhysics}
# ============================================================================

"""
    VelocityInlet{T}

Prescribed velocity at inlet boundary.

This is an alias for `FixedValue{FluidPhysics, T}`.

# Examples
```julia
VelocityInlet(1.0)           # Fixed velocity of 1.0
VelocityInlet(x -> x[2])     # Parabolic profile
```
"""
const VelocityInlet{T} = FixedValue{FluidPhysics, T}

# Note: VelocityInlet(v) constructor works automatically via the FixedValue inner constructor

Base.show(io::IO, bc::VelocityInlet) = print(io, "VelocityInlet: $(bc.value)")

# ============================================================================
# VelocityOutlet (Neumann) - alias for ZeroGradient{FluidPhysics}
# ============================================================================

"""
    VelocityOutlet

Zero-gradient velocity outlet: ∂v/∂n = 0

This is an alias for `ZeroGradient{FluidPhysics}`.

Use this for outflow boundaries where velocity gradients should be zero.

# Examples
```julia
VelocityOutlet()  # Zero-gradient outflow
```
"""
const VelocityOutlet = ZeroGradient{FluidPhysics}

# Note: VelocityOutlet() constructor works automatically via the const alias

Base.show(io::IO, ::VelocityOutlet) = print(io, "VelocityOutlet")

# ============================================================================
# PressureOutlet (Dirichlet) - physics-specific type
# ============================================================================

"""
    PressureOutlet{T} <: Dirichlet

Prescribed pressure at outlet boundary.

This is a distinct type (not an alias for `FixedValue{FluidPhysics}`) because
in incompressible flow, pressure and velocity are separate fields. A
`PressureOutlet` applies to the pressure DOF, while `VelocityInlet` applies
to velocity DOFs.

# Examples
```julia
PressureOutlet(0.0)  # Zero gauge pressure at outlet
```
"""
struct PressureOutlet{T} <: Dirichlet
    p::T
end

physics_domain(::Type{<:PressureOutlet}) = FluidPhysics()
(bc::PressureOutlet)() = bc.p

Base.show(io::IO, bc::PressureOutlet) = print(io, "PressureOutlet: $(bc.p)")
