# ============================================================================
# VelocityInlet (Dirichlet)
# ============================================================================

"""
    VelocityInlet{T} <: Dirichlet

Prescribed velocity at inlet boundary.

Internally represented as `FixedValue{FluidPhysics, T}`.

# Examples
```julia
# Constant velocity
bc = VelocityInlet(1.0)

# Spatially varying velocity
bc = VelocityInlet([1.0, 1.2, 1.1])

# Function-based velocity
bc = VelocityInlet(x -> 1.0 * (1.0 - x[2]^2))  # Parabolic profile
```
"""
const VelocityInlet{T} = FixedValue{FluidPhysics, T}

# Constructor for convenience
VelocityInlet(value::T) where {T} = FixedValue{FluidPhysics, T}(value)

# Time evolution - return closure for ODE: (du, u, p, t) -> modify u
# function make_bc(boundary::VelocityInlet, surf, domain::Domain{Dim}; kwargs...) where {Dim}
#     s = domain.cloud[surf]
#     surf_ids = only(s.points.indices)
#     v = boundary.v
#     (du, u, p, t) -> (u[surf_ids] .= v; nothing)
# end

Base.show(io::IO, bc::VelocityInlet) = print(io, "VelocityInlet: $(bc.value)")

# ============================================================================
# PressureOutlet (Dirichlet)
# ============================================================================

"""
    PressureOutlet{T} <: Dirichlet

Prescribed pressure at outlet boundary.

This is a physics-specific boundary condition that targets the pressure field.
"""
struct PressureOutlet{T} <: Dirichlet
    p::T
end

(bc::PressureOutlet)() = bc.p

# Physics domain trait
physics_domain(::Type{<:PressureOutlet}) = FluidPhysics()

# Time evolution - return closure for ODE: (du, u, p, t) -> modify u
# function make_bc(boundary::PressureOutlet, surf, domain::Domain{Dim}; kwargs...) where {Dim}
#     s = domain.cloud[surf]
#     ids = only(s.points.indices)
#     p_val = boundary.p
#     (du, u, p, t) -> (u[ids] .= p_val; nothing)
# end

Base.show(io::IO, bc::PressureOutlet) = print(io, "PressureOutlet: $(bc.p)")

# ============================================================================
# VelocityOutlet (Neumann with zero gradient)
# ============================================================================

"""
    VelocityOutlet <: Neumann

Zero-gradient velocity outlet: ∂v/∂n = 0

Internally represented as `ZeroGradient{FluidPhysics}`.

Commonly used for outflow boundaries where the velocity profile is fully developed.

# Usage
```julia
bc = VelocityOutlet()
```
"""
const VelocityOutlet = ZeroGradient{FluidPhysics}

Base.show(io::IO, ::VelocityOutlet) = print(io, "VelocityOutlet (zero gradient)")
