# ============================================================================
# Generic Boundary Condition Types
# ============================================================================

"""
    PrescribedValue{F<:Function} <: Dirichlet

Generic Dirichlet BC that prescribes a value via a function.

The function has signature `f(x, t) -> value` where:
- `x`: spatial coordinate of the boundary point
- `t`: time
- Returns the prescribed value at that location and time

# Fields
- `f`: Function with signature (x, t) -> value
- `name`: Display name (e.g., `:Temperature`, `:PrescribedValue`)

For built-in physics, use named constructors (e.g., `Temperature`, `VelocityInlet`).
For custom PDEs, use the unparameterized constructors directly: `PrescribedValue(0.0)`.
"""
struct PrescribedValue{F <: Function} <: Dirichlet
    f::F
    name::Symbol
end

# Unparameterized constructors for custom PDEs
PrescribedValue(v::Number) = PrescribedValue((x, t) -> v, :PrescribedValue)
PrescribedValue(f::Function) = PrescribedValue{typeof(f)}(f, :PrescribedValue)

# BC evaluation: bc(x, t) returns the prescribed value at (x, t)
(bc::PrescribedValue)(x, t) = bc.f(x, t)

Base.show(io::IO, bc::PrescribedValue) = print(io, bc.name)

# ============================================================================

"""
    PrescribedFlux{F<:Function} <: Neumann

Generic Neumann BC that prescribes a flux (normal derivative) via a function.

The flux condition is: ∂u/∂n = f(x, t)

The function has signature `f(x, t) -> flux_value` where:
- `x`: spatial coordinate of the boundary point
- `t`: time
- Returns the prescribed flux value at that location and time

# Fields
- `f`: Function with signature (x, t) -> flux
- `name`: Display name (e.g., `:HeatFlux`, `:PrescribedFlux`)

For built-in physics, use named constructors (e.g., `HeatFlux`, `Traction`).
For custom PDEs, use the unparameterized constructors directly: `PrescribedFlux(1.0)`.
"""
struct PrescribedFlux{F <: Function} <: Neumann
    f::F
    name::Symbol
end

# Unparameterized constructors for custom PDEs
PrescribedFlux(v::Number) = PrescribedFlux((x, t) -> v, :PrescribedFlux)
PrescribedFlux(f::Function) = PrescribedFlux{typeof(f)}(f, :PrescribedFlux)

# BC evaluation: bc(x, t) returns the prescribed flux at (x, t)
(bc::PrescribedFlux)(x, t) = bc.f(x, t)

Base.show(io::IO, bc::PrescribedFlux) = print(io, bc.name)

# ============================================================================

"""
    ZeroFlux <: Neumann

Generic Neumann BC with zero flux: ∂u/∂n = 0.

Represents symmetry, insulation, or fully-developed flow depending on context.

# Fields
- `name`: Display name (e.g., `:Adiabatic`, `:VelocityOutlet`, `:ZeroFlux`)

For built-in physics, use named constructors (e.g., `Adiabatic`, `VelocityOutlet`).
For custom PDEs, use the unparameterized constructor directly: `ZeroFlux()`.
"""
struct ZeroFlux <: Neumann
    name::Symbol
end

# Unparameterized constructor for custom PDEs
ZeroFlux() = ZeroFlux(:ZeroFlux)

# BC evaluation: always returns 0
(::ZeroFlux)(x, t) = 0.0

Base.show(io::IO, bc::ZeroFlux) = print(io, bc.name)
