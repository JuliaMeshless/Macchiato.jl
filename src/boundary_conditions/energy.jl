# ============================================================================
# Energy Boundary Conditions
# ============================================================================
# User-friendly aliases for generic parametric BC types in energy problems.

# ============================================================================
# Temperature (Dirichlet) - alias for FixedValue{EnergyPhysics}
# ============================================================================

"""
    Temperature{T}

Prescribed temperature boundary condition.

This is an alias for `FixedValue{EnergyPhysics, T}`.

# Examples
```julia
Temperature(100.0)           # Fixed temperature of 100
Temperature(x -> sin(x[1]))  # Spatially varying temperature
```
"""
const Temperature{T} = FixedValue{EnergyPhysics, T}

# Note: Temperature(value) constructor works automatically via the FixedValue inner constructor

Base.show(io::IO, bc::Temperature) = print(io, "Temperature: $(bc.value)")

# ============================================================================
# HeatFlux (Neumann) - alias for FixedGradient{EnergyPhysics}
# ============================================================================

"""
    HeatFlux{Q}

Prescribed heat flux boundary condition (∂T/∂n = q).

This is an alias for `FixedGradient{EnergyPhysics, Q}`.

# Examples
```julia
HeatFlux(500.0)              # Constant heat flux
HeatFlux(x -> x[1] * 100.0)  # Spatially varying flux
```
"""
const HeatFlux{Q} = FixedGradient{EnergyPhysics, Q}

# Note: HeatFlux(q) constructor works automatically via the FixedGradient inner constructor

Base.show(io::IO, bc::HeatFlux) = print(io, "HeatFlux: $(bc.gradient)")

# ============================================================================
# Adiabatic (Neumann) - alias for ZeroGradient{EnergyPhysics}
# ============================================================================

"""
    Adiabatic

Thermally insulated boundary: ∂T/∂n = 0

This is an alias for `ZeroGradient{EnergyPhysics}`.

The numerical method (standard derivative vs shadow points) is controlled
by the `scheme` parameter passed to `LinearProblem`, not by the BC itself.

# Examples
```julia
Adiabatic()  # Insulated boundary
```
"""
const Adiabatic = ZeroGradient{EnergyPhysics}

# Note: Adiabatic() constructor works automatically via the const alias

Base.show(io::IO, ::Adiabatic) = print(io, "Adiabatic")

# ============================================================================
# Convection (Robin) - physics-specific type
# ============================================================================

"""
    Convection{H, K, T} <: Robin

Convective heat transfer: h·T + k·∂T/∂n = h·T∞

Newton's law of cooling at boundary. This is a physics-specific Robin BC
with semantic field names, not a generic `MixedBC`.

# Fields
- `h`: Heat transfer coefficient [W/(m²·K)]
- `k`: Thermal conductivity [W/(m·K)]
- `T∞`: Ambient temperature [K or °C]

# Examples
```julia
Convection(25.0, 0.6, 20.0)  # h=25, k=0.6, T∞=20
```
"""
struct Convection{H, K, T} <: Robin
    h::H   # heat transfer coefficient
    k::K   # thermal conductivity
    T∞::T  # ambient temperature

    function Convection(h::H, k::K, T∞::T) where {H, K, T}
        h < 0 && throw(ArgumentError("Heat transfer coefficient must be non-negative"))
        k <= 0 && throw(ArgumentError("Thermal conductivity must be positive"))
        return new{H, K, T}(h, k, T∞)
    end
end

physics_domain(::Type{<:Convection}) = EnergyPhysics()
α(bc::Convection) = bc.h
β(bc::Convection) = bc.k
(bc::Convection)() = bc.h * bc.T∞

function Base.show(io::IO, bc::Convection)
    return print(io, "Convection: h=$(bc.h), k=$(bc.k), T∞=$(bc.T∞)")
end
