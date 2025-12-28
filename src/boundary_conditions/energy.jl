# ============================================================================
# Temperature (Dirichlet)
# ============================================================================

"""
    Temperature(value)

Prescribed temperature BC. Value can be a Number or Function `(x, t) -> value`.
"""
Temperature(value::Number) = PrescribedValue{EnergyPhysics}(value)
Temperature(f::Function) = PrescribedValue{EnergyPhysics}(f)

Base.show(io::IO, bc::PrescribedValue{EnergyPhysics}) = print(io, "Temperature")

# ============================================================================
# HeatFlux (Neumann)
# ============================================================================

"""
    HeatFlux(flux)

Prescribed heat flux BC: ∂T/∂n = q. Flux can be a Number or Function `(x, t) -> flux`.
"""
HeatFlux(flux::Number) = PrescribedFlux{EnergyPhysics}(flux)
HeatFlux(f::Function) = PrescribedFlux{EnergyPhysics}(f)

Base.show(io::IO, bc::PrescribedFlux{EnergyPhysics}) = print(io, "HeatFlux")

# ============================================================================
# Convection (Robin)
# ============================================================================

"""
    Convection(h, k, T∞)

Convective heat transfer: h·T + k·∂T/∂n = h·T∞(x,t).
T∞ can be a Number or Function `(x, t) -> ambient_temp`.
"""
struct Convection{H, K, F<:Function} <: Robin
    h::H
    k::K
    T∞::F

    function Convection(h::H, k::K, T∞::F) where {H, K, F<:Function}
        h < 0 && throw(ArgumentError("Heat transfer coefficient must be non-negative"))
        k <= 0 && throw(ArgumentError("Thermal conductivity must be positive"))
        return new{H, K, F}(h, k, T∞)
    end
end

Convection(h, k, T∞::Number) = Convection(h, k, (x, t) -> T∞)

physics_domain(::Type{<:Convection}) = EnergyPhysics()

α(bc::Convection) = bc.h
β(bc::Convection) = bc.k
(bc::Convection)(x, t) = bc.h * bc.T∞(x, t)

function Base.show(io::IO, bc::Convection)
    T∞_val = bc.T∞(zeros(3), 0.0)
    print(io, "Convection: h=$(bc.h), k=$(bc.k), T∞≈$T∞_val")
end

# ============================================================================
# Adiabatic (Neumann with zero flux)
# ============================================================================

"""
    Adiabatic()

Thermally insulated boundary: ∂T/∂n = 0.
"""
const Adiabatic = ZeroFlux{EnergyPhysics}

Base.show(io::IO, ::ZeroFlux{EnergyPhysics}) = print(io, "Adiabatic")
