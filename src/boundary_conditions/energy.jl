# ============================================================================
# Temperature (Dirichlet)
# ============================================================================

"""
    Temperature{T} <: Dirichlet

Prescribed temperature boundary condition.

Internally represented as `FixedValue{EnergyPhysics, T}`.

# Examples
```julia
# Constant temperature
bc = Temperature(100.0)

# Spatially varying temperature
bc = Temperature([100.0, 120.0, 110.0])

# Function-based temperature
bc = Temperature(x -> 100.0 + 10.0 * x[1])
```
"""
const Temperature{T} = FixedValue{EnergyPhysics, T}

# Constructor for convenience
Temperature(value::T) where {T} = FixedValue{EnergyPhysics, T}(value)

# (bc::Temperature{<:Function})(x, t) = bc.temperature(x, t)

# Time evolution - return closure for ODE: (du, u, p, t) -> modify u
# function make_bc(boundary::Temperature, surf, domain, ids; kwargs...)
#     T = boundary.temperature
#     (du, u, p, t) -> (u[ids] .= T; nothing)
# end
# function make_bc(boundary::Temperature{<:Function}, surf, domain, ids; kwargs...)
#     T_func = boundary.temperature
#     (du, u, p, t) -> (u[ids] .= T_func(surf, t); nothing)
# end

Base.show(io::IO, bc::Temperature) = print(io, "Temperature: $(bc.value)")

# ============================================================================
# HeatFlux (Neumann)
# ============================================================================

"""
    HeatFlux{Q} <: Neumann

Prescribed heat flux boundary condition (∂T/∂n = q).

Internally represented as `Flux{EnergyPhysics, Q}`.

# Examples
```julia
# Constant heat flux
bc = HeatFlux(100.0)

# Spatially varying heat flux
bc = HeatFlux([100.0, 120.0, 110.0])

# Function-based heat flux
bc = HeatFlux(x -> 100.0 * sin(x[1]))
```
"""
const HeatFlux{Q} = Flux{EnergyPhysics, Q}

# Constructor for convenience
HeatFlux(flux::Q) where {Q} = Flux{EnergyPhysics, Q}(flux)

# (bc::HeatFlux{<:Function})(x, t) = bc.heat_flux(x, t)

# function make_bc!(A, b, boundary::HeatFlux, surf, domain, ids; kwargs...)
#     make_bc_neumann!(A, b, surf, domain, ids, boundary.heat_flux;
#         shadow_op = boundary.shadow_op, kwargs...)
# end
# function make_bc!(A, b, boundary::HeatFlux{<:Function}, surf, domain, ids; kwargs...)
#     q_func = boundary.heat_flux
#     make_bc_neumann!(A, b, surf, domain, ids, q_func;
#         shadow_op = boundary.shadow_op, kwargs...)
# end

Base.show(io::IO, bc::HeatFlux) = print(io, "HeatFlux: $(bc.flux)")

# ============================================================================
# Convection (Robin)
# ============================================================================

"""
    Convection{H, K, T} <: Robin

Convective heat transfer: h·T + k·∂T/∂n = h·T∞

Newton's law of cooling at boundary.

This is a physics-specific boundary condition with semantic field names.
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

# Physics domain trait
physics_domain(::Type{<:Convection}) = EnergyPhysics()

# Robin BC coefficients
α(bc::Convection) = bc.h
β(bc::Convection) = bc.k
(bc::Convection)() = bc.h * bc.T∞

function Base.show(io::IO, bc::Convection)
    print(io, "Convection: h=$(bc.h), k=$(bc.k), T∞=$(bc.T∞)")
end

# ============================================================================
# Adiabatic (Neumann with zero flux)
# ============================================================================

"""
    Adiabatic <: Neumann

Thermally insulated boundary: ∂T/∂n = 0

Internally represented as `ZeroGradient{EnergyPhysics}`.

The numerical method (standard derivative vs shadow points) is controlled
by the `scheme` parameter passed to `LinearProblem`, not by the BC itself.

# Usage
```julia
bc = Adiabatic()
```
"""
const Adiabatic = ZeroGradient{EnergyPhysics}

# # Time evolution - specialized implementations
# function make_bc(boundary::Adiabatic{<:ShadowPoints}, surf, domain, ids; kwargs...)
#     shadow_points = generate_shadows(surf, boundary.shadow_op)
#     coords = _coords(domain.cloud)
#     method = KNearestSearch(domain.cloud, 40)
#     adjl = search.(shadow_points, Ref(method))
#     d = regrid(_ustrip(coords), _ustrip(_coords(shadow_points)); adjl = adjl)
#     update_weights!(d)

#     function bc(du, u, p, t)
#         u[ids] .= d(u)
#         return nothing
#     end
#     return bc
# end

Base.show(io::IO, ::Adiabatic) = print(io, "Adiabatic")
