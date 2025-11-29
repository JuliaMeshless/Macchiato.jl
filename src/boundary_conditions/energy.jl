# ============================================================================
# Temperature (Dirichlet)
# ============================================================================

"""
    Temperature{T} <: Dirichlet

Prescribed temperature boundary condition.
"""
struct Temperature{T} <: Dirichlet
    temperature::T
end

(bc::Temperature)() = bc.temperature
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

Base.show(io::IO, bc::Temperature) = print(io, "Temperature: $(bc.temperature)")

# ============================================================================
# HeatFlux (Neumann)
# ============================================================================

"""
    HeatFlux{Q} <: Neumann

Prescribed heat flux boundary condition (∂T/∂n = q).
"""
struct HeatFlux{Q} <: Neumann
    heat_flux::Q
end

HeatFlux(q) = HeatFlux(q)

(bc::HeatFlux)() = bc.heat_flux
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

Base.show(io::IO, bc::HeatFlux) = print(io, "HeatFlux: $(bc.heat_flux)")

# ============================================================================
# Convection (Robin)
# ============================================================================

"""
    Convection{H, K, T} <: Robin

Convective heat transfer: h·T + k·∂T/∂n = h·T∞

Newton's law of cooling at boundary.
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

# bc_type(bc::Convection) = Robin(bc.h, bc.k)
α(bc::Convection) = bc.h
β(bc::Convection) = bc.k
(bc::Convection)() = bc.h * bc.T∞

# function make_bc!(A, b, boundary::Convection, surf, domain, ids; kwargs...)
#     make_bc_robin!(A, b, boundary.h, boundary.k, surf, domain, ids, bc_value(boundary);
#         shadow_op = boundary.shadow_op, kwargs...)
# end

function Base.show(io::IO, bc::Convection)
    print(io, "Convection: h=$(bc.h), k=$(bc.k), T∞=$(bc.T∞)")
end

# ============================================================================
# Adiabatic (Neumann with zero flux)
# ============================================================================

"""
    Adiabatic <: Neumann

Thermally insulated boundary: ∂T/∂n = 0
"""
struct Adiabatic <: Neumann end

# Constructor
Adiabatic() = Adiabatic()

# bc_type(::Adiabatic) = Neumann()
(bc::Adiabatic)() = 0.0  # Zero flux

# LinearProblem: Pass shadow_op to make_bc_neumann!
# function make_bc!(A, b, boundary::Adiabatic, surf, domain, ids; kwargs...)
#     make_bc_neumann!(
#         A, b, surf, domain, ids, 0.0; shadow_op = boundary.shadow_op, kwargs...)
# end

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

# function make_bc(boundary::Adiabatic, surf, domain, ids; kwargs...)
#     println("creating Adiabatic BC")
#     d = directional(_coords(domain.cloud), _coords(surf), normals(surf); kwargs...)
#     update_weights!(d)
#     w = d.weights
#     wi = diag(w)
#     w[diagind(w)] .= 0
#     dropzeros!(w)

#     function bc(du, u, p, t)
#         # TODO is this correct?
#         #du[surf_ids] .= d(u) .- u[surf_ids]
#         u[ids] .= (w * u) ./ wi
#         return nothing
#     end
#     return bc
# end

Base.show(io::IO, ::Adiabatic) = print(io, "Adiabatic")
