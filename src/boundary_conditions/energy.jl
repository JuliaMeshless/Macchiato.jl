"""
    EnergyBoundaryCondition <: AbstractBoundaryCondition

Abstract type for energy boundary conditions.
"""
abstract type EnergyBoundaryCondition <: AbstractBoundaryCondition end

# ============================================================================
# Temperature (Dirichlet)
# ============================================================================

struct Temperature{T} <: EnergyBoundaryCondition
    temperature::T
end

bc_type(::Temperature) = Dirichlet()
bc_value(bc::Temperature) = bc.temperature

# Time evolution - return closure for ODE: (du, u, p, t) -> modify u
function make_bc(boundary::Temperature, surf, domain, ids; kwargs...)
    T = boundary.temperature
    (du, u, p, t) -> (u[ids] .= T; nothing)
end

Base.show(io::IO, bc::Temperature) = print(io, "Temperature: $(bc.temperature)")

# ============================================================================
# HeatFlux (Neumann)
# ============================================================================

"""
    HeatFlux <: EnergyBoundaryCondition

Prescribed heat flux boundary condition - Neumann type.

# Constructors
- `HeatFlux(q)`: Standard directional derivative method
- `HeatFlux(q, shadow_op)`: Use shadow points for derivative approximation

# Fields
- `heat_flux`: Prescribed normal heat flux value
- `shadow_op`: Optional shadow points operator
"""
struct HeatFlux{Q, T} <: EnergyBoundaryCondition
    heat_flux::Q
    shadow_op::T
end

HeatFlux(q) = HeatFlux(q, nothing)

bc_type(::HeatFlux) = Neumann()
bc_value(bc::HeatFlux) = bc.heat_flux

function make_bc!(A, b, boundary::HeatFlux, surf, domain, ids; kwargs...)
    make_bc_neumann!(A, b, surf, domain, ids, boundary.heat_flux;
        shadow_op = boundary.shadow_op, kwargs...)
end

Base.show(io::IO, bc::HeatFlux) = print(io, "HeatFlux: $(bc.heat_flux)")

# ============================================================================
# Convection (Robin)
# ============================================================================

"""
Convection boundary condition - Robin type.

Represents convective heat transfer to surrounding fluid at the boundary.
The heat flux at the boundary is governed by Newton's law of cooling:
    q = h(T - T∞)

For the heat equation with thermal conductivity k:
    -k ∂T/∂n = h(T - T∞)

Rearranging to Robin form (α*T + β*∂T/∂n = g):
    h*T + k*∂T/∂n = h*T∞

where:
- α = h (heat transfer coefficient)
- β = k (thermal conductivity of the material)
- g = h*T∞ (prescribed value)

# Constructors
- `Convection(h, k, T∞)`: Standard directional derivative method
- `Convection(h, k, T∞, shadow_op)`: Use shadow points for derivative approximation

# Fields
- `h`: Heat transfer coefficient [W/(m²·K)]
- `k`: Thermal conductivity of the material [W/(m·K)]
- `T∞`: Ambient/surrounding temperature [K]
- `shadow_op`: Optional shadow points operator
"""
struct Convection{H, K, T, S} <: EnergyBoundaryCondition
    h::H   # heat transfer coefficient
    k::K   # thermal conductivity
    T∞::T  # ambient temperature
    shadow_op::S

    function Convection(h::H, k::K, T∞::T, shadow_op::S = nothing) where {H, K, T, S}
        h < 0 && throw(ArgumentError("Heat transfer coefficient must be non-negative"))
        k <= 0 && throw(ArgumentError("Thermal conductivity must be positive"))
        return new{H, K, T, S}(h, k, T∞, shadow_op)
    end
end

bc_type(bc::Convection) = Robin(bc.h, bc.k)
bc_value(bc::Convection) = bc.h * bc.T∞

function make_bc!(A, b, boundary::Convection, surf, domain, ids; kwargs...)
    make_bc_robin!(A, b, boundary.h, boundary.k, surf, domain, ids, bc_value(boundary);
        shadow_op = boundary.shadow_op, kwargs...)
end

function Base.show(io::IO, bc::Convection)
    print(io, "Convection: h=$(bc.h), k=$(bc.k), T∞=$(bc.T∞)")
end

# ============================================================================
# Adiabatic (Neumann with zero flux)
# ============================================================================

"""
    Adiabatic <: EnergyBoundaryCondition

Adiabatic boundary condition - homogeneous Neumann with zero heat flux.
Represents a thermally insulated boundary where ∂ₙT = 0.

# Constructors
- `Adiabatic()`: Standard directional derivative method
- `Adiabatic(Δ)`: 1st order shadow points with spacing Δ
- `Adiabatic(Δ, order)`: Shadow points with specified order (1 or 2)

# Fields
- `shadow_op`: Optional shadow points operator for high-accuracy derivatives
"""
struct Adiabatic{T} <: EnergyBoundaryCondition
    shadow_op::T
end

# Constructors
Adiabatic() = Adiabatic(nothing)
Adiabatic(Δ::Number) = Adiabatic(ShadowPoints(Δ, 1))
Adiabatic(Δ::Number, order::T) where {T <: Int} = Adiabatic(ShadowPoints(Δ, order))

bc_type(::Adiabatic) = Neumann()
bc_value(::Adiabatic) = 0.0  # Zero flux

# LinearProblem: Pass shadow_op to make_bc_neumann!
function make_bc!(A, b, boundary::Adiabatic, surf, domain, ids; kwargs...)
    make_bc_neumann!(
        A, b, surf, domain, ids, 0.0; shadow_op = boundary.shadow_op, kwargs...)
end

# Time evolution - specialized implementations
function make_bc(boundary::Adiabatic{<:ShadowPoints}, surf, domain, ids; kwargs...)
    shadow_points = generate_shadows(surf, boundary.shadow_op)
    coords = _coords(domain.cloud)
    method = KNearestSearch(domain.cloud, 40)
    adjl = search.(shadow_points, Ref(method))
    d = regrid(_ustrip(coords), _ustrip(_coords(shadow_points)); adjl = adjl)
    update_weights!(d)

    function bc(du, u, p, t)
        u[ids] .= d(u)
        return nothing
    end
    return bc
end

function make_bc(boundary::Adiabatic, surf, domain, ids; kwargs...)
    println("creating Adiabatic BC")
    d = directional(_coords(domain.cloud), _coords(surf), normals(surf); kwargs...)
    update_weights!(d)
    w = d.weights
    wi = diag(w)
    w[diagind(w)] .= 0
    dropzeros!(w)

    function bc(du, u, p, t)
        # TODO is this correct?
        #du[surf_ids] .= d(u) .- u[surf_ids]
        u[ids] .= (w * u) ./ wi
        return nothing
    end
    return bc
end

Base.show(io::IO, ::Adiabatic) = print(io, "Adiabatic")
