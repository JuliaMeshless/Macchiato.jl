# ============================================================================
# VelocityInlet (Dirichlet)
# ============================================================================

"""
    VelocityInlet{T} <: Dirichlet

Prescribed velocity at inlet boundary.
"""
struct VelocityInlet{T} <: Dirichlet
    v::T
end

(bc::VelocityInlet)() = bc.v

# Time evolution - return closure for ODE: (du, u, p, t) -> modify u
# function make_bc(boundary::VelocityInlet, surf, domain::Domain{Dim}; kwargs...) where {Dim}
#     s = domain.cloud[surf]
#     surf_ids = only(s.points.indices)
#     v = boundary.v
#     (du, u, p, t) -> (u[surf_ids] .= v; nothing)
# end

Base.show(io::IO, bc::VelocityInlet) = print(io, "VelocityInlet: $(bc.v)")

# ============================================================================
# PressureOutlet (Dirichlet)
# ============================================================================

"""
    PressureOutlet{T} <: Dirichlet

Prescribed pressure at outlet boundary.
"""
struct PressureOutlet{T} <: Dirichlet
    p::T
end

(bc::PressureOutlet)() = bc.p

# Time evolution - return closure for ODE: (du, u, p, t) -> modify u
# function make_bc(boundary::PressureOutlet, surf, domain::Domain{Dim}; kwargs...) where {Dim}
#     s = domain.cloud[surf]
#     ids = only(s.points.indices)
#     p_val = boundary.p
#     (du, u, p, t) -> (u[ids] .= p_val; nothing)
# end

Base.show(io::IO, bc::PressureOutlet) = print(io, "PressureOutlet: $(bc.p)")
