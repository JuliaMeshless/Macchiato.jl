abstract type FluidBoundaryCondition <: AbstractBoundaryCondition end

# ============================================================================
# VelocityInlet (Dirichlet)
# ============================================================================

struct VelocityInlet{T} <: FluidBoundaryCondition
    v::T
end

bc_type(::VelocityInlet) = Dirichlet()
bc_value(bc::VelocityInlet) = bc.v

function make_bc(boundary::VelocityInlet, surf, domain::Domain{Dim}; kwargs...) where {Dim}
    s = domain.cloud[surf]
    surf_ids = only(s.points.indices)
    v = boundary.v
    # Return closure for ODE time evolution: (du, u, p, t) -> modify u in-place
    (du, u, p, t) -> (u[surf_ids] .= v; nothing)
end

# ============================================================================
# PressureOutlet (Dirichlet)
# ============================================================================

struct PressureOutlet{T} <: FluidBoundaryCondition
    p::T
end

bc_type(::PressureOutlet) = Dirichlet()
bc_value(bc::PressureOutlet) = bc.p

function make_bc(boundary::PressureOutlet, surf, domain::Domain{Dim}; kwargs...) where {Dim}
    s = domain.cloud[surf]
    ids = only(s.points.indices)
    p_val = boundary.p
    # Return closure for ODE time evolution: (du, u, p, t) -> modify u in-place
    (du, u, p, t) -> (u[ids] .= p_val; nothing)
end
