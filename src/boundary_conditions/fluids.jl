"""
    FluidBoundaryCondition <: AbstractBoundaryCondition

Abstract type for fluid boundary conditions.
"""
abstract type FluidBoundaryCondition <: AbstractBoundaryCondition end

# ============================================================================
# VelocityInlet (Dirichlet for velocity)
# ============================================================================

"""
    VelocityInlet{T} <: FluidBoundaryCondition

Velocity inlet boundary condition - Dirichlet type (α=1, β=0).
Prescribes the velocity value at an inlet boundary.
"""
struct VelocityInlet{T} <: FluidBoundaryCondition
    v::T
end

# Accessor functions
(bc::VelocityInlet)() = bc.v
(bc::VelocityInlet{<:Function})(x, t) = bc.v(x, t)

# Helpers
bc_type(::VelocityInlet) = Dirichlet()
bc_value(bc::VelocityInlet) = bc.v

# General make_bc! delegates to apply_bc!
function make_bc!(A, b, boundary::VelocityInlet, surf, domain, ids)
    apply_bc!(A, b, boundary, surf, domain, ids, bc_value(boundary))
end

function make_bc(boundary::VelocityInlet, surf, domain::Domain{Dim}; kwargs...) where {Dim}
    s = domain.cloud[surf]
    surf_ids = only(s.points.indices)
    offset = length(domain.cloud)
    function bc(du, u, p, t)
        u[ids] .= boundary.v
        return nothing
    end
    return bc
end

# ============================================================================
# PressureOutlet (Dirichlet for pressure)
# ============================================================================

"""
    PressureOutlet{T} <: FluidBoundaryCondition

Pressure outlet boundary condition - Dirichlet type (α=1, β=0).
Prescribes the pressure value at an outlet boundary.
"""
struct PressureOutlet{T} <: FluidBoundaryCondition
    p::T
end

# Accessor functions
(bc::PressureOutlet)() = bc.p
(bc::PressureOutlet{<:Function})(x, t) = bc.p(x, t)

# Helpers
bc_type(::PressureOutlet) = Dirichlet()
bc_value(bc::PressureOutlet) = bc.p

# General make_bc! delegates to apply_bc!
function make_bc!(A, b, boundary::PressureOutlet, surf, domain, ids)
    apply_bc!(A, b, boundary, surf, domain, ids, bc_value(boundary))
end

function make_bc(boundary::PressureOutlet, surf, domain::Domain{Dim}; kwargs...) where {Dim}
    s = domain.cloud[surf]
    ids = only(s.points.indices)
    function bc(du, u, p, t)
        u[ids] .= boundary.p
        return nothing
    end
    return bc
end
