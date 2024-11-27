
abstract type FluidBoundaryCondition <: AbstractBoundaryCondition end

struct VelocityInlet{T} <: FluidBoundaryCondition
    v::T
end
(bc::VelocityInlet)() = bc.v
(bc::VelocityInlet{<:Function})(x, t) = bc.v(x, t)

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

struct PressureOutlet{T} <: FluidBoundaryCondition
    p::T
end
(bc::PressureOutlet)() = bc.p
(bc::PressureOutlet{<:Function})(x, t) = bc.p(x, t)

function make_bc(boundary::PressureOutlet, surf, domain::Domain{Dim}; kwargs...) where {Dim}
    s = domain.cloud[surf]
    ids = only(s.points.indices)
    function bc(du, u, p, t)
        u[ids] .= boundary.p
        return nothing
    end
    return bc
end
