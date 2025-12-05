"""
    Wall{T} <: Dirichlet

No-slip wall boundary condition for fluid flows.

Enforces velocity = 0 at solid boundaries.
"""
struct Wall{T} <: Dirichlet
    v::T
end

"""
    Wall()

Stationary wall (no-slip): velocity = 0
"""
Wall() = Wall(nothing)

physics_domain(::Type{<:Wall}) = WallPhysics()
(bc::Wall)() = bc.v === nothing ? 0 : bc.v

Base.show(io::IO, ::Wall) = print(io, "Wall (no-slip)")
