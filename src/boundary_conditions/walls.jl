"""
    Wall{T} <: Dirichlet

No-slip wall (v=0) or moving wall with prescribed velocity.
Uses `WallPhysics` domain (compatible with fluid and energy models).
"""
struct Wall{T} <: Dirichlet
    v::T
end

"""Stationary wall (no-slip): velocity = 0."""
Wall() = Wall(nothing)

(bc::Wall)() = bc.v === nothing ? 0 : bc.v

# Physics domain trait - WallPhysics is compatible with both fluids and energy
physics_domain(::Type{<:Wall}) = WallPhysics()

Base.show(io::IO, bc::Wall{Nothing}) = print(io, "Wall (no-slip)")
Base.show(io::IO, bc::Wall) = print(io, "Wall (moving): v=$(bc.v)")
