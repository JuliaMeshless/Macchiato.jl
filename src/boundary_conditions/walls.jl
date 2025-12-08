"""
    Wall{T} <: Dirichlet

No-slip wall boundary condition for fluid flows.

Enforces velocity = 0 at solid boundaries (or prescribed velocity for moving walls).

This BC uses `WallPhysics` domain, making it compatible with both fluid and energy models.

# Examples
```julia
# Stationary wall (no-slip)
bc = Wall()

# Moving wall with prescribed velocity
bc = Wall(1.0)
```
"""
struct Wall{T} <: Dirichlet
    v::T
end

"""
    Wall()

Stationary wall (no-slip): velocity = 0
"""
Wall() = Wall(nothing)

(bc::Wall)() = bc.v === nothing ? 0 : bc.v

# Physics domain trait - WallPhysics is compatible with both fluids and energy
physics_domain(::Type{<:Wall}) = WallPhysics()

Base.show(io::IO, bc::Wall{Nothing}) = print(io, "Wall (no-slip)")
Base.show(io::IO, bc::Wall) = print(io, "Wall (moving): v=$(bc.v)")
