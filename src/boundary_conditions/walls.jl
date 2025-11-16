"""
Wall boundary condition for fluid flows - Dirichlet type for velocity.

Represents a solid boundary where the no-slip condition is enforced.
For incompressible Navier-Stokes equations, the velocity at the wall
matches the wall velocity:
    v = v_wall

# Cases
- Stationary wall (no-slip): `Wall()` or `Wall(0)` → v = 0
- Moving wall: `Wall(v_wall)` → v = v_wall

# Fields
- `v`: Wall velocity (nothing or 0 for stationary, vector/scalar for moving wall)
"""
struct Wall{T} <: AbstractBoundaryCondition
    v::T
end

Wall() = Wall(nothing)

bc_type(::Wall) = Dirichlet()
bc_value(bc::Wall) = bc.v === nothing ? 0 : bc.v
