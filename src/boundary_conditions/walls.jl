"""
    Wall <: AbstractBoundaryCondition

Wall boundary condition for fluid flows - Dirichlet type for velocity (α=1, β=0).
Represents a solid boundary which fluid does not penetrate and maintains a no-slip 
or specified velocity condition. The wall can be stationary or moving.

# Fields
- `v::T`: Velocity value (nothing for stationary wall, or a velocity vector/function)
"""
struct Wall{T} <: AbstractBoundaryCondition
    v::T
end

# Constructor for stationary wall
Wall() = Wall(nothing)

# Helpers
bc_type(::Wall) = Dirichlet()
bc_value(bc::Wall) = bc.v

# General make_bc! delegates to apply_bc!
function make_bc!(A, b, boundary::Wall, surf, domain, ids)
    apply_bc!(A, b, boundary, surf, domain, ids, bc_value(boundary))
end
