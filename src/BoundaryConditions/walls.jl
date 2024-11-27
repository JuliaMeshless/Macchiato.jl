"""
    Wall <: AbstractBoundaryCondition

Wall is a solid boundary which fluid does not penetrate and maintains a no-slip condition. The wall can be stationary or moving.
"""
struct Wall{T} <: AbstractBoundaryCondition
    v::T
end

# convienience constructors

"""
    Wall()

Creates a stationary wall.
"""
Wall() = Wall(nothing)
