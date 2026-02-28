"""
    Wall(velocity)
    Wall()

No-slip wall or moving wall BC. Value can be a Number or Function `(x, t) -> velocity`.
No arguments creates stationary wall (v=0).
"""
Wall(v::Number) = PrescribedValue((x, t) -> v, :Wall)
Wall(f::Function) = PrescribedValue{typeof(f)}(f, :Wall)
Wall() = PrescribedValue((x, t) -> 0.0, :Wall)
