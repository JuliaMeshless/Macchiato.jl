"""
    Wall(velocity)
    Wall()

No-slip wall or moving wall BC. Value can be a Number or Function `(x, t) -> velocity`.
No arguments creates stationary wall (v=0). Uses `WallPhysics` domain.
"""
Wall(velocity::Number) = PrescribedValue{WallPhysics}(velocity)
Wall(f::Function) = PrescribedValue{WallPhysics}(f)
Wall() = PrescribedValue{WallPhysics}(0.0)

Base.show(io::IO, bc::PrescribedValue{WallPhysics}) = begin
    v = bc(zeros(3), 0.0)
    if v ≈ 0.0
        print(io, "Wall (no-slip)")
    else
        print(io, "Wall (moving): v≈$v")
    end
end
