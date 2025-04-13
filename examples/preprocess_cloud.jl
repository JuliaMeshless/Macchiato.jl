using MeshlessMultiphysics
import MeshlessMultiphysics as MM
using RadialBasisFunctions
using WhatsThePoint
import WhatsThePoint as WTP
using StaticArrays
using LinearAlgebra
using Unitful: m, °, ustrip

# Include our extension file to add support for Meshes.Vec in _angle
include(joinpath(@__DIR__, "wtp_extensions.jl"))

##
# create boundary points

L = (1m, 1m)

dx = 1 / 129 * m # boundary point spacing
S = ConstantSpacing(dx)
rx = dx:dx:(L[1] - dx)
ry = dx:dx:(L[2] - dx)

p_bot = map(i -> WTP.Point(i, 0m), rx)
p_right = map(i -> WTP.Point(L[1], i), ry)
p_top = map(i -> WTP.Point(i, L[2]), reverse(rx))
p_left = map(i -> WTP.Point(0m, i), reverse(ry))

n_bot = map(i -> WTP.Vec(0.0, -1.0), rx)
n_right = map(i -> WTP.Vec(1.0, 0.0), ry)
n_top = map(i -> WTP.Vec(0.0, 1.0), rx)
n_left = map(i -> WTP.Vec(-1.0, 0.0), ry)

p = vcat(p_bot, p_right, p_top, p_left) # points
n = vcat(n_bot, n_right, n_top, n_left) # normals
a = fill(dx, length(p)) # areas

part = PointBoundary(p, n, a)

# Restore the original call
split_surface!(part, ustrip(75°))
combine_surfaces!(part, :surface3, :surface4)

Δ = dx
cloud = WhatsThePoint.discretize(part, ConstantSpacing(Δ), alg = VanDerSandeFornberg())

conv = repel!(cloud, ConstantSpacing(Δ); α = Δ / 20, max_iters = 500)

# generate a new set of vectors from cloud
#starting from cloud isolate cloud.volume
internal_points = cloud.volume.points
internal_coords = ustrip.(MM._coords(cloud.volume))

is_boundary = zeros(Bool, length(internal_points) + length(cloud.boundary.points))
is_Neumann = zeros(Bool, length(cloud.boundary.points))
boundary_points = SVector{2, Float64}[]
normals = SVector{2, Float64}[]
surface_name = Vector{Symbol}(undef, length(cloud.boundary.points))
surfaces = cloud.boundary.surfaces
bd_point_counter = [0]
for key in keys(surfaces)
    for point in surfaces[key].geoms
        bd_point_counter[1] += 1
        # println(point)
        push!(
            boundary_points, SVector{2}(
                ustrip(point.point.coords.x), ustrip(point.point.coords.y)))
        push!(normals,
            SVector{2}(
                ustrip(point.normal.coords[1]), ustrip(point.normal.coords[2])))
        is_boundary[length(internal_points) + bd_point_counter[1]] = true
        is_Neumann[bd_point_counter[1]] = false #TODO:remove hardcoding
    end
end
