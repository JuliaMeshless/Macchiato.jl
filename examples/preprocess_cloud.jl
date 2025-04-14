using MeshlessMultiphysics
import MeshlessMultiphysics as MM
using RadialBasisFunctions
import RadialBasisFunctions as RBF
using WhatsThePoint
import WhatsThePoint as WTP
using StaticArrays
using LinearAlgebra
using ChunkSplitters
using SparseArrays
using LinearSolve
using IterativeSolvers, IncompleteLU
using Unitful: m, °, ustrip

# Include our extension file to add support for Meshes.Vec in _angle
include(joinpath(@__DIR__, "wtp_extensions.jl"))
include(joinpath(@__DIR__, "RBF_extensions.jl"))

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
surface_name = Symbol[]
surface_index = zeros(Int, length(internal_points) + length(cloud.boundary.points))
for i in 1:length(internal_points)
    push!(surface_name, :volume)
    surface_index[i] = i
end
boundary_coords = SVector{2, Float64}[]
normals = SVector{2, Float64}[]

surfaces = cloud.boundary.surfaces
bd_point_counter = [0]
for key in keys(surfaces)
    for (bnd_index, point) in enumerate(surfaces[key].geoms)
        bd_point_counter[1] += 1
        # println(point)
        push!(
            boundary_coords, SVector{2}(
                ustrip(point.point.coords.x), ustrip(point.point.coords.y)))
        push!(normals,
            SVector{2}(
                ustrip(point.normal.coords[1]), ustrip(point.normal.coords[2])))
        is_boundary[length(internal_points) + bd_point_counter[1]] = true
        # is_Neumann[bd_point_counter[1]] = false #TODO:remove hardcoding
        push!(surface_name, key)
        surface_index[length(internal_points) + bd_point_counter[1]] = bnd_index
    end
end

is_Neumann = zeros(Bool, length(internal_points) + length(cloud.boundary.points))
all_normals = vcat(zeros(SVector{2}, length(internal_points)), normals)
all_coords = vcat(internal_coords, boundary_coords)
rbf_basis = RBF.PHS(3; poly_deg = 2)
mon = RBF.MonomialBasis(2, rbf_basis.poly_deg)
Lrbf = RBF.∇²(rbf_basis)
Lmon = RBF.∇²(mon)
k = RBF.autoselect_k(all_coords, rbf_basis)
adjl = RBF.find_neighbors(all_coords, k)

lhs, rhs = _build_weights(
    all_coords, all_normals, is_boundary, is_Neumann, adjl, rbf_basis, Lrbf, Lmon, mon)

# physics models and boundary conditions
h = 250 / 1e6 # W / (mm^2 K)
T∞ = 25 + 273.15 # K
k = 40 / 1e3 # W / (mm K)
ρ = 7833 / 1e9  # kg/mm^3
cₚ = 0.465 * 1e3 # J / (kg K)
α = k / (cₚ * ρ) # mm^2 / s

bcs = Dict(
    :surface1 => Temperature(10), :surface2 => Temperature(0), :surface3 => Temperature(5))

boundary_values = zeros(sum(is_boundary))
boundary2global = findall(is_boundary)
global2boundary = cumsum(is_boundary)
for i in eachindex(boundary_values)
    global_index = boundary2global[i]
    surface_sym = surface_name[global_index]
    boundary_values[i] = bcs[surface_sym].temperature
end

# @printf("ILU left-hand-side ")
#     t0_ILU  = time()
#     ilu_LHS = ilu(lhs,τ=6)
#     t1_ILU  = time()
# @printf("done, elapsed time = %f s\n", t1_ILU-t0_ILU)
solution, history = bicgstabl(lhs, rhs * boundary_values, 2; reltol = 1e-10, log = true)
if history.isconverged
    println("Solver converged in $(history.iters) iterations")
else
    @warn "Solver did not converge after $(history.iters) iterations"
end

#scatter sol on internal coords
using Plots
plotlyjs()
p = scatter(
    [coord[1] for coord in internal_coords], [coord[2] for coord in internal_coords],
    zcolor = solution, color = :turbo, aspect_ratio = :equal,
    title = "Temperature Distribution", colorbar = true, markersize = 1, markerstrokewidth = 0)
plot!(p, layout = (1, 1), size = (800, 600), margin = 10Plots.mm)
display(p)