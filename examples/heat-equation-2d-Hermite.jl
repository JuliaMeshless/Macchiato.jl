using MeshlessMultiphysics
import MeshlessMultiphysics as MM
using RadialBasisFunctions
import RadialBasisFunctions as RBF
names(RBF, all = true)
using WhatsThePoint
import WhatsThePoint as WTP
using Accessors
using NearestNeighbors
using StaticArrays
using LinearAlgebra
using SparseArrays
using LinearSolve
using IterativeSolvers
using Unitful: m, °, ustrip
include("visualize_results.jl")

function get_new_domain_info(cloud)
    #this also introduces dependency from Accessors.jl
    #TODO:consider using shorter vectors (with length of boundary points only)
    surfaces = cloud.boundary.surfaces
    all_coords = ustrip.(MM._coords(cloud))
    all_normals = zeros(typeof(all_coords[1]), size(all_coords))
    is_boundary = falses(length(all_coords))
    is_Neumann = falses(length(all_coords))
    surface_name = Vector{Symbol}(undef, length(all_coords))

    all_coords_tree = KDTree(all_coords)
    boundary_coordinates = zeros(typeof(all_coords[1]))
    for key in keys(surfaces)
        for bnd_geom in surfaces[key].geoms
            #TODO: check if @reset reduces performance, think about alternatives
            #TODO: extend to generic case not just for 2D
            @reset boundary_coordinates[1] = ustrip(bnd_geom.point.coords.x)
            @reset boundary_coordinates[2] = ustrip(bnd_geom.point.coords.y)
            idx, dist = knn(all_coords_tree, boundary_coordinates, 1)
            if dist[1] > 1e-5
                @warn "Boundary point $(boundary_coordinates) is too far from the nearest point in the cloud"
            else
                global_index = idx[1] # get the index of the nearest point in the cloud
                @reset all_normals[global_index] .= ustrip(bnd_geom.normal)

                is_boundary[global_index] = true
                is_Neumann[global_index] = false #TODO:remove hardcoding
                surface_name[global_index] = key
            end
        end
    end

    return all_coords, all_normals, is_boundary, is_Neumann, surface_name
end

function get_boundary_values(bcs, is_boundary, surface_name)
    boundary_values = zeros(sum(is_boundary))
    boundary2global = findall(is_boundary)
    global2boundary = cumsum(is_boundary)
    for i in eachindex(boundary_values)
        global_index = boundary2global[i]
        surface_sym = surface_name[global_index]
        boundary_values[i] = bcs[surface_sym].temperature
    end
    return boundary_values, boundary2global, global2boundary
end

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
split_surface!(part, 75°)
combine_surfaces!(part, :surface3, :surface4)

Δ = dx
cloud = WhatsThePoint.discretize(part, ConstantSpacing(Δ), alg = VanDerSandeFornberg())
conv = repel!(cloud, ConstantSpacing(Δ); α = Δ / 20, max_iters = 1000)

#=
Here I am having trouble building the stencils with this cloud structure
this is because in the cloud structure the separation beteween internal 
and boundary points is too strict, probably it would be more efficient to have
a single array of points and a boolean array indicating whether the point is a boundary point or not
and having points ordered in such a way that closer points are next to each other
the equivalent of MM._coords(cloud) for boundary points appears to be
cloud.boundary.points.geoms 
=#
(all_coords, all_normals, is_boundary, is_Neumann, surface_name) = get_new_domain_info(cloud)

print("sum of is_boundary: ", sum(is_boundary), " -should be 512- \n")

rbf_basis = RBF.PHS(3; poly_deg = 2)
mon = RBF.MonomialBasis(2, rbf_basis.poly_deg)
Lrbf = RBF.∇²(rbf_basis)
Lmon = RBF.∇²(mon)
k = RBF.autoselect_k(all_coords, rbf_basis)
adjl = RBF.find_neighbors(all_coords, k)

lhs, rhs = RBF._build_weights(
    all_coords, all_normals, is_boundary, is_Neumann, adjl, rbf_basis, Lrbf, Lmon, mon)

bcs = Dict(
    :surface1 => Temperature(10), :surface2 => Temperature(0), :surface3 => Temperature(5))

boundary_values, boundary2global, global2boundary = get_boundary_values(
    bcs, is_boundary, surface_name)

T, history = bicgstabl(lhs, -rhs * boundary_values; reltol = 1e-10, log = true)

# physics models and boundary conditions
h = 250 / 1e6 # W / (mm^2 K)
T∞ = 25 + 273.15 # K
k = 40 / 1e3 # W / (mm K)
ρ = 7833 / 1e9  # kg/mm^3
cₚ = 0.465 * 1e3 # J / (kg K)
α = k / (cₚ * ρ) # mm^2 / s

T_full = zeros(length(all_coords))

for i in eachindex(boundary_values)
    global_index = boundary2global[i]
    T_full[global_index] = boundary_values[i]
end
internal_indices = findall(.!is_boundary)
T_full[internal_indices] = T

domain = MM.Domain(cloud, bcs, SolidEnergy(k = k, ρ = ρ, cₚ = cₚ))
viz_2d(domain, T_full; markersize = markersize, size = figsize, levels = 32)