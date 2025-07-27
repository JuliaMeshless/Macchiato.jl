using GLMakie
using MeshlessMultiphysics
import MeshlessMultiphysics as MM
using RadialBasisFunctions
import RadialBasisFunctions as RBF
names(RBF, all = true)
using Accessors
using NearestNeighbors
using StaticArrays
using LinearAlgebra
using SparseArrays
using LinearSolve
using IterativeSolvers
using Unitful: ustrip
include("create_2d_geometry.jl")
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

function get_b(rhs, domain, is_boundary)
    #TODO: make more efficient by avoiding constructing the boundary values array

    boundary_values = zeros(sum(is_boundary))
    # boundary2global = findall(is_boundary)
    global2boundary = cumsum(is_boundary)

    for b in domain.boundaries
        ids, bcs = b.second
        local_ids = global2boundary[ids]
        boundary_values[local_ids] .= bcs.temperature
    end

    return rhs * boundary_values, boundary_values #, boundary2global, global2boundary
end

function make_system_Hermite(domain; kwargs...)

    # domain info might be included in the domain structure (TODO)
    all_coords, all_normals, is_boundary, is_Neumann, surface_name = get_new_domain_info(domain.cloud)

    rbf_basis = RBF.PHS(3; poly_deg = 2)
    mon = RBF.MonomialBasis(2, rbf_basis.poly_deg)
    Lrbf = RBF.∇²(rbf_basis)
    Lmon = RBF.∇²(mon)
    k_stencil = RBF.autoselect_k(all_coords, rbf_basis)
    adjl = RBF.find_neighbors(all_coords, k_stencil)

    A, rhs = RBF._build_weights(
        all_coords, all_normals, is_boundary, is_Neumann, adjl, rbf_basis, Lrbf, Lmon, mon)

    #this part is very much suboptimal (TODO: integrate better)
    (; k, ρ, cₚ) = only(domain.models)
    α = k / (cₚ * ρ)
    A = α * A

    b, boundary_values = get_b(rhs, domain, is_boundary)
    b = -α * b
    return A, b, boundary_values
end

function LinearProblemHermite(domain)
    A, b, boundary_values = make_system_Hermite(domain)
    return LinearSolve.LinearProblem(dropzeros(A), b), boundary_values
end

part, cloud, Δ = create_2d_geometry()

bcs = Dict(
    :surface1 => Temperature(10), :surface2 => Temperature(0), :surface3 => Temperature(5))

# physics models and boundary conditions
h = 250 / 1e6 # W / (mm^2 K)
T∞ = 25 + 273.15 # K
k = 40 / 1e3 # W / (mm K)
ρ = 7833 / 1e9  # kg/mm^3
cₚ = 0.465 * 1e3 # J / (kg K)
α = k / (cₚ * ρ) # mm^2 / s

domain = MM.Domain(cloud, bcs, SolidEnergy(k = k, ρ = ρ, cₚ = cₚ))

prob, boundary_values = LinearProblemHermite(domain)
@time sol = solve(prob)
T = sol.u
T_full = vcat(boundary_values, T)
figsize = (1500, 1500)
markersize = 0.0025
viz_2d(domain, T_full; markersize = markersize, size = figsize, levels = 32)
