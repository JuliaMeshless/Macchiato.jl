using MeshlessMultiphysics
using GLMakie
GLMakie.activate!()
import MeshlessMultiphysics as MM
using RadialBasisFunctions
import RadialBasisFunctions as RBF
names(RBF, all = true)
using WhatsThePoint
import WhatsThePoint as WTP
const WTP = WhatsThePoint
using Accessors
using NearestNeighbors
using StaticArrays
using LinearAlgebra
using SparseArrays
using LinearSolve
using IterativeSolvers
using Unitful: m, °, ustrip
include("create_2d_geometry.jl")
include("visualize_results.jl")
include("visualize_surfaces.jl")
include("exact_solution.jl")

#at the moment need a weaker Domain struct in order to enforce specific bc at any point
#this is more accurate and should allow comparison with manufactured solutions
struct WeakDomain
    cloud::PointCloud
    boundaries::Dict{Symbol, Tuple{UnitRange, Vector{Float64}}}
    models::Vector{<:AbstractModel}
    name::Symbol
end

function WeakDomain(
        cloud::PointCloud, boundaries, models)
    @assert all(v -> v isa Vector{Float64}, values(boundaries)) "All boundary values must be Vector{Float64}"
    models = models isa Vector ? models : [models]

    ids_boundaries = Dict{Symbol, Tuple{UnitRange, Vector{Float64}}}()
    offset = 0
    for surf_name in names(cloud.boundary)
        N = length(cloud[surf_name])
        ids = offset .+ (1:N)
        offset += N
        ids_boundaries[surf_name] = (ids, boundaries[surf_name])
    end

    # display(ids_boundaries)  # Remove or comment out for production
    println("Boundary conditions: ", ids_boundaries)

    return WeakDomain(cloud, ids_boundaries, models, :domain1)
end

function get_new_domain_info(domain)
    surfaces = domain.cloud.boundary.surfaces
    all_coords = ustrip.(MM._coords(domain.cloud))
    all_normals = zeros(typeof(all_coords[1]), size(all_coords))
    is_boundary = falses(length(all_coords))
    is_Neumann = falses(length(all_coords))
    surface_name = Vector{Symbol}(undef, length(all_coords))

    all_coords_tree = KDTree(all_coords)
    boundary_coordinates = zeros(typeof(all_coords[1]))
    for key in keys(surfaces)
        for bnd_geom in surfaces[key].geoms
            @reset boundary_coordinates[1] = ustrip(bnd_geom.point.coords.x)
            @reset boundary_coordinates[2] = ustrip(bnd_geom.point.coords.y)
            idx, dist = knn(all_coords_tree, boundary_coordinates, 1)
            if dist[1] > 1e-5
                @warn "Boundary point $(boundary_coordinates) is too far from the nearest point in the cloud"
            else
                global_index = idx[1]
                @reset all_normals[global_index] .= ustrip(bnd_geom.normal)
                is_boundary[global_index] = true
                # Set Neumann for surface2, Dirichlet otherwise
                is_Neumann[global_index] = (key == :surface2)
                surface_name[global_index] = key
            end
        end
    end
    return all_coords, all_normals, is_boundary, is_Neumann, surface_name
end

function get_b(rhs, domain, is_boundary)
    boundary_values = zeros(sum(is_boundary))
    global2boundary = cumsum(is_boundary)

    for (ids, vals) in values(domain.boundaries)
        local_ids = global2boundary[ids]
        for (i, v) in enumerate(vals)
            boundary_values[local_ids[i]] = v
        end
    end
    return rhs * boundary_values, boundary_values
end

function make_system_Hermite(domain, source_term; kwargs...)
    all_coords, all_normals, is_boundary, is_Neumann, surface_name = get_new_domain_info(domain)
    rbf_basis = RBF.PHS(3; poly_deg = 2)
    mon = RBF.MonomialBasis(2, rbf_basis.poly_deg)
    Lrbf = RBF.∇²(rbf_basis)
    Lmon = RBF.∇²(mon)
    k_stencil = RBF.autoselect_k(all_coords, rbf_basis)
    adjl = RBF.find_neighbors(all_coords, k_stencil)
    A, rhs = RBF._build_weights(
        all_coords, all_normals, is_boundary, is_Neumann, adjl, rbf_basis, Lrbf, Lmon, mon)
    (; k, ρ, cₚ) = only(domain.models)
    α = k / (cₚ * ρ)
    A = A#α * A
    b, boundary_values = get_b(rhs, domain, is_boundary)
    b = source_term - b #α * source_term - α * b
    return A, b, boundary_values
end

function LinearProblemHermite(domain, source_term)
    A, b, boundary_values = make_system_Hermite(domain, source_term)
    return LinearSolve.LinearProblem(dropzeros(A), b), boundary_values
end

part, cloud, Δ = create_2d_geometry()

fig1 = visualize_surfaces(part)
save("surfaces.png", fig1)

h = 250 / 1e6
T∞ = 25 + 273.15
k = 40 / 1e3
ρ = 7833 / 1e9
cₚ = 0.465 * 1e3
α = k / (cₚ * ρ)

bcs_dict = Dict(
    :surface1 => :Dirichlet, :surface2 => :Neumann, :surface3 => :Dirichlet)
bcs = calculate_bcs(cloud, bcs_dict)
# println("Boundary conditions: ", bcs)
source_term = calculate_source_term(cloud)
# println("Source term: ", source_term)

domain = WeakDomain(cloud, bcs, [SolidEnergy(k = k, ρ = ρ, cₚ = cₚ)])
prob, boundary_values = LinearProblemHermite(domain, source_term)
@time sol = solve(prob)
T = sol.u
T_full = vcat(boundary_values, T)
figsize = (1500, 1500)
markersize = 0.0025
fig2 = viz_2d(domain, T_full; markersize = markersize, size = figsize, levels = 32)
save("temperatures.png", fig2)

# Compute and visualize error
internal_coords = [c for c in MM._coords(cloud.volume)]
exact_T_vec = [exact_solution(ustrip(c[1]), ustrip(c[2])) for c in internal_coords]
error_vec = abs.(T .- exact_T_vec)
print("error norm= $(norm(error_vec))")
