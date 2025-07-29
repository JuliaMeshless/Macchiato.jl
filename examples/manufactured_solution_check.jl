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
include("hermite_utils.jl") # for get_new_domain_info

part, cloud, Δ = create_2d_geometry()

fig1 = visualize_surfaces(part)
save("surfaces.png", fig1)

h = 250 / 1e6
T∞ = 25 + 273.15
k = 40 / 1e3
ρ = 7833 / 1e9
cₚ = 0.465 * 1e3
α = k / (cₚ * ρ)

# Unified boundary condition type dictionary for domain info and BC logic
bcs_type_dict = Dict(
    :surface1 => BoundaryCondition(:Dirichlet, (x, y) -> exact_solution(x, y)),
    :surface2 => BoundaryCondition(:Neumann, (x, y, nx, ny) -> exact_flux(x, y, nx, ny)),
    :surface3 => BoundaryCondition(:Dirichlet, (x, y) -> exact_solution(x, y))
)

bcs = make_bcs(bcs_type_dict)
source_term = calculate_source_term(cloud)

domain = MM.Domain(cloud, bcs, SolidEnergy(k = k, ρ = ρ, cₚ = cₚ))
prob, boundary_values = LinearProblemHermite(domain, bcs_type_dict, source_term)
@time sol = solve(prob)
T = sol.u
T_full = vcat(boundary_values, T)
figsize = (1500, 1500)
markersize = 0.0025
viz_2d(domain, T_full; markersize = markersize, size = figsize, levels = 32)

# Compute and visualize error
internal_coords = [c for c in MM._coords(cloud.volume)]
exact_T_vec = [exact_solution(ustrip(c[1]), ustrip(c[2])) for c in internal_coords]
error_vec = abs.(T .- exact_T_vec)
print("error norm= $(norm(error_vec))")
