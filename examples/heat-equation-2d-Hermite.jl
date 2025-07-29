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
include("hermite_utils.jl") # for get_new_domain_info

part, cloud, Δ = create_2d_geometry()

# Generalized boundary condition type dictionary for domain info and BC logic.
# You can set :Dirichlet or :Neumann for each surface, and provide a function value.
bcs_type_dict = Dict(
    :surface1 => BoundaryCondition(:Dirichlet, (x, y) -> 10.0),
    :surface2 => BoundaryCondition(:Neumann, (x, y, nx, ny) -> 0.0),
    :surface3 => BoundaryCondition(:Dirichlet, (x, y) -> 5.0)
)

bcs = make_bcs(bcs_type_dict)

# physics models and boundary conditions
h = 250 / 1e6 # W / (mm^2 K)
T∞ = 25 + 273.15 # K
k = 40 / 1e3 # W / (mm K)
ρ = 7833 / 1e9  # kg/mm^3
cₚ = 0.465 * 1e3 # J / (kg K)
α = k / (cₚ * ρ) # mm^2 / s

domain = MM.Domain(cloud, bcs, SolidEnergy(k = k, ρ = ρ, cₚ = cₚ))

prob, boundary_values = LinearProblemHermite(domain, bcs_type_dict)
@time sol = solve(prob)
T = sol.u
T_full = vcat(boundary_values, T)
figsize = (1500, 1500)
markersize = 0.0025
viz_2d(domain, T_full; markersize = markersize, size = figsize, levels = 32)
