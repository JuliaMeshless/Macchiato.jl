using MeshlessMultiphysics
const MM = MeshlessMultiphysics
const BC = MM.BoundaryConditions
using RadialBasisFunctions
using PointClouds
using StaticArrays
using LinearAlgebra
using FileIO
using DifferentialEquations
using LinearSolve
using CUDA
using CUDA.CUSPARSE
using SparseArrays
import GLMakie
println("using $(BLAS.get_num_threads()) CPU threads")

##

part = PointPart(joinpath(@__DIR__, "geometry/rectangle3d-04.stl")) # in mm
split_surface!(part, 75; views = true)
combine_surfaces!(part, :surface3, :surface4, :surface5, :surface6)

figsize = (2000, 1500)
markersize = 0.015
visualize(part; markersize = markersize, size = figsize)

Δ = 0.04
cloud = PointClouds.discretize(part, ConstantSpacing(Δ), alg = VanDerSandeFornberg())
cloud = load(joinpath(@__DIR__, "rectangle-0.04.jld2"), "cloud")
noderepel!(cloud, ConstantSpacing(Δ); α = Δ / 100, max_iters = 1e3)
visualize(cloud; markersize = markersize, size = figsize)

##

# physics models and boundary conditions
h = 250 / 1e6 # W / (mm^2 K)
T∞ = 25 + 273.15 # K
k = 40 / 1e3 # W / (mm K)
ρ = 7833 / 1e9  # kg/mm^3
cₚ = 0.465 * 1e3 # J / (kg K)
α = k / (cₚ * ρ) # mm^2 / s

#bcs = Dict(:surface1 => Adiabatic(Δ / 4), :surface2 => Temperature(50))
bcs = Dict(:surface1 => Adiabatic(Shadow(ConstantSpacing(Δ / 5), 1)),
    :surface2 => Adiabatic(Shadow(ConstantSpacing(Δ / 5), 1)),
    :surface3 => Temperature(50))
domain = Domain(cloud, bcs, SolidEnergy(k = k, ρ = ρ, cₚ = cₚ))

u0 = fill(25.0, length(domain.cloud))
vol_ids = only(domain.cloud.volume.points.indices)
u0[vol_ids] .= 0.0

# iterative solve using DiffEquations.jl
#prob = MM.Solvers.MultiphysicsProblem(domain, u0, (0.0, 1e-7))
#dt = 0.001 * (Δ)^2 / α
#@time sol = solve(prob, Euler(), dt = dt, save_everystep = false, save_end = true)
#T = Vector(sol.u[end])

# direct solve using LinearSolve.jl
prob = MM.Solvers.LinearProblem(domain)
linsolve = init(prob)
@time sol = solve!(linsolve)
T = sol.u

half = findall(x -> x[2] > 0, coordinates.(cloud.points))
visualize(
    coordinates.(cloud.points[half]), T[half], markersize = markersize, size = figsize)
