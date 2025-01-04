using MeshlessMultiphysics
const MM = MeshlessMultiphysics
using RadialBasisFunctions
using PointClouds
using StaticArrays
using LinearAlgebra
using DifferentialEquations
using LinearSolve
using CUDA
using CUDA.CUSPARSE
import GLMakie
using Unitful: m, ustrip
println("using $(BLAS.get_num_threads()) CPU threads")

##

part = PointBoundary(joinpath(@__DIR__, "geometry/rectangle3d-04.stl"))
split_surface!(part, 75)
combine_surfaces!(part, :surface3, :surface4, :surface5, :surface6)

figsize = (1500, 750)
markersize = 0.015
#visualize(part; markersize = markersize, size = figsize)

Δ = 0.04m
#Δ *= 3
cloud = PointClouds.discretize(part, ConstantSpacing(Δ), alg = VanDerSandeFornberg())
#cloud = load(joinpath(@__DIR__, "rectangle-0.04.jld2"), "cloud")
repel!(cloud, ConstantSpacing(Δ); α = Δ / 100, max_iters = 1e2)
visualize(cloud; markersize = 0.7markersize, size = figsize)

##

# physics models and boundary conditions
h = 250 / 1e6 # W / (mm^2 K)
T∞ = 25 + 273.15 # K
k = 40 / 1e3 # W / (mm K)
ρ = 7833 / 1e9  # kg/mm^3
cₚ = 0.465 * 1e3 # J / (kg K)
α = k / (cₚ * ρ) # mm^2 / s

bcs = Dict(:surface1 => Adiabatic(ShadowPoints(ConstantSpacing(Δ / 5), 1)),
    :surface2 => Adiabatic(ShadowPoints(ConstantSpacing(Δ / 5), 1)),
    :surface3 => Temperature(50))
domain = Domain(cloud, bcs, SolidEnergy(k = k, ρ = ρ, cₚ = cₚ))

start = maximum(domain.boundaries) do b
    b[2][1][end]
end + 1
vol_ids = start:(start + length(domain.cloud.volume) - 1)

u0 = zeros(length(domain.cloud))
u0[vol_ids] .= 30
visualize(cloud.volume.points.geoms, u0[vol_ids], markersize = markersize, size = figsize)

# iterative solve using DiffEquations.jl
prob = MM.MultiphysicsProblem(domain, u0, (0.0, 2e-4))
dt = 0.001 * (ustrip(Δ))^2 / α
@time sol = solve(prob, Euler(), dt = dt, save_everystep = false, save_end = true)
T = Vector(sol.u[end])
u0 = copy(T)

half = findall(x -> x[2] > 0m, MM._coords(cloud.volume.points.geoms))
visualize(
    cloud.volume.points.geoms[half], T[Ref(start - 1) .+ half], markersize = markersize, size = figsize)

##

# direct solve using LinearSolve.jl
prob = MM.Solvers.LinearProblem(domain)
linsolve = init(prob)
@time sol = solve!(linsolve)
T = sol.u

half = findall(x -> x[2] > 0, _coords.(cloud.points))
visualize(
    _coords.(cloud.points[half]), T[half], markersize = markersize, size = figsize)
