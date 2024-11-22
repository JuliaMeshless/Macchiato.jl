using MeshlessMultiphysics
const MM = MeshlessMultiphysics
const BC = MM.BoundaryConditions
using RadialBasisFunctions
using PointClouds
using StaticArrays
using LinearAlgebra
using FileIO
using LoopVectorization
using DifferentialEquations
using LinearSolve
using CUDA
using CUDA.CUSPARSE
using SparseArrays
using IncompleteLU
import GLMakie
using WriteVTK
println("using $(BLAS.get_num_threads()) CPU threads")

##

function viz_half(cloud, T; kwargs...)
    points = coordinates.(cloud.points)
    half = findall(x -> x[2] > 0.02, points)
    visualize(points[half], T[half]; kwargs...)
end

part = PointPart(joinpath(@__DIR__, "geometry/bifurcation-0.0005.stl"); views = true) # in mm
split_surface!(part, 75; views = true)

markersize = 0.0002
figsize = (2000, 1500)
visualize(part; markersize = markersize, size = figsize)

#cloud = PointClouds.discretize(part, ConstantSpacing(Δ), alg = VanDerSandeFornberg())
cloud = load(joinpath(@__DIR__, "geometry/bifurcation-0.0005-4-surfs.jld2"), "cloud")
Δ = 0.0005

#conv = noderepel!(cloud, ConstantSpacing(Δ); α = Δ / 50, max_iters = 1_000)
#f = lineplot(conv, title = "Node Repel Convergence Normalized by Spacing Function",
#    xlabel = "Iteration", ylabel = "Normalized Distance")
#display(f)

#vol = vfilter(p -> isinside(p, cloud), cloud.volume.points)
#cloud = PointCloud(part)
#N = length(cloud.points)
#append!(cloud.points, vol)
#cloud.volume = PointVolume(view(cloud.points, (N + 1):length(cloud.points)))

visualize(cloud; markersize = markersize, size = figsize)

#PointClouds.export_cloud("examples/bifurcation", cloud)

##

# physics models and boundary conditions
h = 250 / 1e6 # W / (mm^2 K)
T∞ = 25 + 273.15 # K
k = 40 / 1e3 # W / (mm K)
ρ = 7833 / 1e9  # kg/mm^3
cₚ = 0.465 * 1e3 # J / (kg K)
α = k / (cₚ * ρ) # mm^2 / s

bcs = Dict(
    :surface1 => Adiabatic(Shadow(ConstantSpacing(Δ), 2)),
    :surface2 => Temperature(2),
    :surface3 => Temperature(0),
    :surface4 => Temperature(1))
domain = MM.Domain(cloud, bcs, SolidEnergy(k = k, ρ = ρ, cₚ = cₚ))

half = findall(x -> x[2] > 0.02, coordinates.(cloud.points))
all_points = coordinates(cloud)

##
# direct solve

#prob = MM.Solvers.LinearProblem(domain)
#linsolve = init(prob)
#@time sol = solve!(linsolve)
#T = sol.u

A, b = MM.Solvers.LinearProblem(domain; k = 40)

#P = ilu(A, τ = 0.1)
#prob = LinearProblem(A, b_new)
#sol = solve(prob, KrylovJL_GMRES(), Pl = P)
#T = sol.u

F = factorize(A)
T = F \ b

visualize(all_points, T, markersize = markersize, size = figsize)
visualize(all_points[half], T[half], markersize = markersize, size = figsize)

∂x = partial(all_points, 1, 1; k = 40)
∂y = partial(all_points, 1, 2; k = 40)
∂z = partial(all_points, 1, 3; k = 40)

visualize(all_points, ∂x(T), markersize = markersize, size = figsize)
visualize(all_points, ∂y(T), markersize = markersize, size = figsize)
visualize(all_points, ∂z(T), markersize = markersize, size = figsize)

##
# iterative solve using DiffEquations.jl
u0 = fill(40.0, length(domain.cloud))
vol_ids = only(domain.cloud.volume.points.indices)
#u0[vol_ids] .= 0.0

T = copy(u0)
prob = MM.Solvers.MultiphysicsProblem(domain, T, (0.0, 5e-7); k = 40)
dt = 0.001 * (Δ)^2 / α
ode_kwargs = (save_everystep = false, save_end = true)
@time sol = solve(prob, Euler(), dt = dt, save_everystep = false, save_end = true)
T = Vector(sol.u[end])

visualize(vol[half], T[half], markersize = markersize, size = figsize)
visualize(vol, T, markersize = markersize, size = figsize)
