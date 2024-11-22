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
using Makie
using Meshes
println("using $(BLAS.get_num_threads()) CPU threads")

##

h = 0.01 / 2
xbounds = (0.0, 5 * 2h)
ybounds = (-h, h)
L = (xbounds[2] - xbounds[1], ybounds[2] - ybounds[1])
base_spacing = h / 15
N = Int.(round.(L ./ base_spacing))
Δ = L ./ N
if !isapprox(Δ[1], Δ[2])
    error("Δx and Δy not the same!")
end
spacing = ConstantSpacing(Δ[1])
sx = Δ[1]

# find the shadow point user_spacing
rx = (xbounds[1] + Δ[1]):Δ[1]:(xbounds[2] - Δ[1])
ry = (ybounds[1] + Δ[2]):Δ[2]:(ybounds[2] - Δ[2])

p_bot = map(i -> Meshes.Point(i, ybounds[1]), rx)
p_right = map(i -> Meshes.Point(xbounds[2], i), ry)
p_top = map(i -> Meshes.Point(i, ybounds[2]), reverse(rx))
p_left = map(i -> Meshes.Point(xbounds[1], i), reverse(ry))

n_bot = map(i -> Meshes.Vec(0.0, -1.0), rx)
n_right = map(i -> Meshes.Vec(1.0, 0.0), ry)
n_top = map(i -> Meshes.Vec(0.0, 1.0), rx)
n_left = map(i -> Meshes.Vec(-1.0, 0.0), ry)

p = vcat(p_bot, p_right, p_top, p_left)
n = vcat(n_bot, n_right, n_top, n_left)
a = ones(length(p))

part = PointPart(p, n, a)
split_surface!(part, 85; views = true)
combine_surfaces!(part, :surface1, :surface3)

structured = false
if structured
    points = [Meshes.Point(x, y) for x in rx for y in ry]
    cloud = PointCloud(part)
    N = length(cloud.points)
    append!(cloud.points, points)
    cloud.volume = PointVolume(view(cloud.points, (N + 1):length(cloud.points)))
else
    cloud = PointClouds.discretize(part, spacing; alg = FornbergFlyer())
end

figsize = (2000, 400)
markersize = 0.00015
noderepel!(cloud, spacing; α = sx / 2, max_iters = 1e2)
visualize(cloud; markersize = markersize, size = figsize)

##

function viz_stencil(cone_adjl, i; kwargs...)
    fig = Figure(; size = figsize)
    ax = Axis(fig[1, 1]; aspect = DataAspect())
    x = getindex.(coords, 1)
    y = getindex.(coords, 2)
    m = meshscatter!(
        ax, x, y; labels = zeros(length(x)), shading = Makie.NoShading, kwargs...)
    stencil = coords[cone_adjl[i]]
    x = getindex.(stencil, 1)
    y = getindex.(stencil, 2)
    m = meshscatter!(
        ax, x, y; labels = ones(length(x)), shading = Makie.NoShading, kwargs...)
    m = meshscatter!(
        ax, x, y; labels = ones(length(x)), shading = Makie.NoShading, kwargs...)
    m = meshscatter!(
        ax, stencil[1][1], stencil[1][2]; labels = ones(length(x)),
        shading = Makie.NoShading, kwargs...)
    return fig
end

viz_stencil(cone_adjl, 70; markersize = markersize)

##

# physics models and boundary conditions
h = 250 / 1e6 # W / (mm^2 K)
T∞ = 25 + 273.15 # K
k = 40 / 1e3 # W / (mm K)
ρ = 7833 / 1e9  # kg/mm^3
cₚ = 0.465 * 1e3 # J / (kg K)
α = k / (cₚ * ρ) # mm^2 / s

#bcs = Dict(:surface1 => Adiabatic(Δ / 4), :surface2 => Temperature(50))
bcs = Dict(
    :surface1 => Temperature(30), :surface2 => Adiabatic(), :surface4 => Temperature(50))
domain = MM.Domain(cloud, bcs, SolidEnergy(k = k, ρ = ρ, cₚ = cₚ))

u0 = fill(25.0, length(domain.cloud))# + length(domain.cloud[:surface1]))
vol_ids = only(domain.cloud.volume.points.indices)
u0[vol_ids] .= 0.0

# iterative solve using DiffEquations.jl
#prob = MM.Solvers.MultiphysicsProblem(domain, u0, (0.0, 1e-7))
#dt = 0.001 * (Δ)^2 / α
#@time sol = solve(prob, Euler(), dt = dt, save_everystep = false, save_end = true)
#T = Meshes.Vector(sol.u[end])

# direct solve using LinearSolve.jl
prob = MM.Solvers.LinearProblem(domain)
linsolve = init(prob)
@time sol = solve!(linsolve)
T = sol.u

half = findall(x -> x[2] > 0.02, coordinates.(cloud.points))

visualize(
    coordinates.(cloud.points[half]), T[half], markersize = markersize, size = figsize)

visualize(
    coordinates.(cloud.points[vol_ids]), T[vol_ids], markersize = markersize, size = figsize)
