using MeshlessMultiphysics
import MeshlessMultiphysics as MM
using RadialBasisFunctions
using WhatsThePoint
import WhatsThePoint as WTP
using StaticArrays
using LinearAlgebra
using DifferentialEquations
using LinearSolve
using CUDA
using CUDA.CUSPARSE
using GLMakie
using UnicodePlots
using Unitful: m, °, ustrip
using JLD2

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
split_surface!(part, 75°)
combine_surfaces!(part, :surface3, :surface4)

figsize = (1500, 1500)
markersize = 0.0025
visualize(part; markersize = markersize, size = figsize)

##

Δ = dx
cloud = WhatsThePoint.discretize(part, ConstantSpacing(Δ), alg = VanDerSandeFornberg())

conv = repel!(cloud, ConstantSpacing(Δ); α = Δ / 20, max_iters = 500)
display(lineplot(conv))

visualize(cloud; markersize = markersize, size = figsize)
#save(joinpath(@__DIR__, "rectangle-0.04.jld2"), Dict("cloud"=>cloud))

##

# physics models and boundary conditions
h = 250 / 1e6 # W / (mm^2 K)
T∞ = 25 + 273.15 # K
k = 40 / 1e3 # W / (mm K)
ρ = 7833 / 1e9  # kg/mm^3
cₚ = 0.465 * 1e3 # J / (kg K)
α = k / (cₚ * ρ) # mm^2 / s

bcs = Dict(
    :surface1 => Temperature(10), :surface2 => Temperature(0), :surface3 => Temperature(5))

domain = MM.Domain(cloud, bcs, SolidEnergy(k = k, ρ = ρ, cₚ = cₚ))

start = maximum(domain.boundaries) do b
    b[2][1][end]
end + 1
vol_ids = start:(start + length(domain.cloud.volume) - 1)

u0 = zeros(length(domain.cloud))

##

# solve using LinearSolve.jl
prob = MM.LinearProblem(domain)
@time sol = solve(prob)
T = sol.u

##
# visualize the solution

# this needs to go into the WhatsThePoint package probably...
function viz(
        domain,
        labels;
        size = (1000, 1000),
        colorrange = WhatsThePoint._get_colorrange(labels),
        colormap = :Spectral,
        levels = 32,
        kwargs...
)
    fig = Figure(; size = size)
    ax = Axis(fig[1, 1]; aspect = DataAspect())

    cmap = Makie.cgrad(colormap, levels; categorical = true)

    for b in domain.boundaries
        ids = b.second[1]
        points = pointify(domain.cloud[b.first])
        c = coords.(points)
        x = map(c -> ustrip(c.x), c)
        y = map(c -> ustrip(c.y), c)
        meshscatter!(
            ax,
            ustrip.(x),
            ustrip.(y);
            color = labels[ids],
            shading = Makie.NoShading,
            colorrange = colorrange,
            colormap = cmap,
            kwargs...
        )
    end

    # volume
    start = maximum(domain.boundaries) do b
        b[2][1][end]
    end + 1
    ids = start:(start + length(domain.cloud.volume) - 1)
    c = coords.(domain.cloud.volume.points)
    x = map(c -> ustrip(c.x), c)
    y = map(c -> ustrip(c.y), c)
    meshscatter!(
        ax,
        ustrip.(x),
        ustrip.(y);
        color = labels[ids],
        shading = Makie.NoShading,
        colorrange = colorrange,
        colormap = cmap,
        kwargs...
    )
    Makie.Colorbar(fig[1, 2]; colorrange = colorrange, colormap = cmap)
    return fig
end

exportvtk("heat-equation-2d", pointify(cloud), [T], ["T"])
viz(domain, T; markersize = markersize, size = figsize, levels = 32)
