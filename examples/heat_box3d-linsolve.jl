using MeshlessMultiphysics
const MM = MeshlessMultiphysics
using RadialBasisFunctions
using WhatsThePoint
using StaticArrays
using LinearAlgebra
using CUDA
using CUDA.CUSPARSE
import GLMakie
using GLMakie
using UnicodePlots
using Unitful: m, ustrip
using JLD2
println("using $(BLAS.get_num_threads()) CPU threads")

##

part = PointBoundary(joinpath(@__DIR__, "geometry/rectangle3d-04.stl"))
split_surface!(part, 75)
combine_surfaces!(part, :surface3, :surface4, :surface5, :surface6)

figsize = (1500, 750)
markersize = 0.015
#visualize(part; markersize = markersize, size = figsize)

Δ = 0.04m
#cloud = load(joinpath(@__DIR__, "rectangle-0.04.jld2"), "cloud")
cloud = WhatsThePoint.discretize(part, ConstantSpacing(Δ), alg = VanDerSandeFornberg())
cloud, conv = repel(cloud, ConstantSpacing(Δ); α = Δ / 100, max_iters = 1e2)
display(lineplot(conv))
visualize(cloud; markersize = 0.7markersize, size = figsize)
#save(joinpath(@__DIR__, "rectangle-0.04.jld2"), Dict("cloud"=>cloud))

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

bcs = Dict(:surface1 => Temperature(50),
    :surface2 => Temperature(50),
    :surface3 => Temperature(0))

bcs = Dict(:surface1 => Temperature(0),
    :surface2 => Temperature(0),
    :surface3 => Temperature(50))

domain = Domain(cloud, bcs, SolidEnergy(k = k, ρ = ρ, cₚ = cₚ))


function viz(
        domain,
        labels,
        f_filter = x -> true;
        size = (1000, 1000),
        colorrange = WhatsThePoint._get_colorrange(labels),
        azimuth = 1.275π,
        elevation = π / 8,
        colormap = :Spectral,
        levels = 32,
        kwargs...
)
    fig = Figure(; size = size)
    ax = Axis3(fig[1, 1]; azimuth = azimuth, elevation = elevation)
    ax.aspect = :data

    cmap = Makie.cgrad(colormap, levels; categorical = true)

    for b in domain.boundaries
        ids = b.second[1]
        pts = points(domain.cloud[b.first])
        c = coords.(pts)
        filtered_ids = findall(f_filter, c)
        c = c[filtered_ids]
        x = map(c -> ustrip(c.x), c)
        y = map(c -> ustrip(c.y), c)
        z = map(c -> ustrip(c.z), c)
        meshscatter!(
            ax,
            ustrip.(x),
            ustrip.(y),
            ustrip.(z);
            color = labels[ids][filtered_ids],
            colorrange = colorrange,
            colormap = cmap,
            kwargs...
        )
        Makie.Colorbar(fig[1, 2]; colorrange = colorrange, colormap = cmap)
    end

    # volume
    start = maximum(domain.boundaries) do b
        b[2][1][end]
    end + 1
    ids = start:(start + length(domain.cloud.volume) - 1)
    c = coords.(points(domain.cloud.volume))
    filtered_ids = findall(f_filter, c)
    c = c[filtered_ids]
    x = map(c -> ustrip(c.x), c)
    y = map(c -> ustrip(c.y), c)
    z = map(c -> ustrip(c.z), c)
    meshscatter!(
        ax,
        ustrip.(x),
        ustrip.(y),
        ustrip.(z);
        color = labels[ids][filtered_ids],
        colorrange = colorrange,
        colormap = cmap,
        kwargs...
    )
    Makie.Colorbar(fig[1, 2]; colorrange = colorrange, colormap = cmap)
    return fig
end

##

# iterative solve using Simulation API
dt = 0.001 * (ustrip(Δ))^2 / α
sim = Simulation(domain; Δt = dt, stop_time = 3e-4, solver = :Euler)
set!(sim, T = 0.0)
@time run!(sim)
T = temperature(sim)

viz(domain, T, c -> c.y > 0m; markersize = markersize, size = figsize)

##

# direct solve using Simulation API
sim = Simulation(domain)
@time run!(sim)
T = temperature(sim)

viz(domain, T, c -> c.y > 0m; markersize = markersize, size = figsize)
