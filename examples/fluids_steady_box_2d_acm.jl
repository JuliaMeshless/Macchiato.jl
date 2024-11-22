using MeshlessMultiphysics
const MM = MeshlessMultiphysics
using MeshlessMultiphysics.Models
using MeshlessMultiphysics.BoundaryConditions

using RadialBasisFunctions
const RBF = RadialBasisFunctions
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
using WriteVTK
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

# find the shadow point user_spacing
rx = (xbounds[1] + Δ[1]):Δ[1]:(xbounds[2] - Δ[1])
ry = (ybounds[1] + Δ[2]):Δ[2]:(ybounds[2] - Δ[2])

sx = Δ[1] / 4

p_bot = map(i -> Point(i, ybounds[1]), rx)
p_right = map(i -> Point(xbounds[2], i), ry)
p_top = map(i -> Point(i, ybounds[2]), reverse(rx))
p_left = map(i -> Point(xbounds[1], i), reverse(ry))

n_bot = map(i -> Vec(0.0, -1.0), rx)
n_right = map(i -> Vec(1.0, 0.0), ry)
n_top = map(i -> Vec(0.0, 1.0), rx)
n_left = map(i -> Vec(-1.0, 0.0), ry)

p = vcat(p_bot, p_right, p_top, p_left)
n = vcat(n_bot, n_right, n_top, n_left)
a = ones(length(p))

part = PointPart(p, n, a)
split_surface!(part, 85; views = true)
combine_surfaces!(part, :surface1, :surface3)

structured = false
if structured
    points = [Point(x, y) for x in rx for y in ry]
    cloud = PointCloud(part)
    N = length(cloud.points)
    append!(cloud.points, points)
    cloud.volume = PointVolume(view(cloud.points, (N + 1):length(cloud.points)))
else
    cloud = PointClouds.discretize(part, ConstantSpacing(Δ[1]); alg = FornbergFlyer())
    noderepel!(cloud, ConstantSpacing(Δ[1]); α = Δ[1] / 10, max_iters = 5e3)
end

figsize = (2000, 400)
markersize = 0.00015
visualize(cloud; markersize = markersize, size = figsize)

##

# physics models and boundary conditions
ρ = 1060.0  # kg/m^3
μ = 0.004  # Pa*s
c = 1
V_in = 1.0
println("Re = $(ρ * V_in * L[2] / μ)")

bcs = Dict(
    :surface4 => VelocityInlet(V_in),
    :surface2 => PressureOutlet(100 * 133.32),
    :surface1 => Wall())
domain = Domain(cloud, bcs, IncompressibleNavierStokes(μ = μ, ρ = ρ))

struct IncompressibleFluid{U, V, P}
    u::U
    v::V
    p::P
end

function Base.similar(u::IncompressibleFluid)
    return IncompressibleFluid(similar(u.u), similar(u.v), similar(u.p))
end
function Base.copyto!(dest::IncompressibleFluid, src::IncompressibleFluid)
    copyto!(dest.u, src.u)
    copyto!(dest.v, src.v)
    copyto!(dest.p, src.p)
    return nothing
end
function Base.:-(a::IncompressibleFluid, b::IncompressibleFluid)
    return IncompressibleFluid(a.u .- b.u, a.v .- b.v, a.p .- b.p)
end
function Base.:+(a::IncompressibleFluid, b::IncompressibleFluid)
    return IncompressibleFluid(a.u .+ b.u, a.v .+ b.v, a.p .+ b.p)
end
function LinearAlgebra.norm(u::IncompressibleFluid, type::Real)
    return IncompressibleFluid(norm(u.u, type), norm(u.v, type), norm(u.p, type))
end
LinearAlgebra.norm(u::IncompressibleFluid) = LinearAlgebra.norm(u, 2)
mag(u::IncompressibleFluid) = sqrt.(u.u .^ 2 .+ u.v .^ 2)

function update_velocity!(u_star, u, domain, dt, ∇², shadow, ∂)
    (; μ, ρ) = only(domain.models)

    # calculate intermediate velocity
    u_star.u .= u.u .+ dt * (μ / ρ * ∇²(u.u) .- advect_x(u.u, u.v) .- ∂[1](u.p) / ρ)
    u_star.v .= u.v .+ dt * (μ / ρ * ∇²(u.v) .- advect_y(u.u, u.v) .- ∂[2](u.p) / ρ)

    # apply no-slip boundary condition to wall
    i = only(domain.cloud[:surface1].points.indices)
    u_star.u[i] .= zero(eltype(u_star.u))
    u_star.v[i] .= zero(eltype(u_star.v))

    # apply velocity inlet boundary condition
    i = only(domain.cloud[:surface4].points.indices)
    n = normals(domain.cloud[:surface4])
    u_star.u[i] .= -V_in * getindex.(n, 1)
    u_star.v[i] .= 0

    # apply pressure outlet boundary condition
    i = only(domain.cloud[:surface2].points.indices)
    u_star.u[i] .= shadow(u.u)
    u_star.v[i] .= shadow(u.v)

    return nothing
end

function update_pressure!(u, domain, dt, ∂, c)
    (; μ, ρ) = only(domain.models)

    u.p .-= (c * c * dt * ρ) .* (∂[1](u.u) .+ ∂[2](u.v))

    # pressure outlet
    i = only(domain.cloud[:surface2].points.indices)
    u.p[i] .= 1

    return nothing
end

function mystep!(u, u_old, domain, dt, ∇², ∂, shadow, c)
    # calculate intermediate velocity
    update_velocity!(u, u_old, domain, dt, ∇², shadow, ∂)

    # calculate pressure correction
    update_pressure!(u, domain, dt, ∂, c)

    return nothing
end

##

N = length(cloud)
dt = 5e-5
basis = PHS(3; poly_deg = 2)
k = RadialBasisFunctions.autoselect_k(coordinates(domain.cloud), basis)
k = 20
t_end = dt * 10

#function run!(u, domain, dt, t_end; k = 40)
vol_ids = only(domain.cloud.volume.points.indices)
vol = coordinates(domain.cloud.volume)
all_points = coordinates(domain.cloud)

# make 2nd partial ops for diffusion term
∇² = laplacian(all_points, basis; k = k)

# make upwinding term
#d = ntuple(dim -> upwind(all_points, dim, dx,basis; k = k), 2)
u∇u = ntuple(dim -> upwind(all_points, dim, basis; Δ = sx, k = k), 2)

Rex = u -> (ρ * Δ[1] * u) / μ
Rey = u -> (ρ * Δ[2] * u) / μ
θu(u) = map(u -> abs(Rex(u)) < 1 ? 0 : 1, u)
θv(v) = map(u -> abs(Rey(u)) < 1 ? 0 : 1, v)
advect_x(u, v) = u∇u[1](u, u, θu(u)) .+ u∇u[2](u, v, θv(v))
advect_y(u, v) = u∇u[1](v, u, θu(u)) .+ u∇u[2](v, v, θv(v))
advect_x(u, v, θ) = u∇u[1](u, u, θ) .+ u∇u[2](u, v, θ)
advect_y(u, v, θ) = u∇u[1](v, u, θ) .+ u∇u[2](v, v, θ)

∂ = ntuple(dim -> partial(all_points, 1, dim, basis; k = k), 2)

# for the outlets, apply nonreflective conditions
s = domain.cloud[:surface2]
shadow_points = generate_shadows(s, Shadow(ConstantSpacing(sx), 1))
shadow = regrid(all_points, shadow_points, basis; k = k)
update_weights!(shadow)

##

t = 0.0
u = IncompressibleFluid(fill(0.0, N), fill(0.0, N), fill(0.0, N))
u_old = similar(u)
copyto!(u_old, u)

##

for _ in 1:1000
    copyto!(u_old, u)
    mystep!(u, u_old, domain, dt, ∇², ∂, shadow, c)
end
visualize(coordinates(cloud), mag(u), markersize = markersize, size = figsize)
visualize(coordinates(cloud), u.p, markersize = markersize, size = figsize)

##

include(joinpath(@__DIR__, "../src/io.jl"))
plot_points = reduce(hcat, all_points)
cells = createvtkcells(plot_points, true)
vtkfile = createvtkfile(
    joinpath(@__DIR__, "../examples/parallel-plates-acm"), plot_points, cells)
addfieldvtk!(vtkfile, "u", u.u)
addfieldvtk!(vtkfile, "v", u.v)
addfieldvtk!(vtkfile, "vmag", mag(u))
addfieldvtk!(vtkfile, "p", u.p)
addfieldvtk!(vtkfile, "advect_x", advect_x(u_old.u, u_old.v))
addfieldvtk!(vtkfile, "advect_y", advect_y(u_old.u, u_old.v))
addfieldvtk!(vtkfile, "u_old", u_old.u)
addfieldvtk!(vtkfile, "v_old", u_old.v)
savevtk!(vtkfile)

##

visualize(coordinates(cloud), mag(u), markersize = markersize, size = figsize)
visualize(coordinates(cloud), mag(u_star), markersize = markersize, size = figsize)
visualize(coordinates(cloud), mag(u_old), markersize = markersize, size = figsize)
visualize(coordinates(cloud), u.u, markersize = markersize, size = figsize)
visualize(coordinates(cloud), u.v, markersize = markersize, size = figsize)
visualize(coordinates(cloud), u.p ./ 133.32, markersize = markersize, size = figsize)

visualize(coordinates(cloud), u_old.v, markersize = markersize, size = figsize)
visualize(all_points, advect_x(u_old.u, u_old.v), markersize = markersize, size = figsize)
visualize(all_points, advect_y(u_old.u, u_old.v), markersize = markersize, size = figsize)

visualize(vol, ∂[2](ϕ), markersize = markersize, size = figsize)
visualize(all_points, ϕ, markersize = markersize, size = figsize)
visualize(all_points, b, markersize = markersize, size = figsize)

##

print("\nTesting derivative operators... ")

func(x) = 1 + sin(4 * x[1]) + cos(3 * x[1]) + sin(2 * x[2])
func_dx(x) = 4 * cos(4 * x[1]) - 3 * sin(3 * x[1])
func_dy(x) = 2 * cos(2 * x[2])
func_dxx(x) = -16 * sin(4 * x[1]) - 9 * cos(3 * x[1])
func_dyy(x) = -4 * sin(2 * x[2])
func_lap(x) = func_dxx(x) + func_dyy(x)

test_vals = func.(all_points)

function calculateError(test, correct)
    sqrt(sum((test - correct) .^ 2) / sum(correct .^ 2))
end

basis = PHS(3; poly_deg = 2)
k = RadialBasisFunctions.autoselect_k(coordinates(domain.cloud), basis)

println("Boundary Nodes")
boundary = mapreduce(coordinates, vcat, surfaces(cloud))
∂ = ntuple(dim -> partial(all_points, boundary, 1, dim, basis; k = k), 2)
∇² = laplacian(all_points, boundary, basis; k = k)
test_dx = ∂[1](test_vals)
test_dy = ∂[2](test_vals)
test_lap = ∇²(test_vals)
correct_dx = func_dx.(boundary)
correct_dy = func_dy.(boundary)
correct_lap = func_lap.(boundary)
println(" dx = $(calculateError(test_dx, correct_dx))")
println(" dy = $(calculateError(test_dy, correct_dy))")
println("lap = $(calculateError(test_lap, correct_lap))")

println("Interior Nodes")
∂ = ntuple(dim -> partial(all_points, vol, 1, dim, basis; k = k), 2)
∂ = ntuple(dim -> ∂virtual(all_points, vol, dim, basis; Δ = Δ[1] / 5, k = k), 2)
∇² = laplacian(all_points, vol, basis; k = k)
test_dx = ∂[1](test_vals)
test_dy = ∂[2](test_vals)
test_lap = ∇²(test_vals)
correct_dx = func_dx.(vol)
correct_dy = func_dy.(vol)
correct_lap = func_lap.(vol)
println(" dx = $(calculateError(test_dx, correct_dx))")
println(" dy = $(calculateError(test_dy, correct_dy))")
println("lap = $(calculateError(test_lap, correct_lap))")
