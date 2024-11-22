using MeshlessMultiphysics
const MM = MeshlessMultiphysics
using MeshlessMultiphysics.Models
using MeshlessMultiphysics.BoundaryConditions

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
using GeoIO
using Meshes
using ProgressMeter
println("using $(BLAS.get_num_threads()) CPU threads")

##

mesh = GeoIO.load(joinpath(@__DIR__, "geometry/bifurcation-0.0005.stl")).geometry

#part = PointPart(joinpath(@__DIR__, "geometry/bifurcation-0.0005.stl"); views = true) # in mm
#split_surface!(part, 75; views = true)
#combine_surfaces!(part, :surface3, :surface4)
#visualize(part; markersize = markersize, size = figsize)

Δ = 0.0005
sx = Δ
#cloud = PointClouds.discretize(part, ConstantSpacing(Δ), alg = VanDerSandeFornberg())
cloud = load(joinpath(@__DIR__, "bifurcation-0.0005-3-surfs.jld2"), "cloud")

markersize = 0.0002
figsize = (2000, 1500)
visualize(cloud; markersize = markersize, size = figsize)
#visualize_normals(cloud; markersize = markersize, size = figsize)

PointClouds.export_cloud("examples/bifurcation", cloud)

##

# physics models and boundary conditions
ρ = 1060  # kg/m^3
μ = 0.004  # Pa*s
V_in = 0.01
c = V_in
println("Re = $(ρ * V_in * 1 / μ)")

bcs = Dict(
    :surface1 => Wall(),
    :surface2 => VelocityInlet(V_in),
    :surface3 => PressureOutlet(100 * 133.32))
domain = MM.Domain(cloud, bcs, IncompressibleNavierStokes(μ = μ, ρ = ρ))

struct IncompressibleFluid{U, V, W, P}
    u::U
    v::V
    w::W
    p::P
end

function Base.similar(u::IncompressibleFluid)
    return IncompressibleFluid(similar(u.u), similar(u.v), similar(u.w), similar(u.p))
end
function Base.copyto!(dest::IncompressibleFluid, src::IncompressibleFluid)
    copyto!(dest.u, src.u)
    copyto!(dest.v, src.v)
    copyto!(dest.w, src.w)
    copyto!(dest.p, src.p)
    return nothing
end
function Base.:-(a::IncompressibleFluid, b::IncompressibleFluid)
    return IncompressibleFluid(a.u .- b.u, a.v .- b.v, a.w .- b.w, a.p .- b.p)
end
function Base.:+(a::IncompressibleFluid, b::IncompressibleFluid)
    return IncompressibleFluid(a.u .+ b.u, a.v .+ b.v, a.w .+ b.w, a.p .+ b.p)
end
function LinearAlgebra.norm(u::IncompressibleFluid, type::Real)
    return IncompressibleFluid(
        norm(u.u, type), norm(u.v, type), norm(u.w, type), norm(u.p, type))
end
LinearAlgebra.norm(u::IncompressibleFluid) = LinearAlgebra.norm(u, 2)
mag(u::IncompressibleFluid) = sqrt.(u.u .^ 2 .+ u.v .^ 2 .+ u.w .^ 2)

function update_velocity!(u_star, u, domain, dt, ∇², shadow, ∂)
    (; μ, ρ) = only(domain.models)

    # calculate intermediate velocity
    u_star.u .= u.u .+ dt * (μ / ρ * ∇²(u.u) .- advect_x(u.u, u.v, u.w) .- ∂[1](u.p) / ρ)
    u_star.v .= u.v .+ dt * (μ / ρ * ∇²(u.v) .- advect_y(u.u, u.v, u.w) .- ∂[2](u.p) / ρ)
    u_star.w .= u.w .+ dt * (μ / ρ * ∇²(u.w) .- advect_z(u.u, u.v, u.w) .- ∂[3](u.p) / ρ)

    # apply no-slip boundary condition to wall
    i = only(domain.cloud[:surface1].points.indices)
    u_star.u[i] .= zero(eltype(u_star.u))
    u_star.v[i] .= zero(eltype(u_star.v))
    u_star.w[i] .= zero(eltype(u_star.w))

    # apply velocity inlet boundary condition
    i = only(domain.cloud[:surface2].points.indices)
    n = normals(domain.cloud[:surface2])
    u_star.u[i] .= -V_in * getindex.(n, 1)
    u_star.v[i] .= -V_in * getindex.(n, 2)
    u_star.w[i] .= -V_in * getindex.(n, 3)

    # apply pressure outlet boundary condition
    i = only(domain.cloud[:surface3].points.indices)
    u_star.u[i] .= shadow(u.u)
    u_star.v[i] .= shadow(u.v)
    u_star.w[i] .= shadow(u.w)

    return nothing
end

function update_pressure!(u, domain, dt, ∂, c)
    (; μ, ρ) = only(domain.models)

    u.p .-= (c * c * dt * ρ) .* (∂[1](u.u) .+ ∂[2](u.v) .+ ∂[3](u.w))

    # pressure outlet
    i = only(domain.cloud[:surface3].points.indices)
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
dt = 1e-4
basis = PHS(3; poly_deg = 2)
k = RadialBasisFunctions.autoselect_k(coordinates(domain.cloud), basis)
k = 40

vol_ids = only(domain.cloud.volume.points.indices)
vol = coordinates(domain.cloud.volume)
all_points = coordinates(domain.cloud)

# make 2nd partial ops for diffusion term
∇² = laplacian(all_points, basis; k = k)

# make upwinding term
u∇u = ntuple(dim -> upwind(all_points, dim, basis; Δ = sx, k = k), 3)

Rex = u -> (ρ * Δ * u) / μ
Rey = u -> (ρ * Δ * u) / μ
Rez = u -> (ρ * Δ * u) / μ
θu(u) = map(u -> abs(Rex(u)) < 1 ? 0 : 1, u)
θv(v) = map(u -> abs(Rey(u)) < 1 ? 0 : 1, v)
θw(w) = map(u -> abs(Rez(u)) < 1 ? 0 : 1, w)
advect_x(u, v, w) = u∇u[1](u, u, θu(u)) .+ u∇u[2](u, v, θv(v)) .+ u∇u[3](u, w, θw(v))
advect_y(u, v, w) = u∇u[1](v, u, θu(u)) .+ u∇u[2](v, v, θv(v)) .+ u∇u[3](v, w, θw(v))
advect_z(u, v, w) = u∇u[1](w, u, θu(u)) .+ u∇u[2](w, v, θv(v)) .+ u∇u[3](w, w, θw(v))
advect_x(u, v, w, θ) = u∇u[1](u, u, θ) .+ u∇u[2](u, v, θ) .+ u∇u[3](u, w, θ)
advect_y(u, v, w, θ) = u∇u[1](v, u, θ) .+ u∇u[2](v, v, θ) .+ u∇u[3](v, w, θ)
advect_z(u, v, w, θ) = u∇u[1](w, u, θ) .+ u∇u[2](w, v, θ) .+ u∇u[3](w, w, θ)

∂ = ntuple(dim -> partial(all_points, 1, dim, basis; k = k), 3)

# for the outlets, apply nonreflective conditions
s = domain.cloud[:surface3]
shadow_points = generate_shadows(s, Shadow(ConstantSpacing(sx), 1))
shadow = regrid(all_points, shadow_points, basis; k = k)
update_weights!(shadow)

##

function run_sim(u, u_old, domain, ∇², ∂, c, shadow, dt; tol = 1e-5, max_steps = 10_000)
    i = 0
    prog = ProgressUnknown(desc = "Working hard:", spinner = true)
    conv = IncompressibleFluid(Inf, Inf, Inf, Inf)
    while (conv.u > tol) || (conv.v > tol)
        next!(prog)
        copyto!(u_old, u)
        mystep!(u, u_old, domain, dt, ∇², ∂, shadow, c)
        if i > max_steps
            println()
            @warn "Maximum steps reached"
            break
        end
        i += 1
        conv = norm(u - u_old, Inf)
    end
    finish!(prog)
    println("Converged in $i steps with u = $(conv.u) and v = $(conv.v)")
    return nothing
end

function viz(x, i = :)
    visualize(coordinates.(cloud.points[i]), x[i], markersize = markersize, size = figsize)
end

half = findall(x -> x[2] > 0.02, coordinates.(cloud.points))

##

t = 0.0
u = IncompressibleFluid(fill(0.0, N), fill(0.0, N), fill(0.0, N), fill(0.0, N))
u_old = similar(u)
copyto!(u_old, u)

##

run_sim(u, u_old, domain, ∇², ∂, c, shadow, dt; max_steps = 10)
visualize(coordinates(cloud), mag(u), markersize = markersize, size = figsize)
#visualize(coordinates(cloud), u.p, markersize = markersize, size = figsize)

viz(mag(u), half)
viz(u.p, half)
