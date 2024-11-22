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
println("using $(BLAS.get_num_threads()) CPU threads")

##

part = PointPart(joinpath(@__DIR__, "geometry/rectangle3d-04.stl"); views = true) # in mm
split_surface!(part, 75; views = true)
combine_surfaces!(part, :surface3, :surface4, :surface5, :surface6)
visualize(part; markersize = 0.02)

Δ = 0.035
#cloud = PointClouds.discretize(part, ConstantSpacing(Δ), alg = VanDerSandeFornberg())
cloud = load(joinpath(@__DIR__, "rectangle-0.04.jld2"), "cloud")

figsize = (1500, 1000)
markersize = 0.02
visualize(cloud; markersize = markersize, size = figsize)

##

# physics models and boundary conditions
ρ = 1060  # kg/m^3
μ = 0.004  # Pa*s
V_in = 0.01
println("Re = $(ρ * V_in * 1 / μ)")

bcs = Dict(
    :surface1 => VelocityInlet(V_in),
    :surface2 => PressureOutlet(100 * 133.32),
    :surface3 => Wall())
domain = Domain(cloud, bcs, IncompressibleNavierStokes(μ = μ, ρ = ρ))

function _make_pressure_correction_system(cloud, dt, sx; kwargs...)
    bcs = Dict(
        :surface1 => Adiabatic(Shadow(ConstantSpacing(sx), 1)),
        :surface3 => Adiabatic(Shadow(ConstantSpacing(sx), 1)),
        :surface2 => Temperature(100 * 133.32))
    domain = Domain(cloud, bcs, SolidEnergy(k = 1, ρ = 1, cₚ = 1))
    A, b = make_system(only(domain.models), domain)
    for (surface_name, boundary_condition) in domain.boundaries
        make_bc!(A, b, boundary_condition, surface_name, domain; kwargs...)
    end
    return dropzeros(A), b
    return factorize(dropzeros(A)), b
end

function _update_pressure_correction_rhs!(b::AbstractVector{B}, u, domain, ∂, dt) where {B}
    (; μ, ρ) = only(domain.models)
    i = only(domain.cloud.volume.points.indices)
    b[i] .= ρ / dt * (∂[1](u.u) + ∂[2](u.v) + ∂[3](u.w))

    # apply no-slip boundary condition to wall
    i = only(domain.cloud[:surface3].points.indices)
    b[i] .= zero(B)

    # apply velocity inlet boundary condition
    i = only(domain.cloud[:surface1].points.indices)
    b[i] .= zero(B)

    # apply pressure outlet boundary condition
    i = only(domain.cloud[:surface2].points.indices)
    b[i] .= 100 * 133.32

    return nothing
end

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
mag(u::IncompressibleFluid) = sqrt.(u.u .^ 2 .+ u.v .^ 2 .+ u.w .^ 2)

function intermediate_velocity!(u_star, u, domain, dt, u∇u, ∇², shadow)
    (; μ, ρ) = only(domain.models)

    # calculate intermediate velocity
    u_star.u .= u.u .+ dt * (μ / ρ * ∇²(u.u) .- u∇u[1](u))
    u_star.v .= u.v .+ dt * (μ / ρ * ∇²(u.v) .- u∇u[2](u))
    u_star.w .= u.w .+ dt * (μ / ρ * ∇²(u.w) .- u∇u[3](u))

    # apply no-slip boundary condition to wall
    i = only(domain.cloud[:surface3].points.indices)
    u_star.u[i] .= zero(eltype(u_star.u))
    u_star.v[i] .= zero(eltype(u_star.v))
    u_star.w[i] .= zero(eltype(u_star.w))

    # apply velocity inlet boundary condition
    i = only(domain.cloud[:surface1].points.indices)
    n = normals(domain.cloud[:surface1])
    V = -V_in
    u_star.u[i] .= V * getindex.(n, 1)
    u_star.v[i] .= V * getindex.(n, 2)
    u_star.w[i] .= V * getindex.(n, 3)

    # apply pressure outlet boundary condition
    i = only(domain.cloud[:surface2].points.indices)
    u_star.u[i] .= shadow(u.u)
    u_star.v[i] .= shadow(u.v)
    u_star.w[i] .= shadow(u.w)

    return nothing
end

function correct_pressure_and_velocity!(u, u_star, ϕ, domain, dt, ∂)
    (; μ, ρ) = only(domain.models)

    u.u .= u_star.u
    u.v .= u_star.v
    u.w .= u_star.w

    # velocity inlet
    i = only(domain.cloud[:surface1].points.indices)
    u.p[i] .= ϕ[i]

    # wall
    i = only(domain.cloud[:surface3].points.indices)
    u.p[i] .= ϕ[i]

    # pressure outlet
    i = only(domain.cloud[:surface2].points.indices)
    u.p[i] .= 100 * 133.32

    i = only(domain.cloud.volume.points.indices)
    u.u[i] .= u_star.u[i] .- dt / ρ * ∂[1](ϕ)
    u.v[i] .= u_star.v[i] .- dt / ρ * ∂[2](ϕ)
    u.w[i] .= u_star.w[i] .- dt / ρ * ∂[3](ϕ)
    u.p[i] .= ϕ[i]
    return nothing
end

function mystep!(u, u_star, ϕ, domain, dt, u∇u, ∇², ∂, shadow, F, b)
    # calculate intermediate velocity
    intermediate_velocity!(u_star, u, domain, dt, u∇u, ∇², shadow)

    # calculate pressure correction
    _update_pressure_correction_rhs!(b, u_star, domain, ∂, dt)
    ldiv!(ϕ, F, b)

    # correct velocity and correct pressure
    correct_pressure_and_velocity!(u, u_star, ϕ, domain, dt, ∂)

    return nothing
end

##

N = length(cloud)
dt = 1e-4
basis = PHS(3; poly_deg = 2)
k = RadialBasisFunctions.autoselect_k(coordinates(domain.cloud), PHS(3; poly_deg = 2))
k = 40
t_end = dt * 10

vol_ids = only(domain.cloud.volume.points.indices)
vol = coordinates(domain.cloud.volume)
all_points = coordinates(domain.cloud)

∇² = laplacian(all_points, basis; k = k)

# make upwinding term
up = ntuple(dim -> upwind(all_points, dim, basis; Δ = sx, k = k), 3)

Re(u) = (ρ * Δ * u) / μ
θ(u) = map(v -> abs(Re(v)) < 1 ? 0 : 1, u)

u∇u = (
    x -> up[1](x.u, x.u, θ(x.u)) + up[2](x.u, x.v, θ(x.v)) + up[3](x.u, x.w, θ(x.w)),
    x -> up[1](x.v, x.u, θ(x.u)) + up[2](x.v, x.v, θ(x.v)) + up[3](x.v, x.w, θ(x.w)),
    x -> up[1](x.w, x.u, θ(x.u)) + up[2](x.w, x.v, θ(x.v)) + up[3](x.w, x.w, θ(x.w)))

# create pressure correction problem
∂ = ntuple(dim -> partial(all_points, vol, 1, dim; k = k), 3)
A, b = _make_pressure_correction_system(domain.cloud, dt, sx / 5; k = k)
F = factorize(A)
ϕ = similar(b)
# for the outlets, apply nonreflective conditions
s = domain.cloud[:surface2]
shadow_points = generate_shadows(s, Shadow(ConstantSpacing(sx), 1))
shadow = regrid(all_points, shadow_points, basis; k = k)
update_weights!(shadow)

##

t = 0.0
u = IncompressibleFluid(fill(0.0, N), fill(0.0, N), fill(0.0, N), fill(0.0, N))
u_old = similar(u)
u_star = similar(u)
copyto!(u_old, u)
copyto!(u_star, u)

##

for _ in 1:3
    copyto!(u_old, u)
    mystep!(u, u_star, ϕ, domain, dt, u∇u, ∇², ∂, shadow, F, b)
end

##

function viz(x, i = half)
    visualize(coordinates.(cloud.points[i]), x[i], markersize = markersize, size = figsize)
end

half = findall(x -> x[2] > 0.2, coordinates.(cloud.points))

viz(mag(u), vol_ids)
viz(ϕ)
