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
cloud = PointClouds.discretize(part, ConstantSpacing(Δ), alg = VanDerSandeFornberg())
#cloud = load(joinpath(@__DIR__, "rectangle-0.04.jld2"), "cloud")
visualize(cloud; markersize = 0.02)

##

# physics models and boundary conditions
ρ = 1060  # kg/m^3
μ = 0.004  # Pa*s
V_in = 0.01
println("Re = $(ρ * V_in * 1 / μ)")

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

bcs = Dict(
    :surface1 => VelocityInlet(V_in),
    :surface2 => PressureOutlet(100 * 133.32),
    :surface3 => Wall())
domain = Domain(cloud, bcs, IncompressibleNavierStokes(μ = μ, ρ = ρ))

∇² = laplacian(all_points, basis; k = k)
∂ = ntuple(dim -> partial(all_points, vol, 1, dim; k = k), 3)
A, b = _make_pressure_correction_system(domain.cloud, dt, sx / 5; k = k)
F = factorize(A)
ϕ = similar(b)
u = IncompressibleFluid(fill(0.0, N), fill(0.0, N), fill(0.0, N), fill(0.0, N))
i = only(domain.cloud[:surface1].points.indices)
u.u[i] .= V_in
_update_pressure_correction_rhs!(b, u, domain, ∂, dt)

##

ldiv!(ϕ, F, b)

##

function viz(x, i = half)
    visualize(coordinates.(cloud.points[i]), x[i],
        markersize = markersize, size = figsize)
end

half = findall(x -> x[2] > 0.2, coordinates.(cloud.points))

viz(ϕ, vol_ids)
