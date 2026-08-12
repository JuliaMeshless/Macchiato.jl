# Shared inputs for the KernelInterpolation.jl comparison benchmarks.
#
# Every node set is built ONCE with KernelInterpolation's own generators and converted
# coordinate-for-coordinate for our side, so both packages see bit-identical nodes.
# Problem definitions (exact solutions, sources, kernels) mirror KernelInterpolation's
# benchmark/benchmarks.jl at commit e36034c3f1e0bf1e9393c8e63740c058629c6aca.

import KernelInterpolation as KI
using StaticArrays: SVector
using WhatsThePoint
import WhatsThePoint as WTP
using Unitful: m
using Random

Random.seed!(1234)  # as in KI's suite (none of the retained benchmarks are random, but kept for fidelity)

to_svectors(nodeset::KI.NodeSet) = [SVector{KI.dim(nodeset), Float64}(x) for x in nodeset]

#-------------------------------------------------------------------------------# interpolation 1D
# KI: nodeset = NodeSet(LinRange(0.0, 2π, 8)); values = sin.(sum.(nodes))

f_interp(x) = sin(sum(x))

const interp1d_nodeset = KI.NodeSet(LinRange(0.0, 2π, 8))
const interp1d_values = f_interp.(interp1d_nodeset)
const interp1d_x = to_svectors(interp1d_nodeset)

#-------------------------------------------------------------------------------# Poisson 2D (RBF-FD)
# KI: −Δu = f on (0,1)², u = g = u_exact on the boundary
#   u_exact(x) = sin(πx₁)·cos(πx₂/2),  f(x) = 5/4·π²·sin(πx₁)·cos(πx₂/2)
# Nodes: 20×20 interior grid on [0.1,0.9]², 10-per-edge boundary grid on [0,1]² (36 nodes).

u_poisson(x) = sinpi(x[1]) * cospi(x[2] / 2)
f_poisson(x) = 5 / 4 * π^2 * sinpi(x[1]) * cospi(x[2] / 2)

const poisson_inner_ns = KI.homogeneous_hypercube(20, (0.1, 0.1), (0.9, 0.9); dim = 2)
const poisson_bdry_ns = KI.homogeneous_hypercube_boundary(10; dim = 2)
const poisson_merged_ns = merge(poisson_inner_ns, poisson_bdry_ns)

const poisson_inner = to_svectors(poisson_inner_ns)
const poisson_bdry = to_svectors(poisson_bdry_ns)
const poisson_all = vcat(poisson_inner, poisson_bdry)  # same ordering as KI's merge(inner, boundary)

#-------------------------------------------------------------------------------# Advection 2D (RBF-FD)
# KI: ∂u/∂t + a·∇u = 0 with a = (0.5, 0.5), inflow Dirichlet on the x=0 and y=0 edges,
#   u(t, x) = exp(−20·Σ(x − a·t − 0.3)²) as initial and boundary data.
# Nodes: 20×20 interior grid on [0.01,1.0]², boundary = union of 20 nodes each on the
# x=0 and y=0 edges (corner deduplicated → 39 nodes).

const adv_a = SVector(0.5, 0.5)

u_adv(t, x) = exp(-20 * sum((x .- adv_a .* t .- 0.3) .^ 2))

const adv_inner_ns = KI.homogeneous_hypercube(20, 0.01, 1.0; dim = 2)
const adv_bdry_ns = KI.NodeSet(
    union(
        [[0.0, y] for y in LinRange(0.0, 1.0, 20)],
        [[x, 0.0] for x in LinRange(0.0, 1.0, 20)],
    )
)

const adv_inner = to_svectors(adv_inner_ns)
const adv_bdry = to_svectors(adv_bdry_ns)
const adv_all = vcat(adv_inner, adv_bdry)  # same ordering as KI's merge(inner, boundary)

#-------------------------------------------------------------------------------# Macchiato point cloud
# Hand-assembled PointCloud carrying the exact same coordinates (no discretize call, so the
# node set is bit-identical to KI's). Normals are only consumed by Neumann/Robin BCs and the
# per-point areas only by shadow-point machinery — both inert here (pure Dirichlet), but the
# constructors require them.

function square_boundary_normal(p::SVector{2, Float64})
    dists = (p[1], 1 - p[1], p[2], 1 - p[2])  # distance to x=0, x=1, y=0, y=1
    normals = (
        WTP.Vec(-1.0, 0.0), WTP.Vec(1.0, 0.0),
        WTP.Vec(0.0, -1.0), WTP.Vec(0.0, 1.0),
    )
    return normals[argmin(dists)]
end

function build_square_cloud(
        interior::Vector{SVector{2, Float64}},
        bdry::Vector{SVector{2, Float64}},
        spacing::Float64,
    )
    bpts = [WTP.Point(p[1] * m, p[2] * m) for p in bdry]
    bnrm = [square_boundary_normal(p) for p in bdry]
    bareas = fill(spacing * m, length(bpts))
    boundary = PointBoundary(bpts, bnrm, bareas)  # single surface, auto-named :surface1
    volume = PointVolume([WTP.Point(p[1] * m, p[2] * m) for p in interior])
    return PointCloud(boundary, volume)
end

const poisson_cloud = build_square_cloud(poisson_inner, poisson_bdry, 1 / 9)
