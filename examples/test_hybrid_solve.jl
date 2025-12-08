using MeshlessMultiphysics
using WhatsThePoint
import WhatsThePoint as WTP
using Unitful: m, °, ustrip
using Test
using DifferentialEquations
using LinearAlgebra

println("="^60)
println("Testing Hybrid Solve Integration")
println("="^60)

# 1. Setup Domain
L = (1m, 1m)
dx = 0.1m
S = ConstantSpacing(dx)

# Create boundary points for a square
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

p = vcat(p_bot, p_right, p_top, p_left)
n = vcat(n_bot, n_right, n_top, n_left)
a = fill(dx, length(p))

part = PointBoundary(p, n, a)
split_surface!(part, 75°)

cloud = WhatsThePoint.discretize(part, ConstantSpacing(dx), alg = VanDerSandeFornberg())
println("Cloud size: ", size(cloud))
println("Surface names: ", names(cloud.boundary))

# 2. Define BCs
# Surface 1 (Bottom): Dirichlet Time Varying T = 10t -> dT/dt = 10
bc_dirichlet = Temperature((x, t) -> 10.0 * t)

# Surface 2 (Right): Neumann Constant q = 0
bc_neumann = Adiabatic()

# Surface 3 (Top): Dirichlet Constant T = 100 -> dT/dt = 0
bc_dirichlet_const = Temperature(100.0)

# Surface 4 (Left): Neumann Constant q = 0
bc_neumann2 = Adiabatic()

boundaries = Dict(
    :surface1 => bc_dirichlet,
    :surface2 => bc_neumann,
    :surface3 => bc_dirichlet_const,
    :surface4 => bc_neumann2
)

model = SolidEnergy(k = 1.0, ρ = 1.0, cₚ = 1.0)
domain = Domain(cloud, boundaries, model)

# 3. Create Problem
prob = MultiphysicsProblem(domain, zeros(length(cloud)), (0.0, 1.0))

# 4. Test f function
du = zeros(length(cloud))
u = zeros(length(cloud))
p = nothing
t = 0.5

prob.f(du, u, p, t)

# Check Dirichlet Time Varying (Surface 1)
ids_1 = domain.boundaries[:surface1][1]
println("Checking Surface 1 (Dirichlet Time Varying)...")
println("  Expected: 10.0")
println("  Actual (first 5): ", du[ids_1[1:min(5, end)]])
@test all(du[ids_1] .≈ 10.0)

# Check Dirichlet Constant (Surface 3)
ids_3 = domain.boundaries[:surface3][1]
println("Checking Surface 3 (Dirichlet Constant)...")
println("  Expected: 0.0")
println("  Actual (first 5): ", du[ids_3[1:min(5, end)]])
@test all(du[ids_3] .== 0.0)

# Check Neumann (Surface 2)
ids_2 = domain.boundaries[:surface2][1]
println("Checking Surface 2 (Neumann u=0)...")
# Should be 0 because u=0
@test all(du[ids_2] .≈ 0.0)

# Now let's set u to something non-zero to see if Neumann nodes evolve
# u = x^2. Laplacian is 2.
coords = MeshlessMultiphysics._coords(cloud)
u_sq = [ustrip(c[1])^2 for c in coords]

prob.f(du, u_sq, p, t)

# Dirichlet nodes should still be overridden
@test all(du[ids_1] .≈ 10.0)
@test all(du[ids_3] .== 0.0)

# Volume nodes
all_boundary_ids = reduce(vcat, [b[1] for b in values(domain.boundaries)])
vol_ids = setdiff(1:length(cloud), all_boundary_ids)

println("Checking Volume nodes (Laplacian of x^2)...")
mean_val = sum(du[vol_ids]) / length(vol_ids)
println("  Mean value: ", mean_val)
# It should be close to 2.0
@test abs(mean_val - 2.0) < 0.5

println("Test Complete")
