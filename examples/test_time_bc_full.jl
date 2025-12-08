using MeshlessMultiphysics
using WhatsThePoint
import WhatsThePoint as WTP
using Unitful: m, °, ustrip
using Test
using DifferentialEquations

println("="^60)
println("Testing Time Varying Boundary Conditions")
println("="^60)

# ------------------------------------------------------------------
# 1. Setup Domain
# ------------------------------------------------------------------
println("\n[1] Creating domain...")

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
println("    Cloud size: ", size(cloud))
println("    Surface names: ", names(cloud.boundary))

surf_name = :surface1

# ------------------------------------------------------------------
# 2. Test FixedValue (Constant in time)
# ------------------------------------------------------------------
println("\n[2] Testing FixedValue (Constant)...")

bc_const = Temperature(100.0)
# Define BCs for all surfaces
boundaries_const = Dict(
    :surface1 => bc_const,
    :surface2 => bc_const,
    :surface3 => bc_const,
    :surface4 => bc_const
)
model = SolidEnergy(k = 1.0, ρ = 1.0, cₚ = 1.0)
domain_const = Domain(cloud, boundaries_const, model)

# Manually create BC functions to avoid MultiphysicsProblem/make_f issues
println("    Creating BC functions...")
bc_funcs_const = []
for (surf_name, (ids, bc)) in domain_const.boundaries
    surf = domain_const.cloud[surf_name]
    push!(bc_funcs_const, make_bc(bc, surf, domain_const, ids))
end

du = zeros(length(cloud))
u = zeros(length(cloud))
p = nothing
t = 0.0

# Apply all BCs
for f in bc_funcs_const
    f(du, u, p, t)
end

ids = domain_const.boundaries[:surface1][1]
@test all(du[ids] .== 0.0)
println("    Passed: du/dt is 0 for constant BC.")

# ------------------------------------------------------------------
# 3. Test TimeVaryingValue (via Temperature with Function)
# ------------------------------------------------------------------
println("\n[3] Testing TimeVaryingValue...")

# T(x, t) = 10.0 * t
# dT/dt = 10.0
time_func = (x, t) -> 10.0 * t
bc_time = Temperature(time_func)

boundaries_time = Dict(
    :surface1 => bc_time,
    :surface2 => bc_time,
    :surface3 => bc_time,
    :surface4 => bc_time
)
domain_time = Domain(cloud, boundaries_time, model)

# Manually create BC functions
println("    Creating BC functions...")
bc_funcs_time = []
for (surf_name, (ids, bc)) in domain_time.boundaries
    surf = domain_time.cloud[surf_name]
    push!(bc_funcs_time, make_bc(bc, surf, domain_time, ids))
end

du .= 0.0
t = 1.0
# Apply all BCs
for f in bc_funcs_time
    f(du, u, p, t)
end

# Check results
ids = domain_time.boundaries[:surface1][1]
max_diff = maximum(abs.(du[ids] .- 10.0))
println("    Max difference from expected (10.0): ", max_diff)
@test all(isapprox.(du[ids], 10.0, atol = 1e-3))
println("    Passed: du/dt matches time derivative.")

# ------------------------------------------------------------------
# 4. Full System Solve
# ------------------------------------------------------------------
println("\n[4] Testing Full System Solve...")

u0 = zeros(length(cloud))
tspan = (0.0, 1.0)
prob = MultiphysicsProblem(domain_time, u0, tspan)

# Solve using Euler for simplicity
dt = 0.01
sol = solve(prob, Euler(), dt = dt)

# Check boundary values at final time
t_final = 1.0
u_final = sol(t_final)

# Expected boundary value: 10.0 * t_final = 10.0
for (surf_name, (ids, bc)) in domain_time.boundaries
    u_boundary = u_final[ids]
    max_err = maximum(abs.(u_boundary .- 10.0 * t_final))
    println("    Surface $surf_name max error: ", max_err)
    @test all(isapprox.(u_boundary, 10.0 * t_final, atol = 1e-2))
end
println("    Passed: Boundary values match expected time evolution.")

println("\n" * "="^60)
println("All tests passed!")
println("="^60)
