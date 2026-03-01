using Macchiato
import Macchiato as MM
using RadialBasisFunctions
using WhatsThePoint
import WhatsThePoint as WTP
using StaticArrays
using LinearAlgebra
using LinearSolve
using Unitful: m, °, ustrip

println("="^60)
println("Testing Shadow Point Boundary Conditions")
println("="^60)

##
# Create a simple square domain
println("\n1. Creating square domain...")

L = (1m, 1m)
dx = 0.1m  # boundary point spacing
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

println("   Created boundary with $(length(p)) points")
println("   Surfaces: ", keys(part.surfaces))

##
# Create cloud
println("\n2. Creating point cloud...")

Δ = dx
cloud = WhatsThePoint.discretize(part, ConstantSpacing(Δ), alg = VanDerSandeFornberg())

println("   Total points in cloud: ", size(cloud))
println("   Volume points: ", length(cloud.volume))
for (key, val) in cloud.boundary.surfaces
    println("   Surface $key: ", length(val), " points")
end

##
# Set up boundary conditions with shadow points
println("\n3. Setting up boundary conditions...")

# Material properties
k = 1.0      # thermal conductivity [W/(m·K)]
ρ = 1.0      # density [kg/m³]
cₚ = 1.0     # specific heat [J/(kg·K)]

# Test Case 1: Neumann BC with shadow points
println("\n   Test Case: Neumann (Adiabatic) with shadow points")
bcs = Dict(
    :surface1 => MM.Temperature(100.0),  # Dirichlet: T = 100 (bottom)
    :surface2 => MM.Temperature(0.0),    # Dirichlet: T = 0 (right)
    :surface3 => MM.Temperature(50.0),   # Dirichlet: T = 50 (top)
    :surface4 => MM.Adiabatic()          # Neumann: ∂T/∂n = 0 (left, adiabatic)
)

println("   Boundary conditions:")
for (surf, bc) in bcs
    println("      $surf => $bc")
end

domain = MM.Domain(cloud, bcs, MM.SolidEnergy(k = k, ρ = ρ, cₚ = cₚ))

##
# Solve with standard derivative (no shadow points)
println("\n4. Solving with standard derivative (no shadow points)...")

prob = MM.LinearProblem(domain)
println("   System size: ", size(prob.A))
println("   Solving...")
sol = solve(prob)
T_standard = sol.u
println("   ✓ Solution obtained")
println("   Temperature range: [$(minimum(T_standard)), $(maximum(T_standard))]")

##
# Now test with shadow points
println("\n5. Testing with FIRST ORDER shadow points...")

# Same BCs as before, but now we pass the scheme to LinearProblem
bcs_shadow1 = Dict(
    :surface1 => MM.Temperature(100.0),
    :surface2 => MM.Temperature(0.0),
    :surface3 => MM.Temperature(50.0),
    :surface4 => MM.Adiabatic()  # Adiabatic BC - scheme is specified in LinearProblem
)

println("   Boundary conditions:")
for (surf, bc) in bcs_shadow1
    println("      $surf => $bc")
end
println("   Scheme: ShadowPoints(dx, 1)")

domain_shadow1 = MM.Domain(cloud, bcs_shadow1, MM.SolidEnergy(k = k, ρ = ρ, cₚ = cₚ))
prob_shadow1 = MM.LinearProblem(
    domain_shadow1; scheme = WTP.ShadowPoints(WTP.ConstantSpacing(dx), 1)
)
println("   System size: ", size(prob_shadow1.A))
println("   Solving...")
sol_shadow1 = solve(prob_shadow1)
T_shadow1 = sol_shadow1.u
println("   ✓ Solution obtained")
println("   Temperature range: [$(minimum(T_shadow1)), $(maximum(T_shadow1))]")

##
# Test with second order shadow points
println("\n6. Testing with SECOND ORDER shadow points...")

bcs_shadow2 = Dict(
    :surface1 => MM.Temperature(100.0),
    :surface2 => MM.Temperature(0.0),
    :surface3 => MM.Temperature(50.0),
    :surface4 => MM.Adiabatic()  # Adiabatic BC - scheme is specified in LinearProblem
)

println("   Boundary conditions:")
for (surf, bc) in bcs_shadow2
    println("      $surf => $bc")
end
println("   Scheme: ShadowPoints(dx, 2)")

domain_shadow2 = MM.Domain(cloud, bcs_shadow2, MM.SolidEnergy(k = k, ρ = ρ, cₚ = cₚ))
prob_shadow2 = MM.LinearProblem(
    domain_shadow2; scheme = WTP.ShadowPoints(WTP.ConstantSpacing(dx), 2)
)
println("   System size: ", size(prob_shadow2.A))
println("   Solving...")
sol_shadow2 = solve(prob_shadow2)
T_shadow2 = sol_shadow2.u
println("   ✓ Solution obtained")
println("   Temperature range: [$(minimum(T_shadow2)), $(maximum(T_shadow2))]")

##
# Test Robin BC (Convection)
println("\n7. Testing Robin BC (Convection) with shadow points...")

h = 10.0  # heat transfer coefficient [W/(m²·K)]
T_inf = 25.0  # ambient temperature [K]

bcs_robin = Dict(
    :surface1 => MM.Temperature(100.0),
    :surface2 => MM.Temperature(0.0),
    :surface3 => MM.Temperature(50.0),
    :surface4 => MM.Convection(h, k, T_inf)  # Robin BC: h*T + k*∂T/∂n = h*T_inf
)

println("   Boundary conditions:")
for (surf, bc) in bcs_robin
    println("      $surf => $bc")
end

domain_robin = MM.Domain(cloud, bcs_robin, MM.SolidEnergy(k = k, ρ = ρ, cₚ = cₚ))
prob_robin = MM.LinearProblem(
    domain_robin; scheme = WTP.ShadowPoints(WTP.ConstantSpacing(dx), 1)
)
println("   System size: ", size(prob_robin.A))
println("   Solving...")
sol_robin = solve(prob_robin)
T_robin = sol_robin.u
println("   ✓ Solution obtained")
println("   Temperature range: [$(minimum(T_robin)), $(maximum(T_robin))]")

println("\n" * "="^60)
println("Test completed!")
println("="^60)
