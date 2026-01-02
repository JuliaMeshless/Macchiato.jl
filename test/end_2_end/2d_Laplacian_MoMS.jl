import MeshlessMultiphysics as MM
using WhatsThePoint
import WhatsThePoint as WTP
using Unitful: ustrip, m
using LinearAlgebra
using LinearSolve
using Test

include("2d_square.jl")

# ============================================================================
# Method of Manufactured Solutions Test for 2D Poisson Equation
# ============================================================================
# Tests the complete solution procedure by solving ∇²u = f with known solution
# and comparing numerical vs analytical results

# ============================================================================
# Step 1: Define manufactured polynomial solution
# ============================================================================

u_exact(x, y) = x * (1 - x) + y * (1 - y)

"""
Laplacian of u_exact:
∂²u/∂x² = -2
∂²u/∂y² = -2
∇²u = -4 (constant source term)
"""
source_term(x, y) = -4.0

# Boundary condition functions (analytical solution at boundaries)
bc_bottom(x, t) = u_exact(ustrip(x[1]), 0.0)  # y = 0: u = x(1-x)
bc_right(x, t) = u_exact(1.0, ustrip(x[2]))   # x = 1: u = y(1-y)
bc_top(x, t) = u_exact(ustrip(x[1]), 1.0)     # y = 1: u = x(1-x)
bc_left(x, t) = u_exact(0.0, ustrip(x[2]))    # x = 0: u = y(1-y)

dx = 1 / 33 * m  # Resolution (33x33 gives 1089 points)
part = create_2d_square_domain(dx)

Δ = dx
cloud = WTP.discretize(part, ConstantSpacing(Δ), alg = VanDerSandeFornberg())
conv = repel!(cloud, ConstantSpacing(Δ); α = Δ / 20, max_iters = 500)

# ============================================================================
# Step 3: Set up boundary conditions from analytical solution
# ============================================================================
#visualize surfaces
#figsize = (800, 800)
#markersize = 0.0025

#surface1 is bottom
#surface2 is right
#surface3 is top
#surface4 is left

# We want to test different types of BCs:
# 1. Dirichlet (Temperature): u = u_exact
# 2. Neumann (HeatFlux): ∂u/∂n = ∇u ⋅ n
# 3. Robin (Convection): h·u + k·∂u/∂n = h·T∞

# Calculate normal derivatives for u_exact = x(1-x) + y(1-y)
# ∇u = (1-2x, 1-2y)
# Bottom (y=0, n=(0,-1)): ∂u/∂n = -(1-2y) = -1
# Right (x=1, n=(1,0)):   ∂u/∂n = (1-2x) = -1
# Top (y=1, n=(0,1)):     ∂u/∂n = (1-2y) = -1
# Left (x=0, n=(-1,0)):   ∂u/∂n = -(1-2x) = -1
# So flux is -1 everywhere.

flux_val = -1.0

# For Robin on Top (y=1):
# Let h = 1, k = 1.
# u + ∂u/∂n = T∞
# u + (-1) = T∞  =>  T∞ = u - 1
bc_top_robin(x, t) = bc_top(x, t) - 1.0

bcs = Dict(
    :surface1 => MM.Temperature(bc_bottom),             # Bottom: Dirichlet
    :surface2 => MM.HeatFlux(flux_val),                 # Right: Neumann
    :surface3 => MM.Convection(1.0, 1.0, bc_top_robin), # Top: Robin
    :surface4 => MM.Temperature(bc_left)                # Left: Dirichlet
)

# ============================================================================
# Step 4: Set up physics model with source term
# ============================================================================

# For Poisson equation ∇²u = f, we use SolidEnergy with α = k/(ρcₚ) = 1
# Steady-state: ∇²u = f/α → (α∇²)u = f
# We need α = 1, so set k = ρ = cₚ = 1

k = 1.0
ρ = 1.0
cₚ = 1.0

# Source term function: f(x, t) = ∇²u_exact = -4
source_function(x, t) = source_term(x[1], x[2])

model = MM.SolidEnergy(k = k, ρ = ρ, cₚ = cₚ, source = source_function)
domain = MM.Domain(cloud, bcs, model)

println("\nModel: ", model)

# ============================================================================
# Step 5: Solve steady-state linear problem
# ============================================================================

prob = MM.LinearProblem(domain)
sol = solve(prob)
u_numerical = sol.u

# ============================================================================
# Step 6: Compute analytical solution at all points
# ============================================================================

u_analytical = map(MM._coords(cloud)) do pt
    u_exact(ustrip(pt.x), ustrip(pt.y))
end

# ============================================================================
# Step 7: Compute errors
# ============================================================================

error = u_numerical .- u_analytical
L2_error = norm(error, 2) / sqrt(length(error))
Linf_error = norm(error, Inf)
relative_L2 = L2_error / norm(u_analytical, 2)

println("\nError Analysis:")
println("  L2 error:      ", L2_error)
println("  L∞ error:      ", Linf_error)
println("  Relative L2:   ", relative_L2)

# Error at boundary points (should be near zero)
boundary_indices = 1:length(cloud.boundary)
boundary_error = norm(error[boundary_indices], Inf)
println("  Boundary L∞:   ", boundary_error)

# ============================================================================
# Step 8: Tests
# ============================================================================

@testset "2D Poisson MoMS" begin
    # Boundary conditions should be exact
    @test boundary_error < 1e-10

    # Interior solution accuracy depends on discretization
    # For 33x33 grid with polynomial solution, expect good accuracy
    @test L2_error < 5e-2  # Reasonable for meshless method
    @test Linf_error < 1e-1
    @test relative_L2 < 0.1  # Less than 10% relative error

    println("\n✓ Poisson equation solver validated with MoMS")
    println("  Solving: ∇²u = -4 with u = x(1-x) + y(1-y)")
end
