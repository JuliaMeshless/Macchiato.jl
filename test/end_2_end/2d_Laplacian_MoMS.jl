import Macchiato as MM
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

# ============================================================================
# Step 2: Boundary conditions from the analytical solution
# ============================================================================
#
# surface1 = bottom, surface2 = right, surface3 = top, surface4 = left.
#
# Three BC types are exercised at once:
#   1. Dirichlet (Temperature): u = u_exact
#   2. Neumann   (HeatFlux):    ∂u/∂n = ∇u ⋅ n
#   3. Robin     (Convection):  h·u + k·∂u/∂n = h·T∞
#
# For u_exact = x(1-x) + y(1-y), ∇u = (1-2x, 1-2y), so on every edge
# ∂u/∂n = -1:
#   bottom (y=0, n=(0,-1)): -(1-2y) = -1     right (x=1, n=(1,0)):  (1-2x) = -1
#   top    (y=1, n=(0,1)):   (1-2y) = -1     left  (x=0, n=(-1,0)): -(1-2x) = -1

flux_val = -1.0

# Robin on the top edge with h = k = 1: u + ∂u/∂n = T∞ ⟹ T∞ = u - 1
bc_top_robin(x, t) = bc_top(x, t) - 1.0

moms_bcs() = Dict(
    :surface1 => MM.Temperature(bc_bottom),             # Bottom: Dirichlet
    :surface2 => MM.HeatFlux(flux_val),                 # Right: Neumann
    :surface3 => MM.Convection(1.0, 1.0, bc_top_robin), # Top: Robin
    :surface4 => MM.Temperature(bc_left),               # Left: Dirichlet
)

"""
    solve_poisson_moms(dx) -> (; L2_error, Linf_error, relative_L2, boundary_error)

Solve ∇²u = -4 on the unit square at resolution `dx` under mixed
Dirichlet/Neumann/Robin conditions, and compare against `u_exact`.

Kept in a function rather than at file scope so a failure here is reported as one
test error instead of escaping `include` and aborting the whole suite.
"""
function solve_poisson_moms(dx = 1 / 33 * m)
    cloud = create_2d_square_cloud(dx)

    # Poisson ∇²u = f via SolidEnergy with α = k/(ρcₚ) = 1
    source_function(x, t) = source_term(x[1], x[2])
    model = MM.SolidEnergy(k = 1.0, ρ = 1.0, cₚ = 1.0, source = source_function)
    domain = MM.Domain(cloud, moms_bcs(), model)

    sol = solve(MM.LinearProblem(domain))
    u_numerical = sol.u

    u_analytical = map(MM._coords(cloud)) do pt
        u_exact(ustrip(pt.x), ustrip(pt.y))
    end

    error = u_numerical .- u_analytical
    return (
        L2_error = norm(error, 2) / sqrt(length(error)),
        Linf_error = norm(error, Inf),
        relative_L2 = (norm(error, 2) / sqrt(length(error))) / norm(u_analytical, 2),
        boundary_error = norm(error[1:length(cloud.boundary)], Inf),
    )
end

@testset "2D Poisson MoMS" begin
    r = solve_poisson_moms()

    println("\nError Analysis:")
    println("  L2 error:      ", r.L2_error)
    println("  L∞ error:      ", r.Linf_error)
    println("  Relative L2:   ", r.relative_L2)
    println("  Boundary L∞:   ", r.boundary_error)

    # Boundary conditions should be satisfied essentially exactly
    @test r.boundary_error < 1.0e-10

    # Interior accuracy depends on discretization; these are reasonable for a
    # 33×33 meshless cloud on a polynomial solution.
    @test r.L2_error < 5.0e-2
    @test r.Linf_error < 1.0e-1
    @test r.relative_L2 < 0.1

    println("\n✓ Poisson equation solver validated with MoMS")
    println("  Solving: ∇²u = -4 with u = x(1-x) + y(1-y)")
end
