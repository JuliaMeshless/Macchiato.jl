import Macchiato as MM
using Macchiato
using WhatsThePoint
import WhatsThePoint as WTP
using Unitful: ustrip, m
using LinearAlgebra
using Test

include("2d_square.jl")

# ============================================================================
# Complex-valued steady solve (regression for eltype-generic Simulation)
# ============================================================================
# The steady path must carry a ComplexF64 system end to end: `make_system`
# decides the eltype, the Dirichlet writer stores complex boundary values, and
# `Simulation._solution` accepts the complex solution vector (its storage is
# Vector{<:Number}, not Vector{Float64}). Time-harmonic problems (e.g. the
# Helmholtz scattering example) depend on this.
#
# Manufactured solution: a complex multiple of the standard polynomial MMS,
#   u = (1 + 2im)·(x(1-x) + y(1-y)),  ∇²u = -4(1 + 2im)
# with Dirichlet data from u_exact on all four surfaces. The matrix is promoted
# to ComplexF64 at assembly — a real factorization cannot back-substitute a
# complex right-hand side, so the whole system commits to the complex eltype.

u_exact_c(x) = (1.0 + 2.0im) * (x[1] * (1 - x[1]) + x[2] * (1 - x[2]))
source_c(x) = -4.0 * (1.0 + 2.0im)

struct ComplexPoisson{S} <: Macchiato.AbstractModel
    source::S
end

Macchiato.num_vars(::ComplexPoisson, _) = 1

function Macchiato.make_system(model::ComplexPoisson, domain; kwargs...)
    x = node_coordinates(domain)
    ∇² = laplacian(x; k = 40)
    A = ComplexF64.(weights(∇²))
    b = ComplexF64[model.source(xᵢ) for xᵢ in x]
    return A, b
end

dx = 1 / 33 * m
part = create_2d_square_domain(dx)
cloud = WTP.discretize(part, ConstantSpacing(dx))

bcs = Dict(
    :surface1 => PrescribedValue((x, t) -> u_exact_c(x)),
    :surface2 => PrescribedValue((x, t) -> u_exact_c(x)),
    :surface3 => PrescribedValue((x, t) -> u_exact_c(x)),
    :surface4 => PrescribedValue((x, t) -> u_exact_c(x)),
)

domain = MM.Domain(cloud, bcs, ComplexPoisson(source_c))
sim = Simulation(domain)
run!(sim)
u = solution(sim)

coords = node_coordinates(domain)
err = u .- u_exact_c.(coords)
boundary_error = maximum(abs, err[1:length(cloud.boundary)])
max_error = maximum(abs, err)
println("complex steady solve: eltype = ", eltype(u),
    ", boundary L∞ = ", round(boundary_error; sigdigits = 3),
    ", L∞ = ", round(max_error; sigdigits = 3))

@testset "complex steady solve" begin
    @test eltype(u) == ComplexF64
    # Dirichlet rows are identity rows — complex boundary values stored exactly
    @test boundary_error < 1.0e-10
    # interior accuracy in line with the real-valued MoMS test at the same dx
    @test max_error < 5.0e-2
    # the imaginary part must actually be solved for, not silently dropped
    # (exact max of |imag(u)| is 1.0 at the domain center)
    @test maximum(abs, imag.(u)) > 0.5
end
