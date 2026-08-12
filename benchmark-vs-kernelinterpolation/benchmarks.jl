# Benchmarks: Macchiato stack (Macchiato + RadialBasisFunctions.jl) vs KernelInterpolation.jl.
#
# For every benchmark the two implementations are written BACK TO BACK (ki_* then ours_*) so a
# reader can verify side by side that the settings are identical. Shared node sets and problem
# definitions live in common.jl. Benchmarks from KI's suite that need features we don't have
# (Wendland/TPS kernels, global collocation, least squares, multiscale, off-node RBF-FD
# evaluation) are skipped — see README.md for the full list with reasons.

using BenchmarkTools
using RadialBasisFunctions
using Macchiato
using LinearSolve
using SciMLBase: ODEProblem
using OrdinaryDiffEqRosenbrock: Rodas5P
using OrdinaryDiffEqNonlinearSolve  # solver dependency of the Rosenbrock initialization, as in KI's suite
using SparseArrays
using LinearAlgebra

const SUITE = BenchmarkGroup()

# Dirichlet boundary block: one identity row per boundary node (boundary nodes are the last
# n_all − n_inner entries of the global ordering). This is what KI's assembly emits for its
# boundary rows with the (default) Lagrange local basis.
function bdry_identity_rows(n_inner::Int, n_all::Int)
    n_bdry = n_all - n_inner
    return sparse(1:n_bdry, (n_inner + 1):n_all, ones(n_bdry), n_bdry, n_all)
end

#-------------------------------------------------------------------------------# interpolation 1D
# Shared settings: 8 nodes on [0, 2π] · values f(x) = sin(sum(x)) · Gaussian kernel
# φ(r) = exp(−(εr)²) with ε = 0.5 (identical formula and shape-parameter convention in both
# packages) · polynomial augmentation of total degree ≤ 2 on BOTH sides · dense symmetric
# saddle-point system, LAPACK Bunch-Kaufman factorization on both sides (KI: Symmetric(A)\b;
# ours: hardcoded bunchkaufman!).
#
# DEVIATION from KI's stock suite (documented in README): KI's Gauss kernel has order 0, so its
# stock benchmark adds NO polynomial. Our Interpolator cannot disable augmentation, so both
# sides run with degree-2 augmentation instead (KI: m = 3 ⇒ degree ≤ 2; ours: poly_deg = 2).

function ki_interpolation_1d(nodeset, values)
    kernel = KI.GaussKernel{KI.dim(nodeset)}(shape_parameter = 0.5)
    return KI.interpolate(nodeset, values, kernel; m = 3)  # m = 3 ⇒ polynomials of degree ≤ 2
end

function ours_interpolation_1d(x, values)
    return Interpolator(x, values, Gaussian(0.5; poly_deg = 2))  # degree ≤ 2 augmentation
end

let g = SUITE["interpolation 1D"] = BenchmarkGroup()
    g["KernelInterpolation"] = @benchmarkable ki_interpolation_1d($interp1d_nodeset, $interp1d_values)
    g["Macchiato stack"] = @benchmarkable ours_interpolation_1d($interp1d_x, $interp1d_values)
end

#-------------------------------------------------------------------------------# RBF-FD operator build 2D
# Shared settings: 436 nodes (400 interior + 36 boundary, identical coordinates) · PHS kernel
# φ(r) = r³ · polynomial augmentation of total degree ≤ 1 (KI: m = order(kernel) = 2 ⇒
# degree ≤ m−1 = 1; ours: poly_deg = 1) · k-nearest-neighbor stencils with k = 25, both via a
# NearestNeighbors.jl KDTree · one local weight system solved per node (436 stencils each side).
#
# Seam caveat (documented in README): KI's RBFFDBasis stops after stencil selection + local
# weights; assembly into a sparse operator matrix happens later, inside solve_stationary. Our
# laplacian() fuses stencil selection + local weights + sparse assembly into one call, so the
# "Macchiato stack" entry includes sparse-matrix assembly that the KI entries exclude.

function ki_rbffd_build_lagrange(nodeset)
    kernel = KI.PolyharmonicSplineKernel{2}(3)
    return KI.RBFFDBasis(
        nodeset, kernel, KI.KNearestNeighbors(25);
        m = KI.order(kernel), local_basis = KI.RBFFDLagrangeBasis(),
    )
end

function ki_rbffd_build_standard(nodeset)
    kernel = KI.PolyharmonicSplineKernel{2}(3)
    return KI.RBFFDBasis(
        nodeset, kernel, KI.KNearestNeighbors(25);
        m = KI.order(kernel), local_basis = KI.RBFFDStandardBasis(),
    )
end

function ours_rbffd_build(nodes)
    # weights are built eagerly inside the constructor (stencils + local solves + assembly)
    return laplacian(nodes; basis = PHS(3; poly_deg = 1), k = 25)
end

let g = SUITE["RBF-FD operator build 2D"] = BenchmarkGroup()
    g["KernelInterpolation (Lagrange basis)"] = @benchmarkable ki_rbffd_build_lagrange($poisson_merged_ns)
    g["KernelInterpolation (standard basis)"] = @benchmarkable ki_rbffd_build_standard($poisson_merged_ns)
    g["Macchiato stack"] = @benchmarkable ours_rbffd_build($poisson_all)
end

#-------------------------------------------------------------------------------# RBF-FD Poisson 2D (prebuilt weights)
# Shared settings: −Δu = f, Dirichlet g = u_exact (see common.jl) · same 436 nodes, PHS r³,
# degree ≤ 1 augmentation, k = 25 stencils as above · stencil weights prebuilt OUTSIDE the
# benchmark on both sides · benchmark = assemble the 436×436 sparse system (interior rows −Δ,
# boundary rows identity), evaluate b = [f; g], and solve with sparse LU — UMFPACK on both
# sides (KI: SparseMatrixCSC \ b; ours: LinearSolve UMFPACKFactorization, passed explicitly
# because it is not Macchiato's default).

function ki_poisson_solve(sd)
    return KI.solve_stationary(sd)  # assembles pde_boundary_matrix + b from the prebuilt basis, then A \ b
end

function ours_poisson_solve(L_int, inner, bdry, n_inner, n_all)
    A = vcat(-L_int, bdry_identity_rows(n_inner, n_all))  # interior rows −∇² ⇒ −Δu = f, boundary rows identity
    b = vcat(f_poisson.(inner), u_poisson.(bdry))
    return solve(LinearProblem(A, b), UMFPACKFactorization())
end

const ki_poisson_pde = KI.PoissonEquation((x, equations) -> f_poisson(x))
ki_poisson_g(x) = u_poisson(x)

const ki_poisson_sd_lagrange = KI.SpatialDiscretization(
    ki_poisson_pde, poisson_inner_ns, ki_poisson_g, poisson_bdry_ns,
    ki_rbffd_build_lagrange(poisson_merged_ns),
)
const ki_poisson_sd_standard = KI.SpatialDiscretization(
    ki_poisson_pde, poisson_inner_ns, ki_poisson_g, poisson_bdry_ns,
    ki_rbffd_build_standard(poisson_merged_ns),
)
const ours_poisson_L_int = weights(
    laplacian(poisson_all; eval_points = poisson_inner, basis = PHS(3; poly_deg = 1), k = 25)
)

let g = SUITE["RBF-FD Poisson 2D"] = BenchmarkGroup()
    g["KernelInterpolation (Lagrange basis)"] = @benchmarkable ki_poisson_solve($ki_poisson_sd_lagrange)
    g["KernelInterpolation (standard basis)"] = @benchmarkable ki_poisson_solve($ki_poisson_sd_standard)
    g["Macchiato stack"] = @benchmarkable ours_poisson_solve(
        $ours_poisson_L_int, $poisson_inner, $poisson_bdry,
        $(length(poisson_inner)), $(length(poisson_all)),
    )
end

#-------------------------------------------------------------------------------# RBF-FD Poisson 2D end-to-end
# Not in KI's stock suite; added because the task is "benchmark this package": the full
# package-level pipeline from raw coordinates to solution, exercising Macchiato proper
# (Domain construction, SolidEnergy model assembly, Dirichlet row replacement, LinearSolve).
# Shared settings identical to the two Poisson benchmarks above; both sides include stencil
# selection + local weights + assembly + UMFPACK solve. SolidEnergy with k = ρ = cₚ = 1
# solves ∇²T = source, so source = −f gives −ΔT = f, matching KI's PoissonEquation.

function ki_poisson_end_to_end(inner_ns, bdry_ns)
    pde = KI.PoissonEquation((x, equations) -> f_poisson(x))
    kernel = KI.PolyharmonicSplineKernel{2}(3)
    sd = KI.SpatialDiscretization(
        pde, inner_ns, ki_poisson_g, bdry_ns, KI.RBFFD(), kernel;
        stencil_selection = KI.KNearestNeighbors(25),
        m = KI.order(kernel), local_basis = KI.RBFFDLagrangeBasis(),
    )
    return KI.solve_stationary(sd)
end

function ours_poisson_end_to_end(cloud)
    model = SolidEnergy(k = 1.0, ρ = 1.0, cₚ = 1.0, source = (x, t) -> -f_poisson(x))
    bcs = Dict(:surface1 => PrescribedValue((x, t) -> u_poisson(x)))
    domain = Domain(cloud, bcs, model)
    prob = LinearProblem(domain; basis = PHS(3; poly_deg = 1), k = 25)
    return solve(prob, UMFPACKFactorization())
end

let g = SUITE["RBF-FD Poisson 2D end-to-end"] = BenchmarkGroup()
    g["KernelInterpolation"] = @benchmarkable ki_poisson_end_to_end($poisson_inner_ns, $poisson_bdry_ns)
    g["Macchiato stack"] = @benchmarkable ours_poisson_end_to_end($poisson_cloud)
end

#-------------------------------------------------------------------------------# RBF-FD Advection 2D rhs!
# Shared settings: ∂u/∂t + a·∇u = 0, a = (0.5, 0.5), inflow Dirichlet data g(t, x) = u_adv(t, x)
# on the x=0 and y=0 edges (see common.jl) · 439 nodes (400 interior + 39 boundary, identical
# coordinates) · PHS r³, degree ≤ 1 augmentation, k = 25 stencils · both sides benchmark the
# in-place ODE right-hand side du = −A·u + b, where A has a·∇ stencil rows at interior nodes
# and identity rows at boundary nodes, and b = (0; g(t, x)) is filled in place before a 5-arg
# mul! — the exact formulation of KI.rhs!. The benchmark state is sol.u[end] of a short
# Rodas5P solve of the same semidiscretization on tspan = (0.0, 0.01), as in KI's suite.
#
# Note (documented in README): Macchiato ships no advection model, so the ours side is
# user-authored from RadialBasisFunctions.jl operators following Macchiato's custom-model
# pattern (docs/src/custom_pdes.md); it is not a shipped Macchiato solver.

function build_ki_advection(inner_ns, bdry_ns, tspan)
    pde = KI.AdvectionEquation((adv_a[1], adv_a[2]), (t, x, equations) -> 0.0)
    kernel = KI.PolyharmonicSplineKernel{2}(3)
    sd = KI.SpatialDiscretization(
        pde, inner_ns, (t, x) -> u_adv(t, x), bdry_ns, KI.RBFFD(), kernel;
        stencil_selection = KI.KNearestNeighbors(25), m = KI.order(kernel),
    )
    semi = KI.Semidiscretization(sd, (t, x, equations) -> u_adv(t, x))
    ode = KI.semidiscretize(semi, tspan)
    sol = solve(ode, Rodas5P())
    return semi, sol.u[end]
end

function build_ours_advection(all_nodes, inner_nodes, bdry_nodes, tspan)
    adv = directional(all_nodes, adv_a; eval_points = inner_nodes, basis = PHS(3; poly_deg = 1), k = 25)
    n_inner = length(inner_nodes)
    A = vcat(weights(adv), bdry_identity_rows(n_inner, length(all_nodes)))

    function ours_rhs!(du, u, p, t)
        # b = (f; g): zero source at interior rows, boundary data g(t, x) at boundary rows,
        # then du = −A·u + b via 5-arg mul! — same structure as KI.rhs!.
        @views fill!(du[1:n_inner], 0)
        for (i, x) in enumerate(bdry_nodes)
            du[n_inner + i] = u_adv(t, x)
        end
        mul!(du, A, u, -1, true)
        return nothing
    end

    u0 = [u_adv(0.0, x) for x in all_nodes]
    sol = solve(ODEProblem(ours_rhs!, u0, tspan), Rodas5P())
    return ours_rhs!, sol.u[end]
end

const adv_tspan = (0.0, 0.01)
const ki_adv_semi, ki_adv_u_end = build_ki_advection(adv_inner_ns, adv_bdry_ns, adv_tspan)
const ours_adv_rhs!, ours_adv_u_end = build_ours_advection(adv_all, adv_inner, adv_bdry, adv_tspan)

let g = SUITE["RBF-FD Advection 2D rhs!"] = BenchmarkGroup()
    g["KernelInterpolation"] = @benchmarkable KI.rhs!(
        $(similar(ki_adv_u_end)), $(copy(ki_adv_u_end)), $ki_adv_semi, $(first(adv_tspan))
    )
    g["Macchiato stack"] = @benchmarkable ours_adv_rhs!(
        $(similar(ours_adv_u_end)), $(copy(ours_adv_u_end)), nothing, $(first(adv_tspan))
    )
end
