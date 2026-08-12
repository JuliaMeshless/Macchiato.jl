# Parity verification: before any timing, assert that both packages — fed the identical
# settings — produce the same numbers. A mismatch means the settings are NOT actually
# identical and the benchmark comparison would be meaningless.

using Printf

function _check(label::String, diff::Float64, tol::Float64, failures::Vector{String})
    status = diff <= tol ? "PASS" : "FAIL"
    @printf("  [%s] %-58s max|Δ| = %.3e  (tol %.1e)\n", status, label, diff, tol)
    diff <= tol || push!(failures, label)
    return nothing
end

function verify()
    println("Verifying setting parity (identical inputs ⇒ matching outputs)…")
    failures = String[]

    # -- interpolation 1D: both interpolants must reproduce the data and agree off-node
    itp_ki = ki_interpolation_1d(interp1d_nodeset, interp1d_values)
    itp_ours = ours_interpolation_1d(interp1d_x, interp1d_values)
    xs_test = LinRange(0.05, 2π - 0.05, 17)
    _check(
        "interpolation 1D: KI vs ours at 17 off-node points",
        maximum(abs(itp_ki([x]) - itp_ours(SVector(x))) for x in xs_test),
        1.0e-8, failures,
    )
    _check(
        "interpolation 1D: ours reproduces nodal data",
        maximum(abs(itp_ours(x) - v) for (x, v) in zip(interp1d_x, interp1d_values)),
        1.0e-8, failures,
    )

    # -- Poisson 2D: all four solution paths must agree at the 400 interior nodes
    u_ref = u_poisson.(poisson_inner)

    itp_lag = ki_poisson_solve(ki_poisson_sd_lagrange)
    itp_std = ki_poisson_solve(ki_poisson_sd_standard)
    u_ki_lag = [itp_lag(x) for x in poisson_inner_ns]
    u_ki_std = [itp_std(x) for x in poisson_inner_ns]

    sol_ours = ours_poisson_solve(
        ours_poisson_L_int, poisson_inner, poisson_bdry,
        length(poisson_inner), length(poisson_all),
    )
    u_ours = sol_ours.u[1:length(poisson_inner)]  # ordering: [interior; boundary]

    sol_mc = ours_poisson_end_to_end(poisson_cloud)
    u_mc = sol_mc.u[(length(poisson_bdry) + 1):end]  # Macchiato ordering: [boundary; interior]

    _check("Poisson 2D: KI Lagrange vs KI standard basis", maximum(abs.(u_ki_lag .- u_ki_std)), 1.0e-6, failures)
    _check("Poisson 2D: KI (Lagrange) vs ours (RBF.jl assembly)", maximum(abs.(u_ki_lag .- u_ours)), 1.0e-6, failures)
    # Looser tolerance: Macchiato's cloud orders nodes boundary-first while KI merges
    # interior-first, and on this symmetric grid the k-NN cutoff has distance ties, so 8 of
    # 436 stencils pick a different (equidistant) neighbor — a benign ~1e-4 effect at
    # identical settings, verified in the benchmark write-up.
    _check("Poisson 2D: KI (Lagrange) vs ours (Macchiato end-to-end)", maximum(abs.(u_ki_lag .- u_mc)), 1.0e-3, failures)
    _check("Poisson 2D: KI vs exact solution (discretization error)", maximum(abs.(u_ki_lag .- u_ref)), 5.0e-2, failures)
    _check("Poisson 2D: ours vs exact solution (discretization error)", maximum(abs.(u_ours .- u_ref)), 5.0e-2, failures)

    # -- Advection 2D: both rhs! evaluations must agree on the SAME state vector
    u_test = [u_adv(0.0, x) for x in adv_all]  # ordering matches KI's merge(inner, boundary)
    du_ki = similar(u_test)
    KI.rhs!(du_ki, u_test, ki_adv_semi, 0.0)
    du_ours = similar(u_test)
    ours_adv_rhs!(du_ours, u_test, nothing, 0.0)
    _check("Advection 2D: KI.rhs! vs ours_rhs! on identical state", maximum(abs.(du_ki .- du_ours)), 1.0e-6, failures)
    # Informational sanity bound, not a parity assertion: the two sides run independent
    # adaptive Rodas5P solves (KI supplies solver hints ours lacks), so the end states agree
    # only to solver tolerance. The parity-critical check is the rhs! one above.
    _check(
        "Advection 2D: ODE end states (independent Rodas5P solves)",
        maximum(abs.(ki_adv_u_end .- ours_adv_u_end)),
        5.0e-2, failures,
    )

    if isempty(failures)
        println("All parity checks passed — settings verified identical across solvers.\n")
    else
        error("Parity verification FAILED for: " * join(failures, "; "))
    end
    return nothing
end
