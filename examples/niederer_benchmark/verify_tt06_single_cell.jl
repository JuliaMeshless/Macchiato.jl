# Single-cell verification of the ten Tusscher-Panfilov 2006 epicardial model
# used by niederer_benchmark.jl. Paces one cell with forward Euler (same integrator
# used per split in the tissue benchmark) and reports V_rest, V_peak, dV/dt_max,
# and APD90 for the last beat.
#
# Expected values (ten Tusscher & Panfilov 2006, Am J Physiol Heart Circ Physiol 291):
#   V_rest     ≈ -85 to -87 mV   (ICs set V to -85.23)
#   V_peak     ≈ +35 to +50 mV
#   dV/dt_max  ≈ 250-300 mV/ms   (paper reports ≈ 288)
#   APD90      ≈ 270-310 ms      at BCL=1000 ms (epicardial)

using Pkg
Pkg.activate(joinpath(@__DIR__, "../"))

using GLMakie: Figure, Axis, lines!, save, axislegend

include(joinpath(@__DIR__, "ten_tusscher_2006.jl"))

function pace(; dt = 0.01, BCL = 1000.0, n_beats = 5, stim_amp = 52.0, stim_dur = 1.0)
    u = tt06_initial_conditions(1)
    du = similar(u)

    nsteps_prepace = round(Int, (n_beats - 1) * BCL / dt)
    for step in 1:nsteps_prepace
        t_local = mod((step - 1) * dt, BCL)
        s = t_local < stim_dur ? stim_amp : 0.0
        tt06_reaction_point!(du, u, 1, 1, s)
        @. u += dt * du
    end

    nsteps = round(Int, BCL / dt)
    ts = Vector{Float64}(undef, nsteps + 1)
    Vs = Vector{Float64}(undef, nsteps + 1)
    ts[1] = 0.0
    Vs[1] = u[1]
    for step in 1:nsteps
        t_local = (step - 1) * dt
        s = t_local < stim_dur ? stim_amp : 0.0
        tt06_reaction_point!(du, u, 1, 1, s)
        @. u += dt * du
        ts[step + 1] = step * dt
        Vs[step + 1] = u[1]
    end

    dVdt = diff(Vs) ./ diff(ts)
    return ts, Vs, dVdt
end

function compute_metrics(ts, Vs, dVdt)
    V_rest = Vs[1]
    V_peak = maximum(Vs)

    # INa upstroke (~250 mV/ms) dwarfs the 52 mV/ms stim pulse, so take the max of
    # the full numerical derivative. If INa failed to fire we would see ≤ stim_amp.
    dVdt_max = maximum(dVdt)

    V_90 = V_rest + 0.1 * (V_peak - V_rest)
    i_up = findfirst(v -> v > 0.0, Vs)
    i_peak = argmax(Vs)
    i_down = findnext(v -> v < V_90, Vs, i_peak)
    APD90 = (i_up === nothing || i_down === nothing) ? NaN : ts[i_down] - ts[i_up]
    return (; V_rest, V_peak, dVdt_max, APD90)
end

function run_and_report(; dt, BCL = 1000.0, n_beats = 10, stim_amp = 52.0, stim_dur = 1.0)
    println("── dt = $dt ms, $n_beats beats @ BCL = $BCL ms ──")
    ts, Vs, dVdt = pace(; dt, BCL, n_beats, stim_amp, stim_dur)
    m = compute_metrics(ts, Vs, dVdt)
    println("  V_rest    = $(round(m.V_rest; digits = 2)) mV    [expect ≈ -85 to -87]")
    println("  V_peak    = $(round(m.V_peak; digits = 2)) mV    [expect ≈ +35 to +55]")
    println("  dV/dt max = $(round(m.dVdt_max; digits = 1)) mV/ms [expect ≈ 250-300]")
    println("  APD90     = $(round(m.APD90; digits = 1)) ms    [expect ≈ 270-310]")
    return ts, Vs, m
end

function main()
    println("Single-cell TT06 epi verification\n")

    # Compare dt = 0.01 (benchmark's step) vs dt = 0.002 (finer check for FE overshoot)
    ts_coarse, Vs_coarse, _ = run_and_report(; dt = 0.01)
    println()
    ts_fine, Vs_fine, _ = run_and_report(; dt = 0.002)

    fig = Figure(; size = (900, 400))
    ax = Axis(
        fig[1, 1]; xlabel = "time (ms)", ylabel = "V (mV)",
        title = "TT06 epi — last beat (BCL=1000 ms)"
    )
    lines!(ax, ts_coarse, Vs_coarse; label = "dt = 0.01 ms")
    lines!(ax, ts_fine, Vs_fine; label = "dt = 0.002 ms")
    axislegend(ax)
    save(joinpath(@__DIR__, "tt06_single_cell_ap.png"), fig)
    println("\nAction potentials saved to tt06_single_cell_ap.png")
    return nothing
end

main()
