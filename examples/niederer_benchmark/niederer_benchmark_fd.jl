# Niederer benchmark — classical finite-difference variant.
#
# Same TT06 reaction and same monodomain physics as niederer_benchmark.jl, but the
# diffusion operator is assembled as a standard 7-point anisotropic Laplacian stencil
# on the structured grid instead of RBF-FD. Intended as a reference to isolate
# numerical behaviour of the RBF-FD diffusion from the reaction model.

using Pkg
Pkg.activate(joinpath(@__DIR__, "../"))

using OrdinaryDiffEqOperatorSplitting
using OrdinaryDiffEqLowOrderRK: Euler, ODEFunction, init, step!, TimeChoiceIterator
using OhMyThreads: tforeach
using StaticArrays
using SparseArrays
using LinearAlgebra
using WriteVTK
using DelimitedFiles
using GLMakie: Figure, Axis3, Legend, meshscatter!, display

include(joinpath(@__DIR__, "ten_tusscher_2006.jl"))

## --- Physical parameters (Niederer 2011, Table 3) — same as RBF-FD variant ---
const Lx, Ly, Lz = 20.0, 7.0, 3.0

const σ_iL, σ_iT = 0.17, 0.019
const σ_eL, σ_eT = 0.62, 0.24
const σ_L = σ_iL * σ_eL / (σ_iL + σ_eL)
const σ_T = σ_iT * σ_eT / (σ_iT + σ_eT)

const β = 140.0
const Cm_tissue = 0.01
const D_L = σ_L / (β * Cm_tissue)
const D_T = σ_T / (β * Cm_tissue)

const I_stim = 50.0
const stim_duration = 2.0
const stim_size = 1.5

const V_activation = 0.0

const P_REF = (
    P1 = (0.0, 0.0, 0.0),
    P2 = (0.0, 7.0, 0.0),
    P3 = (20.0, 7.0, 0.0),
    P4 = (20.0, 0.0, 0.0),
    P5 = (0.0, 0.0, 3.0),
    P6 = (0.0, 7.0, 3.0),
    P7 = (20.0, 0.0, 3.0),
    P8 = (20.0, 7.0, 3.0),
    P9 = (10.0, 3.5, 1.5),
)

##

"""
    create_geometry(dx) -> NamedTuple

Tensor-product grid with **natural** (ix, iy, iz) ordering — flat index
`k = ix + (iy-1)*nx + (iz-1)*nx*ny`. Unlike the RBF-FD variant, we don't split
boundary vs interior because the FD stencil addresses the grid directly.
"""
function create_geometry(dx)
    nx = round(Int, Lx / dx) + 1
    ny = round(Int, Ly / dx) + 1
    nz = round(Int, Lz / dx) + 1
    xs = range(0.0, Lx; length = nx)
    ys = range(0.0, Ly; length = ny)
    zs = range(0.0, Lz; length = nz)
    N = nx * ny * nz

    all_pts = Vector{SVector{3, Float64}}(undef, N)
    for iz in 1:nz, iy in 1:ny, ix in 1:nx
        k = ix + (iy - 1) * nx + (iz - 1) * nx * ny
        all_pts[k] = SVector(xs[ix], ys[iy], zs[iz])
    end

    # n_bnd / N_vol kept only for logging parity with the RBF-FD variant.
    n_bnd = N - (nx - 2) * (ny - 2) * (nz - 2)
    N_vol = N - n_bnd

    println("Grid: $nx × $ny × $nz = $N points ($n_bnd boundary + $N_vol volume)")
    return (; all_pts, N, N_vol, n_bnd, nx, ny, nz, dx)
end

function find_stimulus_region(all_pts)
    return findall(all_pts) do p
        p[1] ≤ stim_size && p[2] ≤ stim_size && p[3] ≤ stim_size
    end
end

"""
    build_fd_diffusion_operator(geo) -> SparseMatrixCSC

Second-order central FD for ∇·(D·∇V) with D = diag(D_L, D_T, D_T) on a uniform
structured grid. No-flux (Neumann) BC via ghost-point mirroring: at a face, the
absent neighbour is replaced by the interior neighbour, which collapses the
half-stencil to `2*D/dx² * (V_interior - V_boundary)`. `sparse(I, J, V)` sums
duplicate `(row, col)` entries so we can emit the ghost contribution as a second
push to the same neighbour index and let assembly do the rest.
"""
function build_fd_diffusion_operator(geo)
    (; nx, ny, nz, dx, N) = geo
    idx(ix, iy, iz) = ix + (iy - 1) * nx + (iz - 1) * nx * ny
    cx = D_L / dx^2
    cy = D_T / dx^2
    cz = D_T / dx^2

    Is = Int[]
    Js = Int[]
    Vs = Float64[]
    sizehint!(Is, 7N)
    sizehint!(Js, 7N)
    sizehint!(Vs, 7N)

    @inline function push_neighbor!(k_center, k_nbr, coef)
        push!(Is, k_center)
        push!(Js, k_nbr)
        push!(Vs, coef)
    end

    for iz in 1:nz, iy in 1:ny, ix in 1:nx
        k = idx(ix, iy, iz)
        diag = 0.0

        # x face (coefficient cx on each neighbour; mirror for Neumann at boundary)
        k_xm = ix > 1 ? idx(ix - 1, iy, iz) : idx(ix + 1, iy, iz)
        k_xp = ix < nx ? idx(ix + 1, iy, iz) : idx(ix - 1, iy, iz)
        push_neighbor!(k, k_xm, cx); diag -= cx
        push_neighbor!(k, k_xp, cx); diag -= cx

        k_ym = iy > 1 ? idx(ix, iy - 1, iz) : idx(ix, iy + 1, iz)
        k_yp = iy < ny ? idx(ix, iy + 1, iz) : idx(ix, iy - 1, iz)
        push_neighbor!(k, k_ym, cy); diag -= cy
        push_neighbor!(k, k_yp, cy); diag -= cy

        k_zm = iz > 1 ? idx(ix, iy, iz - 1) : idx(ix, iy, iz + 1)
        k_zp = iz < nz ? idx(ix, iy, iz + 1) : idx(ix, iy, iz - 1)
        push_neighbor!(k, k_zm, cz); diag -= cz
        push_neighbor!(k, k_zp, cz); diag -= cz

        push_neighbor!(k, k, diag)
    end

    return sparse(Is, Js, Vs, N, N)
end

function setup_problem(geo, stim_ids, W_diff)
    (; N) = geo
    is_stimulated = falses(N)
    is_stimulated[stim_ids] .= true

    function diffusion!(du, u, p, t)
        fill!(du, 0)
        mul!(view(du, 1:N), W_diff, view(u, 1:N))
        return nothing
    end

    function reaction!(du, u, p, t)
        active_stim_dvdt = t < stim_duration ? I_stim / (β * Cm_tissue) : 0.0
        tforeach(1:N) do i
            stim_dvdt = is_stimulated[i] ? active_stim_dvdt : 0.0
            tt06_reaction_point!(du, u, i, N, stim_dvdt)
        end
        return nothing
    end

    f_split = GenericSplitFunction(
        (ODEFunction(diffusion!), ODEFunction(reaction!)),
        (collect(1:N), collect(1:(NSTATES * N))),
    )

    u0 = tt06_initial_conditions(N)
    tspan = (0.0, 80.0)
    return OperatorSplittingProblem(f_split, u0, tspan)
end

function run_simulation(prob, geo; dt = 0.01, sample_interval = 0.1, log_interval = 5.0)
    (; N) = geo

    alg = LieTrotterGodunov((Euler(), Euler()))
    integrator = init(prob, alg; dt = dt, alias_u0 = false, adaptive = false)

    activation_time = fill(NaN, N)
    t_end = prob.tspan[2]
    sample_times = 0.0:sample_interval:t_end
    next_log = log_interval

    println("Solving (dt = $dt ms, t_end = $t_end ms, sampling every $sample_interval ms)...")
    @time for (u, t) in TimeChoiceIterator(integrator, sample_times)
        V = @view u[1:N]
        @inbounds for i in 1:N
            if isnan(activation_time[i]) && V[i] > V_activation
                activation_time[i] = t
            end
        end
        if t ≥ next_log - 1.0e-9
            Vmin, Vmax = extrema(V)
            n_act = count(!isnan, activation_time)
            println(
                "  t = $(round(t; digits = 2)) ms | V ∈ [$(round(Vmin; digits = 1)), $(round(Vmax; digits = 1))] mV | activated: $n_act / $N"
            )
            next_log += log_interval
        end
    end

    return integrator.u, activation_time
end

function nearest_point_index(all_pts, target)
    tx, ty, tz = target
    best_i = 1
    best_d2 = Inf
    @inbounds for i in eachindex(all_pts)
        p = all_pts[i]
        d2 = (p[1] - tx)^2 + (p[2] - ty)^2 + (p[3] - tz)^2
        if d2 < best_d2
            best_d2 = d2
            best_i = i
        end
    end
    return best_i, sqrt(best_d2)
end

function report_reference_points(all_pts, activation_time)
    println()
    println("Activation times at Niederer reference points:")
    println("  Pi   target (mm)           snapped (mm)                dist   AT (ms)")
    for (name, target) in pairs(P_REF)
        idx, dist = nearest_point_index(all_pts, target)
        p = all_pts[idx]
        at = activation_time[idx]
        at_str = isnan(at) ? "not activated" : string(round(at; digits = 2))
        println(
            "  $(rpad(name, 4)) ",
            "($(rpad(round(target[1]; digits = 2), 5)), $(rpad(round(target[2]; digits = 2), 4)), $(rpad(round(target[3]; digits = 2), 4))) ",
            "($(rpad(round(p[1]; digits = 3), 6)), $(rpad(round(p[2]; digits = 3), 5)), $(rpad(round(p[3]; digits = 3), 5))) ",
            "$(rpad(round(dist; digits = 3), 6)) $at_str"
        )
    end
end

function write_diagonal_csv(filename, all_pts, activation_time; n_samples = 50)
    p1 = SVector(P_REF.P1...)
    p8 = SVector(P_REF.P8...)
    total_length = norm(p8 - p1)
    rows = Matrix{Float64}(undef, n_samples, 5)
    for i in 1:n_samples
        s = (i - 1) / (n_samples - 1)
        target = p1 + s * (p8 - p1)
        idx, _ = nearest_point_index(all_pts, Tuple(target))
        p = all_pts[idx]
        rows[i, 1] = s * total_length
        rows[i, 2] = p[1]
        rows[i, 3] = p[2]
        rows[i, 4] = p[3]
        rows[i, 5] = activation_time[idx]
    end
    open(filename, "w") do io
        println(io, "distance_mm,x_mm,y_mm,z_mm,activation_time_ms")
        writedlm(io, rows, ',')
    end
    println("Diagonal P1→P8 activation curve written to $filename")
    return nothing
end

function write_results(filename, geo, u_final, activation_time)
    (; N, nx, ny, nz) = geo
    V_final = u_final[1:N]

    # Natural ordering — direct reshape onto (nx, ny, nz).
    V_grid = reshape(copy(V_final), nx, ny, nz)
    at_grid = reshape(copy(activation_time), nx, ny, nz)

    xs = range(0.0, Lx; length = nx)
    ys = range(0.0, Ly; length = ny)
    zs = range(0.0, Lz; length = nz)

    vtk_grid(filename, collect(xs), collect(ys), collect(zs)) do vtk
        vtk["V"] = V_grid
        vtk["activation_time"] = at_grid
    end
    println("Results written to $filename.vtr")
    return nothing
end

function visualize_geometry(geo; markersize = 0.15)
    (; all_pts) = geo
    on_boundary = map(all_pts) do p
        p[1] == 0.0 || p[1] == Lx || p[2] == 0.0 || p[2] == Ly ||
            p[3] == 0.0 || p[3] == Lz
    end
    bnd = all_pts[on_boundary]
    vol = all_pts[.!on_boundary]

    fig = Figure(; size = (1000, 700))
    ax = Axis3(
        fig[1, 1]; aspect = :data,
        title = "Niederer cuboid FD ($(length(bnd)) boundary + $(length(vol)) interior)",
        xlabel = "x (mm)", ylabel = "y (mm)", zlabel = "z (mm)",
    )

    pb = meshscatter!(ax,
        [p[1] for p in bnd], [p[2] for p in bnd], [p[3] for p in bnd];
        markersize = markersize, color = :red,
    )
    pv = meshscatter!(ax,
        [p[1] for p in vol], [p[2] for p in vol], [p[3] for p in vol];
        markersize = markersize, color = :blue,
    )
    Legend(fig[1, 2], [pb, pv], ["boundary", "interior"])
    return fig
end

function main()
    dx = 0.1  # Niederer reference resolution

    geo = create_geometry(dx)
    (; all_pts, N, N_vol, n_bnd) = geo

    stim_ids = find_stimulus_region(all_pts)
    println("Domain: $N points ($N_vol volume, $n_bnd boundary), $(length(stim_ids)) stimulated")

    fig = visualize_geometry(geo)
    display(fig)
    println("Rotate to inspect the point cloud. Press Enter to continue to simulation setup...")
    readline()

    W_diff = build_fd_diffusion_operator(geo)
    prob = setup_problem(geo, stim_ids, W_diff)

    # Classical FD is stable at dt=0.01 for dx=0.1 (anisotropic CFL ~ 0.04 ms).
    u_final, activation_time = run_simulation(prob, geo; dt = 0.01, log_interval = 5.0)

    report_reference_points(all_pts, activation_time)
    write_diagonal_csv(joinpath(@__DIR__, "niederer_diagonal_fd.csv"), all_pts, activation_time)
    write_results(joinpath(@__DIR__, "niederer_result_fd"), geo, u_final, activation_time)

    return nothing
end

main()
