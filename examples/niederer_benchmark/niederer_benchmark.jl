# Niederer benchmark for cardiac electrophysiology
# Niederer et al. (2011) "Verification of cardiac tissue electrophysiology simulators
# using an N-version benchmark", Phil. Trans. R. Soc. A 369:4331-4351
#
# Solves the monodomain equation on a 3D cuboid using:
#   - RadialBasisFunctions.jl for the meshless anisotropic diffusion operator
#   - OrdinaryDiffEqOperatorSplitting.jl for Lie-Trotter time integration
#   - ten Tusscher-Panfilov 2006 epicardial cell model

using Pkg
Pkg.activate(joinpath(@__DIR__, "../"))

using RadialBasisFunctions
using OrdinaryDiffEqOperatorSplitting
using OrdinaryDiffEqLowOrderRK: Euler, ODEFunction, init, TimeChoiceIterator
using OhMyThreads: tforeach
using StaticArrays
using SparseArrays
using LinearAlgebra
using WriteVTK
using DelimitedFiles
using WhatsThePoint
import Meshes
using Meshes: SimpleMesh, connect, Triangle, Point, refine, TriRefinement
using Unitful: mm, ustrip

include(joinpath(@__DIR__, "ten_tusscher_2006.jl"))

## --- Physical parameters (Niederer 2011, Table 3) ---
const Lx, Ly, Lz = 20.0, 7.0, 3.0   # domain size (mm); fibers along x

# Intra-/extra-cellular conductivities, σ in S/m == mS/mm (Table 3)
const σ_iL, σ_iT = 0.17, 0.019
const σ_eL, σ_eT = 0.62, 0.24
# Monodomain conductivities: σ = σ_i σ_e / (σ_i + σ_e)
const σ_L = σ_iL * σ_eL / (σ_iL + σ_eL)   # ≈ 0.13342 mS/mm
const σ_T = σ_iT * σ_eT / (σ_iT + σ_eT)   # ≈ 0.01761 mS/mm

const β = 140.0                       # surface-to-volume ratio (1/mm)
const Cm_tissue = 0.01                # membrane capacitance (µF/mm²)
const D_L = σ_L / (β * Cm_tissue)     # diffusion along fiber (mm²/ms)
const D_T = σ_T / (β * Cm_tissue)     # diffusion transverse (mm²/ms)

# Stimulus (Table 3): 50 000 µA/cm³ for 2 ms in a 1.5 mm cube at the corner.
# Converted to µA/mm³ so units line up with β [1/mm] and Cm [µF/mm²]:
# stim_dvdt = I_stim / (β Cm) = 50 / 1.4 ≈ 35.71 mV/ms.
const I_stim = 50.0                   # stimulus amplitude (µA/mm³ = 50 000 µA/cm³)
const stim_duration = 2.0             # stimulus duration (ms)
const stim_size = 1.5                 # stimulus cube side length (mm)

# Activation threshold per paper §3(f): first crossing of V through 0 mV.
const V_activation = 0.0

# Niederer benchmark reference points (Figure 1b). 8 corners + centre.
const P_REF = (
    P1 = (0.0, 0.0, 0.0),    # stimulus corner
    P2 = (0.0, 7.0, 0.0),
    P3 = (20.0, 7.0, 0.0),
    P4 = (20.0, 0.0, 0.0),
    P5 = (0.0, 0.0, 3.0),
    P6 = (0.0, 7.0, 3.0),
    P7 = (20.0, 0.0, 3.0),
    P8 = (20.0, 7.0, 3.0),   # far corner, last to activate
    P9 = (10.0, 3.5, 1.5),   # centre
)

##

function extract_coords(ptcloud)
    return map(WhatsThePoint.points(ptcloud)) do p
        c = Meshes.coords(p)
        SVector(ustrip(mm, c.x), ustrip(mm, c.y), ustrip(mm, c.z))
    end
end

"""
    create_geometry(dx) -> NamedTuple

Generate a scattered point cloud over the Niederer cuboid with target spacing
`dx` mm using WhatsThePoint's Van der Sande-Fornberg (VdSF) algorithm. Boundary
points come from the centroids of a triangulated cuboid surface refined so each
edge is ≲ dx; interior points are quasi-uniform scattered nodes packed by VdSF.

Scattered nodes avoid the symmetric/degenerate stencils a regular lattice gives
the PHS + polynomial RBF-FD basis. Returns `(; all_pts, vol_pts, N, N_vol,
n_bnd, vol_ids, dx)` with boundary points first (indices 1:n_bnd) and interior
points second.
"""
function create_geometry(dx)
    # 1. Triangulated cuboid surface (12 triangles, 2 per face). Vertices are
    # tagged with mm so the default-meters convention in Meshes.jl doesn't
    # clash with the ConstantSpacing(dx * mm) passed below — VdSF's internal
    # projection grid sizes off extent/spacing and blows up on a unit mismatch.
    vertices = Point.(
        [
            (0.0mm, 0.0mm, 0.0mm), (Lx * mm, 0.0mm, 0.0mm), (Lx * mm, Ly * mm, 0.0mm), (0.0mm, Ly * mm, 0.0mm),
            (0.0mm, 0.0mm, Lz * mm), (Lx * mm, 0.0mm, Lz * mm), (Lx * mm, Ly * mm, Lz * mm), (0.0mm, Ly * mm, Lz * mm),
        ]
    )
    triangles = [
        connect((1, 3, 2), Triangle), connect((1, 4, 3), Triangle),
        connect((5, 6, 7), Triangle), connect((5, 7, 8), Triangle),
        connect((1, 2, 6), Triangle), connect((1, 6, 5), Triangle),
        connect((3, 4, 8), Triangle), connect((3, 8, 7), Triangle),
        connect((1, 5, 8), Triangle), connect((1, 8, 4), Triangle),
        connect((2, 3, 7), Triangle), connect((2, 7, 6), Triangle),
    ]
    mesh = SimpleMesh(vertices, triangles)

    # 2. Refine until the mean boundary triangle edge ≈ dx so boundary
    # centroid spacing matches the target interior spacing. Using one fewer
    # round than `ceil(log2(max_edge / dx))` keeps boundary density from
    # dominating interior density; otherwise boundary-point stencils pick up
    # 40 near-coplanar neighbors and the quadratic polynomial augmentation
    # goes rank-deficient on the xy/xz/yz cross-terms.
    max_edge = sqrt(Lx^2 + Ly^2 + Lz^2)
    n_refine = max(ceil(Int, log2(max_edge / dx)) - 1, 1)
    for _ in 1:n_refine
        mesh = refine(mesh, TriRefinement())
    end
    println("Refined boundary mesh: $(length(collect(Meshes.elements(mesh)))) triangles")

    # 3. VdSF scattered volume fill at target spacing.
    #
    # NOT Orthtree, despite it being the preferred 3D algorithm elsewhere: Orthtree
    # emits its volume points in metres regardless of the mesh's units, so on the
    # mm-tagged geometry below `PointCloud` rejects the mm boundary / m volume
    # pair ("boundary and volume CRS still differ"). Revisit once WhatsThePoint
    # carries the mesh unit through the octree fill.
    boundary = PointBoundary(mesh)
    cloud = WhatsThePoint.discretize(boundary, ConstantSpacing(dx * mm); alg = VanDerSandeFornberg())

    # 4. Extract coordinates in raw mm. Drop out-of-domain points (rare) and
    # a thin z ≈ 0 sliver where VdSF's initial height field places spurious
    # points right on top of boundary centroids (would cause near-duplicate
    # stencil nodes). The sliver tolerance is small (~dx/20) so interior
    # density is preserved.
    bnd_pts = extract_coords(WhatsThePoint.boundary(cloud))
    vol_pts_raw = extract_coords(cloud.volume)
    tol = dx / 20
    vol_pts = filter(vol_pts_raw) do p
        tol ≤ p[1] ≤ Lx - tol && tol ≤ p[2] ≤ Ly - tol && tol ≤ p[3] ≤ Lz - tol
    end

    # Defensive de-edging: the downstream BC code assumes each boundary point
    # has a single unambiguous face. Triangle centroids are strictly interior to
    # a triangle (hence to one cuboid face), so this should be a no-op in practice.
    moved = dedge_boundary_points!(bnd_pts)
    moved > 0 && println("De-edged $moved boundary point(s) that straddled multiple faces")

    all_pts = vcat(bnd_pts, vol_pts)
    N = length(all_pts)
    N_vol = length(vol_pts)
    n_bnd = length(bnd_pts)
    vol_ids = (n_bnd + 1):N
    dropped = length(vol_pts_raw) - N_vol
    println("VdSF cloud: $N points ($n_bnd boundary + $N_vol volume, $dropped out-of-domain dropped)")

    return (; all_pts, vol_pts, N, N_vol, n_bnd, vol_ids, dx)
end

"""
    dedge_boundary_points!(bnd_pts; tol = 1e-3) -> Int

If a boundary point has ≥ 2 coordinates within `tol` mm of a cuboid face,
keep the coordinate nearest an extreme on its chosen face and push the
others `2·tol` inward so the point ends up unambiguously on a single face.
Returns the number of points moved.
"""
function dedge_boundary_points!(bnd_pts; tol = 1.0e-3)
    moved = 0
    for i in eachindex(bnd_pts)
        p = bnd_pts[i]
        dists = (p[1], Lx - p[1], p[2], Ly - p[2], p[3], Lz - p[3])
        on_face = map(d -> d < tol, dists)
        count(on_face) ≤ 1 && continue

        closest = argmin(dists)
        kept_axis = (closest - 1) ÷ 2 + 1  # 1,2→x ; 3,4→y ; 5,6→z
        x, y, z = p[1], p[2], p[3]
        if kept_axis != 1
            x < tol && (x = 2 * tol)
            x > Lx - tol && (x = Lx - 2 * tol)
        end
        if kept_axis != 2
            y < tol && (y = 2 * tol)
            y > Ly - tol && (y = Ly - 2 * tol)
        end
        if kept_axis != 3
            z < tol && (z = 2 * tol)
            z > Lz - tol && (z = Lz - 2 * tol)
        end
        bnd_pts[i] = SVector(x, y, z)
        moved += 1
    end
    return moved
end

function find_stimulus_region(all_pts)
    ids = findall(all_pts) do p
        p[1] ≤ stim_size && p[2] ≤ stim_size && p[3] ≤ stim_size
    end
    isempty(ids) && throw(ArgumentError("no points fell inside the $(stim_size) mm stimulus cube; reduce dx"))
    return ids
end

"""
    boundary_normals(bnd_pts; tol = 1e-3) -> Vector{SVector{3, Float64}}

Return the outward unit normal for each boundary point. Asserts each point
is on exactly one of the six cuboid faces — run `dedge_boundary_points!`
first so that edge/corner straddlers get a single unambiguous face.
"""
function boundary_normals(bnd_pts; tol = 1.0e-3)
    n = length(bnd_pts)
    normals = Vector{SVector{3, Float64}}(undef, n)
    for i in 1:n
        p = bnd_pts[i]
        flags = (
            p[1] < tol, Lx - p[1] < tol,
            p[2] < tol, Ly - p[2] < tol,
            p[3] < tol, Lz - p[3] < tol,
        )
        count(flags) == 1 || throw(
            ArgumentError("boundary point $i at $p touches $(count(flags)) faces, expected 1")
        )
        if flags[1]
            normals[i] = SVector(-1.0, 0.0, 0.0)
        elseif flags[2]
            normals[i] = SVector(1.0, 0.0, 0.0)
        elseif flags[3]
            normals[i] = SVector(0.0, -1.0, 0.0)
        elseif flags[4]
            normals[i] = SVector(0.0, 1.0, 0.0)
        elseif flags[5]
            normals[i] = SVector(0.0, 0.0, -1.0)
        else
            normals[i] = SVector(0.0, 0.0, 1.0)
        end
    end
    return normals
end

"""
    build_shadow_indices(all_pts, bnd_pts, normals, dx, n_bnd) -> (shadow_ids, shadow_dists)

For each boundary point `i`, find the nearest *interior* cloud point to the
shadow position `x_i - dx * n_i` (one spacing inward along the inward normal).
Returns the global index `shadow_ids[i]` into `all_pts` and the actual
distance `shadow_dists[i]` from boundary point to its shadow.
"""
function build_shadow_indices(all_pts, bnd_pts, normals, dx, n_bnd)
    n = length(bnd_pts)
    shadow_ids = Vector{Int}(undef, n)
    shadow_dists = Vector{Float64}(undef, n)
    N = length(all_pts)
    for i in 1:n
        target = bnd_pts[i] - dx * normals[i]
        best_j = 0
        best_d2 = Inf
        @inbounds for j in (n_bnd + 1):N
            pj = all_pts[j]
            d2 = (pj[1] - target[1])^2 + (pj[2] - target[2])^2 + (pj[3] - target[3])^2
            if d2 < best_d2
                best_d2 = d2
                best_j = j
            end
        end
        shadow_ids[i] = best_j
        shadow_dists[i] = sqrt(best_d2)
    end
    return shadow_ids, shadow_dists
end

"""
    build_ghost_points(bnd_pts, normals, dx) -> Vector{SVector{3, Float64}}

Place one ghost node per boundary point, offset outward by `dx` along the
outward normal: `g_i = x_i + dx * n_i`. The ghost sits outside the domain
and has no independent DOF — its value is tied to the corresponding
`shadow_ids[i]` interior node via the 1st-order zero-Neumann identity
`V_ghost = V_shadow` (see `build_tying_matrix`).
"""
function build_ghost_points(bnd_pts, normals, dx)
    return [bnd_pts[i] + dx * normals[i] for i in eachindex(bnd_pts)]
end

"""
    build_tying_matrix(N, n_bnd, shadow_ids) -> SparseMatrixCSC

Sparse `(N + n_bnd) × N` matrix that maps the extended state (original DOFs
plus ghost DOFs) to the reduced state (original DOFs only) under the tying
`V_ghost_j = V_shadow_ids[j]`:

- `T[i, i] = 1` for `i ∈ 1..N` (identity on original DOFs)
- `T[N + j, shadow_ids[j]] = 1` for `j ∈ 1..n_bnd` (ghost j → its shadow col)

Applying `W_ext * T` folds ghost columns of the extended Laplacian back into
the shadow columns — the scattered-RBF analogue of FD's ghost-point mirror.
"""
function build_tying_matrix(N, n_bnd, shadow_ids)
    rows = Vector{Int}(undef, N + n_bnd)
    cols = Vector{Int}(undef, N + n_bnd)
    vals = ones(Float64, N + n_bnd)
    @inbounds for i in 1:N
        rows[i] = i
        cols[i] = i
    end
    @inbounds for j in 1:n_bnd
        rows[N + j] = N + j
        cols[N + j] = shadow_ids[j]
    end
    return sparse(rows, cols, vals, N + n_bnd, N)
end

"""
    build_diffusion_operator(all_pts, ghost_pts, n_bnd, shadow_ids)

Build the anisotropic monodomain diffusion operator `∇ · (D ∇V)` with
`D = diag(D_L, D_T, D_T)` (fibers along x) on the extended cloud
`data_pts = vcat(all_pts, ghost_pts)`, evaluated at `all_pts` only. Then fold
ghost columns back to shadow columns via `W_bc = W_ext * T`, giving an
`N × N` operator whose boundary rows are **proper Laplacian rows seeing a
mirror-extended V field** — the scattered-RBF equivalent of FD's half-stencil
`(2/dx²)·(V₁ − V₀)` at a no-flux wall.

`k = 60` is deliberately larger than the k≈30 often quoted for 3D RBF-FD:
on a scattered cloud, boundary-point stencils with only 40 neighbors tend
to land entirely on a single face (quadratic cross-terms xy/xz/yz go
near-rank-deficient → stencil weights blow up to 10^7). Bumping k to 60
forces each stencil to reach into the interior, cutting the worst weight
by ~5 orders of magnitude.
"""
function build_diffusion_operator(all_pts, ghost_pts, n_bnd, shadow_ids; k = 60, poly_deg = 2)
    N = length(all_pts)
    D_vec = SVector(D_L, D_T, D_T)
    basis = PHS(3; poly_deg = poly_deg)
    data_pts = vcat(all_pts, ghost_pts)
    diff_op = custom(
        data_pts, @operator(∇ ⋅ (D_vec * ∇));
        eval_points = all_pts, k = k, basis = basis,
    )
    W_ext = diff_op.weights                           # N × (N + n_bnd)
    T = build_tying_matrix(N, n_bnd, shadow_ids)      # (N + n_bnd) × N
    W_bc = W_ext * T                                  # N × N
    row_sum_err = maximum(abs, vec(sum(W_bc; dims = 2)))
    max_w = maximum(abs, nonzeros(W_bc))
    println("W_diff_bc: size=$(size(W_bc)), nnz=$(nnz(W_bc)), max |row-sum|=$(row_sum_err), max |W|=$(round(max_w; digits = 1))")
    return W_bc
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
    tspan = (0.0, 80.0)  # enough for all 9 reference points to activate at dx=0.5
    return OperatorSplittingProblem(f_split, u0, tspan)
end

function run_simulation(prob, geo; dt = 0.05, sample_interval = 0.1, log_interval = 1.0)
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

"""
    nearest_point_index(all_pts, target) -> Int

Linear-search nearest neighbor of `target` in `all_pts`. O(N) per query; fine for
the handful of benchmark points we query.
"""
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
    return
end

function write_diagonal_csv(filename, all_pts, activation_time; n_samples = 50)
    p1 = SVector(P_REF.P1...)
    p8 = SVector(P_REF.P8...)
    total_length = norm(p8 - p1)
    rows = Matrix{Float64}(undef, n_samples, 5)  # distance, x, y, z, AT
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
    (; all_pts, N) = geo
    V_final = u_final[1:N]

    # Unstructured point-cloud VTK: one VTK_VERTEX cell per point. Works for any
    # scattered node layout; Paraview/VisIt render these as point glyphs.
    points_matrix = reduce(hcat, all_pts)  # 3 × N
    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in 1:N]

    vtk_grid(filename, points_matrix, cells) do vtk
        vtk["V"] = V_final
        vtk["activation_time"] = activation_time
    end
    println("Results written to $filename.vtu")
    return nothing
end

function main()
    # dx=0.5 is the shipping configuration: stable, activation times within
    # ~2 ms of FD on the 8 boundary reference points, 80 ms simulation runs
    # in ~17 s. Refining to dx=0.25 tightens P4 but exposes a plateau-phase
    # V drift (stencil outliers + explicit-Euler cumulative error over
    # 160 000 steps). Reaching Niederer's dx=0.1 with explicit time stepping
    # would take ~20 hours per run; it needs semi-implicit diffusion, not
    # available in this iteration.
    dx = 0.5

    geo = create_geometry(dx)
    (; all_pts, N, N_vol, n_bnd) = geo

    stim_ids = find_stimulus_region(all_pts)
    println("Domain: $N points ($N_vol volume, $n_bnd boundary), $(length(stim_ids)) stimulated")

    bnd_pts = view(all_pts, 1:n_bnd)
    normals = boundary_normals(bnd_pts)
    shadow_ids, shadow_dists = build_shadow_indices(all_pts, bnd_pts, normals, dx, n_bnd)
    println(
        "Shadow distances (mm): min=", round(minimum(shadow_dists); digits = 3),
        " max=", round(maximum(shadow_dists); digits = 3),
        " mean=", round(sum(shadow_dists) / length(shadow_dists); digits = 3),
    )

    ghost_pts = build_ghost_points(bnd_pts, normals, dx)
    println("Ghost nodes: $(length(ghost_pts)) (one per boundary, outward by dx=$dx mm)")

    W_diff = build_diffusion_operator(all_pts, ghost_pts, n_bnd, shadow_ids)
    prob = setup_problem(geo, stim_ids, W_diff)

    # Explicit-Euler CFL: dt * max|W| must stay ≲ 2. At dx=0.5, max|W|≈195,
    # so dt=0.005 leaves plenty of margin. Scaling is roughly 1/dx².
    u_final, activation_time = run_simulation(prob, geo; dt = 0.005, log_interval = 5.0)

    report_reference_points(all_pts, activation_time)
    write_diagonal_csv(joinpath(@__DIR__, "niederer_diagonal.csv"), all_pts, activation_time)
    write_results(joinpath(@__DIR__, "niederer_result"), geo, u_final, activation_time)

    return nothing
end

main()
