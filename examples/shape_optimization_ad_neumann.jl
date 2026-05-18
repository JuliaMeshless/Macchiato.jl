# ============================================================================
# Phase 2: Neumann BC Differentiation — AD/FD Validation
# ============================================================================
# Validates the generic differentiable Neumann BC chain:
#   pts → _build_weights(Partial(1,d)) → W_dx, W_dy
#       → model callback (computes coefficients)
#       → batch_overwrite_sparse_rows! (generic primitive)
#       → PDESolveIFT → u → compute_von_mises → L
#
# Architecture: the primitive is model-agnostic — it takes weight matrices +
# coefficients. The elasticity-specific physics lives in `build_traction_data!`,
# which runs inside the Mooncake trace and is automatically differentiable.
#
# Setup: cantilever beam with mixed BCs (same as cantilever_beam_2d.jl)
#   - Left (x=0):   Displacement (Dirichlet)
#   - Right (x=L):  Traction (Neumann, parabolic shear)
#   - Top/bottom:   TractionFree (Neumann, zero traction)
#
# Level 1: normals frozen at reference config.
# ============================================================================
using Pkg
Pkg.activate(@__DIR__)

using Macchiato
import Macchiato as MM
using WhatsThePoint
import WhatsThePoint as WTP
using RadialBasisFunctions: PHS, find_neighbors
import RadialBasisFunctions: _build_weights, Partial, MixedPartial
using Mooncake
using StaticArrays
using SparseArrays
using LinearAlgebra
using FiniteDifferences
using Unitful: m, °, ustrip

# ============================================================================
# Parameters
# ============================================================================

L     = 8.0
D     = 1.0
P     = 1000.0
E_val = 1.0e7
ν_val = 0.3
I_val = 2D^3 / 3

model = LinearElasticity(E = E_val, ν = ν_val)
μ, λstar = lame_parameters(model)

u_exact(x, y) = -P / (6E_val * I_val) * y * ((6L - 3x) * x + (2 + ν_val) * (y^2 - D^2))
v_exact(x, y) =  P / (6E_val * I_val) * (3ν_val * y^2 * (L - x) + (4 + 5ν_val) * D^2 * x + (3L - x) * x^2)

# ============================================================================
# Domain Setup
# ============================================================================

dx = 0.5m

rx = dx:dx:((L * m) - dx)
ry = dx:dx:((2D * m) - dx)

p_bot   = [WTP.Point(i, -D * m) for i in rx]
n_bot   = [WTP.Vec(0.0, -1.0) for _ in rx]
p_right = [WTP.Point(L * m, -D * m + i) for i in ry]
n_right = [WTP.Vec(1.0, 0.0) for _ in ry]
p_top   = [WTP.Point(i, D * m) for i in reverse(rx)]
n_top   = [WTP.Vec(0.0, 1.0) for _ in rx]
p_left  = [WTP.Point(0.0m, -D * m + i) for i in reverse(ry)]
n_left  = [WTP.Vec(-1.0, 0.0) for _ in ry]

pts  = vcat(p_bot, p_right, p_top, p_left)
nrms = vcat(n_bot, n_right, n_top, n_left)
areas = fill(dx, length(pts))

part = PointBoundary(pts, nrms, areas)
split_surface!(part, 75°)
cloud = WTP.discretize(part, ConstantSpacing(dx), alg = VanDerSandeFornberg())

bc_right_fn(x, t) = (0.0, P * (D^2 - x[2]^2) / (2I_val))
bc_left_fn(x, t)  = (u_exact(x[1], x[2]), v_exact(x[1], x[2]))

bcs = Dict(
    :surface1 => TractionFree(),
    :surface2 => Traction(bc_right_fn),
    :surface3 => TractionFree(),
    :surface4 => Displacement(bc_left_fn),
)

domain = MM.Domain(cloud, bcs, model)
N = length(cloud)
println("Points: $N  (DOFs: $(2N))")

# ============================================================================
# Extract AD setup from domain
# ============================================================================

basis = PHS(3; poly_deg = 3)
k = 35

coords = MM._ustrip(MM._coords(cloud))
pts_flat = vcat([collect(p) for p in coords]...)

adjl_all = find_neighbors(coords, k)
active = active_dofs(domain)
dirichlet_dofs, dirichlet_vals = build_dirichlet_info(domain)

# Separate Neumann boundary data (pre-computed, frozen at reference config)
neumann_ids_all = Int[]
neumann_nx_all  = Float64[]
neumann_ny_all  = Float64[]
neumann_tx_all  = Float64[]
neumann_ty_all  = Float64[]
neumann_surf_names = Symbol[]

for (surf_name, (ids, bc)) in domain.boundaries
    if bc isa Macchiato.Dirichlet
        continue
    end
    push!(neumann_surf_names, surf_name)
    surf = domain.cloud[surf_name]
    normals_surf = ustrip.(normal(surf))
    for (local_i, global_i) in enumerate(ids)
        push!(neumann_ids_all, global_i)
        push!(neumann_nx_all, normals_surf[local_i][1])
        push!(neumann_ny_all, normals_surf[local_i][2])
        x_coord = MM.get_node_coords(surf, local_i)
        tx, ty = bc(x_coord, 0.0)
        push!(neumann_tx_all, tx)
        push!(neumann_ty_all, ty)
    end
end

# Neumann adjacency: reuse adjl_all for sparsity-pattern consistency
neumann_adjl = adjl_all[neumann_ids_all]

n_neumann = length(neumann_ids_all)
println("Dirichlet DOFs: $(length(dirichlet_dofs))  Neumann pts: $n_neumann")
println("active DOFs: $(count(active)) / $(length(active))")

# ============================================================================
# Model-specific coefficient builder (runs inside Mooncake trace)
# ============================================================================
# Computes the (col_ptr, cols, coeffs, weight_rows, b_vals) arrays for
# batch_overwrite_sparse_rows!. This is plain Julia — no primitives needed.
# Mooncake traces through it automatically.

function build_traction_data!(
    col_ptr::Vector{Int},
    a_cols::Vector{Int},
    w_cols::Vector{Int},
    coeffs::Vector{Float64},
    b_vals::Vector{Float64},
    rows::Vector{Int},
    weight_rows::Vector{Int},
    adjl_neumann::Vector{Vector{Int}},
    normal_nx::Vector{Float64},
    normal_ny::Vector{Float64},
    λstar::Float64,
    μ_lame::Float64,
    tx_vals::Vector{Float64},
    ty_vals::Vector{Float64},
    N_total::Int,
)
    n_pts = length(adjl_neumann)
    M = 2  # W_dx, W_dy

    ptr = 1
    for local_i in 1:n_pts
        stencil = adjl_neumann[local_i]
        k_sten = length(stencil)
        nx = normal_nx[local_i]
        ny = normal_ny[local_i]

        # Row 1: x-equation
        r1 = 2 * local_i - 1
        weight_rows[r1] = local_i
        b_vals[r1] = tx_vals[local_i]
        col_ptr[r1] = ptr
        for j in 1:k_sten
            # u_x column: A column = stencil[j], weight column = stencil[j]
            a_cols[ptr] = stencil[j]
            w_cols[ptr] = stencil[j]
            coeffs[(ptr - 1) * M + 1] = nx * (λstar + 2μ_lame)
            coeffs[(ptr - 1) * M + 2] = ny * μ_lame
            ptr += 1
            # v column: A column = stencil[j] + N, weight column = stencil[j]
            a_cols[ptr] = stencil[j] + N_total
            w_cols[ptr] = stencil[j]
            coeffs[(ptr - 1) * M + 1] = ny * μ_lame
            coeffs[(ptr - 1) * M + 2] = nx * λstar
            ptr += 1
        end
        col_ptr[r1 + 1] = ptr

        # Row 2: y-equation
        r2 = 2 * local_i
        weight_rows[r2] = local_i
        b_vals[r2] = ty_vals[local_i]
        col_ptr[r2] = ptr
        for j in 1:k_sten
            a_cols[ptr] = stencil[j]
            w_cols[ptr] = stencil[j]
            coeffs[(ptr - 1) * M + 1] = ny * λstar
            coeffs[(ptr - 1) * M + 2] = nx * μ_lame
            ptr += 1
            a_cols[ptr] = stencil[j] + N_total
            w_cols[ptr] = stencil[j]
            coeffs[(ptr - 1) * M + 1] = nx * μ_lame
            coeffs[(ptr - 1) * M + 2] = ny * (λstar + 2μ_lame)
            ptr += 1
        end
        col_ptr[r2 + 1] = ptr
    end
    return nothing
end

# ============================================================================
# Sanity check: differentiable assembly vs mainstream
# ============================================================================

println("\n--- Differentiable assembly vs mainstream ---")

# Reference via mainstream API
A_ref, b_ref = make_system(model, domain; k = k, basis = basis)
for (surf_name, (ids, bc)) in domain.boundaries
    surf = domain.cloud[surf_name]
    MM.make_bc!(A_ref, b_ref, bc, surf, domain, ids; λstar = λstar, μ_lame = μ, k = k, basis = basis)
end
u_ref = lu(A_ref) \ b_ref

ux_ana = [u_exact(p[1], p[2]) for p in coords]
uy_ana = [v_exact(p[1], p[2]) for p in coords]
err_ux_ref = norm(u_ref[1:N] - ux_ana) / norm(ux_ana)
err_uy_ref = norm(u_ref[(N+1):(2N)] - uy_ana) / norm(uy_ana)
println("  Mainstream solve: err_ux = $err_ux_ref, err_uy = $err_uy_ref")

# Differentiable path
neumann_boundary_pts = coords[neumann_ids_all]
neumann_adjl_check = find_neighbors(coords, neumann_boundary_pts, k)

A_diff = make_system_differentiable(model, pts_flat, N, adjl_all, basis, λstar, μ)
b_diff = zeros(2N)
apply_dirichlet!(A_diff, b_diff, dirichlet_dofs, dirichlet_vals)

W_dx = _build_weights(Partial(1, 1), coords, neumann_boundary_pts, neumann_adjl_check, basis)
W_dy = _build_weights(Partial(1, 2), coords, neumann_boundary_pts, neumann_adjl_check, basis)

# Build generic assembly data
n_neumann_rows = 2 * n_neumann
neumann_rows = vcat(neumann_ids_all, neumann_ids_all .+ N)
n_total_entries = n_neumann * 4 * maximum(length, neumann_adjl_check)
col_ptr = zeros(Int, n_neumann_rows + 1)
a_cols  = zeros(Int, n_total_entries)
w_cols  = zeros(Int, n_total_entries)
coeffs_flat = zeros(Float64, 2 * n_total_entries)  # M=2 weights
b_vals_neumann = zeros(Float64, n_neumann_rows)
weight_rows_neumann = zeros(Int, n_neumann_rows)

build_traction_data!(
    col_ptr, a_cols, w_cols, coeffs_flat, b_vals_neumann,
    neumann_rows, weight_rows_neumann,
    neumann_adjl_check,
    neumann_nx_all, neumann_ny_all,
    λstar, μ,
    neumann_tx_all, neumann_ty_all, N,
)

batch_overwrite_sparse_rows!(
    A_diff, b_diff,
    neumann_rows, col_ptr, a_cols, w_cols,
    SparseMatrixCSC{Float64, Int}[W_dx, W_dy],
    coeffs_flat, weight_rows_neumann, b_vals_neumann,
)

solver_ref = PDESolveIFT(active)
u_diff = solver_ref(A_diff, b_diff)

err_u = norm(u_diff - u_ref) / norm(u_ref)
println("  ‖u_diff - u_ref‖ / ‖u_ref‖ = $err_u")
if err_u < 1e-8
    println("  PASS")
else
    @warn "  mismatch"
end

# ============================================================================
# AD/FD validation
# ============================================================================

println("\n--- AD/FD validation ---")

# Probe: von Mises at interior points
interior_coords = MM._coords(cloud.volume)
interior_N = length(interior_coords)
probe_local_idx = 1:min(3, interior_N)
vol_start = sum(length(domain.cloud[sn]) for sn in WTP.names(cloud.boundary))
probe_idx = (vol_start) .+ (probe_local_idx .- 1)
probe_idx = probe_idx[probe_idx .<= N]

solver = PDESolveIFT(active)

function loss_closure(pts_in)
    pts_vec = [SVector{2,Float64}(pts_in[2i - 1], pts_in[2i]) for i in 1:N]

    # PDE system
    A = make_system_differentiable(model, pts_in, N, adjl_all, basis, λstar, μ)
    b = zeros(eltype(pts_in), 2N)

    # Dirichlet BCs
    apply_dirichlet!(A, b, dirichlet_dofs, dirichlet_vals)

    # Neumann BCs: build derivative weights
    neumann_pts = pts_vec[neumann_ids_all]
    neumann_adjl_dyn = find_neighbors(pts_vec, neumann_pts, k)

    W_dx_dyn = _build_weights(Partial(1, 1), pts_vec, neumann_pts, neumann_adjl_dyn, basis)
    W_dy_dyn = _build_weights(Partial(1, 2), pts_vec, neumann_pts, neumann_adjl_dyn, basis)

    # Build generic assembly data (model-specific, runs inside trace)
    n_neumann_rows_l = 2 * n_neumann
    neumann_rows_l = vcat(neumann_ids_all, neumann_ids_all .+ N)
    n_entries_l = n_neumann * 4 * maximum(length, neumann_adjl_dyn)
    col_ptr_l = zeros(Int, n_neumann_rows_l + 1)
    a_cols_l  = zeros(Int, n_entries_l)
    w_cols_l  = zeros(Int, n_entries_l)
    coeffs_l  = zeros(Float64, 2 * n_entries_l)
    b_vals_l  = zeros(Float64, n_neumann_rows_l)
    wr_l      = zeros(Int, n_neumann_rows_l)

    build_traction_data!(
        col_ptr_l, a_cols_l, w_cols_l, coeffs_l, b_vals_l,
        neumann_rows_l, wr_l,
        neumann_adjl_dyn,
        neumann_nx_all, neumann_ny_all,
        λstar, μ,
        neumann_tx_all, neumann_ty_all, N,
    )

    # Generic primitive (model-agnostic)
    batch_overwrite_sparse_rows!(
        A, b,
        neumann_rows_l, col_ptr_l, a_cols_l, w_cols_l,
        SparseMatrixCSC{Float64, Int}[W_dx_dyn, W_dy_dyn],
        coeffs_l, wr_l, b_vals_l,
    )

    # Solve
    u = solver(A, b)
    σ_vm = compute_von_mises(u, N, pts_in, adjl_all, basis, λstar, μ)
    return sum(σ_vm[probe_idx] .^ 2)
end

L0 = loss_closure(pts_flat)
println("  Loss at reference config: $L0")

# AD gradient
println("  Building Mooncake rrule...")
rule = Mooncake.build_rrule(loss_closure, pts_flat)
_, (_, grad_ad) = Mooncake.value_and_gradient!!(rule, loss_closure, pts_flat)
println("  done  (norm = $(norm(grad_ad)))")

# FD gradient
println("  Computing FD gradient...")
fdm = FiniteDifferences.central_fdm(5, 1)
grad_fd = FiniteDifferences.grad(fdm, loss_closure, pts_flat)[1]
println("  done  (norm = $(norm(grad_fd)))")

# ============================================================================
# Validation
# ============================================================================

rel_err = norm(grad_ad .- grad_fd) / norm(grad_fd)

println("\n========================================")
println("Phase 2: Neumann AD Validation")
println("========================================")
println("N=$N  k=$k  poly_deg=3")
println("Dirichlet DOFs: $(length(dirichlet_dofs))  Neumann pts: $n_neumann")
println("AD/FD relative error: $(round(rel_err; sigdigits=4))")
if rel_err < 1e-3
    println("PASS (rtol < 1e-3)")
else
    @warn "FAIL: rtol = $rel_err"
end
