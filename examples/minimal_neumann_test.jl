# ============================================================================
# Minimal test: Neumann gradient through batch_overwrite_sparse_rows!
# ============================================================================
# Builds the full system from scratch each call. Tests:
#   pts → _build_weights → W_dx, W_dy
#       → batch_overwrite_sparse_rows! → A
#       → PDESolveIFT → u → sum(u.^2)
#
# Fixed adjl (frozen at reference config) keeps sparsity consistent.
# ============================================================================
using Pkg
Pkg.activate(@__DIR__)

using Macchiato, Mooncake, RadialBasisFunctions, StaticArrays, SparseArrays,
      LinearAlgebra, FiniteDifferences
import RadialBasisFunctions: _build_weights, Partial, find_neighbors

N = 25
pts = [SVector{2}(0.1 + 0.8i / 5, 0.1 + 0.8j / 5) for i in 1:5 for j in 1:5]
pts_flat = vcat([collect(p) for p in pts]...)

basis = PHS(3; poly_deg = 3)
k = 20
adjl = find_neighbors(pts, k)

model = LinearElasticity(E = 1.0e7, ν = 0.3)
μ, λstar = lame_parameters(model)

# Left column = Dirichlet (clamped), all other points = Neumann (free)
on_left(p) = p[1] ≈ 0.1
dirichlet_idx = findall(on_left, pts)
neumann_idx = setdiff(1:N, dirichlet_idx)
n_neumann = length(neumann_idx)

neumann_adjl = adjl[neumann_idx]
neumann_rows = vcat(neumann_idx, neumann_idx .+ N)

dirichlet_dofs = Int[]
dirichlet_vals = Float64[]
for gi in dirichlet_idx
    push!(dirichlet_dofs, gi);     push!(dirichlet_vals, 0.01)  # u_x
    push!(dirichlet_dofs, gi + N); push!(dirichlet_vals, 0.0)   # u_y
end

active = trues(2N)
active[dirichlet_dofs] .= false
solver = PDESolveIFT(active)

# Pre-compute Neumann coefficient data (frozen normals = (1,0) for simplicity)
n_total_rows = 2 * n_neumann
n_entries = n_neumann * 4 * k
col_ptr_neumann = zeros(Int, n_total_rows + 1)
a_cols_neumann  = zeros(Int, n_entries)
w_cols_neumann  = zeros(Int, n_entries)
coeffs_neumann  = zeros(Float64, 2 * n_entries)
b_vals_neumann  = zeros(Float64, n_total_rows)
wr_neumann      = zeros(Int, n_total_rows)

let M = 2, ptr = 1
for (li, gi) in enumerate(neumann_idx)
    st = neumann_adjl[li]
    ks = length(st)
    # Row x-eq: [∂u/∂x for u-cols, ∂v/∂y for v-cols]
    r1 = 2li - 1
    wr_neumann[r1] = li
    col_ptr_neumann[r1] = ptr
    for j in 1:ks
        a_cols_neumann[ptr] = st[j];      w_cols_neumann[ptr] = st[j]
        coeffs_neumann[(ptr-1)*M+1] = 1.0; coeffs_neumann[(ptr-1)*M+2] = 0.0
        ptr += 1
        a_cols_neumann[ptr] = st[j] + N;  w_cols_neumann[ptr] = st[j]
        coeffs_neumann[(ptr-1)*M+1] = 0.0; coeffs_neumann[(ptr-1)*M+2] = 1.0
        ptr += 1
    end
    col_ptr_neumann[r1+1] = ptr
    # Row y-eq: [∂u/∂y for u-cols, ∂v/∂x for v-cols]
    r2 = 2li
    wr_neumann[r2] = li
    col_ptr_neumann[r2] = ptr
    for j in 1:ks
        a_cols_neumann[ptr] = st[j];      w_cols_neumann[ptr] = st[j]
        coeffs_neumann[(ptr-1)*M+1] = 0.0; coeffs_neumann[(ptr-1)*M+2] = 1.0
        ptr += 1
        a_cols_neumann[ptr] = st[j] + N;  w_cols_neumann[ptr] = st[j]
        coeffs_neumann[(ptr-1)*M+1] = 1.0; coeffs_neumann[(ptr-1)*M+2] = 0.0
        ptr += 1
    end
    col_ptr_neumann[r2+1] = ptr
end
end

# ============================================================================
# Loss — builds A from scratch, no copy
# ============================================================================

function neumann_loss(pts_in)
    pts_vec = [SVector{2,Float64}(pts_in[2i - 1], pts_in[2i]) for i in 1:N]

    # Build PDE system (differentiable)
    A = make_system_differentiable(model, pts_in, N, adjl, basis, λstar, μ)
    b = zeros(eltype(pts_in), 2N)

    # Dirichlet BCs
    apply_dirichlet!(A, b, dirichlet_dofs, dirichlet_vals)

    # Neumann BCs
    neumann_pts_v = pts_vec[neumann_idx]
    W_dx = _build_weights(Partial(1, 1), pts_vec, neumann_pts_v, neumann_adjl, basis)
    W_dy = _build_weights(Partial(1, 2), pts_vec, neumann_pts_v, neumann_adjl, basis)

    batch_overwrite_sparse_rows!(
        A, b, neumann_rows, col_ptr_neumann, a_cols_neumann, w_cols_neumann,
        SparseMatrixCSC{Float64, Int}[W_dx, W_dy],
        coeffs_neumann, wr_neumann, b_vals_neumann,
    )

    u = solver(A, b)
    return sum(abs2, u)
end

L0 = neumann_loss(pts_flat)
println("Loss: $L0")

# ============================================================================
# AD/FD
# ============================================================================
println("Building Mooncake rrule...")
rule = Mooncake.build_rrule(neumann_loss, pts_flat)
_, (_, grad_ad) = Mooncake.value_and_gradient!!(rule, neumann_loss, pts_flat)
println("AD norm: $(norm(grad_ad))")

println("Computing FD...")
fdm = FiniteDifferences.central_fdm(5, 1)
grad_fd = FiniteDifferences.grad(fdm, neumann_loss, pts_flat)[1]
println("FD norm: $(norm(grad_fd))")

rel_err = norm(grad_ad .- grad_fd) / norm(grad_fd)
println("\nAD/FD rel_err: $(round(rel_err; sigdigits=4))")
println(rel_err < 1e-3 ? "PASS" : "FAIL")
