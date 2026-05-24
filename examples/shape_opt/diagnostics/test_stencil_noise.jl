# Test: does increasing k reduce boundary gradient noise on 17x9 grid?
using Macchiato, StaticArrays, LinearAlgebra, SparseArrays, Printf
using RadialBasisFunctions
import RadialBasisFunctions: _build_weights, Partial, MixedPartial, find_neighbors

const L_val = 8.0; const D_val = 1.0; const P = 1000.0
const I_val = 2 * D_val^3 / 3
const nx, ny = 17, 9
xs = range(0.0, L_val; length = nx)
ys = range(-D_val, D_val; length = ny)
points0 = [SVector{2, Float64}(x, y) for x in xs for y in ys]
const N = length(points0)
model = LinearElasticity(E = 1e7, ν = 0.3)
μ, λstar = Macchiato.lame_parameters(model)
basis = PHS(3; poly_deg = 3)

is_left(p)   = p[1] ≈ 0.0
is_right(p)  = p[1] ≈ L_val
is_top(p)    = p[2] ≈ D_val && !is_left(p)
is_bottom(p) = p[2] ≈ -D_val && !is_left(p)

dirichlet_idx = findall(is_left, points0)
neumann_idx   = findall(p -> !is_left(p) && (is_right(p) || is_top(p) || is_bottom(p)), points0)
top_idx       = findall(is_top, points0)
bottom_idx    = findall(is_bottom, points0)
right_idx     = findall(is_right, points0)

dirichlet_dofs = vcat(dirichlet_idx, dirichlet_idx .+ N)
dirichlet_vals = zeros(2 * length(dirichlet_idx))

active = trues(2N)
for i in dirichlet_idx; active[i] = false; active[i + N] = false; end

interior_idx = findall(p -> !is_left(p) && !is_right(p) && !is_top(p) && !is_bottom(p), points0)
interior_rows = falses(N)
for i in interior_idx; interior_rows[i] = true; end

# Neumann layout
neumann_normals = [
    is_right(points0[i])  ? SVector(1.0, 0.0) :
    is_top(points0[i])    ? SVector(0.0, 1.0) :
    SVector(0.0, -1.0)
    for i in neumann_idx
]
neumann_tractions = [
    is_right(points0[i]) ?
        SVector(0.0, P * (D_val^2 - points0[i][2]^2) / (2 * I_val)) :
        SVector(0.0, 0.0)
    for i in neumann_idx
]

pts_flat = collect(reduce(vcat, [collect(p) for p in points0]))

for k_test in [35, 60, 100, 140]
    t_start = time()
    adjl = find_neighbors(points0, k_test)
    neumann_adjl = adjl[neumann_idx]

    trac_jac = [
        is_right(points0[neumann_idx[i]]) ?
            SMatrix{2,2,Float64}(0.0,0.0,0.0,-P*points0[neumann_idx[i]][2]/I_val) :
            SMatrix{2,2,Float64}(0.0,0.0,0.0,0.0)
        for i in eachindex(neumann_idx)]

    tlayout = Macchiato.build_traction_layout(
        neumann_idx, neumann_adjl, neumann_normals, neumann_tractions, λstar, μ, N)

    b_ref = zeros(2N)
    for (i, d) in enumerate(dirichlet_dofs); b_ref[d] = dirichlet_vals[i]; end
    for k in eachindex(tlayout.rows)
        b_ref[tlayout.rows[k]] = tlayout.b_vals[k]
    end

    result = Macchiato.shape_gradient(
        pts_flat, model, N, adjl, basis,
        active, dirichlet_dofs, dirichlet_vals,
        _ -> b_ref;
        interior_rows = interior_rows,
        traction_layout = tlayout,
        neumann_ids = neumann_idx,
        neumann_adjl = neumann_adjl,
        traction_jacobians = trac_jac,
        normal_jacobians = nothing,
    )
    g = copy(result.Δpts)
    Macchiato.extract_load_sensitivities!(g, tlayout, result.u, neumann_idx, trac_jac)

    t_elapsed = time() - t_start

    # Boundary gradient noise diagnostics
    # Build boundary loop
    bot = sort(findall(is_bottom, points0); by = i -> points0[i][1])
    rgt = sort(findall(is_right,  points0); by = i -> points0[i][2])
    top = sort(findall(is_top,    points0); by = i -> -points0[i][1])
    lft = sort(findall(is_left,   points0); by = i -> -points0[i][2])
    loop = Int[]
    append!(loop, lft[end:end])
    append!(loop, bot)
    append!(loop, rgt[2:end])
    append!(loop, top)
    append!(loop, lft[1:end-1])
    seen = Set{Int}(); bnd = Int[]
    for i in loop; if !(i in seen); push!(bnd, i); push!(seen, i); end; end

    g_y_loop = [g[2 * bnd[k]] for k in 1:length(bnd)]
    alt = [(-1.0)^k for k in 1:length(bnd)]
    nyq_ratio = abs(dot(g_y_loop, alt)) / max(norm(g_y_loop), 1e-15)

    # Symmetry
    top_pts = [(i, points0[i][1]) for i in top_idx if !is_right(points0[i])]
    bot_pts = [(i, points0[i][1]) for i in bottom_idx if !is_right(points0[i])]
    sym_max = 0.0
    for (it, xt) in top_pts
        ib = nothing
        for (ib_cand, xb) in bot_pts
            if abs(xt - xb) < 1e-10; ib = ib_cand; break; end
        end
        ib === nothing && continue
        sym_max = max(sym_max, abs(g[2*it] + g[2*ib]))
    end

    # Smoothness: fraction of gradient norm in high-frequency modes
    # Compute "roughness" as norm of 2nd difference along each edge
    function edge_noise(edge_idx)
        n_e = length(edge_idx)
        n_e < 3 && return 0.0
        d2 = sum(abs2, g[2*edge_idx[k]] - 2*g[2*edge_idx[k+1]] + g[2*edge_idx[k+2]]
                 for k in 1:n_e-2)
        raw = sum(abs2, g[2*i] for i in edge_idx)
        return sqrt(d2 / max(raw, 1e-15))
    end

    top_edge = [i for i in top_idx if !is_right(points0[i])]  # exclude corner
    bot_edge = [i for i in bottom_idx if !is_right(points0[i])]

    noise_top = edge_noise(sort(top_edge; by = i -> -points0[i][1]))
    noise_bot = edge_noise(sort(bot_edge; by = i -> points0[i][1]))

    @printf("k=%-3d  ‖g‖=%.2e  Nyquist=%.2e  sym=%.2e  noise_top=%.3f  noise_bot=%.3f  time=%.2fs\n",
            k_test, norm(g), nyq_ratio, sym_max, noise_top, noise_bot, t_elapsed)
end

# Print boundary gradient walk for the best k
println()
println("=== Gradient walk along bottom edge (k=100) ===")
k_best = 100
adjl = find_neighbors(points0, k_best)
neumann_adjl_b = adjl[neumann_idx]
tlayout_b = Macchiato.build_traction_layout(
    neumann_idx, neumann_adjl_b, neumann_normals, neumann_tractions, λstar, μ, N)
trac_jac_b = [
    is_right(points0[neumann_idx[i]]) ?
        SMatrix{2,2,Float64}(0.0,0.0,0.0,-P*points0[neumann_idx[i]][2]/I_val) :
        SMatrix{2,2,Float64}(0.0,0.0,0.0,0.0)
    for i in eachindex(neumann_idx)]
b_ref_b = zeros(2N)
for (i, d) in enumerate(dirichlet_dofs); b_ref_b[d] = dirichlet_vals[i]; end
for k in eachindex(tlayout_b.rows); b_ref_b[tlayout_b.rows[k]] = tlayout_b.b_vals[k]; end
res_b = Macchiato.shape_gradient(
    pts_flat, model, N, adjl, basis,
    active, dirichlet_dofs, dirichlet_vals,
    _ -> b_ref_b;
    interior_rows = interior_rows,
    traction_layout = tlayout_b,
    neumann_ids = neumann_idx,
    neumann_adjl = neumann_adjl_b,
    traction_jacobians = trac_jac_b,
    normal_jacobians = nothing,
)
g_b = copy(res_b.Δpts)
Macchiato.extract_load_sensitivities!(g_b, tlayout_b, res_b.u, neumann_idx, trac_jac_b)

bot_sorted = sort(bottom_idx; by = i -> points0[i][1])
println("Bottom edge (L→R):")
for i in bot_sorted
    @printf("  x=%.1f  g_y=%+.2e\n", points0[i][1], g_b[2i])
end
top_sorted = sort(top_idx; by = i -> -points0[i][1])
println("Top edge (R→L):")
for i in top_sorted
    @printf("  x=%.1f  g_y=%+.2e\n", points0[i][1], g_b[2i])
end
