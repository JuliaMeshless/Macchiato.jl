# Complex-step gradient validation for Phase 3 cantilever
# Compares AD gradient with complex-step FD (machine-precision accurate)
using Macchiato, StaticArrays, LinearAlgebra, SparseArrays, Printf
using RadialBasisFunctions
import RadialBasisFunctions: _build_weights, Partial, MixedPartial, find_neighbors

# === Problem setup (matching Phase 3 example) ===
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
const k = 35
adjl = find_neighbors(points0, k)

# BC classification
is_left(p)   = p[1] ≈ 0.0
is_right(p)  = p[1] ≈ L_val
is_top(p)    = p[2] ≈ D_val && !is_left(p)
is_bottom(p) = p[2] ≈ -D_val && !is_left(p)

dirichlet_idx = findall(is_left, points0)
neumann_idx   = findall(p -> !is_left(p) && (is_right(p) || is_top(p) || is_bottom(p)), points0)

dirichlet_dofs = vcat(dirichlet_idx, dirichlet_idx .+ N)
dirichlet_vals = zeros(2 * length(dirichlet_idx))

# Neumann normals + tractions
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
neumann_adjl = adjl[neumann_idx]

traction_layout = Macchiato.build_traction_layout(
    neumann_idx, neumann_adjl, neumann_normals, neumann_tractions, λstar, μ, N)

pts_flat = collect(reduce(vcat, [collect(p) for p in points0]))

# === Forward solve (generic over number type) ===
function forward_solve_generic(pts_flat::AbstractVector{T}) where {T<:Number}
    pts_v = [SVector{2, T}(pts_flat[2i-1], pts_flat[2i]) for i in 1:N]
    W_d2x  = _build_weights(Partial(2, 1),      pts_v, pts_v, adjl, basis)
    W_d2y  = _build_weights(Partial(2, 2),      pts_v, pts_v, adjl, basis)
    W_d2xy = _build_weights(MixedPartial(1, 2), pts_v, pts_v, adjl, basis)
    neu_pts = pts_v[neumann_idx]
    W_dx = _build_weights(Partial(1, 1), pts_v, neu_pts, neumann_adjl, basis)
    W_dy = _build_weights(Partial(1, 2), pts_v, neu_pts, neumann_adjl, basis)

    A = Macchiato.assemble_elasticity_from_weights(W_d2x, W_d2y, W_d2xy, N, λstar, μ)
    b = zeros(T, 2N)
    Macchiato.apply_dirichlet!(A, b, dirichlet_dofs, dirichlet_vals)
    Macchiato.apply_traction!(A, b, traction_layout, W_dx, W_dy)
    u = A \ b
    return u, b
end

function compliance_generic(pts_flat::AbstractVector{T}) where {T<:Number}
    u, b = forward_solve_generic(pts_flat)
    return dot(b, u)
end

# === Real reference solution ===
u_ref, b_ref = forward_solve_generic(pts_flat)
C_ref = dot(b_ref, u_ref)
@printf("Reference compliance (real): %.8f\n", C_ref)
@printf("  ‖u‖ = %.4e  ‖b‖ = %.4e\n", norm(u_ref), norm(b_ref))

# === Complex-step gradient on a few free entries ===
# Test on BOTH interior (Dirichlet rows excluded) and boundary entries
active_v = trues(2N)
for i in dirichlet_idx; active_v[i] = false; active_v[i+N] = false; end

top_idx    = findall(is_top, points0)
bottom_idx = findall(is_bottom, points0)
right_idx  = findall(is_right, points0)
interior_idx = findall(p -> !is_left(p) && !is_right(p) && !is_top(p) && !is_bottom(p), points0)
boundary_idx = vcat(top_idx, bottom_idx, right_idx)

# Pick a few interior points (x and y DOFs) and a few boundary points (y DOFs only)
test_entries = Int[]
for i in interior_idx[1:4:end]   # interior: test both x and y
    push!(test_entries, 2i-1, 2i)
end
for i in boundary_idx[1:4:end]   # boundary: test y only
    push!(test_entries, 2i)
end
n_test = length(test_entries)
println("Testing $(n_test) entries ($(count(j->j in [2i-1 for i in interior_idx], test_entries)) interior, $(n_test - count(j->j in [2i-1 for i in interior_idx], test_entries)) boundary)")

h = 1e-20   # complex step size (machine precision via imag part)
g_complex = zeros(n_test)
t_start = time()

for (k, j) in enumerate(test_entries)
    # Perturb single coordinate by i*h
    pts_c = complex(pts_flat)     # promotes to ComplexF64
    pts_c[j] += im * h
    C = compliance_generic(pts_c)
    g_complex[k] = imag(C) / h
end

t_elapsed = time() - t_start
@printf("Complex-step: %d entries in %.2f s (%.1f ms/entry)\n",
        n_test, t_elapsed, 1000 * t_elapsed / n_test)

# === Full AD gradient (matching Phase 3 example exactly) ===
println("Computing full AD gradient (all sensitivity paths)...")
b_ref_vec = zeros(2N)
for (i, d) in enumerate(dirichlet_dofs); b_ref_vec[d] = dirichlet_vals[i]; end
for k in eachindex(traction_layout.rows)
    b_ref_vec[traction_layout.rows[k]] = traction_layout.b_vals[k]
end

active = trues(2N)
for i in dirichlet_idx; active[i] = false; active[i+N] = false; end

interior_rows = falses(N)
for i in interior_idx; interior_rows[i] = true; end

# Full sensitivity: traction Jacobians, normal Jacobians, D_act
trac_jac = [
    is_right(points0[neumann_idx[i]]) ?
        SMatrix{2,2,Float64}(0.0,0.0,0.0,-P*points0[neumann_idx[i]][2]/I_val) :
        SMatrix{2,2,Float64}(0.0,0.0,0.0,0.0)
    for i in eachindex(neumann_idx)]

# At reference: D_act = D = 1, normals are canonical
result = Macchiato.shape_gradient(
    pts_flat, model, N, adjl, basis,
    active, dirichlet_dofs, dirichlet_vals,
    _ -> b_ref_vec;
    interior_rows = interior_rows,
    traction_layout = traction_layout,
    neumann_ids = neumann_idx,
    neumann_adjl = neumann_adjl,
    traction_jacobians = trac_jac,
    normal_jacobians = nothing,  # frozen normals at reference
)
g_ad = copy(result.Δpts)
# u-side b contribution (∂L/∂b = u for compliance)
Macchiato.extract_load_sensitivities!(g_ad, traction_layout, result.u, neumann_idx, trac_jac)
# At reference config, D_act = D = 1.0, so D_act sensitivity is zero
# (no contribution because ∂D/∂y terms are zero when |y_top|=|y_bot|=D)

# === Comparison ===
g_ad_probe = g_ad[test_entries]

println()
println("=== Comparison (complex-step vs AD) ===")
println(@sprintf("  %6s  %18s  %18s  %12s", "DOF", "complex", "AD", "rel_err"))
_comp_errors = Float64[]
for (k, j) in enumerate(test_entries)
    gc = g_complex[k]
    ga = g_ad_probe[k]
    denom = max(abs(gc), abs(ga), 1e-15)
    rel = abs(gc - ga) / denom
    push!(_comp_errors, rel)
    println(@sprintf("  %6d  %+18.8e  %+18.8e  %12.2e", j, gc, ga, rel))
end

rel_full = norm(g_complex .- g_ad_probe) / max(norm(g_complex), 1e-15)
println()
@printf("  Rel error (2-norm on %d entries): %.4e\n", n_test, rel_full)
@printf("  Max per-component rel error: %.4e\n", maximum(_comp_errors))

if rel_full < 1e-10
    println("  VERDICT: AD gradient is machine-precision correct.")
elseif rel_full < 1e-5
    println("  VERDICT: AD gradient is correct (sub-FD-noise level).")
else
    println("  VERDICT: possible issue — investigate further.")
end
