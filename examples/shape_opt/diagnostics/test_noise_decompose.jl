# Decompose boundary gradient noise by operator
using Macchiato, StaticArrays, LinearAlgebra, SparseArrays, Printf
using RadialBasisFunctions
import RadialBasisFunctions: _build_weights_and_cache, Partial, MixedPartial, find_neighbors

const L_val = 8.0; const D_val = 1.0; const P = 1000.0; const I_val = 2 * D_val^3 / 3
const nx, ny = 17, 9
xs = range(0.0, L_val; length = nx)
ys = range(-D_val, D_val; length = ny)
points0 = [SVector{2, Float64}(x, y) for x in xs for y in ys]
const N = length(points0)
model = LinearElasticity(E = 1e7, ν = 0.3)
μ, λstar = Macchiato.lame_parameters(model)
basis = PHS(3; poly_deg = 3)
k = 35
adjl = find_neighbors(points0, k)

is_left(p)   = p[1] ≈ 0.0
is_right(p)  = p[1] ≈ L_val
is_top(p)    = p[2] ≈ D_val && !is_left(p)
is_bottom(p) = p[2] ≈ -D_val && !is_left(p)

didx = findall(is_left, points0)
nidx = findall(p -> !is_left(p) && (is_right(p) || is_top(p) || is_bottom(p)), points0)
iidx = findall(p -> !is_left(p) && !is_right(p) && !is_top(p) && !is_bottom(p), points0)
top_idx = findall(is_top, points0)
bot_idx = findall(is_bottom, points0)

ddofs = vcat(didx, didx .+ N)
dvals = zeros(2 * length(didx))
active = trues(2N)
for i in didx; active[i] = false; active[i + N] = false; end
int_rows = falses(N)
for i in iidx; int_rows[i] = true; end
pts = points0

# Neumann setup
nnorms = Vector{SVector{2,Float64}}(undef, length(nidx))
for (k, i) in enumerate(nidx)
    if is_right(points0[i])
        nnorms[k] = SVector(1.0, 0.0)
    elseif is_top(points0[i])
        nnorms[k] = SVector(0.0, 1.0)
    else
        nnorms[k] = SVector(0.0, -1.0)
    end
end
ntraks = Vector{SVector{2,Float64}}(undef, length(nidx))
for (k, i) in enumerate(nidx)
    if is_right(points0[i])
        ty = P * (D_val^2 - points0[i][2]^2) / (2 * I_val)
        ntraks[k] = SVector(0.0, ty)
    else
        ntraks[k] = SVector(0.0, 0.0)
    end
end
nadjl = adjl[nidx]
tlayout = Macchiato.build_traction_layout(nidx, nadjl, nnorms, ntraks, λstar, μ, N)

trac_jac = Vector{SMatrix{2,2,Float64,4}}(undef, length(nidx))
for (k, i) in enumerate(nidx)
    if is_right(points0[i])
        trac_jac[k] = SMatrix{2,2,Float64}(0.0, 0.0, 0.0, -P * points0[i][2] / I_val)
    else
        trac_jac[k] = SMatrix{2,2,Float64}(0.0, 0.0, 0.0, 0.0)
    end
end

b_ref = zeros(2N)
for (i, d) in enumerate(ddofs); b_ref[d] = dvals[i]; end
for k in eachindex(tlayout.rows); b_ref[tlayout.rows[k]] = tlayout.b_vals[k]; end

pts_flat = collect(reduce(vcat, [collect(p) for p in pts]))
result = Macchiato.shape_gradient(
    pts_flat, model, N, adjl, basis,
    active, ddofs, dvals, _ -> b_ref;
    interior_rows = int_rows, traction_layout = tlayout,
    neumann_ids = nidx, neumann_adjl = nadjl,
    traction_jacobians = trac_jac, normal_jacobians = nothing)

# Decompose: rebuild each operator's pullback contribution separately
npts = pts[nidx]
W_d2x, cd2x = _build_weights_and_cache(Partial(2, 1), pts, pts, adjl, basis)
W_d2y, cd2y = _build_weights_and_cache(Partial(2, 2), pts, pts, adjl, basis)
W_d2xy, cdxy = _build_weights_and_cache(MixedPartial(1, 2), pts, pts, adjl, basis)
W_dx, cdx = _build_weights_and_cache(Partial(1, 1), pts, npts, nadjl, basis)
W_dy, cdy = _build_weights_and_cache(Partial(1, 2), pts, npts, nadjl, basis)

ΔW_d2x, ΔW_d2y, ΔW_d2xy = Macchiato.allocate_weight_gradients(W_d2x, W_d2x, W_d2x)
Macchiato.extract_weight_sensitivities_elasticity!(
    ΔW_d2x, ΔW_d2y, ΔW_d2xy, W_d2x, result.η, result.u, active, λstar, μ;
    interior_rows = int_rows)
ΔW_dx, ΔW_dy = Macchiato.allocate_weight_gradients(W_dx, W_dy)
Macchiato.extract_neumann_sensitivities!(ΔW_dx, ΔW_dy, tlayout, result.η, result.u, active)

g_d2x = zeros(2N)
Macchiato._propagate_weight_gradient!(g_d2x, ΔW_d2x, W_d2x, cd2x, pts, pts, adjl, basis, Partial(2, 1))
g_d2y = zeros(2N)
Macchiato._propagate_weight_gradient!(g_d2y, ΔW_d2y, W_d2y, cd2y, pts, pts, adjl, basis, Partial(2, 2))
g_d2xy = zeros(2N)
Macchiato._propagate_weight_gradient!(g_d2xy, ΔW_d2xy, W_d2xy, cdxy, pts, pts, adjl, basis, MixedPartial(1, 2))
g_dx = zeros(2N)
Macchiato._propagate_weight_gradient!(g_dx, ΔW_dx, W_dx, cdx, pts, npts, nadjl, basis, Partial(1, 1);
                                        eval_offset = nidx)
g_dy = zeros(2N)
Macchiato._propagate_weight_gradient!(g_dy, ΔW_dy, W_dy, cdy, pts, npts, nadjl, basis, Partial(1, 2);
                                        eval_offset = nidx)

function noise_ratio(g, idx)
    n = length(idx)
    n < 3 && return 0.0
    d2 = sum(abs2, g[2*idx[k]] - 2*g[2*idx[k+1]] + g[2*idx[k+2]] for k in 1:n-2)
    raw = sum(abs2, g[2*i] for i in idx)
    return sqrt(d2 / max(raw, 1e-15))
end

bot_sorted = sort(bot_idx; by = i -> pts[i][1])
top_sorted = sort(top_idx; by = i -> -pts[i][1])  # CCW order

println("Per-operator noise on bottom edge:")
@printf("  %6s  %8s  %10s\n", "op", "noise", "‖g‖")
for (name, g) in [("d2x", g_d2x), ("d2y", g_d2y), ("d2xy", g_d2xy),
                   ("dx", g_dx), ("dy", g_dy)]
    @printf("  %6s  %8.3f  %10.2e\n", name, noise_ratio(g, bot_sorted), norm(g))
end
total = g_d2x + g_d2y + g_d2xy + g_dx + g_dy
@printf("  %6s  %8.3f  %10.2e\n", "total", noise_ratio(total, bot_sorted), norm(total))
@printf("  %6s  %8.3f  %10.2e\n", "AD", noise_ratio(result.Δpts, bot_sorted), norm(result.Δpts))

# Show gradient walk for the noisiest operator
println()
println("Bottom edge g_y from dx operator (1st deriv, Neumann rows):")
for i in bot_sorted
    @printf("  i=%d  x=%.1f  g_y=%+.2e\n", i, pts[i][1], g_dx[2i])
end

println()
println("Top edge g_y from dx operator:")
for i in top_sorted
    @printf("  i=%d  x=%.1f  g_y=%+.2e\n", i, pts[i][1], g_dx[2i])
end

# Noisiest interior operator
println()
println("Bottom edge g_y from d2x operator (2nd deriv, interior rows):")
for i in bot_sorted
    @printf("  i=%d  x=%.1f  g_y=%+.2e\n", i, pts[i][1], g_d2x[2i])
end
