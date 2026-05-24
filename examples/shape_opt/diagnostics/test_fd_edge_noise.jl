# ============================================================================
# Decisive experiment: is the boundary-gradient Nyquist noise intrinsic to the
# discrete gradient, or an artifact of the per-operator pullback decomposition?
# ============================================================================
# Context: docs/boundary_gradient_noise.md §"Decisive experiment".
#
# Two hypotheses make opposite, falsifiable predictions for the *finite
# difference* gradient — which is ground truth and never forms the 5
# per-operator pullbacks:
#
#   A (operator-pullback cancellation): the adjoint manufactures the Nyquist
#     noise; FD should be smooth (noise_ratio < 1.0) while AD ≈ 3.8.  ⇒ build
#     `_pullback_weights_combined!` in RadialBasisFunctions.
#
#   B (intrinsic ℓ² shape gradient): the noise is the true discrete gradient;
#     FD should reproduce it (noise_ratio ≈ AD).  ⇒ do NOT build the combined
#     pullback; invest in mesh-independent Helmholtz / smooth design space.
#
# Method: reuse the *exact* validated loss/gradient pair from the Phase 3
# example (`loss_compl_only` / `grad_compl_only_ad`, the same functions the
# strided 1e-6 FD check certifies), but FD **every contiguous edge node**
# (non-strided) so the metric can see adjacent-node alternation. This is the
# adjacency-resolved check the doc flags as missing from the strided probe.
#
# Run with:  jlrun test_fd_edge_noise.jl     (from examples/)
# ============================================================================

# Setup-only: pull in the Phase 3 grid/BCs/layout/forward_solve + the validated
# loss/grad pair, but skip its runtime FD check, iter-0 diagnostic, and 40-iter
# optimization loop (which would also move `pts_flat` off the reference).
MACCHIATO_SETUP_ONLY = true
include("../shape_optimization_phase3_cantilever.jl")

using Printf, LinearAlgebra

# ----------------------------------------------------------------------------
# noise_ratio — identical metric to examples/test_noise_decompose.jl.
# Second-difference energy / total energy of the y-components along `idx`
# (global point indices, ordered by spatial adjacency along the edge).
# sqrt(d2/raw); ≈ 4.0 ⇔ pure ±alternation (Nyquist), → 0 ⇔ smooth.
# ----------------------------------------------------------------------------
function noise_ratio(g, idx)
    n = length(idx)
    n < 3 && return 0.0
    d2  = sum(abs2, g[2 * idx[k]] - 2 * g[2 * idx[k + 1]] + g[2 * idx[k + 2]] for k in 1:(n - 2))
    raw = sum(abs2, g[2 * i] for i in idx)
    return sqrt(d2 / max(raw, 1e-15))
end

# Edge node lists, ordered along the edge by ascending x (adjacency = list order).
bot_sorted = sort(bottom_idx; by = i -> points0[i][1])
top_sorted = sort(top_idx;    by = i -> points0[i][1])

println()
println(repeat("=", 76))
println("DECISIVE EXPERIMENT — full non-strided FD edge-noise at $(nx)×$(ny) (N=$N)")
println(repeat("=", 76))
println("  bottom edge: $(length(bot_sorted)) contiguous nodes   top edge: $(length(top_sorted)) nodes")

# ----------------------------------------------------------------------------
# AD gradient (the thing currently validated to ~1e-6 against the strided FD).
# Reference geometry. grad_compl_only_ad restores the layout internally.
# ----------------------------------------------------------------------------
println()
println("Computing AD gradient (grad_compl_only_ad at reference geometry)...")
t_ad = @elapsed g_ad = grad_compl_only_ad(pts_flat)
@printf("  done in %.2f s   ‖g_ad‖ = %.4e\n", t_ad, norm(g_ad))

# ----------------------------------------------------------------------------
# Full central-difference gradient on every edge y-coordinate.
# loss_compl_only(pp) calls update_layout_for!(pp) internally, so each probe is
# self-consistent with live tractions + live polyline normals at the perturbed
# geometry — i.e. exactly the loss grad_compl_only_ad differentiates.
# ----------------------------------------------------------------------------
function fd_edge_gradient(edge_idx::Vector{Int}, h::Float64)
    g = zeros(2N)
    nsolve = 0
    for i in edge_idx
        pp = copy(pts_flat)
        pp[2i] = pts_flat[2i] + h
        Lp = loss_compl_only(pp)
        pp[2i] = pts_flat[2i] - h
        Lm = loss_compl_only(pp)
        g[2i] = (Lp - Lm) / (2h)
        nsolve += 2
    end
    return g, nsolve
end

const H_PRIMARY = 1.0e-4   # same step as the validated strided check (line ~612)

println()
println("Running full FD on bottom + top edges (h = $H_PRIMARY, central)...")
t_fd = @elapsed begin
    g_fd_bot, ns_bot = fd_edge_gradient(bot_sorted, H_PRIMARY)
    g_fd_top, ns_top = fd_edge_gradient(top_sorted, H_PRIMARY)
end
update_layout_for!(pts_flat)   # restore reference layout after the FD sweep
@printf("  %d forward solves in %.2f s (%.2f s/solve)\n",
        ns_bot + ns_top, t_fd, t_fd / (ns_bot + ns_top))

# Combine the two edge FD vectors into one (disjoint supports).
g_fd = g_fd_bot .+ g_fd_top

# ----------------------------------------------------------------------------
# Compare noise_ratio: FD (ground truth) vs AD (uses the pullback decomposition)
# ----------------------------------------------------------------------------
println()
println("noise_ratio (≈4.0 = pure Nyquist alternation, →0 = smooth):")
@printf("  %-10s  %12s  %12s\n", "edge", "FD", "AD")
for (name, idx) in (("bottom", bot_sorted), ("top", top_sorted))
    @printf("  %-10s  %12.4f  %12.4f\n", name, noise_ratio(g_fd, idx), noise_ratio(g_ad, idx))
end

# ----------------------------------------------------------------------------
# Adjacency-resolved pointwise FD-vs-AD agreement on contiguous edge nodes —
# the check the doc notes the strided 20-point probe cannot do. If FD and AD
# agree node-by-node here, they are the *same noisy vector*.
# ----------------------------------------------------------------------------
println()
println("Adjacency-resolved FD vs AD on contiguous edge nodes (y-component):")
for (name, idx) in (("bottom", bot_sorted), ("top", top_sorted))
    ad_e = [g_ad[2i] for i in idx]
    fd_e = [g_fd[2i] for i in idx]
    relerr = norm(ad_e .- fd_e) / max(norm(fd_e), 1e-15)
    @printf("  %-10s  ‖AD-FD‖/‖FD‖ = %.4e   ‖FD‖ = %.4e\n", name, relerr, norm(fd_e))
end

# Per-node walk on the bottom edge so alternation is visible by eye.
println()
println("Bottom-edge walk (x, FD g_y, AD g_y):")
@printf("  %5s  %6s  %14s  %14s\n", "i", "x", "FD g_y", "AD g_y")
for i in bot_sorted
    @printf("  %5d  %6.2f  %+14.4e  %+14.4e\n", i, points0[i][1], g_fd[2i], g_ad[2i])
end

# ----------------------------------------------------------------------------
# Robustness: re-run the bottom edge at a second step to confirm the FD noise
# is not a step-size artifact (a true discrete-gradient mode is step-stable).
# ----------------------------------------------------------------------------
const H_CHECK = 5.0e-5
println()
println("Step-robustness check on bottom edge (h = $H_CHECK)...")
g_fd_bot2, _ = fd_edge_gradient(bot_sorted, H_CHECK)
update_layout_for!(pts_flat)
@printf("  noise_ratio(FD bottom, h=%.0e) = %.4f   (h=%.0e gave %.4f)\n",
        H_CHECK, noise_ratio(g_fd_bot2, bot_sorted),
        H_PRIMARY, noise_ratio(g_fd_bot, bot_sorted))

# ----------------------------------------------------------------------------
# Verdict (per docs/boundary_gradient_noise.md §"Decisive experiment" table).
# ----------------------------------------------------------------------------
nr_fd = max(noise_ratio(g_fd, bot_sorted), noise_ratio(g_fd, top_sorted))
nr_ad = max(noise_ratio(g_ad, bot_sorted), noise_ratio(g_ad, top_sorted))
println()
println(repeat("=", 76))
println("VERDICT")
println(repeat("=", 76))
@printf("  max FD noise_ratio = %.3f     max AD noise_ratio = %.3f\n", nr_fd, nr_ad)
if nr_fd >= 2.0 && abs(nr_fd - nr_ad) < 1.5
    println("""
  → FD reproduces the AD noise. The Nyquist content is INTRINSIC to the discrete
    ℓ² shape gradient (Hypothesis B). A combined pullback computes the same
    quantity and CANNOT remove it.
    ACTION: do NOT build `_pullback_weights_combined!`. Invest in a
    mesh-independent Helmholtz filter (physical r, rescaled direction) and/or a
    smooth boundary design space.""")
elseif nr_fd < 1.0 && nr_ad >= 2.0
    println("""
  → FD is smooth while AD is noisy. The adjoint is producing noise the true
    gradient lacks (Hypothesis A); the strided probe hid it.
    ACTION: investigate the pullback; the combined pullback is justified. Also
    make the FD validation adjacency-aware.""")
else
    println("""
  → Inconclusive by the doc's thresholds (FD=$(round(nr_fd,digits=3)),
    AD=$(round(nr_ad,digits=3))). Inspect the per-node walk and the
    adjacency-resolved rel_err above before deciding.""")
end
println(repeat("=", 76))
