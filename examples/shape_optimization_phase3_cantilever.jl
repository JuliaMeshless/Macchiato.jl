# ============================================================================
# Phase 3: Cantilever Compliance Minimization (plan_manual_adjoint.md §Phase 3)
# ============================================================================
# End-to-end shape optimization on a 9×5 cantilever (N=45):
#   - Left edge:   clamped (Dirichlet, u=v=0)
#   - Right edge:  parabolic shear traction  t_y(y) = P·(D²-y²)/(2I)
#   - Top/bottom:  traction-free
#   - Loss:        compliance = bᵀu          (∂L/∂u = b)
#   - Constraint:  area ≈ A₀                 (two-sided quadratic penalty)
#   - Design:      y-coords of top, bottom, right boundary points
#                  (interior + left edge + all x-coords frozen)
#
# Pipeline: shape_gradient → boundary-loop sensitivity filter → free-DOF mask
# → Armijo line search → pts update.
#
# Physics upgrades vs. the original Phase 3 draft (which froze b):
#   1. Dead-load with shape-dependent tractions: `traction_layout.b_vals` is
#      recomputed each iteration from the current y-coords, and per-Neumann
#      2×2 Jacobians ∂t/∂p are passed to `shape_gradient`, which now includes
#      the ηᵀ·∂b/∂pts contribution (`extract_load_sensitivities!`).
#   2. Sensitivity filtering: a single α=0.25 3-point average along the
#      boundary loop suppresses the Nyquist-mode checkerboard artifact that
#      the unfiltered gradient was exploiting.
#
# Connectivity (adjl), traction normals, and coefficient stencils stay frozen
# (Level 1 — see plan §Key architectural decisions). Live normals are Level 2.
# ============================================================================
using Pkg
Pkg.activate(@__DIR__)

using Macchiato
using Mooncake
using RadialBasisFunctions
import RadialBasisFunctions: _build_weights, Partial, MixedPartial, find_neighbors
using StaticArrays
using SparseArrays
using LinearAlgebra
using Printf
using UnicodePlots
using CairoMakie
using FiniteDifferences

# ============================================================================
# Problem parameters
# ============================================================================

const L     = 8.0
const D     = 1.0
const P     = 1000.0
const E_val = 1.0e7
const ν_val = 0.3
const I_val = 2D^3 / 3

model    = LinearElasticity(E = E_val, ν = ν_val)
μ, λstar = lame_parameters(model)

# ============================================================================
# Reference grid + BC classification (matches Phase B)
# ============================================================================

const nx, ny = 9, 5
xs0 = range(0.0, L; length = nx)
ys0 = range(-D, D; length = ny)
points0 = [SVector{2}(x, y) for x in xs0 for y in ys0]
const N = length(points0)

is_left(p)   = p[1] ≈ 0.0
is_right(p)  = p[1] ≈ L
is_top(p)    = p[2] ≈ D  && !is_left(p)
is_bottom(p) = p[2] ≈ -D && !is_left(p)
is_dirichlet(p) = is_left(p)
is_neumann(p)   = !is_dirichlet(p) && (is_right(p) || is_top(p) || is_bottom(p))
is_interior(p)  = !is_dirichlet(p) && !is_neumann(p)

dirichlet_idx = findall(is_dirichlet, points0)
neumann_idx   = findall(is_neumann,   points0)
interior_idx  = findall(is_interior,  points0)
top_idx       = findall(is_top,       points0)
bottom_idx    = findall(is_bottom,    points0)
right_idx     = findall(is_right,     points0)

println("Cantilever: N=$N  (Dirichlet=$(length(dirichlet_idx))  " *
        "Neumann=$(length(neumann_idx))  interior=$(length(interior_idx)))")

# 2N DOF masks
active_v = trues(2N)
for i in dirichlet_idx
    active_v[i]     = false
    active_v[i + N] = false
end
active = BitVector(active_v)

interior_rows = falses(N)
for i in interior_idx
    interior_rows[i] = true
end

# ============================================================================
# Design mask: which entries of Δpts the optimizer is allowed to use
# ============================================================================
# Layout of pts_flat: [x_1, y_1, x_2, y_2, …]. Index of y_i is 2i.
# Free design variables: y-coords of top, bottom, right boundary points —
# *except* the two right corners (L, ±D). Those corners are kept frozen
# because their reference normals (1, 0) are baked into `traction_layout`
# and would misrepresent the geometry the moment a corner moves outward
# (Level 2 / live-normals territory, see plan §Phase D).

is_corner(p) = is_right(p) && (p[2] ≈ D || p[2] ≈ -D)

free_mask = falses(2N)
for i in vcat(top_idx, bottom_idx, right_idx)
    is_corner(points0[i]) && continue
    free_mask[2i] = true          # y-component only
end
n_free = count(free_mask)
println("Design variables (free y-coords, corners excluded): $n_free")

# ============================================================================
# Dirichlet + Neumann BC values
# ============================================================================

dirichlet_dofs = vcat(dirichlet_idx, dirichlet_idx .+ N)
dirichlet_vals = zeros(2 * length(dirichlet_idx))

function normal_of(p)
    if is_right(p)
        return SVector{2}(1.0, 0.0)
    elseif is_top(p)
        return SVector{2}(0.0, 1.0)
    elseif is_bottom(p)
        return SVector{2}(0.0, -1.0)
    else
        error("not a Neumann point: $p")
    end
end

# Classify Neumann points by *index* once at init. This is critical: using
# spatial predicates (`p[1] ≈ L`) inside the live-load loop is fragile —
# even a 1e-4 FD perturbation of a right-edge x-coord makes `is_right(p)`
# return false, which discontinuously drops the traction and corrupts the
# FD reference gradient. Index-based classification stays stable under any
# perturbation we ever apply.
const neumann_is_right_idx = [is_right(points0[i]) for i in neumann_idx]

# Traction as a function of (local Neumann index, current point coords). The
# right-edge parabolic shear depends on the y-coord; all other Neumann points
# carry zero traction regardless of location.
function traction_at(i_local::Int, p)
    if neumann_is_right_idx[i_local]
        return SVector{2}(0.0, P * (D^2 - p[2]^2) / (2 * I_val))
    else
        return SVector{2}(0.0, 0.0)
    end
end

# 2×2 Jacobian J[a, b] = ∂t_a / ∂p_b. Only right edge contributes (∂ty/∂y).
function traction_jacobian_at(i_local::Int, p)
    if neumann_is_right_idx[i_local]
        return SMatrix{2, 2, Float64}(0.0, 0.0, 0.0, -P * p[2] / I_val)
    else
        return SMatrix{2, 2, Float64}(0.0, 0.0, 0.0, 0.0)
    end
end

neumann_normals   = [normal_of(points0[i])             for i in neumann_idx]
neumann_tractions = [traction_at(i, points0[neumann_idx[i]])
                     for i in eachindex(neumann_idx)]

# ============================================================================
# RBF setup — connectivity is frozen for the whole loop
# ============================================================================

basis = PHS(3; poly_deg = 3)
const k = 35
adjl         = find_neighbors(points0, k)
neumann_adjl = adjl[neumann_idx]

traction_layout = build_traction_layout(
    neumann_idx, neumann_adjl, neumann_normals, neumann_tractions,
    λstar, μ, N,
)

# ============================================================================
# Boundary polygon (CCW) for shoelace area
# ============================================================================
# Order: bottom edge L→R, right edge bottom→top, top edge R→L, left edge top→bot.
# Each segment skips its starting corner (already added by the previous one).

function build_boundary_loop(points)
    bot = sort(findall(is_bottom, points); by = i -> points[i][1])
    rgt = sort(findall(is_right,  points); by = i -> points[i][2])
    top = sort(findall(is_top,    points); by = i -> -points[i][1])
    lft = sort(findall(is_left,   points); by = i -> -points[i][2])
    # Bottom-left corner is in `lft` (left edge). Bottom-right in `bot` and `rgt`.
    # Start at bottom-left, traverse CCW, avoid duplicates.
    loop = Int[]
    append!(loop, lft[end:end])          # bottom-left (start)
    append!(loop, bot)                    # bottom edge (includes bottom-right)
    append!(loop, rgt[2:end])             # right edge skipping bottom-right
    append!(loop, top)                    # top edge (right→left, includes top-right? top excludes left)
    append!(loop, lft[1:end-1])           # left edge top→bot, skip bottom-left (already added)
    # Drop any duplicates that snuck in (defensive)
    seen = Set{Int}()
    out = Int[]
    for i in loop
        if !(i in seen)
            push!(out, i)
            push!(seen, i)
        end
    end
    return out
end

const boundary_loop = build_boundary_loop(points0)
@assert length(boundary_loop) == 2 * (nx - 1) + 2 * (ny - 1)
println("Boundary polygon vertices: $(length(boundary_loop))  (expected $(2*(nx-1)+2*(ny-1)))")

# Shoelace area of a CCW polygon given pts_flat
function polygon_area(pts_flat::AbstractVector, loop::Vector{Int})
    A2 = 0.0
    n_b = length(loop)
    @inbounds for k in 1:n_b
        i1 = loop[k]
        i2 = loop[mod1(k + 1, n_b)]
        x1, y1 = pts_flat[2i1-1], pts_flat[2i1]
        x2, y2 = pts_flat[2i2-1], pts_flat[2i2]
        A2 += x1 * y2 - x2 * y1
    end
    return 0.5 * A2
end

# ∂Area/∂pts (full 2N), zero on interior + non-loop entries
function polygon_area_grad!(g::AbstractVector, pts_flat::AbstractVector, loop::Vector{Int})
    fill!(g, 0.0)
    n_b = length(loop)
    @inbounds for k in 1:n_b
        i_prev = loop[mod1(k - 1, n_b)]
        i_next = loop[mod1(k + 1, n_b)]
        i      = loop[k]
        # dA/dx_i = 0.5 (y_{i+1} - y_{i-1});  dA/dy_i = 0.5 (x_{i-1} - x_{i+1})
        g[2i-1] = 0.5 * (pts_flat[2i_next]     - pts_flat[2i_prev])
        g[2i]   = 0.5 * (pts_flat[2i_prev - 1] - pts_flat[2i_next - 1])
    end
    return g
end

# ============================================================================
# Forward solve helper (frozen layout) — for line search & loss eval
# ============================================================================

function forward_solve(pts_flat::Vector{Float64})
    pts_v = [SVector{2}(pts_flat[2i - 1], pts_flat[2i]) for i in 1:N]
    W_d2x  = _build_weights(Partial(2, 1),      pts_v, pts_v, adjl, basis)
    W_d2y  = _build_weights(Partial(2, 2),      pts_v, pts_v, adjl, basis)
    W_d2xy = _build_weights(MixedPartial(1, 2), pts_v, pts_v, adjl, basis)
    neu_pts = pts_v[neumann_idx]
    W_dx = _build_weights(Partial(1, 1), pts_v, neu_pts, neumann_adjl, basis)
    W_dy = _build_weights(Partial(1, 2), pts_v, neu_pts, neumann_adjl, basis)
    A = assemble_elasticity_from_weights(W_d2x, W_d2y, W_d2xy, N, λstar, μ)
    b = zeros(2N)
    apply_dirichlet!(A, b, dirichlet_dofs, dirichlet_vals)
    apply_traction!(A, b, traction_layout, W_dx, W_dy)
    u = lu(A) \ b
    return (u = u, b = b)
end

compliance(u, b) = dot(b, u)

# ============================================================================
# Reference solve + initial loss
# ============================================================================

pts_flat = collect(reduce(vcat, [collect(p) for p in points0]))
sol0 = forward_solve(pts_flat)
C0   = compliance(sol0.u, sol0.b)
V0   = polygon_area(pts_flat, boundary_loop)
V_target = V0           # keep area at the reference value

# Penalty parameter — scaled so the gradient magnitudes are commensurate at iter 0.
# Order of magnitude: ‖∇C‖ ~ shape_gradient norm at iter 0; ‖∇Vpen‖ ~ 2ρ·(V-V₀)·‖∇V‖.
# At iter 0 V=V₀ so penalty=0 but its slope still matters once we step.
const ρ_pen = 1.0e3

println()
println("Reference state")
println(@sprintf("  ‖u‖           = %.4e", norm(sol0.u)))
println(@sprintf("  compliance C₀ = %.6e", C0))
println(@sprintf("  area V₀       = %.6e   (target = %.6e)", V0, V_target))

# ============================================================================
# Live loads + sensitivity filter
# ============================================================================
# We update `traction_layout.b_vals` from the current geometry each iteration
# (dead-load formulation with shape-dependent traction values), and pass the
# matching per-point 2×2 Jacobians to `shape_gradient` so the manual adjoint
# picks up the `ηᵀ·∂b/∂pts` term. Coefficients/normals stay frozen — that's
# Phase D Level 2 territory.

function update_traction_loads!(layout::TractionLayout,
                                  pts_flat::Vector{Float64},
                                  neumann_ids::Vector{Int},
                                  traction_fn)
    @inbounds for i_local in eachindex(neumann_ids)
        g = neumann_ids[i_local]
        p = SVector{2}(pts_flat[2g - 1], pts_flat[2g])
        t = traction_fn(i_local, p)
        layout.b_vals[2i_local - 1] = t[1]
        layout.b_vals[2i_local]     = t[2]
    end
    return layout
end

function build_traction_jacobians(pts_flat::Vector{Float64},
                                   neumann_ids::Vector{Int},
                                   jac_fn)
    return [
        jac_fn(i_local, SVector{2}(pts_flat[2neumann_ids[i_local] - 1],
                                     pts_flat[2neumann_ids[i_local]]))
        for i_local in eachindex(neumann_ids)
    ]
end

# Sensitivity filter — 3-point boundary-loop average on the y-gradient.
# A single α=0.25 pass nulls the Nyquist (every-other-vertex) mode, killing
# the checkerboard artifact while preserving smooth design modes. Honors
# `free_mask`: frozen DOFs neither contribute to nor receive averaged values.
function filter_along_boundary!(out::Vector{Float64},
                                  in_::Vector{Float64},
                                  loop::Vector{Int},
                                  free_mask::BitVector;
                                  α::Float64 = 0.25)
    out .= in_
    n_loop = length(loop)
    @inbounds for k in 1:n_loop
        i_curr = loop[k]
        free_mask[2 * i_curr] || continue
        i_prev = loop[mod1(k - 1, n_loop)]
        i_next = loop[mod1(k + 1, n_loop)]
        gp = free_mask[2 * i_prev] ? in_[2 * i_prev] : in_[2 * i_curr]
        gn = free_mask[2 * i_next] ? in_[2 * i_next] : in_[2 * i_curr]
        out[2 * i_curr] = α * gp + (1 - 2α) * in_[2 * i_curr] + α * gn
    end
    return out
end

const smoothing_α = 0.25

# ============================================================================
# Total loss + gradient (live tractions, filtered gradient)
# ============================================================================

# total_loss(pts_flat) = compliance(u(pts_flat)) + ρ·(V - V_target)²
function total_loss(pts_flat::Vector{Float64})
    update_traction_loads!(traction_layout, pts_flat, neumann_idx, traction_at)
    sol = forward_solve(pts_flat)
    V   = polygon_area(pts_flat, boundary_loop)
    return compliance(sol.u, sol.b) + ρ_pen * (V - V_target)^2
end

function total_gradient!(grad::Vector{Float64}, pts_flat::Vector{Float64})
    update_traction_loads!(traction_layout, pts_flat, neumann_idx, traction_at)
    jacobians = build_traction_jacobians(pts_flat, neumann_idx, traction_jacobian_at)
    # ∂L/∂u = b at the *current* shape. Compute it cheaply (no PDE solve):
    # zero everywhere except Dirichlet rows (= dirichlet_vals) and Neumann
    # rows (= layout.b_vals).
    b_now = zeros(2N)
    @inbounds for (i, d) in enumerate(dirichlet_dofs)
        b_now[d] = dirichlet_vals[i]
    end
    @inbounds for k in eachindex(traction_layout.rows)
        b_now[traction_layout.rows[k]] = traction_layout.b_vals[k]
    end
    result = shape_gradient(
        pts_flat, model, N, adjl, basis,
        active,
        dirichlet_dofs, dirichlet_vals,
        _ -> b_now;
        interior_rows      = interior_rows,
        traction_layout    = traction_layout,
        neumann_ids        = neumann_idx,
        neumann_adjl       = neumann_adjl,
        traction_jacobians = jacobians,
    )
    grad .= result.Δpts
    # Explicit ∂L/∂b·∂b/∂pts contribution. For compliance L = bᵀu, ∂L/∂b = u,
    # so this is uᵀ·∂b/∂pts — same sparse VJP as inside shape_gradient but
    # with u in place of η (which handled the implicit ηᵀ·∂b/∂pts piece).
    extract_load_sensitivities!(grad, traction_layout, result.u, neumann_idx, jacobians)
    g_area = similar(grad)
    polygon_area_grad!(g_area, pts_flat, boundary_loop)
    V = polygon_area(pts_flat, boundary_loop)
    @. grad += 2 * ρ_pen * (V - V_target) * g_area
    # Apply boundary-loop smoother, then mask to free DOFs.
    grad_smooth = similar(grad)
    filter_along_boundary!(grad_smooth, grad, boundary_loop, free_mask; α = smoothing_α)
    grad_smooth .*= free_mask
    grad .= grad_smooth
    return (u = result.u, V = V)
end

# ============================================================================
# Validation: AD compliance gradient (with ∂b/∂pts) vs central FD
# ============================================================================
# Confirms `extract_load_sensitivities!` is wired correctly. Compliance only
# (no penalty, no filter, no mask) at the reference geometry.

function loss_compl_only(pts_flat::Vector{Float64})
    update_traction_loads!(traction_layout, pts_flat, neumann_idx, traction_at)
    sol = forward_solve(pts_flat)
    return compliance(sol.u, sol.b)
end

function grad_compl_only_ad(pts_flat::Vector{Float64})
    update_traction_loads!(traction_layout, pts_flat, neumann_idx, traction_at)
    jacobians = build_traction_jacobians(pts_flat, neumann_idx, traction_jacobian_at)
    b_now = zeros(2N)
    @inbounds for (i, d) in enumerate(dirichlet_dofs)
        b_now[d] = dirichlet_vals[i]
    end
    @inbounds for k in eachindex(traction_layout.rows)
        b_now[traction_layout.rows[k]] = traction_layout.b_vals[k]
    end
    result = shape_gradient(
        pts_flat, model, N, adjl, basis,
        active,
        dirichlet_dofs, dirichlet_vals,
        _ -> b_now;
        interior_rows      = interior_rows,
        traction_layout    = traction_layout,
        neumann_ids        = neumann_idx,
        neumann_adjl       = neumann_adjl,
        traction_jacobians = jacobians,
    )
    g = copy(result.Δpts)
    # Explicit u-side b contribution (∂L/∂b = u for compliance).
    extract_load_sensitivities!(g, traction_layout, result.u, neumann_idx, jacobians)
    return g
end

println()
println("Validating compliance gradient (AD with ∂b/∂pts vs central FD)...")
grad_ad_check = grad_compl_only_ad(pts_flat)
t_fd_check = @elapsed begin
    grad_fd_check = FiniteDifferences.grad(
        FiniteDifferences.central_fdm(5, 1), loss_compl_only, pts_flat,
    )[1]
end
rel_err = norm(grad_ad_check .- grad_fd_check) / norm(grad_fd_check)
println(@sprintf("  ‖grad_AD - grad_FD‖ / ‖grad_FD‖ = %.4e   (FD took %.2f s)",
                  rel_err, t_fd_check))
if rel_err < 1.0e-4
    println("  PASS — ∂b/∂pts contribution is correct.")
else
    @warn "AD/FD mismatch — gradient implementation may be off ($rel_err)"
    diff = grad_ad_check .- grad_fd_check
    perm = sortperm(abs.(diff); rev = true)
    println("  Top 10 component mismatches:")
    for k in 1:min(10, length(perm))
        i = perm[k]
        which = i > N ? "v" : "u"
        pt    = i > N ? i - N : i
        # Reconstruct (x, y) from pts_flat via the global flat index. The flat
        # index 2k-1, 2k are the x, y of point k. But i here is in the
        # *block-stacked* convention used by η/u (1:N is x-eq, N+1:2N is y-eq).
        # For diagnosis, just show which DOF this corresponds to.
        println(@sprintf("    DOF %3d (%s_%d):  AD = %+10.3e   FD = %+10.3e   Δ = %+10.3e",
                          i, which, pt, grad_ad_check[i], grad_fd_check[i], diff[i]))
    end
end
update_traction_loads!(traction_layout, pts_flat, neumann_idx, traction_at)

# ============================================================================
# Optimization loop — gradient descent with Armijo backtracking
# ============================================================================

max_iter   = 40
α_init     = 1.0e-7
c_armijo   = 1.0e-4
β_backtrack = 0.5
max_ls_steps = 25

# Physical guardrails — stop if the discrete system loses physical meaning
# (e.g. boundary self-intersection from over-aggressive line search). With the
# live-load + filtered formulation we expect to reach a clean optimum well
# before tripping these.
const u_norm_max  = 5.0                          # ~37× the initial ‖u‖
const compl_floor = 1.0e-4 * 71.8                # ~0.01% of C₀ — paranoia bound

grad         = zeros(2N)
hist_loss    = Float64[]
hist_compl   = Float64[]
hist_volume  = Float64[]
hist_gradnorm = Float64[]
hist_step    = Float64[]
warm_times   = Float64[]

# Warm-up call to absorb cold-compile cost (Mooncake-extension path)
_t_cold = @elapsed total_gradient!(grad, pts_flat)
println(@sprintf("\nGradient cold call: %.3f s   ‖∇L‖₀ = %.4e", _t_cold, norm(grad)))

println()
println("Optimization loop")
println(@sprintf("  ρ_pen = %.3e   α₀ = %.3e   max_iter = %d", ρ_pen, α_init, max_iter))
println()
println(@sprintf("%4s  %12s  %12s  %12s  %12s  %8s  %8s",
                  "iter", "loss", "compliance", "area", "‖∇L‖", "α", "ls_t[ms]"))

let α = α_init
    for iter in 1:max_iter
        t_grad = @elapsed begin
            info = total_gradient!(grad, pts_flat)
        end
        push!(warm_times, t_grad)

        sol_cur = forward_solve(pts_flat)
        L_cur   = compliance(sol_cur.u, sol_cur.b) + ρ_pen * (info.V - V_target)^2
        gnorm   = norm(grad)

        push!(hist_loss,     L_cur)
        push!(hist_compl,    compliance(sol_cur.u, sol_cur.b))
        push!(hist_volume,   info.V)
        push!(hist_gradnorm, gnorm)

        # Armijo: L(x + α·p) ≤ L(x) + c·α·∇Lᵀ·p   with p = -grad
        p = -grad
        descent_slope = dot(grad, p)              # = -‖grad‖² (≤ 0)
        α_try = min(α * 2, 1.0e-3)                # gentle re-grow each iter
        accepted = false
        L_new = L_cur
        pts_try = similar(pts_flat)
        t_ls = @elapsed begin
            for ls in 1:max_ls_steps
                @. pts_try = pts_flat + α_try * p
                L_new = total_loss(pts_try)
                if L_new ≤ L_cur + c_armijo * α_try * descent_slope
                    accepted = true
                    break
                end
                α_try *= β_backtrack
            end
        end
        push!(hist_step, α_try)

        if !accepted
            println(@sprintf("%4d  %12.4e  %12.4e  %12.4e  %12.4e  %8.2e  %8.2f  (LS FAIL)",
                              iter, L_cur, hist_compl[end], info.V, gnorm, α_try, 1e3 * t_ls))
            break
        end

        pts_flat .= pts_try
        α = α_try

        println(@sprintf("%4d  %12.4e  %12.4e  %12.4e  %12.4e  %8.2e  %8.2f",
                          iter, L_cur, hist_compl[end], info.V, gnorm, α_try, 1e3 * t_ls))

        # Physical guardrail — bail if compliance turns negative or u blows up.
        sol_check = forward_solve(pts_flat)
        C_check   = compliance(sol_check.u, sol_check.b)
        if C_check < compl_floor || C_check < 0 || norm(sol_check.u) > u_norm_max
            println(@sprintf("  stop: physical guardrail tripped (C=%.3e, ‖u‖=%.3e)",
                              C_check, norm(sol_check.u)))
            break
        end
        # Natural convergence — gradient norm fell to a tiny fraction of init.
        if gnorm < 1.0e-3 * hist_gradnorm[1]
            println(@sprintf("  stop: gradient norm converged (‖∇L‖ = %.4e)", gnorm))
            break
        end
    end
end

# Final state
sol_f = forward_solve(pts_flat)
C_f   = compliance(sol_f.u, sol_f.b)
V_f   = polygon_area(pts_flat, boundary_loop)
push!(hist_loss,     C_f + ρ_pen * (V_f - V_target)^2)
push!(hist_compl,    C_f)
push!(hist_volume,   V_f)

println()
println("============================================")
println("Phase 3: Final state")
println("============================================")
println(@sprintf("  iterations completed = %d", length(warm_times)))
println(@sprintf("  compliance:  %.6e → %.6e   (Δ = %.2f%%)",
                  C0, C_f, 100 * (C_f - C0) / C0))
println(@sprintf("  area:        %.6e → %.6e   (Δ = %.2f%%)",
                  V0, V_f, 100 * (V_f - V0) / V0))
println(@sprintf("  ‖∇L‖:        %.4e → %.4e",
                  hist_gradnorm[1], hist_gradnorm[end]))
println(@sprintf("  warm gradient time (median): %.2f ms",
                  1e3 * sort(warm_times)[max(1, length(warm_times) ÷ 2)]))

# ============================================================================
# Success criteria check (plan_manual_adjoint.md §Phase 3)
# ============================================================================

println()
println("Success criteria:")
n_iter        = length(warm_times)
compl_drop    = (C0 - C_f) / C0
crit_runs     = n_iter ≥ 5 && all(isfinite, hist_compl)
crit_mono     = all(diff(hist_compl[1:end-1]) .≤ 1e-10)
crit_vol      = abs(V_f - V_target) / V_target < 0.02
crit_reduce   = compl_drop ≥ 0.20
crit_warm     = (length(warm_times) > 1 ? minimum(warm_times[2:end]) : warm_times[1]) < 0.02
println(@sprintf("  [%s] runs ≥5 iters, no NaN (got %d)",          crit_runs   ? "ok" : "  ", n_iter))
println(@sprintf("  [%s] compliance monotone (Armijo enforced)",   crit_mono   ? "ok" : "  "))
println(@sprintf("  [%s] area within 2%% of target",                crit_vol    ? "ok" : "  "))
println(@sprintf("  [%s] compliance reduction ≥ 20%% (got %.1f%%)", crit_reduce ? "ok" : "  ", 100 * compl_drop))
println(@sprintf("  [%s] warm gradient < 20 ms",                    crit_warm   ? "ok" : "  "))

# ============================================================================
# Terminal convergence plots (UnicodePlots) for quick visual check
# ============================================================================

println()
println(lineplot(1:length(hist_compl), hist_compl;
                  title = "Compliance vs iter", xlabel = "iter",
                  ylabel = "C", canvas = BrailleCanvas, height = 12))
println(lineplot(1:length(hist_gradnorm), hist_gradnorm;
                  title = "‖∇L‖ vs iter", xlabel = "iter", ylabel = "‖∇L‖",
                  yscale = :log10, canvas = BrailleCanvas, height = 12))

# ============================================================================
# CairoMakie figure: initial vs final shape + convergence
# ============================================================================

fig = Figure(size = (1100, 700))

# Initial geometry
ax1 = Axis(fig[1, 1]; title = "Initial geometry (rectangle)",
           aspect = DataAspect(), xlabel = "x", ylabel = "y")
init_xs = [points0[i][1] for i in boundary_loop]
init_ys = [points0[i][2] for i in boundary_loop]
push!(init_xs, init_xs[1]); push!(init_ys, init_ys[1])
lines!(ax1, init_xs, init_ys; color = :black, linewidth = 2)
scatter!(ax1, [p[1] for p in points0], [p[2] for p in points0];
         color = :gray, markersize = 4)

# Final geometry
ax2 = Axis(fig[1, 2]; title = "Optimized geometry (tapered)",
           aspect = DataAspect(), xlabel = "x", ylabel = "y")
pts_final = [SVector{2}(pts_flat[2i-1], pts_flat[2i]) for i in 1:N]
fin_xs = [pts_final[i][1] for i in boundary_loop]
fin_ys = [pts_final[i][2] for i in boundary_loop]
push!(fin_xs, fin_xs[1]); push!(fin_ys, fin_ys[1])
lines!(ax2, fin_xs, fin_ys; color = :crimson, linewidth = 2)
scatter!(ax2, [p[1] for p in pts_final], [p[2] for p in pts_final];
         color = :gray, markersize = 4)

# Convergence: compliance
ax3 = Axis(fig[2, 1]; title = "Compliance vs iteration",
           xlabel = "iter", ylabel = "C")
lines!(ax3, 1:length(hist_compl), hist_compl; linewidth = 2)

# Convergence: gradient norm
ax4 = Axis(fig[2, 2]; title = "‖∇L‖ vs iteration",
           xlabel = "iter", ylabel = "‖∇L‖", yscale = log10)
lines!(ax4, 1:length(hist_gradnorm), hist_gradnorm; linewidth = 2)

fig_path = joinpath(@__DIR__, "phase3_cantilever_result.png")
save(fig_path, fig)
println()
println("Figure saved: $fig_path")
