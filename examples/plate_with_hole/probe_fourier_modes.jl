# ============================================================================
# Probe: per-Fourier-mode SNR of the adjoint shape gradient on a FIXED cloud.
#
# THE question for multi-DOF shape opt: when we parameterize the hole boundary
# by radial Fourier modes r(θ)=r₀+Σ aₘcos(mθ)+bₘsin(mθ) and contract the noisy
# nodal gradient onto those modes, how many modes carry trustworthy signal
# before the Nyquist noise floor swamps them?
#
# This isolates the noise question from the optimizer's bookkeeping (no
# morphing, no remeshing, no stepping). Everything is measured on the single
# iter-0 ellipse cloud with FIXED adjacency.
#
#   adjoint:  dC/daₘ = Σⱼ g_rad,ⱼ cos(mθⱼ)   [contract nodal grad onto mode m]
#   FD     :  dC/daₘ = (C(+ε·cos(mθ)·r̂) − C(−ε·cos(mθ)·r̂)) / 2ε
#
# Both differentiate the SAME map (fixed adjl, normals rebuilt from the
# perturbed polyline), so a mismatch is pure noise corruption of the
# contraction — not a modelling difference. θⱼ and r̂ⱼ are the ACTUAL polar
# angle/direction of each hole node, used identically in both paths.
#
# Run:  jlrun plate_with_hole/probe_fourier_modes.jl   (from examples/)
# ============================================================================
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Macchiato
import RadialBasisFunctions: _build_weights, Partial, MixedPartial, find_neighbors
using RadialBasisFunctions: PHS
using StaticArrays, LinearAlgebra, Printf

const Lx = 4.0; const Ly = 4.0
const a0 = 0.40; const b0 = 0.20
const dx = 0.05
const σ∞ = 1.0
const nθ = max(48, round(Int, 2π * sqrt((a0^2 + b0^2) / 2) / dx))
model = LinearElasticity(E = 1.0e7, ν = 0.3)
μ, λstar = lame_parameters(model)
basis = PHS(3; poly_deg = 3); const k = 35

const margin   = 1 + 1.2 * dx / min(a0, b0)
const xs_grid  = collect((-Lx/2+dx):dx:(Lx/2-dx/2))
const ys_grid  = collect((-Ly/2+dx):dx:(Ly/2-dx/2))

# ---- outer boundary --------------------------------------------------------
const outer_pts = SVector{2,Float64}[]; const outer_tag = Symbol[]
for y in (-Ly/2):dx:(Ly/2)
    push!(outer_pts, SVector(-Lx/2, y)); push!(outer_tag, :xlo)
    push!(outer_pts, SVector(Lx/2, y));  push!(outer_tag, :xhi)
end
for x in (-Lx/2+dx):dx:(Lx/2-dx)
    push!(outer_pts, SVector(x, Ly/2));  push!(outer_tag, :yhi)
    push!(outer_pts, SVector(x, -Ly/2)); push!(outer_tag, :ylo)
end
const n_outer = length(outer_pts)
outer_normal(i) = outer_tag[i]===:xhi ? SVector(1.0,0.0) :
                  outer_tag[i]===:xlo ? SVector(-1.0,0.0) :
                  outer_tag[i]===:yhi ? SVector(0.0,1.0) : SVector(0.0,-1.0)

# ---- fixed interior cloud around the iter-0 ellipse ------------------------
ellipse_val(p, a, b) = (p[1]/a)^2 + (p[2]/b)^2
interior_pts = SVector{2,Float64}[]
for x in xs_grid, y in ys_grid
    ellipse_val(SVector(x,y), a0, b0) > margin^2 || continue
    push!(interior_pts, SVector(x,y))
end
const n_int = length(interior_pts)

hole0 = [SVector(a0*cos(2π*j/nθ), b0*sin(2π*j/nθ)) for j in 0:(nθ-1)]
const pts0 = vcat(interior_pts, outer_pts, hole0)
const N = length(pts0)
const interior_idx = collect(1:n_int)
const outer_idx    = collect((n_int+1):(n_int+n_outer))
const hole_idx     = collect((n_int+n_outer+1):N)
const neumann_idx  = vcat(outer_idx, hole_idx)
const hole_pos     = collect(1:nθ)

nearest(p0,pool) = pool[argmin([hypot(pts0[i][1]-p0[1], pts0[i][2]-p0[2]) for i in pool])]
const pin_ux1 = nearest((0.0, 0.8), interior_idx)
const pin_ux2 = nearest((0.0,-0.8), interior_idx)
const pin_uy1 = nearest((0.8, 0.0), interior_idx)
const dirichlet_dofs = [pin_ux1, pin_ux2, pin_uy1 + N]
const active = let a = trues(2N); a[pin_ux1]=a[pin_ux2]=a[pin_uy1+N]=false; a end
const interior_rows = let r = falses(N); for i in interior_idx; r[i]=true; end; r end

# FIXED adjacency — the whole point is to hold the stencils constant.
const adjl0 = find_neighbors(pts0, k)
const neumann_adjl0 = adjl0[neumann_idx]

const _ZJV = SVector(0.0, 0.0)
zero_njac(g) = NormalJacobian(g, g, _ZJV, _ZJV, _ZJV, _ZJV)

flat(v) = reduce(vcat, [[p[1], p[2]] for p in v])

function build_layout(hole_pts)
    pts_local = vcat(interior_pts, outer_pts, hole_pts)
    hn, hjacs = polyline_normals(flat(pts_local), hole_idx, hole_pos)
    njacs = vcat(NormalJacobian[zero_njac(i) for i in outer_idx], hjacs)
    normals = SVector{2,Float64}[]; tractions = SVector{2,Float64}[]
    for i in 1:n_outer
        nn = outer_normal(i); push!(normals, nn); push!(tractions, σ∞ .* nn)
    end
    append!(normals, hn); append!(tractions, fill(SVector(0.0,0.0), nθ))
    layout = build_traction_layout(neumann_idx, neumann_adjl0, normals, tractions, λstar, μ, N)
    return pts_local, layout, njacs
end

# Forward-only compliance for FD (fixed adjl0, normals rebuilt from polyline).
function solve_C(hole_pts)
    pts_local, layout, _ = build_layout(hole_pts)
    Wx  = _build_weights(Partial(2,1),    pts_local, pts_local, adjl0, basis)
    Wy  = _build_weights(Partial(2,2),    pts_local, pts_local, adjl0, basis)
    Wxy = _build_weights(MixedPartial(1,2), pts_local, pts_local, adjl0, basis)
    neu = pts_local[neumann_idx]
    Wdx = _build_weights(Partial(1,1), pts_local, neu, neumann_adjl0, basis)
    Wdy = _build_weights(Partial(1,2), pts_local, neu, neumann_adjl0, basis)
    A = assemble_elasticity_from_weights(Wx, Wy, Wxy, N, λstar, μ)
    b = zeros(2N)
    apply_dirichlet!(A, b, dirichlet_dofs, zeros(length(dirichlet_dofs)))
    apply_traction!(A, b, layout, Wdx, Wdy)
    u = lu(A) \ b
    return dot(b, u)
end

# Adjoint nodal gradient at the hole nodes.
function adjoint_g_hole(hole_pts)
    pts_local, layout, njacs = build_layout(hole_pts)
    b = zeros(2N)
    for kk in eachindex(layout.rows); b[layout.rows[kk]] = layout.b_vals[kk]; end
    res = shape_gradient(flat(pts_local), model, N, adjl0, basis, active,
                         dirichlet_dofs, zeros(3), _ -> b;
                         interior_rows = interior_rows, traction_layout = layout,
                         neumann_ids = neumann_idx, neumann_adjl = neumann_adjl0,
                         traction_jacobians = nothing, normal_jacobians = njacs)
    C = dot(b, res.u)
    g_hole = [SVector(res.Δpts[2*gi-1], res.Δpts[2*gi]) for gi in hole_idx]
    return C, g_hole
end

# ============================================================================
# Run
# ============================================================================
println("=== Per-Fourier-mode SNR probe (fixed cloud, fixed adjl) ===")
println("N=$N  nθ=$nθ  (Nyquist mode = $(nθ ÷ 2))\n")

C0, g_hole = adjoint_g_hole(hole0)

# Actual polar angle and outward unit radial of each hole node.
θ   = [atan(hole0[j][2], hole0[j][1]) for j in 1:nθ]
r̂   = [hole0[j] / hypot(hole0[j]...)  for j in 1:nθ]
g_rad = [dot(g_hole[j], r̂[j]) for j in 1:nθ]

@printf("C0 = %.6e\n", C0)
@printf("‖g_hole‖ = %.4e   ‖g_rad‖ = %.4e\n\n", norm(reduce(vcat,[[g[1],g[2]] for g in g_hole])), norm(g_rad))

# ---- full radial-gradient spectrum (free, from one adjoint solve) ----------
# aₘ = (2/nθ) Σ g_rad cos(mθ),  bₘ = (2/nθ) Σ g_rad sin(mθ)
# Report the per-mode amplitude √(aₘ²+bₘ²); this is the contracted gradient
# magnitude for mode m. Watch where energy concentrates vs the Nyquist mode.
println("--- radial-gradient Fourier spectrum (amplitude per mode) ---")
println("  m      |a_m|        |b_m|        amp=√(a²+b²)")
mode_amp = Float64[]
for m in 0:(nθ ÷ 2)
    am = (2.0/nθ) * sum(g_rad[j]*cos(m*θ[j]) for j in 1:nθ)
    bm = (m==0) ? 0.0 : (2.0/nθ) * sum(g_rad[j]*sin(m*θ[j]) for j in 1:nθ)
    amp = hypot(am, bm)
    push!(mode_amp, amp)
    if m <= 8 || m >= (nθ÷2 - 2) || m in (12, 16, 20)
        @printf("  %2d   %+.4e   %+.4e   %.4e\n", m, am, bm, amp)
    end
end
lowband  = sqrt(sum(abs2, mode_amp[3:7]))     # modes 2..6 (1-indexed +1)
highband = sqrt(sum(abs2, mode_amp[end-5:end]))# top 6 modes near Nyquist
@printf("\n  ‖modes 2..6‖ = %.4e    ‖top-6 (near Nyquist)‖ = %.4e    ratio hi/lo = %.2f\n\n",
        lowband, highband, highband/lowband)

# ---- FD validation of selected modes ---------------------------------------
# Perturb hole node j radially by ε·cos(mθⱼ)·r̂ⱼ (an exact mode-m radius
# deformation), re-solve, central difference. Compare to the adjoint
# contraction onto the identical mode.
println("--- adjoint vs FD per mode (cos branch; fixed cloud) ---")
println("   m     dC/da_m (adj)    dC/da_m (FD)     rel.err     ratio")
const ε = 1e-5
for m in [1, 2, 3, 4, 5, 6, 8, 12, 20]
    dCda_adj = sum(g_rad[j]*cos(m*θ[j]) for j in 1:nθ)
    hp = [hole0[j] + ε*cos(m*θ[j]) .* r̂[j] for j in 1:nθ]
    hm = [hole0[j] - ε*cos(m*θ[j]) .* r̂[j] for j in 1:nθ]
    Cp = solve_C(hp); Cm = solve_C(hm)
    dCda_fd = (Cp - Cm) / (2ε)
    relerr = abs(dCda_adj - dCda_fd) / max(abs(dCda_fd), 1e-30)
    @printf("  %2d   %+.6e   %+.6e   %.2e   %.3f\n",
            m, dCda_adj, dCda_fd, relerr, dCda_adj/dCda_fd)
end

println("\nReading:")
println("  • If low modes (2-6) match FD to ~1e-2 and amplitude decays toward")
println("    Nyquist  → multi-DOF is viable, just cap the mode count.")
println("  • If the spectrum is flat/rising toward Nyquist, or low modes")
println("    disagree with FD → the contraction itself is noise-limited and")
println("    needs a Sobolev (Helmholtz) inner product, not just truncation.")
