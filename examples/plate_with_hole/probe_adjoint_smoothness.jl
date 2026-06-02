# ============================================================================
# probe_adjoint_smoothness.jl
#
# MEASURES (does not assume) the spectral content of u, η, and the RHS b along
# the hole boundary, to determine whether K⁻¹ amplifies or dampens high
# frequencies.  This directly tests the claim "λ is already smooth."
#
# Three questions:
#   Q1: Is b (the RHS / load vector) smooth along the hole?
#   Q2: Is u = K⁻¹b smooth along the hole?  (hi/lo of u vs hi/lo of b)
#   Q3: Is η = K⁻ᵀb smooth along the hole?  (for compliance, ∂C/∂u = b)
#
# If u/η have MORE high-freq than b → K⁻¹ AMPLIFIES → Tikhonov is relevant.
# If u/η have LESS/SAME high-freq as b → K⁻¹ is smoothing → FD-smoothed ∂K/∂r
#   is the right target (the noise lives in ∂W/∂x, not in K⁻¹).
#
# We also compute the full discrete shape gradient spectrum for comparison.
#
# Run:  jlrun plate_with_hole/probe_adjoint_smoothness.jl   (from examples/)
# ============================================================================
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Macchiato
import RadialBasisFunctions: _build_weights, Partial, MixedPartial, find_neighbors
using RadialBasisFunctions: PHS
using StaticArrays
using LinearAlgebra
using Printf

# ---- problem setup ----------------------------------------------------------
const Lx        = 4.0
const Ly        = 4.0
const a0        = 0.40
const b0        = 0.20
const dx        = 0.05
const σ∞        = 1.0

model   = LinearElasticity(E = 1.0e7, ν = 0.3)
μ, λstar = lame_parameters(model)
basis   = PHS(3; poly_deg = 3)
const k = 35

ellipse_val(p, a, b) = (p[1]/a)^2 + (p[2]/b)^2
const margin = 1 + 1.2*dx/min(a0, b0)
const nθ = max(48, round(Int, 2π*sqrt((a0^2 + b0^2)/2)/dx))

ref_interior = SVector{2,Float64}[]
let xs = (-Lx/2+dx):dx:(Lx/2-dx/2), ys = (-Ly/2+dx):dx:(Ly/2-dx/2)
    for x in xs, y in ys
        ellipse_val(SVector(x,y), a0, b0) > margin^2 || continue
        push!(ref_interior, SVector(x,y))
    end
end
const n_int = length(ref_interior)

outer_pts = SVector{2,Float64}[]; outer_tag = Symbol[]
for y in (-Ly/2):dx:(Ly/2)
    push!(outer_pts, SVector(-Lx/2, y)); push!(outer_tag, :xlo)
    push!(outer_pts, SVector(Lx/2, y));  push!(outer_tag, :xhi)
end
for x in (-Lx/2+dx):dx:(Lx/2-dx)
    push!(outer_pts, SVector(x, Ly/2));  push!(outer_tag, :yhi)
    push!(outer_pts, SVector(x, -Ly/2)); push!(outer_tag, :ylo)
end
const n_outer = length(outer_pts)

const θ_vals = [2π*(j-1)/nθ for j in 1:nθ]
hole0 = [SVector(a0*cos(t), b0*sin(t)) for t in θ_vals]

const ref_pts = vcat(ref_interior, outer_pts, hole0)
const N = length(ref_pts)
const interior_idx = collect(1:n_int)
const outer_idx    = collect((n_int+1):(n_int+n_outer))
const hole_idx     = collect((n_int+n_outer+1):N)
const neumann_idx  = vcat(outer_idx, hole_idx)
const hole_pos     = collect(1:nθ)

nearest(p0, pool) = pool[argmin([hypot(ref_pts[i][1]-p0[1], ref_pts[i][2]-p0[2]) for i in pool])]
const pin_ux1 = nearest((0.0, 0.8), interior_idx)
const pin_ux2 = nearest((0.0,-0.8), interior_idx)
const pin_uy1 = nearest((0.8, 0.0), interior_idx)
const dirichlet_dofs = [pin_ux1, pin_ux2, pin_uy1 + N]
const active = let a = trues(2N); a[pin_ux1]=a[pin_ux2]=a[pin_uy1+N]=false; a end
const interior_rows = let r = falses(N); for i in interior_idx; r[i]=true; end; r end

const _ZJV = SVector(0.0, 0.0)
zero_njac(g) = NormalJacobian(g, g, _ZJV, _ZJV, _ZJV, _ZJV)
flat(v) = reduce(vcat, [[p[1], p[2]] for p in v])
outer_normal(i) = outer_tag[i]===:xhi ? SVector(1.0,0.0) :
                  outer_tag[i]===:xlo ? SVector(-1.0,0.0) :
                  outer_tag[i]===:yhi ? SVector(0.0,1.0) : SVector(0.0,-1.0)

const adjl_ref = find_neighbors(ref_pts, k)
const pts = ref_pts

# ---- build forward system ---------------------------------------------------
neumann_adjl_ref = adjl_ref[neumann_idx]
hn, hjacs = polyline_normals(flat(pts), hole_idx, hole_pos)
njacs = vcat(NormalJacobian[zero_njac(i) for i in outer_idx], hjacs)
normals = SVector{2,Float64}[]
tractions = SVector{2,Float64}[]
for i in 1:n_outer
    nn = outer_normal(i); push!(normals, nn); push!(tractions, σ∞ .* nn)
end
append!(normals, hn); append!(tractions, fill(SVector(0.0,0.0), nθ))
layout = build_traction_layout(neumann_idx, neumann_adjl_ref, normals, tractions, λstar, μ, N)

# Assemble
W_d2x  = _build_weights(Partial(2, 1),      pts, pts, adjl_ref, basis)
W_d2y  = _build_weights(Partial(2, 2),      pts, pts, adjl_ref, basis)
W_d2xy = _build_weights(MixedPartial(1, 2), pts, pts, adjl_ref, basis)
neu_pts = pts[neumann_idx]
W_dx   = _build_weights(Partial(1, 1), pts, neu_pts, neumann_adjl_ref, basis)
W_dy   = _build_weights(Partial(1, 2), pts, neu_pts, neumann_adjl_ref, basis)

A_raw = assemble_elasticity_from_weights(W_d2x, W_d2y, W_d2xy, N, λstar, μ)
b_raw = zeros(2N)
A = copy(A_raw)
b = copy(b_raw)
apply_dirichlet!(A, b, dirichlet_dofs, zeros(length(dirichlet_dofs)))
apply_traction!(A, b, layout, W_dx, W_dy)

F = lu(A)
u = F \ b
η = F' \ b    # adjoint: Kᵀη = b (for compliance, ∂C/∂u = b)

# ---- spectral tools ---------------------------------------------------------
# Fourier amplitude of a scalar field sampled at θ_vals
famp(g, m) = m == 0 ? abs(sum(g)/nθ) :
    hypot((2/nθ)*sum(g[j]*cos(m*θ_vals[j]) for j in 1:nθ),
          (2/nθ)*sum(g[j]*sin(m*θ_vals[j]) for j in 1:nθ))

function spectrum(g, label)
    lo  = sqrt(sum(famp(g, m)^2 for m in 2:5))
    hi  = sqrt(sum(famp(g, m)^2 for m in (nθ÷2-3):(nθ÷2)))
    tot = sqrt(sum(famp(g, m)^2 for m in 0:(nθ÷2)))
    println("  $(rpad(label,14)) ‖·‖=$(round(tot,sigdigits=4))  lo(m2..5)=$(round(lo,sigdigits=4))  hi(Nyq)=$(round(hi,sigdigits=4))  hi/lo=$(round(hi/max(lo,1e-30),digits=3))")
    return (lo=lo, hi=hi, tot=tot)
end

# ---- Q1: is b (the RHS) smooth? ---------------------------------------------
# b is 2N: [b_x; b_y].  Extract components at hole nodes.
bx_hole = [b[i]     for i in hole_idx]
by_hole = [b[i + N] for i in hole_idx]
# Radial component of the load vector at hole nodes
br_hole = [bx_hole[j]*cos(θ_vals[j]) + by_hole[j]*sin(θ_vals[j]) for j in 1:nθ]
# Tangential
bt_hole = [-bx_hole[j]*sin(θ_vals[j]) + by_hole[j]*cos(θ_vals[j]) for j in 1:nθ]

println("=== Q1: RHS b spectral content along hole boundary (nθ=$nθ) ===")
r_br = spectrum(br_hole, "b_radial")
r_bt = spectrum(bt_hole, "b_tang")
println("  (hole is traction-free, so b ≈ 0 on hole rows; this is the baseline)\n")

# ---- Q2: is u = K⁻¹b smooth? ------------------------------------------------
ux_hole = [u[i]     for i in hole_idx]
uy_hole = [u[i + N] for i in hole_idx]
ur_hole = [ux_hole[j]*cos(θ_vals[j]) + uy_hole[j]*sin(θ_vals[j]) for j in 1:nθ]
ut_hole = [-ux_hole[j]*sin(θ_vals[j]) + uy_hole[j]*cos(θ_vals[j]) for j in 1:nθ]

println("=== Q2: u = K⁻¹b spectral content along hole boundary ===")
r_ur = spectrum(ur_hole, "u_radial")
r_ut = spectrum(ut_hole, "u_tang")

# amplification ratio: hi/lo of u divided by hi/lo of b
# (if u is smoother than b, ratio < 1; if rougher, ratio > 1)
if r_br.lo > 1e-30
    @printf("  amplification (u_rad hi/lo) / (b_rad hi/lo) = %.2f\n",
            (r_ur.hi/r_ur.lo) / (r_br.hi/r_br.lo))
end
println()

# ---- Q3: is η = K⁻ᵀb smooth? ------------------------------------------------
ηx_hole = [η[i]     for i in hole_idx]
ηy_hole = [η[i + N] for i in hole_idx]
ηr_hole = [ηx_hole[j]*cos(θ_vals[j]) + ηy_hole[j]*sin(θ_vals[j]) for j in 1:nθ]
ηt_hole = [-ηx_hole[j]*sin(θ_vals[j]) + ηy_hole[j]*cos(θ_vals[j]) for j in 1:nθ]

println("=== Q3: η = K⁻ᵀb spectral content along hole boundary ===")
r_ηr = spectrum(ηr_hole, "η_radial")
r_ηt = spectrum(ηt_hole, "η_tang")
@printf("  ‖η - u‖∞ = %.2e\n", norm(η - u, Inf))
println()

# ---- Q4: full discrete shape gradient ---------------------------------------
res = shape_gradient(flat(pts), model, N, adjl_ref, basis, active,
                     dirichlet_dofs, zeros(3), _ -> b;
                     interior_rows = interior_rows, traction_layout = layout,
                     neumann_ids = neumann_idx, neumann_adjl = neumann_adjl_ref,
                     traction_jacobians = nothing, normal_jacobians = njacs)
Δpts = res.Δpts
g_hole = [SVector(Δpts[2*i-1], Δpts[2*i]) for i in hole_idx]
g_rad = [(g_hole[j][1]*cos(θ_vals[j]) + g_hole[j][2]*sin(θ_vals[j]))
         for j in 1:nθ]

println("=== Q4: Full discrete shape gradient (radial component) ===")
r_grad = spectrum(g_rad, "grad(rad)")

# ---- spectral breakdown across ALL modes ------------------------------------
println("\n=== Fourier amplitude table (every 2nd mode, 0..nθ/2) ===")
@printf("  %3s  %14s  %14s  %14s  %14s  %14s\n",
        "m", "u_rad", "u_tan", "η_rad", "η_tan", "grad(rad)")
for m in 0:2:(nθ÷2)
    @printf("  %3d  %14.4e  %14.4e  %14.4e  %14.4e  %14.4e\n",
            m, famp(ur_hole, m), famp(ut_hole, m),
            famp(ηr_hole, m), famp(ηt_hole, m), famp(g_rad, m))
end

# ---- verdict ----------------------------------------------------------------
println("\n======== VERDICT ========")
println("  u_rad hi/lo  = $(round(r_ur.hi/max(r_ur.lo,1e-30),digits=3))")
println("  u_tan hi/lo  = $(round(r_ut.hi/max(r_ut.lo,1e-30),digits=3))")
println("  η_rad hi/lo  = $(round(r_ηr.hi/max(r_ηr.lo,1e-30),digits=3))")
println("  η_tan hi/lo  = $(round(r_ηt.hi/max(r_ηt.lo,1e-30),digits=3))")
println("  grad   hi/lo = $(round(r_grad.hi/max(r_grad.lo,1e-30),digits=3))")

u_rough  = r_ur.hi/max(r_ur.lo,1e-30) > 0.5 || r_ut.hi/max(r_ut.lo,1e-30) > 0.5
η_rough  = r_ηr.hi/max(r_ηr.lo,1e-30) > 0.5 || r_ηt.hi/max(r_ηt.lo,1e-30) > 0.5
g_rough  = r_grad.hi/max(r_grad.lo,1e-30) > 2.0

if u_rough || η_rough
    println("\n  → u/η HAVE significant high-frequency content.")
    println("    K⁻¹ is NOT just smoothing — it contributes noise.")
    println("    Tikhonov regularization of K⁻¹ is RELEVANT.")
end
if g_rough && !(u_rough || η_rough)
    println("\n  → u/η are SMOOTH but the gradient is ROUGH.")
    println("    The noise lives in ∂K/∂r (∂W/∂x), not in K⁻¹.")
    println("    FD-smoothed pullback is the right target.")
end
if g_rough && (u_rough || η_rough)
    println("\n  → BOTH u/η and ∂K/∂r contribute noise.")
    println("    Both regularisation approaches deserve investigation.")
end
