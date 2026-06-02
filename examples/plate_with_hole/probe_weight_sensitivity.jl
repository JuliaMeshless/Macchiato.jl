# ============================================================================
# Probe B — ISOLATE ∂W/∂xⱼ: is the Nyquist rise manufactured by differentiating
# the RBF-FD weights w.r.t. node position, with NO PDE involved?
#
# The discrete shape gradient is, for any PDE,
#     dJ/dxⱼ = (smooth physical part)  +  ηᵀ (∂W/∂xⱼ) u
#                                          └── RBF-FD weight sensitivity
# Replace the PDE solution u and adjoint η by FIXED SMOOTH ANALYTIC fields and
# keep only the weight-sensitivity contraction:
#     S(geometry) = ηᵀ W f ,     dS/dr_j = ηᵀ (∂W/∂r_j) f
# with W the RBF-FD Laplacian.  f, η are smooth ⇒ any roughness in dS/dr_j over
# the hole nodes is purely the geometry-sensitivity of the differentiation
# weights — physics-free.
#
# Verdict: if dS/dr_j rises toward the Nyquist mode for SMOOTH f, η, the
# mechanism is the RBF-FD weight differentiation itself.
#
# Run:  jlrun plate_with_hole/probe_weight_sensitivity.jl   (from examples/)
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
const nθ = max(48, round(Int, 2π*sqrt((a0^2 + b0^2)/2)/dx))
basis = PHS(3; poly_deg = 3); const k = 35

const margin  = 1 + 1.2*dx/min(a0, b0)
const xs_grid = collect((-Lx/2+dx):dx:(Lx/2-dx/2))
const ys_grid = collect((-Ly/2+dx):dx:(Ly/2-dx/2))

const outer_pts = SVector{2,Float64}[]
for y in (-Ly/2):dx:(Ly/2)
    push!(outer_pts, SVector(-Lx/2, y)); push!(outer_pts, SVector(Lx/2, y))
end
for x in (-Lx/2+dx):dx:(Lx/2-dx)
    push!(outer_pts, SVector(x, Ly/2)); push!(outer_pts, SVector(x, -Ly/2))
end
const n_outer = length(outer_pts)

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
const hole_idx = collect((n_int+n_outer+1):N)
const adjl_ref = find_neighbors(pts0, k)
flat(v) = reduce(vcat, [[p[1], p[2]] for p in v])

# Smooth analytic fields, sampled at the nodes (low-order, fully resolved).
sf(p) = exp(0.30*p[1]) * cos(0.40*p[2])     # f
sη(p) = exp(0.20*p[2]) * sin(0.30*p[1])     # η

# Surrogate S = ηᵀ (Wxx+Wyy) f  — purely the RBF-FD Laplacian contracted on
# smooth fields.
function surrogate_S(hole_pts, fvec, ηvec)
    pts = vcat(interior_pts, outer_pts, hole_pts)
    Wxx = _build_weights(Partial(2,1), pts, pts, adjl_ref, basis)
    Wyy = _build_weights(Partial(2,2), pts, pts, adjl_ref, basis)
    return dot(ηvec, (Wxx + Wyy) * fvec)
end

# ============================================================================
println("=== ∂W/∂xⱼ isolation: weight-sensitivity spectrum (physics-free) ===")
println("N=$N  nθ=$nθ  (Nyquist mode = $(nθ÷2))\n")

θ = [atan(hole0[j][2], hole0[j][1]) for j in 1:nθ]
r̂ = [hole0[j]/hypot(hole0[j]...)    for j in 1:nθ]

# fields fixed at the reference cloud (do NOT resample as the node moves: we
# want the weight sensitivity, ∂W/∂x, not ∂f/∂x).
const fvec = [sf(p) for p in pts0]
const ηvec = [sη(p) for p in pts0]

# sanity: how well does W reproduce the analytic Laplacian of f? (resolution)
let pts = pts0
    Wxx = _build_weights(Partial(2,1), pts, pts, adjl_ref, basis)
    Wyy = _build_weights(Partial(2,2), pts, pts, adjl_ref, basis)
    lap_num = (Wxx + Wyy) * fvec
    lap_exa = [(0.30^2 - 0.40^2)*sf(p) for p in pts]   # ∇²(e^{ax}cos(by)) = (a²-b²)f
    rel = norm(lap_num[hole_idx] - lap_exa[hole_idx]) / norm(lap_exa[hole_idx])
    @printf("RBF-FD Laplacian of the smooth field f, rel.err on hole nodes = %.2e\n", rel)
    @printf("(f is fully resolved; any roughness below is from ∂W/∂x, not f.)\n\n")
end

const ε = 1e-5
g_rad = zeros(nθ)
for j in 1:nθ
    hp = copy(hole0); hp[j] = hole0[j] + ε .* r̂[j]
    hm = copy(hole0); hm[j] = hole0[j] - ε .* r̂[j]
    g_rad[j] = (surrogate_S(hp, fvec, ηvec) - surrogate_S(hm, fvec, ηvec)) / (2ε)
end

println("--- weight-sensitivity radial spectrum dS/dr_j (amplitude per mode) ---")
println("  m      amp=√(a²+b²)")
mode_amp = Float64[]
for m in 0:(nθ÷2)
    am = (2.0/nθ)*sum(g_rad[j]*cos(m*θ[j]) for j in 1:nθ)
    bm = (m==0) ? 0.0 : (2.0/nθ)*sum(g_rad[j]*sin(m*θ[j]) for j in 1:nθ)
    push!(mode_amp, hypot(am, bm))
    (m <= 8 || m >= nθ÷2-2 || m in (12,16,20)) && @printf("  %2d   %.4e\n", m, hypot(am,bm))
end
low  = sqrt(sum(abs2, mode_amp[3:7]))
high = sqrt(sum(abs2, mode_amp[end-5:end]))
@printf("\n  ‖modes 2..6‖ = %.4e   ‖top-6 near Nyquist‖ = %.4e   ratio hi/lo = %.2f\n",
        low, high, high/low)
println("\nVerdict: rise toward Nyquist with SMOOTH f, η ⇒ the roughness is the")
println("geometry-sensitivity of the RBF-FD differentiation weights (∂W/∂x).")
