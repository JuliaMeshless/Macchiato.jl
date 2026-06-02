# ============================================================================
# Test 2b — does the ∂W/∂x artifact climb with the operator's DERIVATIVE ORDER?
#
# The shape-gradient pullback ∂W/∂x is ONE derivative order higher than the
# forward operator: differentiating an order-q weight w.r.t. node position
# probes order-(q+1) behaviour of the RBF kernel.  For PHS r³ the high
# derivatives become singular (~1/r) at the stencil centre, which should amplify
# high-frequency content.  Prediction: the artifact (hi/low of dS/dr_j) grows
# with q.
#
# Same physics-free surrogate as Test 1: S = ηᵀ (W_ℒ f), smooth f,η, fixed cloud.
# Sweep ℒ = Identity (q=0, pullback→1st), Partial(1,1) (q=1→2nd),
# Partial(2,1)+Partial(2,2)=Laplacian (q=2→3rd).  Report TOTAL and INTERIOR-row
# hi/low (Test 1 showed the high-freq lives in the interior rows).
#
# Run:  jlrun plate_with_hole/probe_operator_order.jl   (from examples/)
# ============================================================================
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Macchiato
import RadialBasisFunctions: _build_weights, Partial, find_neighbors, Identity, derivative_order
using RadialBasisFunctions: PHS
using StaticArrays, LinearAlgebra, Printf, Statistics

const Lx = 4.0; const Ly = 4.0
const a0 = 0.40; const b0 = 0.20
const dx = 0.05
const nθ = max(48, round(Int, 2π*sqrt((a0^2 + b0^2)/2)/dx))
basis = PHS(3; poly_deg = 3); const k = 35

const margin  = 1 + 1.2*dx/min(a0, b0)
const xs_grid = collect((-Lx/2+dx):dx:(Lx/2-dx/2))
const ys_grid = collect((-Ly/2+dx):dx:(Ly/2-dx/2))
ellipse_val(p, a, b) = (p[1]/a)^2 + (p[2]/b)^2

const outer_pts = SVector{2,Float64}[]
for y in (-Ly/2):dx:(Ly/2)
    push!(outer_pts, SVector(-Lx/2, y)); push!(outer_pts, SVector(Lx/2, y))
end
for x in (-Lx/2+dx):dx:(Lx/2-dx)
    push!(outer_pts, SVector(x, Ly/2)); push!(outer_pts, SVector(x, -Ly/2))
end
const n_outer = length(outer_pts)
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
const hole_idx     = collect((n_int+n_outer+1):N)
const adjl = find_neighbors(pts0, k)

sf(p) = exp(0.30*p[1]) * cos(0.40*p[2])
sη(p) = exp(0.20*p[2]) * sin(0.30*p[1])
const fvec = [sf(p) for p in pts0]
const ηvec = [sη(p) for p in pts0]
θ  = [atan(hole0[j][2], hole0[j][1]) for j in 1:nθ]
r̂  = [hole0[j]/hypot(hole0[j]...)    for j in 1:nθ]

function famp(f, m)
    m == 0 && return abs(sum(f)/nθ)
    hypot((2/nθ)*sum(f[j]*cos(m*θ[j]) for j in 1:nθ),
          (2/nθ)*sum(f[j]*sin(m*θ[j]) for j in 1:nθ))
end
hiband(f) = sqrt(sum(famp(f,m)^2 for m in (nθ÷2-3):(nθ÷2)))
loband(f) = sqrt(sum(famp(f,m)^2 for m in 2:5))
totnorm(f) = norm(f .- sum(f)/nθ)
Sset(Wf, set) = dot(view(ηvec, set), view(Wf, set))

# build W f for a list of operators summed (Laplacian = two Partials)
applyW(ops, pts) = sum(_build_weights(op, pts, pts, adjl, basis) for op in ops) * fvec

function sweep(label, ops, qorder)
    ε = 1e-5
    g_tot = zeros(nθ); g_int = zeros(nθ)
    for j in 1:nθ
        hp = copy(hole0); hp[j] = hole0[j] + ε .* r̂[j]
        hm = copy(hole0); hm[j] = hole0[j] - ε .* r̂[j]
        Wfp = applyW(ops, vcat(interior_pts, outer_pts, hp))
        Wfm = applyW(ops, vcat(interior_pts, outer_pts, hm))
        g_tot[j] = (dot(ηvec, Wfp)              - dot(ηvec, Wfm))              / (2ε)
        g_int[j] = (Sset(Wfp, interior_idx)     - Sset(Wfm, interior_idx))     / (2ε)
    end
    @printf("  %-14s q=%d (pullback→%d)  | TOTAL hi/low=%6.2f (‖g‖=%.2e)  | INTERIOR hi/low=%6.2f (hi=%.2e)\n",
            label, qorder, qorder+1,
            hiband(g_tot)/max(loband(g_tot),1e-30), totnorm(g_tot),
            hiband(g_int)/max(loband(g_int),1e-30), hiband(g_int))
    return (qorder, hiband(g_tot)/max(loband(g_tot),1e-30), hiband(g_int)/max(loband(g_int),1e-30))
end

println("=== ∂W/∂x artifact vs operator derivative order ===")
println("N=$N  nθ=$nθ  (Nyquist m=$(nθ÷2)).  Smooth f,η; fixed cloud, k=$k.\n")
@printf("derivative_order: Identity=%d  Partial(1,1)=%d  Partial(2,1)=%d\n\n",
        derivative_order(Identity()), derivative_order(Partial(1,1)), derivative_order(Partial(2,1)))

r0 = sweep("Identity",          [Identity()],                 0)
r1 = sweep("Partial(1,1)",      [Partial(1,1)],               1)
r2 = sweep("Laplacian(2,1+2,2)",[Partial(2,1), Partial(2,2)], 2)

println("\nReading:")
println("  • If TOTAL/INTERIOR hi/low climbs 0→1→2, the artifact is driven by the")
println("    higher-order RBF derivatives the gradient pullback requires (q→q+1):")
println("    the cure is smoother kernels / higher poly_deg and node-layout, not")
println("    boundary stabilization.")
@printf("\n  hi/low (INTERIOR) by order:  q0=%.2f  q1=%.2f  q2=%.2f\n", r0[3], r1[3], r2[3])