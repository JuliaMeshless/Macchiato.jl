# ============================================================================
# Test 1 — is the ∂W/∂x artifact dominated by the one-sided BOUNDARY stencils?
#
# Physics-free surrogate (as in probe_weight_sensitivity.jl):
#     S = ηᵀ W f ,  W = Wxx+Wyy,  f,η smooth.
# But split the sum over rows by node TYPE:
#     S = S_hole(one-sided)  +  S_interior(centered)  +  S_outer(one-sided)
#   S_set = Σ_{i∈set} η_i (Wf)_i
# and FD-differentiate each w.r.t. a hole node's radial position.  If the
# broadband / high-frequency content of dS/dr_j lives in S_hole (the one-sided
# boundary rows), the artifact is a boundary-stencil instability — the kind
# ghost-node / Hermite stabilization targets, and which simply enlarging the
# (still one-sided) stencil should NOT fix.
#
# k-sweep (35→70→140) revisits the old "bigger stencils don't help" finding in
# this clean frame.
#
# Run:  jlrun plate_with_hole/probe_artifact_rowdecomp.jl   (from examples/)
# ============================================================================
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Macchiato
import RadialBasisFunctions: _build_weights, Partial, find_neighbors
using RadialBasisFunctions: PHS
using StaticArrays, LinearAlgebra, Printf, Statistics

const Lx = 4.0; const Ly = 4.0
const a0 = 0.40; const b0 = 0.20
const dx = 0.05
const nθ = max(48, round(Int, 2π*sqrt((a0^2 + b0^2)/2)/dx))
basis = PHS(3; poly_deg = 3)

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
const outer_idx    = collect((n_int+1):(n_int+n_outer))
const hole_idx     = collect((n_int+n_outer+1):N)

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

# dot of η,(Wf) restricted to a row-index set
Sset(Wf, set) = dot(view(ηvec, set), view(Wf, set))

function decompose(k)
    adjl = find_neighbors(pts0, k)
    ρ = mean(maximum(hypot(pts0[i][1]-pts0[c][1], pts0[i][2]-pts0[c][2]) for c in adjl[i])
             for i in hole_idx)
    Wf(pts) = ((Wxx,Wyy)=( _build_weights(Partial(2,1),pts,pts,adjl,basis),
                           _build_weights(Partial(2,2),pts,pts,adjl,basis));
               (Wxx+Wyy)*fvec)
    ε = 1e-5
    g_hole = zeros(nθ); g_int = zeros(nθ); g_out = zeros(nθ)
    for j in 1:nθ
        hp = copy(hole0); hp[j] = hole0[j] + ε .* r̂[j]
        hm = copy(hole0); hm[j] = hole0[j] - ε .* r̂[j]
        Wfp = Wf(vcat(interior_pts, outer_pts, hp))
        Wfm = Wf(vcat(interior_pts, outer_pts, hm))
        g_hole[j] = (Sset(Wfp, hole_idx)     - Sset(Wfm, hole_idx))     / (2ε)
        g_int[j]  = (Sset(Wfp, interior_idx) - Sset(Wfm, interior_idx)) / (2ε)
        g_out[j]  = (Sset(Wfp, outer_idx)    - Sset(Wfm, outer_idx))    / (2ε)
    end
    return ρ, g_hole, g_int, g_out
end

println("=== ∂W/∂x artifact — row-type decomposition + k-sweep ===")
println("N=$N  nθ=$nθ  (Nyquist m=$(nθ÷2))")
println("Row sets: HOLE (one-sided), INTERIOR (centered), OUTER (one-sided).\n")

for k in (35, 70, 140)
    ρ, gh, gi, go = decompose(k)
    @printf("---- k = %d   stencil radius ρ = %.4f  (≈ %.1f node-spacings) ----\n",
            k, ρ, ρ/dx)
    @printf("  %-9s %12s %12s %12s %10s\n", "row set", "‖g‖", "low(2-5)", "hi(top4)", "hi/low")
    for (name, g) in (("HOLE", gh), ("INTERIOR", gi), ("OUTER", go))
        @printf("  %-9s %12.3e %12.3e %12.3e %10.2f\n",
                name, totnorm(g), loband(g), hiband(g), hiband(g)/max(loband(g),1e-30))
    end
    tot = gh .+ gi .+ go
    @printf("  %-9s %12.3e %12.3e %12.3e %10.2f\n",
            "TOTAL", totnorm(tot), loband(tot), hiband(tot), hiband(tot)/max(loband(tot),1e-30))
    println()
end

println("Reading:")
println("  • If HOLE rows carry most of ‖g‖ and the high (hi/low) content, the")
println("    artifact is a one-sided boundary-stencil instability ⇒ Hermite/ghost.")
println("  • If hi/low for HOLE rows stays high as k grows 35→140, enlarging the")
println("    (still one-sided) stencil does NOT cure it (matches the old finding).")
