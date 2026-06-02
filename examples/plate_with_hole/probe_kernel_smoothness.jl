# ============================================================================
# Cure-test — does a SMOOTHER kernel remove the q=2 ∂W/∂x artifact?
#
# Test 2b showed the artifact is driven by the gradient pullback probing 3rd-
# order kernel derivatives (q=2 operator → q+1=3).  PHS r³ is only C², so its
# 3rd derivative is singular (~1/r) at the stencil centre.  PHS5=r⁵ (C⁴) and
# PHS7=r⁷ (C⁶) have SMOOTH 3rd derivatives, so the q=2 artifact should drop
# sharply — a parameter-free fix (discrete kernel choice, no tuning knob).
#
# For each kernel (poly_deg=3): measure
#   (a) forward accuracy: RBF-FD Laplacian of a smooth field vs analytic ∇²f
#       (interior rows AND one-sided hole rows separately), and
#   (b) the q=2 artifact: ∂(ηᵀ(Wxx+Wyy)f)/∂r_j — ‖g‖ and hi/low (total+interior).
#
# Run:  jlrun plate_with_hole/probe_kernel_smoothness.jl   (from examples/)
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
const k = 35

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

sf(p) = exp(0.30*p[1]) * cos(0.40*p[2])              # ∇²f = (0.3²−0.4²) f
sη(p) = exp(0.20*p[2]) * sin(0.30*p[1])
const fvec = [sf(p) for p in pts0]
const ηvec = [sη(p) for p in pts0]
const lap_exact = [(0.30^2 - 0.40^2)*sf(p) for p in pts0]
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

function lap_of(basis, pts)
    (_build_weights(Partial(2,1), pts, pts, adjl, basis) +
     _build_weights(Partial(2,2), pts, pts, adjl, basis)) * fvec
end

function assess(name, basis)
    # (a) forward accuracy
    lapf = lap_of(basis, pts0)
    err_int  = norm(lapf[interior_idx] - lap_exact[interior_idx]) / norm(lap_exact[interior_idx])
    err_hole = norm(lapf[hole_idx]     - lap_exact[hole_idx])     / norm(lap_exact[hole_idx])
    # (b) q=2 artifact
    ε = 1e-5; g_tot = zeros(nθ); g_int = zeros(nθ)
    for j in 1:nθ
        hp = copy(hole0); hp[j] = hole0[j] + ε .* r̂[j]
        hm = copy(hole0); hm[j] = hole0[j] - ε .* r̂[j]
        Wfp = lap_of(basis, vcat(interior_pts, outer_pts, hp))
        Wfm = lap_of(basis, vcat(interior_pts, outer_pts, hm))
        g_tot[j] = (dot(ηvec, Wfp)          - dot(ηvec, Wfm))          / (2ε)
        g_int[j] = (Sset(Wfp, interior_idx) - Sset(Wfm, interior_idx)) / (2ε)
    end
    @printf("  %-7s | accuracy: int=%.2e hole=%.2e | artifact: ‖g‖=%.3e  hi/low(tot)=%.2f  hi/low(int)=%.2f  hi_int=%.3e\n",
            name, err_int, err_hole, totnorm(g_tot),
            hiband(g_tot)/max(loband(g_tot),1e-30),
            hiband(g_int)/max(loband(g_int),1e-30), hiband(g_int))
    return (name, totnorm(g_tot), hiband(g_int))
end

println("=== Kernel smoothness vs q=2 ∂W/∂x artifact (poly_deg=3, k=$k) ===")
println("N=$N  nθ=$nθ.  PHS3=r³(C²)  PHS5=r⁵(C⁴)  PHS7=r⁷(C⁶).")
println("Pullback of the Laplacian (q=2) needs the 3rd kernel derivative.\n")

r3 = assess("PHS3", PHS(3; poly_deg = 3))
r5 = assess("PHS5", PHS(5; poly_deg = 3))
r7 = assess("PHS7", PHS(7; poly_deg = 3))

@printf("\nArtifact ‖g‖:   PHS3=%.3e  PHS5=%.3e  PHS7=%.3e   (PHS5/PHS3=%.2f, PHS7/PHS3=%.2f)\n",
        r3[2], r5[2], r7[2], r5[2]/r3[2], r7[2]/r3[2])
@printf("Interior hi-band: PHS3=%.3e  PHS5=%.3e  PHS7=%.3e   (PHS5/PHS3=%.2f, PHS7/PHS3=%.2f)\n",
        r3[3], r5[3], r7[3], r5[3]/r3[3], r7[3]/r3[3])
println("\nReading:")
println("  • If PHS5/PHS7 cut ‖g‖ and the interior hi-band sharply while forward")
println("    accuracy stays good ⇒ smoother kernel is the parameter-free fix:")
println("    use it for shape-opt so the q+1 pullback derivative is non-singular.")