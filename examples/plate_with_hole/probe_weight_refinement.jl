# ============================================================================
# Refinement scan of the ∂W/∂x artifact (physics-free).
#
# Repeats the probe_weight_sensitivity surrogate  S = ηᵀ(Wxx+Wyy)f  (smooth f,η)
# at several node spacings dx, with the boundary resolution scaled too
# (nθ ∝ 1/dx), and reports the radial weight-sensitivity spectrum dS/dr_j.
#
# Goal — extract the auto-calibration law:
#   1. Does the artifact (high-mode band) shrink as h→0 (consistency error)?
#   2. Does the low-mode (smooth) part converge?
#   3. Is the crossover frequency invariant when expressed in STENCIL-RADIUS
#      units (ρ/λ)?  If yes, the stencil radius is the parameter-free smoothing
#      length for a generic Riesz/Sobolev fix.
#
# Run:  jlrun plate_with_hole/probe_weight_refinement.jl   (from examples/)
# ============================================================================
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Macchiato
import RadialBasisFunctions: _build_weights, Partial, find_neighbors
using RadialBasisFunctions: PHS
using StaticArrays, LinearAlgebra, Printf, Statistics

const Lx = 4.0; const Ly = 4.0
const a0 = 0.40; const b0 = 0.20
basis = PHS(3; poly_deg = 3); const k = 35
const r_eff = sqrt((a0^2 + b0^2)/2)

sf(p) = exp(0.30*p[1]) * cos(0.40*p[2])
sη(p) = exp(0.20*p[2]) * sin(0.30*p[1])

ellipse_val(p, a, b) = (p[1]/a)^2 + (p[2]/b)^2

function run_dx(dx)
    nθ     = max(8, round(Int, 2π*r_eff/dx))          # boundary scales with dx
    margin = 1 + 1.2*dx/min(a0, b0)
    xs = collect((-Lx/2+dx):dx:(Lx/2-dx/2))
    ys = collect((-Ly/2+dx):dx:(Ly/2-dx/2))

    interior = SVector{2,Float64}[]
    for x in xs, y in ys
        ellipse_val(SVector(x,y), a0, b0) > margin^2 || continue
        push!(interior, SVector(x,y))
    end
    outer = SVector{2,Float64}[]
    for y in (-Ly/2):dx:(Ly/2)
        push!(outer, SVector(-Lx/2, y)); push!(outer, SVector(Lx/2, y))
    end
    for x in (-Lx/2+dx):dx:(Lx/2-dx)
        push!(outer, SVector(x, Ly/2)); push!(outer, SVector(x, -Ly/2))
    end
    hole = [SVector(a0*cos(2π*j/nθ), b0*sin(2π*j/nθ)) for j in 0:(nθ-1)]
    pts  = vcat(interior, outer, hole)
    N = length(pts); n_int = length(interior)
    hole_idx = collect((N-nθ+1):N)
    adjl = find_neighbors(pts, k)

    # stencil radius ρ and boundary node spacing s on the hole
    ρ = mean(maximum(hypot(pts[i][1]-pts[c][1], pts[i][2]-pts[c][2]) for c in adjl[i])
             for i in hole_idx)
    s = mean(hypot(hole[mod1(j+1,nθ)][1]-hole[j][1], hole[mod1(j+1,nθ)][2]-hole[j][2])
             for j in 1:nθ)

    fvec = [sf(p) for p in pts]; ηvec = [sη(p) for p in pts]

    function S(hpts)
        p2 = vcat(interior, outer, hpts)
        Wxx = _build_weights(Partial(2,1), p2, p2, adjl, basis)
        Wyy = _build_weights(Partial(2,2), p2, p2, adjl, basis)
        return dot(ηvec, (Wxx + Wyy) * fvec)
    end

    θ  = [atan(hole[j][2], hole[j][1]) for j in 1:nθ]
    r̂  = [hole[j]/hypot(hole[j]...)    for j in 1:nθ]
    ε  = 1e-5
    g  = zeros(nθ)
    for j in 1:nθ
        hp = copy(hole); hp[j] = hole[j] + ε .* r̂[j]
        hm = copy(hole); hm[j] = hole[j] - ε .* r̂[j]
        g[j] = (S(hp) - S(hm)) / (2ε)
    end

    amp = [ (m==0 ? abs((1/nθ)*sum(g)) :
             hypot((2/nθ)*sum(g[j]*cos(m*θ[j]) for j in 1:nθ),
                   (2/nθ)*sum(g[j]*sin(m*θ[j]) for j in 1:nθ)))
            for m in 0:(nθ÷2) ]
    return (dx=dx, nθ=nθ, N=N, ρ=ρ, s=s, amp=amp, θ=θ, g=g)
end

println("=== ∂W/∂x refinement scan ===\n")
results = [run_dx(dx) for dx in (0.10, 0.0707, 0.05)]

@printf("%8s %5s %7s %8s %8s %8s | %10s %10s %10s %8s\n",
        "dx", "nθ", "N", "ρ(sten)", "s(bnd)", "ρ/s",
        "m2 amp", "low(2-6)", "hi(top6)", "hi/low")
println("-"^96)
for r in results
    A = r.amp
    m2   = A[3]
    low  = sqrt(sum(abs2, A[3:min(7,end)]))
    hi   = sqrt(sum(abs2, A[max(1,end-5):end]))
    @printf("%8.4f %5d %7d %8.4f %8.4f %8.3f | %10.3e %10.3e %10.3e %8.2f\n",
            r.dx, r.nθ, r.N, r.ρ, r.s, r.ρ/r.s, m2, low, hi, hi/low)
end

# crossover frequency in stencil-radius units: smallest m where the amplitude
# stops decaying and joins the artifact floor (amp[m] > 0.5*max(high half)).
println("\nPer-resolution spectrum tail + crossover (λ_m = 2π·r̄/m, ρ/λ_m):")
for r in results
    A = r.amp; M = length(A)-1
    floor_amp = median(A[(M÷2+1):end])
    mcross = findfirst(m -> A[m+1] >= floor_amp, 2:M)
    mcross = mcross === nothing ? M : mcross+1
    λ = 2π*r_eff/mcross
    @printf("  dx=%.4f: artifact floor≈%.3e, crossover m*=%2d  →  λ*=%.4f  ρ/λ*=%.3f  (Nyq m=%d)\n",
            r.dx, floor_amp, mcross, λ, r.ρ/λ, r.nθ÷2)
end

println("\nReading:")
println("  • hi(top6) shrinking with dx ⇒ artifact is a finite-h consistency error.")
println("  • m2 amp converging ⇒ the smooth/physical part is consistent.")
println("  • ρ/λ* roughly constant across dx ⇒ stencil radius is the natural,")
println("    auto-calibrated smoothing length for the generic Riesz fix.")
