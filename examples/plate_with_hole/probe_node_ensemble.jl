# ============================================================================
# Node-distribution + placement-ensemble test.
#
# The artifact is ∂W/∂(node position).  Two ideas, one probe:
#  (a) Cartesian grid is a poor RBF-FD cloud (aligned/anisotropic stencils,
#      Test-1 staircase).  Does a single JITTERED (irregular) interior change it?
#  (b) The artifact is placement-correlated; the physical gradient is placement-
#      invariant.  Average the q=2 ∂W/∂x surrogate over K jittered interior
#      clouds (boundary fixed) — does ‖g_ensemble‖ collapse vs ‖g_single‖?
#
# If the ensemble collapses the artifact ⇒ placement-averaging is a parameter-
# free cure that ENABLES per-node freedom.  If not ⇒ stable bias ⇒ fall back to
# a smooth reduced design space.
#
# Run:  jlrun plate_with_hole/probe_node_ensemble.jl   (from examples/)
# ============================================================================
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Macchiato
import RadialBasisFunctions: _build_weights, Partial, find_neighbors
using RadialBasisFunctions: PHS
using StaticArrays, LinearAlgebra, Printf, Statistics, Random

const Lx = 4.0; const Ly = 4.0
const a0 = 0.40; const b0 = 0.20
const dx = 0.05
const nθ = max(48, round(Int, 2π*sqrt((a0^2 + b0^2)/2)/dx))
basis = PHS(3; poly_deg = 3); const k = 35
const K = 6           # ensemble size
const jit = 0.35      # interior jitter as a fraction of dx

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
const cart_interior = let v = SVector{2,Float64}[]
    for x in xs_grid, y in ys_grid
        ellipse_val(SVector(x,y), a0, b0) > margin^2 || continue
        push!(v, SVector(x,y))
    end
    v
end
const n_int = length(cart_interior)
const hole0 = [SVector(a0*cos(2π*j/nθ), b0*sin(2π*j/nθ)) for j in 0:(nθ-1)]
const interior_idx = collect(1:n_int)
const hole_idx     = collect((n_int+n_outer+1):(n_int+n_outer+nθ))

sf(p) = exp(0.30*p[1]) * cos(0.40*p[2])
sη(p) = exp(0.20*p[2]) * sin(0.30*p[1])
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

jitter(seed) = (rng = MersenneTwister(seed);
    [p + SVector((rand(rng)-0.5)*2*jit*dx, (rand(rng)-0.5)*2*jit*dx) for p in cart_interior])

# q=2 artifact (radial sensitivity of ηᵀ(Wxx+Wyy)f) on a given interior cloud
function artifact(interior)
    base = vcat(interior, outer_pts)
    adjl = find_neighbors(vcat(base, hole0), k)
    fvec = [sf(p) for p in vcat(base, hole0)]
    ηvec = [sη(p) for p in vcat(base, hole0)]
    lap(pts) = (_build_weights(Partial(2,1), pts, pts, adjl, basis) +
                _build_weights(Partial(2,2), pts, pts, adjl, basis)) * fvec
    ε = 1e-5; g = zeros(nθ)
    for j in 1:nθ
        hp = copy(hole0); hp[j] = hole0[j] + ε .* r̂[j]
        hm = copy(hole0); hm[j] = hole0[j] - ε .* r̂[j]
        g[j] = (dot(ηvec, lap(vcat(base, hp))) - dot(ηvec, lap(vcat(base, hm)))) / (2ε)
    end
    return g
end

println("=== Node-distribution + placement-ensemble test (q=2 artifact) ===")
println("N=$(n_int+n_outer+nθ)  nθ=$nθ  K=$K  jitter=±$(jit)·dx\n")

g_cart = artifact(cart_interior)
@printf("Cartesian grid : ‖g‖=%.3e  low=%.3e  hi=%.3e  hi/low=%.2f\n",
        totnorm(g_cart), loband(g_cart), hiband(g_cart), hiband(g_cart)/loband(g_cart))

println("\nJittered (irregular) interior clouds:")
gset = Vector{Vector{Float64}}()
for s in 1:K
    g = artifact(jitter(1000 + s)); push!(gset, g)
    @printf("  cloud %d : ‖g‖=%.3e  hi/low=%.2f\n", s, totnorm(g), hiband(g)/loband(g))
end

g_ens = sum(gset) ./ K
mean_single = mean(totnorm.(gset))
@printf("\nSingle jittered (mean over clouds): ‖g‖=%.3e\n", mean_single)
@printf("ENSEMBLE AVERAGE (K=%d):            ‖g‖=%.3e  low=%.3e  hi=%.3e  hi/low=%.2f\n",
        K, totnorm(g_ens), loband(g_ens), hiband(g_ens), hiband(g_ens)/loband(g_ens))
@printf("\n  ‖g_ensemble‖ / ‖g_single‖ = %.3f      (→0 means the artifact averages out)\n",
        totnorm(g_ens)/mean_single)
@printf("  ‖g_ensemble‖ / ‖g_cartesian‖ = %.3f\n", totnorm(g_ens)/totnorm(g_cart))

println("\nReading:")
println("  • Cartesian vs jittered ‖g‖: if jittered is similar, the artifact is")
println("    intrinsic, not a Cartesian-grid artifact of the probe.")
println("  • ‖g_ensemble‖/‖g_single‖ ≪ 1 ⇒ placement-averaging cancels the artifact")
println("    (it was placement-noise) ⇒ a parameter-free per-node cure.")
println("  • ratio ≈ 1 ⇒ stable bias ⇒ use a smooth reduced design space instead.")