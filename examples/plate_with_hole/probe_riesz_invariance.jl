# ============================================================================
# Generic parameter-free fix + discretization-invariance test.
#
# 1. Discrete adjoint gradient (here: thermal compliance, FD radial — physically
#    normalized, so comparable across resolutions).  Any PDE works identically.
# 2. Convert to a shape-gradient DENSITY  d_j = (dJ/dr_j)/w_j , w_j = arc-length
#    quadrature weight, so d is an intensive field on the boundary (the discrete
#    estimate of the continuous Hadamard density).
# 3. Riesz/Sobolev step in a Laplace–Beltrami metric:
#        (I − ℓ² Δ_Γ) g_smooth = d ,    ℓ = c·ρ
#    Δ_Γ = boundary Laplace–Beltrami (here the periodic arc-length 1D Laplacian;
#    in 3D the surface LB — same RBF-FD-discretizable operator).  ρ = local
#    stencil radius (auto, ∝h).  c = ONE fixed O(1) constant, set once, never
#    per-case.  ℓ tracks the 1/h^p artifact growth automatically.
#
# TEST: the raw nodal gradient is ill-posed (the ∂W/∂x artifact sharpens like
# 1/h^p).  A correct, auto-calibrated fix must make the descent direction
# DISCRETIZATION-INVARIANT: g_smooth's low modes must converge as dx→0, and its
# high-frequency band must be suppressed, with the SAME c at every resolution.
#
# Run:  jlrun plate_with_hole/probe_riesz_invariance.jl   (from examples/)
# ============================================================================
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Macchiato
import RadialBasisFunctions: _build_weights, Partial, find_neighbors
using RadialBasisFunctions: PHS
using StaticArrays, LinearAlgebra, SparseArrays, Printf, Statistics

const Lx = 4.0; const Ly = 4.0
const a0 = 0.40; const b0 = 0.20
const G  = 1.0
const c_riesz = 1.0          # the ONE fixed constant: ℓ = c_riesz · ρ(stencil)
basis = PHS(3; poly_deg = 3); const k = 35
const r_eff = sqrt((a0^2 + b0^2)/2)
ellipse_val(p, a, b) = (p[1]/a)^2 + (p[2]/b)^2
flat(v) = reduce(vcat, [[p[1], p[2]] for p in v])

# ---- build cloud for a given dx (boundary scales: nθ ∝ 1/dx) ---------------
function build_cloud(dx)
    nθ     = max(8, round(Int, 2π*r_eff/dx))
    margin = 1 + 1.2*dx/min(a0, b0)
    xs = collect((-Lx/2+dx):dx:(Lx/2-dx/2)); ys = collect((-Ly/2+dx):dx:(Ly/2-dx/2))
    interior = SVector{2,Float64}[]
    for x in xs, y in ys
        ellipse_val(SVector(x,y), a0, b0) > margin^2 || continue
        push!(interior, SVector(x,y))
    end
    outer = SVector{2,Float64}[]; tag = Symbol[]
    for y in (-Ly/2):dx:(Ly/2)
        push!(outer, SVector(-Lx/2,y)); push!(tag,:xlo)
        push!(outer, SVector(Lx/2,y));  push!(tag,:xhi)
    end
    for x in (-Lx/2+dx):dx:(Lx/2-dx)
        push!(outer, SVector(x,Ly/2));  push!(tag,:yhi)
        push!(outer, SVector(x,-Ly/2)); push!(tag,:ylo)
    end
    hole = [SVector(a0*cos(2π*j/nθ), b0*sin(2π*j/nθ)) for j in 0:(nθ-1)]
    pts  = vcat(interior, outer, hole)
    N = length(pts); n_int = length(interior); n_out = length(outer)
    interior_idx = collect(1:n_int)
    outer_idx    = collect((n_int+1):(n_int+n_out))
    hole_idx     = collect((n_int+n_out+1):N)
    adjl = find_neighbors(pts, k)
    pin  = interior_idx[argmin([hypot(pts[i][1], pts[i][2]-1.5) for i in interior_idx])]
    ρ = mean(maximum(hypot(pts[i][1]-pts[c][1], pts[i][2]-pts[c][2]) for c in adjl[i])
             for i in hole_idx)
    return (; dx, nθ, N, pts, interior, outer, tag, hole, n_int, n_out,
            interior_idx, outer_idx, hole_idx, adjl, pin, ρ)
end

outer_normal(t) = t===:xhi ? SVector(1.0,0.0) : t===:xlo ? SVector(-1.0,0.0) :
                  t===:yhi ? SVector(0.0,1.0) : SVector(0.0,-1.0)

# ---- thermal problem: DIRICHLET drive (well-posed, no pin), insulated hole --
#   left T=+1, right T=-1, top/bottom + hole insulated, interior ∇²T=0.
#   Objective J = heat outflow through the right edge = ∮_xhi (n·∇T) ds
#   (a proper arc-length-weighted flux integral ⇒ resolution-convergent).
function solve_J(C, hole_pts)
    pts = vcat(C.interior, C.outer, hole_pts); N = C.N; dx = C.dx
    Wxx = _build_weights(Partial(2,1), pts, pts, C.adjl, basis)
    Wyy = _build_weights(Partial(2,2), pts, pts, C.adjl, basis)
    Wdx = _build_weights(Partial(1,1), pts, pts, C.adjl, basis)
    Wdy = _build_weights(Partial(1,2), pts, pts, C.adjl, basis)
    hn, _ = polyline_normals(flat(pts), C.hole_idx, collect(1:C.nθ))

    insul = Int[]; dir = Int[]; nx = zeros(N); ny = zeros(N)
    for (kk,i) in enumerate(C.outer_idx)
        t = C.tag[kk]
        if t === :xlo || t === :xhi
            push!(dir, i)
        else
            push!(insul, i); nn = outer_normal(t); nx[i]=nn[1]; ny[i]=nn[2]
        end
    end
    for (kk,i) in enumerate(C.hole_idx); push!(insul, i); nn=hn[kk]; nx[i]=nn[1]; ny[i]=nn[2]; end

    Rint = sparse(C.interior_idx, C.interior_idx, ones(C.n_int), N, N)
    Rins = sparse(insul, insul, ones(length(insul)), N, N)
    Rdir = sparse(dir,   dir,   ones(length(dir)),   N, N)
    K = Rint*(Wxx+Wyy) + Rins*(spdiagm(0=>nx)*Wdx + spdiagm(0=>ny)*Wdy) + Rdir
    b = zeros(N)
    for (kk,i) in enumerate(C.outer_idx)
        C.tag[kk]===:xlo && (b[i]=+1.0); C.tag[kk]===:xhi && (b[i]=-1.0)
    end
    T = lu(K) \ b
    flux = Wdx * T                      # n=(1,0) on the right edge ⇒ n·∇T = ∂T/∂x
    right = [i for (kk,i) in enumerate(C.outer_idx) if C.tag[kk]===:xhi]
    return dx * sum(flux[i] for i in right)
end

# ---- radial shape-gradient DENSITY on the hole loop ------------------------
function grad_density(C)
    nθ = C.nθ; hole = C.hole
    θ  = [atan(hole[j][2], hole[j][1]) for j in 1:nθ]
    r̂  = [hole[j]/hypot(hole[j]...)    for j in 1:nθ]
    s  = [hypot(hole[mod1(j+1,nθ)][1]-hole[j][1], hole[mod1(j+1,nθ)][2]-hole[j][2]) for j in 1:nθ]
    w  = [(s[mod1(j-1,nθ)] + s[j])/2 for j in 1:nθ]          # arc-length quadrature
    ε  = 1e-5; g_rad = zeros(nθ)
    for j in 1:nθ
        hp = copy(hole); hp[j] = hole[j] + ε .* r̂[j]
        hm = copy(hole); hm[j] = hole[j] - ε .* r̂[j]
        g_rad[j] = (solve_J(C, hp) - solve_J(C, hm)) / (2ε)
    end
    return θ, g_rad ./ w, s          # density = nodal grad / quadrature weight
end

# ---- Laplace–Beltrami Riesz smoothing on the loop --------------------------
# (I − ℓ²Δ_Γ) g = d ,  Δ_Γ = periodic arc-length 1D Laplacian (negative-def).
function riesz_smooth(d, s, ρ)
    nθ = length(d); ℓ = c_riesz * ρ
    L = zeros(nθ, nθ)
    for j in 1:nθ
        sm = s[mod1(j-1,nθ)]; sp = s[j]; h = (sm+sp)/2
        L[j, mod1(j-1,nθ)] +=  1.0/(sm*h)
        L[j, j]            += -(1.0/(sm*h) + 1.0/(sp*h))
        L[j, mod1(j+1,nθ)] +=  1.0/(sp*h)
    end
    return (I - ℓ^2 .* L) \ d
end

# ---- Fourier amplitude of a nodal field over θ -----------------------------
function famp(f, θ, m)
    nθ = length(f)
    m == 0 && return abs(sum(f)/nθ)
    hypot((2/nθ)*sum(f[j]*cos(m*θ[j]) for j in 1:nθ),
          (2/nθ)*sum(f[j]*sin(m*θ[j]) for j in 1:nθ))
end
hiband(f,θ,nθ) = sqrt(sum(famp(f,θ,m)^2 for m in (nθ÷2-3):(nθ÷2)))
loband(f,θ)    = sqrt(sum(famp(f,θ,m)^2 for m in 2:5))
fullnorm(f)    = norm(f .- sum(f)/length(f))     # drop the m0 (area) part
# signed low-mode coefficient vector (m=2..6) — a resolution-independent
# fingerprint of the descent DIRECTION; compared by cosine similarity.
function coeffs(f, θ; mmax=6)
    nθ = length(f); v = Float64[]
    for m in 2:mmax
        push!(v, (2/nθ)*sum(f[j]*cos(m*θ[j]) for j in 1:nθ))
        push!(v, (2/nθ)*sum(f[j]*sin(m*θ[j]) for j in 1:nθ))
    end
    return v
end
cossim(a,b) = dot(a,b)/(norm(a)*norm(b) + 1e-300)

# ============================================================================
println("=== Laplace–Beltrami Riesz fix — discretization-invariance test ===")
println("ℓ = c·ρ with c = $c_riesz (fixed). Density = (dJ/dr)/arc-weight.\n")

dxs = (0.10, 0.0707, 0.05)
rows = []
for dx in dxs
    C = build_cloud(dx)
    θ, d, s = grad_density(C)
    g = riesz_smooth(d, s, C.ρ)
    push!(rows, (; dx, nθ=C.nθ, ρ=C.ρ, θ, d, g))
end

# (1) artifact suppression + low-mode dominance — both scale-free ratios.
println("Artifact suppression (hi-band / low-band) and low-mode dominance:")
@printf("%8s %4s %8s | %12s %12s | %12s %12s\n",
        "dx","nθ","ρ","raw hi/low","smo hi/low","raw low-frac","smo low-frac")
println("-"^80)
for r in rows
    rhl = hiband(r.d,r.θ,r.nθ)/loband(r.d,r.θ)
    shl = hiband(r.g,r.θ,r.nθ)/loband(r.g,r.θ)
    rlf = loband(r.d,r.θ)/fullnorm(r.d)
    slf = loband(r.g,r.θ)/fullnorm(r.g)
    @printf("%8.4f %4d %8.4f | %12.3f %12.3f | %12.3f %12.3f\n",
            r.dx, r.nθ, r.ρ, rhl, shl, rlf, slf)
end

# (2) does the descent DIRECTION converge across resolutions? cosine similarity
# of the (scale-free) low-mode coefficient vectors between consecutive dx.
println("\nDescent-direction stability across resolutions (cosine of m2..6 coeff vectors):")
for i in 2:length(rows)
    cr = cossim(coeffs(rows[i-1].d, rows[i-1].θ), coeffs(rows[i].d, rows[i].θ))
    cg = cossim(coeffs(rows[i-1].g, rows[i-1].θ), coeffs(rows[i].g, rows[i].θ))
    @printf("  dx %.4f→%.4f:  raw cos = %+.4f   smoothed cos = %+.4f\n",
            rows[i-1].dx, rows[i].dx, cr, cg)
end

println("\nReading:")
println("  • smo hi/low ≪ raw hi/low and smo low-frac→1  ⇒ the Riesz step removes")
println("    the ∂W/∂x artifact and leaves a low-mode (smooth) descent direction.")
println("  • smoothed cos → 1 (vs lower/erratic raw cos)  ⇒ the descent DIRECTION is")
println("    discretization-invariant with a single fixed c, ℓ=ρ (auto-calibrated).")
