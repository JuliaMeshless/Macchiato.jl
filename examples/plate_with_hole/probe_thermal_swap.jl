# ============================================================================
# Probe A — THERMAL SWAP: is the Nyquist rise in the shape gradient generic
# across physics, or special to elasticity?
#
# Same plate-with-hole cloud as probe_fourier_modes.jl, but the PDE is scalar
# Laplace ∇²T = 0 (steady heat).  Insulated hole (∂T/∂n=0); balanced through-
# flux on left/right outer edges (heat-flow analog of the traction load);
# insulated top/bottom; one pinned node for the temperature datum.
# Objective J = bᵀT (thermal compliance).
#
# Shape gradient by FINITE DIFFERENCE on the hole nodes (radial), with FIXED
# adjacency and normals rebuilt — identical protocol to probe_fourier_modes.jl.
# FD is the true discrete gradient (FD=AD established for elasticity), and being
# physics-agnostic it is the clean way to compare a DIFFERENT PDE.
#
# Verdict: if dJ/dr_j shows the SAME rise toward the Nyquist mode as elasticity,
# the mechanism is generic to RBF-FD shape gradients (not the physics).
#
# Run:  jlrun plate_with_hole/probe_thermal_swap.jl   (from examples/)
# ============================================================================
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Macchiato
import RadialBasisFunctions: _build_weights, Partial, MixedPartial, find_neighbors
using RadialBasisFunctions: PHS
using StaticArrays, LinearAlgebra, SparseArrays, Printf

const Lx = 4.0; const Ly = 4.0
const a0 = 0.40; const b0 = 0.20
const dx = 0.05
const G  = 1.0                      # through-flux magnitude
const nθ = max(48, round(Int, 2π*sqrt((a0^2 + b0^2)/2)/dx))
basis = PHS(3; poly_deg = 3); const k = 35

const margin  = 1 + 1.2*dx/min(a0, b0)
const xs_grid = collect((-Lx/2+dx):dx:(Lx/2-dx/2))
const ys_grid = collect((-Ly/2+dx):dx:(Ly/2-dx/2))

# ---- cloud (identical layout to probe_fourier_modes.jl) --------------------
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
const hole_pos     = collect(1:nθ)
const adjl_ref     = find_neighbors(pts0, k)

# datum pin (interior node nearest centre-ish, away from hole)
nearest(p0,pool) = pool[argmin([hypot(pts0[i][1]-p0[1], pts0[i][2]-p0[2]) for i in pool])]
const pin = nearest((0.0, 1.5), interior_idx)

flat(v) = reduce(vcat, [[p[1], p[2]] for p in v])

# ---- thermal solve, compliance --------------------------------------------
# K T = b.  interior: ∇²T=0.  boundary rows: n·∇T = b_i  (flux).  pin: T=0.
function solve_J(hole_pts)
    pts = vcat(interior_pts, outer_pts, hole_pts)
    Wxx = _build_weights(Partial(2,1),     pts, pts, adjl_ref, basis)
    Wyy = _build_weights(Partial(2,2),     pts, pts, adjl_ref, basis)
    Wdx = _build_weights(Partial(1,1),     pts, pts, adjl_ref, basis)
    Wdy = _build_weights(Partial(1,2),     pts, pts, adjl_ref, basis)
    Wlap = Wxx + Wyy

    hn, _ = polyline_normals(flat(pts), hole_idx, hole_pos)

    nx = zeros(N); ny = zeros(N)
    for (kk,i) in enumerate(outer_idx); nn = outer_normal(kk); nx[i]=nn[1]; ny[i]=nn[2]; end
    for (kk,i) in enumerate(hole_idx);  nn = hn[kk];           nx[i]=nn[1]; ny[i]=nn[2]; end

    Rint = sparse(interior_idx, interior_idx, ones(n_int), N, N)
    bnd  = vcat(outer_idx, hole_idx)
    Rbnd = sparse(bnd, bnd, ones(length(bnd)), N, N)
    NX = spdiagm(0 => nx); NY = spdiagm(0 => ny)

    K = Rint*Wlap + Rbnd*(NX*Wdx + NY*Wdy)
    b = zeros(N)
    for (kk,i) in enumerate(outer_idx)          # balanced through-flux on x-edges
        outer_tag[kk]===:xlo && (b[i] = +G)
        outer_tag[kk]===:xhi && (b[i] = -G)
    end                                          # top/bottom + hole: insulated (b=0)

    K[pin, :] .= 0.0; K[pin, pin] = 1.0; b[pin] = 0.0   # temperature datum
    T = lu(K) \ b                                        # sparse (UMFPACK)
    return dot(b, T)
end

# ============================================================================
println("=== Thermal-swap shape-gradient spectrum (FD, fixed cloud) ===")
println("N=$N  nθ=$nθ  (Nyquist mode = $(nθ÷2))\n")

θ = [atan(hole0[j][2], hole0[j][1]) for j in 1:nθ]
r̂ = [hole0[j]/hypot(hole0[j]...)    for j in 1:nθ]

const ε = 1e-5
J0 = solve_J(hole0)
@printf("J0 = %.6e   (thermal compliance)\n", J0)
g_rad = zeros(nθ)
for j in 1:nθ
    hp = copy(hole0); hp[j] = hole0[j] + ε .* r̂[j]
    hm = copy(hole0); hm[j] = hole0[j] - ε .* r̂[j]
    g_rad[j] = (solve_J(hp) - solve_J(hm)) / (2ε)
end

println("\n--- radial shape-gradient Fourier spectrum (amplitude per mode) ---")
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
println("\nVerdict: ratio hi/lo ≫ 1 (rise toward Nyquist) ⇒ the noise mechanism is")
println("generic to RBF-FD shape gradients, not specific to elasticity.")
