# ============================================================================
# Stencil-conditioning check — does local shift/scale (the "unitary stencil"
# idea) have headroom?
#
# RBF.jl-1 builds the local saddle system [Φ P; Pᵀ 0] with the polynomial block
# evaluated at RAW GLOBAL coordinates (no shift to the eval point, no scaling).
# For a plate x,y∈[-2,2] with dx=0.05 the two blocks differ in scale by ~1e5,
# so the system is needlessly ill-conditioned — which inflates both the weights'
# roundoff AND the computed sensitivity ∂W/∂x.
#
# This rebuilds a few representative local collocation matrices (PHS3 r³ +
# poly_deg 3, k=35) in three coordinate frames and reports cond():
#   (a) GLOBAL   : raw coordinates (what RBF.jl-1 uses)
#   (b) SHIFTED  : centred at the eval point  (x - x_e)
#   (c) SHIFT+SCALE: (x - x_e)/h, h = mean stencil distance  ("unitary stencil")
#
# If (a) ≫ (c), local shift/scale is real, parameter-free headroom (a similarity
# transform: weights are mathematically unchanged, only conditioning/roundoff
# improves — and with a FIXED scale there are no position-dependent derivative
# terms, so the shape gradient stays clean).
#
# Run:  jlrun plate_with_hole/probe_stencil_conditioning.jl   (from examples/)
# ============================================================================
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
import RadialBasisFunctions: find_neighbors
using StaticArrays, LinearAlgebra, Printf

const Lx=4.0; const Ly=4.0; const a0=0.40; const b0=0.20; const dx=0.05
const nθ = max(48, round(Int, 2π*sqrt((a0^2+b0^2)/2)/dx)); const k=35
const margin = 1 + 1.2*dx/min(a0,b0)
ellipse_val(p,a,b)=(p[1]/a)^2+(p[2]/b)^2

interior = SVector{2,Float64}[]
for x in (-Lx/2+dx):dx:(Lx/2-dx/2), y in (-Ly/2+dx):dx:(Ly/2-dx/2)
    ellipse_val(SVector(x,y),a0,b0) > margin^2 || continue
    push!(interior, SVector(x,y))
end
outer = SVector{2,Float64}[]
for y in (-Ly/2):dx:(Ly/2); push!(outer,SVector(-Lx/2,y)); push!(outer,SVector(Lx/2,y)); end
for x in (-Lx/2+dx):dx:(Lx/2-dx); push!(outer,SVector(x,Ly/2)); push!(outer,SVector(x,-Ly/2)); end
hole = [SVector(a0*cos(2π*j/nθ), b0*sin(2π*j/nθ)) for j in 0:(nθ-1)]
pts = vcat(interior, outer, hole); N=length(pts)
n_int=length(interior); n_out=length(outer)
adjl = find_neighbors(pts, k)

# degree-3 2D monomials at a point p
monos(p) = (x=p[1]; y=p[2]; SVector(1.0, x, y, x^2, x*y, y^2, x^3, x^2*y, x*y^2, y^3))
const nmon = 10

# build saddle matrix for a stencil given a coordinate transform
function saddle(stencil_pts, transform)
    q = [transform(p) for p in stencil_pts]
    kk = length(q)
    Φ = [hypot((q[i]-q[j])...)^3 for i in 1:kk, j in 1:kk]      # PHS3 r³
    P = reduce(vcat, [monos(q[i])' for i in 1:kk])              # kk × nmon
    M = [Φ P; P' zeros(nmon,nmon)]
    return cond(M)
end

# representative eval points: interior-near-hole, interior-far, hole-boundary, outer
near = interior[argmin([minimum(hypot((interior[i]-h)...) for h in hole) for i in 1:n_int])]
far  = interior[argmax([minimum(hypot((interior[i]-h)...) for h in hole) for i in 1:n_int])]
samples = [(:interior_near_hole, findfirst(==(near), pts)),
           (:interior_far,       findfirst(==(far),  pts)),
           (:hole_boundary,      n_int+n_out+1),
           (:outer_boundary,     n_int+1)]

println("=== Local stencil conditioning: GLOBAL vs SHIFTED vs SHIFT+SCALE ===")
println("PHS3 r³ + poly_deg 3, k=$k.  (RBF.jl-1 uses the GLOBAL frame.)\n")
@printf("%-20s %14s %14s %14s %10s\n", "stencil @", "cond GLOBAL", "cond SHIFTED", "cond SH+SCALE", "‖x_e‖")
println("-"^76)
for (name, idx) in samples
    nbr = adjl[idx]
    spts = pts[nbr]
    xe = pts[idx]
    h = sum(hypot((p-xe)...) for p in spts)/length(spts)
    cg = saddle(spts, p -> p)                       # global
    cs = saddle(spts, p -> p - xe)                  # shifted to eval point
    cz = saddle(spts, p -> (p - xe)/h)              # shifted + scaled (unitary)
    @printf("%-20s %14.3e %14.3e %14.3e %10.3f\n", string(name), cg, cs, cz, hypot(xe...))
end
println("\nReading: if GLOBAL ≫ SHIFT+SCALE, local nondimensionalization is real,")
println("parameter-free headroom (standard RBF-FD practice RBF.jl-1 omits). Whether")
println("it cleans the ARTIFACT depends on how much of ∂W/∂x is roundoff (effect of")
println("conditioning) vs the kernel-intrinsic true sensitivity — but it can only help.")