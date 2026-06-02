# ============================================================================
# (2) Fairness test — isolate LSQ (over-determination) with the SAME PHS+poly
# basis, removing the poly-vs-PHS confound of the GMLS probe.
#
# One framework, local coords, PHS3 (r³) + poly_deg-3 (M=10 monomials), fixed
# stencil n=50.  The interpolant uses m RBF centers (nearest m of the stencil)
# + M polys, fit to all n stencil values:
#   • m+M = n  → square  → EXACT (interpolatory) differentiation
#   • m+M < n  → over-determined → LEAST-SQUARES differentiation
# Only the over-determination changes (kernel, poly degree, stencil all fixed),
# so a drop in the q=2 ∂W/∂x artifact is unambiguously the LSQ effect.
#
# weight row:  ℒf(x_e) ≈ c_evalᵀ (BᵀB)⁻¹ Bᵀ f  ⇒  w = B (BᵀB)⁻¹ c_eval
#   B[i,:]      = [ ‖ξ_i−ξ_c‖³ (centers) | monomials(ξ_i) ]   (ξ = local coords)
#   c_eval      = [ 9‖ξ_c‖ (∇²r³ at eval) | ∇²monomials(0) ]
#
# Run:  jlrun plate_with_hole/probe_lsq_rbf.jl   (from examples/)
# ============================================================================
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
import RadialBasisFunctions: _build_weights, Partial, find_neighbors
using RadialBasisFunctions: PHS
using StaticArrays, LinearAlgebra, SparseArrays, Printf

const Lx=4.0; const Ly=4.0; const a0=0.40; const b0=0.20; const dx=0.05
const nθ = max(48, round(Int, 2π*sqrt((a0^2+b0^2)/2)/dx))
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
const n_int=length(interior); const n_out=length(outer)
hole0 = [SVector(a0*cos(2π*j/nθ), b0*sin(2π*j/nθ)) for j in 0:(nθ-1)]
const pts0 = vcat(interior, outer, hole0); const N=length(pts0)
const interior_idx=collect(1:n_int); const hole_idx=collect((n_int+n_out+1):N)

sf(p)=exp(0.30*p[1])*cos(0.40*p[2]); sη(p)=exp(0.20*p[2])*sin(0.30*p[1])
const fvec=[sf(p) for p in pts0]; const ηvec=[sη(p) for p in pts0]
const lap_exact=[(0.30^2-0.40^2)*sf(p) for p in pts0]
θ=[atan(hole0[j][2],hole0[j][1]) for j in 1:nθ]; r̂=[hole0[j]/hypot(hole0[j]...) for j in 1:nθ]
famp(f,m)= m==0 ? abs(sum(f)/nθ) :
    hypot((2/nθ)*sum(f[j]*cos(m*θ[j]) for j in 1:nθ), (2/nθ)*sum(f[j]*sin(m*θ[j]) for j in 1:nθ))
hiband(f)=sqrt(sum(famp(f,m)^2 for m in (nθ÷2-3):(nθ÷2))); loband(f)=sqrt(sum(famp(f,m)^2 for m in 2:5))
totnorm(f)=norm(f .- sum(f)/nθ)

monos(ξ)=(x=ξ[1];y=ξ[2]; @SVector[1.0,x,y,x^2,x*y,y^2,x^3,x^2*y,x*y^2,y^3]); const M=10
const lp = @SVector[0.0,0,0,2,0,2,0,0,0,0]

# PHS3+poly3 differentiation, m centers (nearest), fixed stencil = adjl_n[i]
function rbf_L(pts, adjl_n, m)
    I=Int[]; J=Int[]; V=Float64[]
    for i in 1:N
        nbr = adjl_n[i]; xe = pts[i]
        ξ = [pts[c]-xe for c in nbr]
        ord = sortperm([hypot(v...) for v in ξ])          # nearest-first
        nbr = nbr[ord]; ξ = ξ[ord]; n = length(nbr)
        cen = ξ[1:m]
        B = Matrix{Float64}(undef, n, m+M)
        for a in 1:n
            for c in 1:m; B[a,c] = hypot((ξ[a]-cen[c])...)^3; end
            B[a, m+1:m+M] = monos(ξ[a])
        end
        ceval = Vector{Float64}(undef, m+M)
        for c in 1:m; ceval[c] = 9*hypot(cen[c]...); end     # ∇²(r³)=9r at eval (local 0)
        ceval[m+1:m+M] = lp
        wrow = B * ((B'B) \ ceval)
        append!(I, fill(i,n)); append!(J, nbr); append!(V, wrow)
    end
    return sparse(I,J,V,N,N)
end

phs_L(pts,adjl,basis)=_build_weights(Partial(2,1),pts,pts,adjl,basis)+_build_weights(Partial(2,2),pts,pts,adjl,basis)

function assess(name, Lbuild)
    lapf=Lbuild(pts0)*fvec
    acc=norm(lapf[interior_idx]-lap_exact[interior_idx])/norm(lap_exact[interior_idx])
    ε=1e-5; g=zeros(nθ)
    for j in 1:nθ
        hp=copy(pts0); hp[hole_idx[j]]=hole0[j]+ε.*r̂[j]
        hm=copy(pts0); hm[hole_idx[j]]=hole0[j]-ε.*r̂[j]
        g[j]=(dot(ηvec,Lbuild(hp)*fvec)-dot(ηvec,Lbuild(hm)*fvec))/(2ε)
    end
    @printf("  %-30s | acc=%.2e | ‖g‖=%.3e  hi/low=%.2f\n", name, acc, totnorm(g), hiband(g)/max(loband(g),1e-30))
    return (totnorm(g), hiband(g)/max(loband(g),1e-30))
end

println("=== (2) LSQ isolated: PHS3+poly3, fixed stencil n=50, vary centers m ===")
println("N=$N  nθ=$nθ  M=$M.  m+M=n ⇒ exact;  m+M<n ⇒ least-squares.\n")

const n_sten = 50
adjl50 = find_neighbors(pts0, n_sten)
basis = PHS(3; poly_deg=3)

assess("PHS colloc _build_weights k=50", p->phs_L(p, adjl50, basis))
println("  -- same framework (local coords, PHS3+poly3), vary over-determination --")
res = Tuple{Int,Float64,Float64}[]
for m in (40, 25, 15)          # m+M = 50(exact), 35, 25 ; osr=n/(m+M)=1.0,1.43,2.0
    r = assess("rbf m=$m centers (m+M=$(m+M), osr=$(round(n_sten/(m+M),digits=2)))", p->rbf_L(p, adjl50, m))
    push!(res, (m, r[1], r[2]))
end

println("\nOver-determination trend (same PHS+poly basis, n=50):")
for (m,g,hl) in res
    @printf("  m=%2d  m+M=%2d  osr=%.2f : ‖g‖=%.3e  hi/low=%.2f\n", m, m+M, n_sten/(m+M), g, hl)
end
println("\nReading: m+M=50 is exact (interpolatory) PHS+poly; m+M=35,25 are LSQ.")
println("If ‖g‖/hi-low drop as osr rises with the SAME PHS basis ⇒ the GMLS win was")
println("the LEAST-SQUARES over-determination, not poly-vs-PHS. Confirms the cure.")