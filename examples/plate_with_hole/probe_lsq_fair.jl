# ============================================================================
# (2, corrected) Separate STENCIL/averaging size from the LSQ construction.
#
# Earlier I confounded the two (GMLS n=50 vs PHS colloc k=35). Here: sweep the
# stencil size s for BOTH methods on the SAME cloud & metric, q=2 artifact:
#   • PHS3+poly3 EXACT collocation, k=s
#   • GMLS poly_deg-3 LEAST-SQUARES, n=s
# If both curves fall with s and the GMLS curve sits clearly below collocation
# at MATCHED s ⇒ LSQ adds a genuine reduction beyond averaging. If they coincide
# ⇒ it was just stencil/averaging size. (Residual poly-vs-PHS confound noted —
# but this answers the size question, which is the one that bit me.)
#
# Run:  jlrun plate_with_hole/probe_lsq_fair.jl   (from examples/)
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

phs_L(pts,adjl,basis)=_build_weights(Partial(2,1),pts,pts,adjl,basis)+_build_weights(Partial(2,2),pts,pts,adjl,basis)
function gmls_L(pts, adjl_n)
    I=Int[]; J=Int[]; V=Float64[]
    for i in 1:N
        nbr=adjl_n[i]; xe=pts[i]; n=length(nbr)
        P=Matrix{Float64}(undef,n,M); for a in 1:n; P[a,:]=monos(pts[nbr[a]]-xe); end
        wrow = P*((P'P)\lp)
        append!(I,fill(i,n)); append!(J,nbr); append!(V,wrow)
    end
    sparse(I,J,V,N,N)
end

function artifact(Lbuild)
    lapf=Lbuild(pts0)*fvec
    acc=norm(lapf[interior_idx]-lap_exact[interior_idx])/norm(lap_exact[interior_idx])
    ε=1e-5; g=zeros(nθ)
    for j in 1:nθ
        hp=copy(pts0); hp[hole_idx[j]]=hole0[j]+ε.*r̂[j]
        hm=copy(pts0); hm[hole_idx[j]]=hole0[j]-ε.*r̂[j]
        g[j]=(dot(ηvec,Lbuild(hp)*fvec)-dot(ηvec,Lbuild(hm)*fvec))/(2ε)
    end
    (acc, totnorm(g), hiband(g)/max(loband(g),1e-30))
end

println("=== (2c) stencil-size-controlled: PHS collocation vs GMLS LSQ ===")
println("N=$N  nθ=$nθ.  Same cloud, same metric; sweep stencil size s.\n")
basis = PHS(3; poly_deg=3)
@printf("%5s | %-28s | %-28s\n", "s", "PHS colloc  (acc, ‖g‖, hi/low)", "GMLS LSQ    (acc, ‖g‖, hi/low)")
println("-"^70)
for s in (25, 50, 75, 100)
    adjs = find_neighbors(pts0, s)
    ap,gp,hp = artifact(p->phs_L(p, adjs, basis))
    ag,gg,hg = artifact(p->gmls_L(p, adjs))
    @printf("%5d | %.2e  %.3e  %.2f      | %.2e  %.3e  %.2f\n", s, ap,gp,hp, ag,gg,hg)
end
println("\nReading: compare the two columns at MATCHED s.")
println("  • GMLS clearly below PHS at every s ⇒ LSQ helps beyond averaging size.")
println("  • curves coincide ⇒ it was stencil/averaging size, LSQ adds little.")
println("  • both fall with s ⇒ averaging size is the dominant, reliable lever.")