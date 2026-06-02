# ============================================================================
# Forward-accuracy check — resolve the GMLS/Hermite confound.
#
# The real-gradient test gave GMLS ‖d‖ 167× and Hermite ‖d‖ 40× larger than
# collocation — suggesting my GMLS/Hermite FORWARD solves may be inaccurate
# (making the LSQ/Hermite gradient verdict confounded, not definitive).
#
# Test on a MANUFACTURED HARMONIC solution T=x²−y² (∇²T=0 exactly), same domain
# and BC structure as the gradient test (Dirichlet left/right, Neumann
# top/bottom/hole). Solve with each method; compare interior T to exact.
#   • all accurate  ⇒ gradient noise is intrinsic; LSQ/Hermite verdict stands.
#   • GMLS/Hermite inaccurate ⇒ my gradient test was unfair to them.
#
# Run:  jlrun plate_with_hole/probe_forward_accuracy.jl   (from examples/)
# ============================================================================
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Macchiato
import RadialBasisFunctions: _build_weights, Partial, find_neighbors,
                             RadialBasisOperator, Laplacian, Dirichlet, Neumann, BoundaryCondition
using RadialBasisFunctions: PHS
using StaticArrays, LinearAlgebra, SparseArrays, Printf

const Lx=4.0; const Ly=4.0; const a0=0.40; const b0=0.20; const dx=0.05
const nθ = max(48, round(Int, 2π*sqrt((a0^2+b0^2)/2)/dx))
const margin = 1 + 1.2*dx/min(a0,b0); const sten=50
ellipse_val(p,a,b)=(p[1]/a)^2+(p[2]/b)^2
interior = SVector{2,Float64}[]
for x in (-Lx/2+dx):dx:(Lx/2-dx/2), y in (-Ly/2+dx):dx:(Ly/2-dx/2)
    ellipse_val(SVector(x,y),a0,b0) > margin^2 || continue
    push!(interior, SVector(x,y))
end
outer = SVector{2,Float64}[]; tag = Symbol[]
for y in (-Ly/2):dx:(Ly/2)
    push!(outer,SVector(-Lx/2,y)); push!(tag,:xlo); push!(outer,SVector(Lx/2,y)); push!(tag,:xhi)
end
for x in (-Lx/2+dx):dx:(Lx/2-dx)
    push!(outer,SVector(x,Ly/2)); push!(tag,:yhi); push!(outer,SVector(x,-Ly/2)); push!(tag,:ylo)
end
const n_int=length(interior); const n_out=length(outer)
hole0 = [SVector(a0*cos(2π*j/nθ), b0*sin(2π*j/nθ)) for j in 0:(nθ-1)]
const pts = vcat(interior,outer,hole0)
const interior_idx=collect(1:n_int)
const outer_idx=collect((n_int+1):(n_int+n_out))
const hole_idx=collect((n_int+n_out+1):(n_int+n_out+nθ))
const N=length(pts)
outer_n(t)= t===:xhi ? SVector(1.0,0.0) : t===:xlo ? SVector(-1.0,0.0) : t===:yhi ? SVector(0.0,1.0) : SVector(0.0,-1.0)
flat(v)=reduce(vcat,[[p[1],p[2]] for p in v])
const adjl = find_neighbors(pts, sten)

# manufactured harmonic field
Tex(p)=p[1]^2 - p[2]^2
gradT(p)=SVector(2p[1], -2p[2])

# classification + normals
rowt=fill(:interior,N); nx=zeros(N); ny=zeros(N)
for (kk,i) in enumerate(outer_idx)
    t=tag[kk]; n=outer_n(t); nx[i]=n[1]; ny[i]=n[2]
    rowt[i] = (t===:xlo||t===:xhi) ? :dir : :neu
end
let (hn,_)=polyline_normals(flat(pts), hole_idx, collect(1:nθ))
    for (kk,i) in enumerate(hole_idx); rowt[i]=:neu; nx[i]=hn[kk][1]; ny[i]=hn[kk][2]; end
end
# manufactured RHS: interior ∇²T=0 ; Dirichlet T=Tex ; Neumann ∂ₙT=∇T·n
bvec=zeros(N)
for i in 1:N
    if rowt[i]===:dir; bvec[i]=Tex(pts[i])
    elseif rowt[i]===:neu; bvec[i]=dot(gradT(pts[i]), SVector(nx[i],ny[i]))
    end  # interior stays 0
end
Texact=[Tex(p) for p in pts]
err(T)=norm(T[interior_idx]-Texact[interior_idx])/norm(Texact[interior_idx])

# ---- PHS collocation -------------------------------------------------------
function solve_phs(basis)
    Wxx=_build_weights(Partial(2,1),pts,pts,adjl,basis); Wyy=_build_weights(Partial(2,2),pts,pts,adjl,basis)
    Wdx=_build_weights(Partial(1,1),pts,pts,adjl,basis); Wdy=_build_weights(Partial(1,2),pts,pts,adjl,basis)
    neu=[i for i in 1:N if rowt[i]===:neu]; dir=[i for i in 1:N if rowt[i]===:dir]
    Rint=sparse(interior_idx,interior_idx,ones(n_int),N,N)
    Rneu=sparse(neu,neu,ones(length(neu)),N,N); Rdir=sparse(dir,dir,ones(length(dir)),N,N)
    K=Rint*(Wxx+Wyy)+Rneu*(spdiagm(0=>nx)*Wdx+spdiagm(0=>ny)*Wdy)+Rdir
    return lu(K)\bvec
end

# ---- GMLS ------------------------------------------------------------------
monos(ξ)=(x=ξ[1];y=ξ[2]; @SVector[1.0,x,y,x^2,x*y,y^2,x^3,x^2*y,x*y^2,y^3]); const Mm=10
const lp_lap=@SVector[0.0,0,0,2,0,2,0,0,0,0]
function gmls_row(nbr,xe,lp)
    n=length(nbr); P=Matrix{Float64}(undef,n,Mm); for a in 1:n; P[a,:]=monos(pts[nbr[a]]-xe); end
    P*((P'P)\lp)
end
function solve_gmls()
    I=Int[];J=Int[];V=Float64[]
    for i in 1:N
        if rowt[i]===:dir; push!(I,i);push!(J,i);push!(V,1.0)
        else lp = rowt[i]===:interior ? lp_lap : SVector(0.0,nx[i],ny[i],0,0,0,0,0,0,0)
            w=gmls_row(adjl[i],pts[i],lp); append!(I,fill(i,length(adjl[i])));append!(J,adjl[i]);append!(V,w)
        end
    end
    lu(sparse(I,J,V,N,N))\bvec
end

# ---- Hermite ---------------------------------------------------------------
function solve_hermite(basis)
    is_bnd=[rowt[i]!==:interior for i in 1:N]
    bcs=BoundaryCondition{Float64}[]; norms=SVector{2,Float64}[]
    for i in 1:N
        is_bnd[i] || continue
        push!(bcs, rowt[i]===:dir ? Dirichlet() : Neumann()); push!(norms, SVector(nx[i],ny[i]))
    end
    op=RadialBasisOperator(Laplacian(), pts, pts, basis, is_bnd, bcs, norms; k=sten, adjl=adjl)
    return op.weights \ bvec
end

println("=== Forward-accuracy on manufactured harmonic T=x²−y² (∇²T=0) ===")
println("N=$N  nθ=$nθ  s=$sten.  Interior L2 relative error vs exact:\n")
basis=PHS(3;poly_deg=3)
@printf("  PHS collocation : rel.err = %.3e\n", err(solve_phs(basis)))
@printf("  GMLS LSQ        : rel.err = %.3e\n", err(solve_gmls()))
@printf("  Hermite         : rel.err = %.3e\n", err(solve_hermite(basis)))
println("\nReading:")
println("  • all small (≲1e-3) ⇒ solves are accurate; the gradient hi/low~3.5 is")
println("    intrinsic and LSQ/Hermite genuinely don't cure it.")
println("  • GMLS/Hermite large ⇒ my solver impls are inaccurate ⇒ the gradient")
println("    verdict was unfair to them; they deserve a correct implementation.")