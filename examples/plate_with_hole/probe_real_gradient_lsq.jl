# ============================================================================
# (1) Does LSQ smooth the REAL PDE shape gradient (not just the surrogate)?
#
# Dirichlet-driven thermal problem (left T=+1, right T=-1, top/bottom/hole
# insulated). Build the forward operator two ways at MATCHED stencil s=50:
#   • PHS3+poly3 exact collocation
#   • GMLS poly_deg-3 least-squares
# Compute the REAL thermal-compliance shape gradient by FD (perturb each hole
# node radially, re-solve, central diff) — this IS the true discrete gradient.
# Compare the radial-gradient-density Fourier spectra. If GMLS gives a lower
# hi/low / smaller artifact on the REAL gradient, the surrogate result transfers
# and the clean-DOF floor rises end-to-end.
#
# Run:  jlrun plate_with_hole/probe_real_gradient_lsq.jl   (from examples/)
# ============================================================================
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Macchiato                       # polyline_normals
import RadialBasisFunctions: _build_weights, Partial, find_neighbors
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
const interior_idx=collect(1:n_int)
const outer_idx=collect((n_int+1):(n_int+n_out))
const hole_idx=collect((n_int+n_out+1):(n_int+n_out+nθ))
const N=n_int+n_out+nθ
outer_n(t)= t===:xhi ? SVector(1.0,0.0) : t===:xlo ? SVector(-1.0,0.0) : t===:yhi ? SVector(0.0,1.0) : SVector(0.0,-1.0)
flat(v)=reduce(vcat,[[p[1],p[2]] for p in v])
θ=[atan(hole0[j][2],hole0[j][1]) for j in 1:nθ]; r̂=[hole0[j]/hypot(hole0[j]...) for j in 1:nθ]

monos(ξ)=(x=ξ[1];y=ξ[2]; @SVector[1.0,x,y,x^2,x*y,y^2,x^3,x^2*y,x*y^2,y^3]); const M=10
const lp_lap=@SVector[0.0,0,0,2,0,2,0,0,0,0]
lp_dx()=@SVector[0.0,1,0,0,0,0,0,0,0,0]; lp_dy()=@SVector[0.0,0,1,0,0,0,0,0,0,0]

famp(g,m)= m==0 ? abs(sum(g)/nθ) :
    hypot((2/nθ)*sum(g[j]*cos(m*θ[j]) for j in 1:nθ), (2/nθ)*sum(g[j]*sin(m*θ[j]) for j in 1:nθ))
hiband(g)=sqrt(sum(famp(g,m)^2 for m in (nθ÷2-3):(nθ÷2))); loband(g)=sqrt(sum(famp(g,m)^2 for m in 2:5))
totnorm(g)=norm(g .- sum(g)/nθ)

# normals + row classification (fixed across hole perturbations except hole normals)
function classify(pts)
    nx=zeros(N); ny=zeros(N); rowt=fill(:interior, N)
    for i in interior_idx; rowt[i]=:interior; end
    for (kk,i) in enumerate(outer_idx)
        t=tag[kk]
        if t===:xlo || t===:xhi; rowt[i]=:dir
        else rowt[i]=:neu; n=outer_n(t); nx[i]=n[1]; ny[i]=n[2]; end
    end
    hn,_ = polyline_normals(flat(pts), hole_idx, collect(1:nθ))
    for (kk,i) in enumerate(hole_idx); rowt[i]=:neu; nx[i]=hn[kk][1]; ny[i]=hn[kk][2]; end
    return nx, ny, rowt
end

# ---- GMLS assembly ---------------------------------------------------------
function gmls_row(pts, nbr, xe, lp)
    n=length(nbr); P=Matrix{Float64}(undef,n,M)
    for a in 1:n; P[a,:]=monos(pts[nbr[a]]-xe); end
    return P*((P'P)\lp)
end
function build_gmls(pts, adjl, nx, ny, rowt)
    I=Int[]; J=Int[]; V=Float64[]; b=zeros(N)
    for i in 1:N
        if rowt[i]===:dir
            push!(I,i); push!(J,i); push!(V,1.0)
            b[i] = (pts[i][1] < 0 ? 1.0 : -1.0)         # left +1, right -1
        else
            lp = rowt[i]===:interior ? lp_lap : (nx[i].*lp_dx() .+ ny[i].*lp_dy())
            w = gmls_row(pts, adjl[i], pts[i], lp)
            append!(I,fill(i,length(adjl[i]))); append!(J,adjl[i]); append!(V,w)
        end
    end
    return sparse(I,J,V,N,N), b
end
function gmls_dx_at(pts, adjl, idxs)   # ∂x weights at selected nodes (for flux)
    rows=Dict{Int,Tuple{Vector{Int},Vector{Float64}}}()
    for i in idxs; rows[i]=(adjl[i], gmls_row(pts, adjl[i], pts[i], lp_dx())); end
    return rows
end

# ---- PHS assembly ----------------------------------------------------------
function build_phs(pts, adjl, basis, nx, ny, rowt)
    Wxx=_build_weights(Partial(2,1),pts,pts,adjl,basis); Wyy=_build_weights(Partial(2,2),pts,pts,adjl,basis)
    Wdx=_build_weights(Partial(1,1),pts,pts,adjl,basis); Wdy=_build_weights(Partial(1,2),pts,pts,adjl,basis)
    Rint=sparse(interior_idx,interior_idx,ones(n_int),N,N)
    neu=[i for i in 1:N if rowt[i]===:neu]; dir=[i for i in 1:N if rowt[i]===:dir]
    Rneu=sparse(neu,neu,ones(length(neu)),N,N); Rdir=sparse(dir,dir,ones(length(dir)),N,N)
    K = Rint*(Wxx+Wyy) + Rneu*(spdiagm(0=>nx)*Wdx+spdiagm(0=>ny)*Wdy) + Rdir
    b=zeros(N); for i in dir; b[i]=(pts[i][1]<0 ? 1.0 : -1.0); end
    return K, b, Wdx
end

const right_idx=[i for (kk,i) in enumerate(outer_idx) if tag[kk]===:xhi]
const adjl0 = find_neighbors(vcat(interior,outer,hole0), sten)

# arc-length quadrature weight per hole node (for density)
function holeweights(hole)
    s=[hypot((hole[mod1(j+1,nθ)]-hole[j])...) for j in 1:nθ]
    [(s[mod1(j-1,nθ)]+s[j])/2 for j in 1:nθ]
end

function J_gmls(pts, adjl)
    nx,ny,rowt=classify(pts); K,b=build_gmls(pts,adjl,nx,ny,rowt)
    T=lu(K)\b; dxr=gmls_dx_at(pts,adjl,right_idx)
    return dx*sum(dot(dxr[i][2], T[dxr[i][1]]) for i in right_idx)
end
function J_phs(pts, adjl, basis)
    nx,ny,rowt=classify(pts); K,b,Wdx=build_phs(pts,adjl,basis,nx,ny,rowt)
    T=lu(K)\b
    return dx*sum(dot(Wdx[i,:], T) for i in right_idx)
end

function grad_spectrum(Jfun)
    ε=1e-5; g=zeros(nθ); w=holeweights(hole0)
    for j in 1:nθ
        hp=copy(hole0); hp[j]=hole0[j]+ε.*r̂[j]; hm=copy(hole0); hm[j]=hole0[j]-ε.*r̂[j]
        ptsp=vcat(interior,outer,hp); ptsm=vcat(interior,outer,hm)
        g[j]=(Jfun(ptsp)-Jfun(ptsm))/(2ε)
    end
    d = g ./ w                          # shape-gradient density
    return totnorm(d), hiband(d)/max(loband(d),1e-30)
end

println("=== (1) REAL thermal shape gradient: PHS colloc vs GMLS LSQ (matched s=$sten) ===")
println("N=$N  nθ=$nθ.  FD radial gradient density on the actual Dirichlet-thermal compliance.\n")
basis=PHS(3;poly_deg=3)
gp,hp = grad_spectrum(p->J_phs(p, adjl0, basis))
gg,hg = grad_spectrum(p->J_gmls(p, adjl0))
@printf("  PHS3 collocation : ‖d‖=%.3e  hi/low=%.2f\n", gp, hp)
@printf("  GMLS LSQ         : ‖d‖=%.3e  hi/low=%.2f\n", gg, hg)
@printf("\n  ratio GMLS/PHS:  ‖d‖ %.2f   hi/low %.2f\n", gg/gp, hg/hp)
println("\nReading: GMLS hi/low and ‖d‖ below PHS ⇒ the LSQ smoothing transfers to")
println("the REAL PDE shape gradient, raising the clean-DOF floor end-to-end.")