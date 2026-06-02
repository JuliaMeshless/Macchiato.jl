# ============================================================================
# (Hermite) Does Hermite-stabilized Neumann reduce the REAL gradient noise?
#
# The real thermal gradient is solve-dominated: dJ/dr ~ K⁻¹(dK/dr), and K⁻¹
# amplifies the boundary/Neumann weight-sensitivity (collocation hi/low=3.80).
# RBF.jl-1's Hermite path bakes the BC (Bu=αu+β∂ₙu) into the local interpolation,
# stabilizing exactly those Neumann rows. Test: does building the forward solve
# with the Hermite operator drop the real gradient's hi/low?
#
# Compare PHS collocation (naive Neumann) vs Hermite (stabilized Neumann), and
# keep GMLS-LSQ for reference. Same Dirichlet-thermal problem, matched stencil.
#
# Run:  jlrun plate_with_hole/probe_hermite_real.jl   (from examples/)
# ============================================================================
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Macchiato                       # polyline_normals
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
const interior_idx=collect(1:n_int)
const outer_idx=collect((n_int+1):(n_int+n_out))
const hole_idx=collect((n_int+n_out+1):(n_int+n_out+nθ))
const N=n_int+n_out+nθ
outer_n(t)= t===:xhi ? SVector(1.0,0.0) : t===:xlo ? SVector(-1.0,0.0) : t===:yhi ? SVector(0.0,1.0) : SVector(0.0,-1.0)
flat(v)=reduce(vcat,[[p[1],p[2]] for p in v])
θ=[atan(hole0[j][2],hole0[j][1]) for j in 1:nθ]; r̂=[hole0[j]/hypot(hole0[j]...) for j in 1:nθ]
famp(g,m)= m==0 ? abs(sum(g)/nθ) :
    hypot((2/nθ)*sum(g[j]*cos(m*θ[j]) for j in 1:nθ), (2/nθ)*sum(g[j]*sin(m*θ[j]) for j in 1:nθ))
hiband(g)=sqrt(sum(famp(g,m)^2 for m in (nθ÷2-3):(nθ÷2))); loband(g)=sqrt(sum(famp(g,m)^2 for m in 2:5))
totnorm(g)=norm(g .- sum(g)/nθ)
const right_idx=[i for (kk,i) in enumerate(outer_idx) if tag[kk]===:xhi]
const adjl0 = find_neighbors(vcat(interior,outer,hole0), sten)
holeweights(h)= (s=[hypot((h[mod1(j+1,nθ)]-h[j])...) for j in 1:nθ]; [(s[mod1(j-1,nθ)]+s[j])/2 for j in 1:nθ])

# row classification + per-boundary BC/normal in boundary-index order
function classify(pts)
    rowt=fill(:interior,N); nx=zeros(N); ny=zeros(N)
    for (kk,i) in enumerate(outer_idx)
        t=tag[kk]
        if t===:xlo||t===:xhi; rowt[i]=:dir; n=outer_n(t)
        else rowt[i]=:neu; n=outer_n(t); end
        nx[i]=n[1]; ny[i]=n[2]
    end
    hn,_=polyline_normals(flat(pts), hole_idx, collect(1:nθ))
    for (kk,i) in enumerate(hole_idx); rowt[i]=:neu; nx[i]=hn[kk][1]; ny[i]=hn[kk][2]; end
    return rowt, nx, ny
end

# ---- PHS collocation thermal solve (naive Neumann) -------------------------
function J_phs(pts, adjl, basis)
    rowt,nx,ny=classify(pts)
    Wxx=_build_weights(Partial(2,1),pts,pts,adjl,basis); Wyy=_build_weights(Partial(2,2),pts,pts,adjl,basis)
    Wdx=_build_weights(Partial(1,1),pts,pts,adjl,basis); Wdy=_build_weights(Partial(1,2),pts,pts,adjl,basis)
    neu=[i for i in 1:N if rowt[i]===:neu]; dir=[i for i in 1:N if rowt[i]===:dir]
    Rint=sparse(interior_idx,interior_idx,ones(n_int),N,N)
    Rneu=sparse(neu,neu,ones(length(neu)),N,N); Rdir=sparse(dir,dir,ones(length(dir)),N,N)
    K=Rint*(Wxx+Wyy)+Rneu*(spdiagm(0=>nx)*Wdx+spdiagm(0=>ny)*Wdy)+Rdir
    b=zeros(N); for i in dir; b[i]=(pts[i][1]<0 ? 1.0 : -1.0); end
    T=lu(K)\b
    return dx*sum(dot(Wdx[i,:],T) for i in right_idx)
end

# ---- Hermite thermal solve (BC baked into interpolation) -------------------
function J_hermite(pts, adjl, basis)
    rowt,nx,ny=classify(pts)
    is_bnd=[rowt[i]!==:interior for i in 1:N]
    bcs=BoundaryCondition{Float64}[]; norms=SVector{2,Float64}[]
    for i in 1:N
        is_bnd[i] || continue
        push!(bcs, rowt[i]===:dir ? Dirichlet() : Neumann())
        push!(norms, SVector(nx[i],ny[i]))
    end
    op = RadialBasisOperator(Laplacian(), pts, pts, basis, is_bnd, bcs, norms; k=sten, adjl=adjl)
    K = op.weights
    b=zeros(N); for i in 1:N; rowt[i]===:dir && (b[i]=(pts[i][1]<0 ? 1.0 : -1.0)); end
    T = K \ b
    Wdx=_build_weights(Partial(1,1),pts,pts,adjl,basis)
    return dx*sum(dot(Wdx[i,:],T) for i in right_idx)
end

function spectrum(Jfun)
    ε=1e-5; g=zeros(nθ); w=holeweights(hole0)
    for j in 1:nθ
        hp=copy(hole0); hp[j]=hole0[j]+ε.*r̂[j]; hm=copy(hole0); hm[j]=hole0[j]-ε.*r̂[j]
        g[j]=(Jfun(vcat(interior,outer,hp))-Jfun(vcat(interior,outer,hm)))/(2ε)
    end
    d=g./w; return totnorm(d), hiband(d)/max(loband(d),1e-30)
end

println("=== (Hermite) REAL thermal gradient: collocation vs Hermite Neumann (s=$sten) ===")
println("N=$N  nθ=$nθ.\n")
basis=PHS(3;poly_deg=3)
gp,hp = spectrum(p->J_phs(p, adjl0, basis))
@printf("  PHS collocation (naive Neumann) : ‖d‖=%.3e  hi/low=%.2f\n", gp, hp)
gh,hh = spectrum(p->J_hermite(p, adjl0, basis))
@printf("  Hermite (stabilized Neumann)    : ‖d‖=%.3e  hi/low=%.2f\n", gh, hh)
@printf("\n  ratio Hermite/colloc:  ‖d‖ %.2f   hi/low %.2f\n", gh/gp, hh/hp)
println("\nReading: Hermite hi/low ≪ collocation ⇒ the real-gradient noise was")
println("solve-amplified Neumann instability (your hypothesis); Hermite is the cure.")
println("If hi/low stays ~high ⇒ even stabilized BCs don't fix it ⇒ smooth design space.")