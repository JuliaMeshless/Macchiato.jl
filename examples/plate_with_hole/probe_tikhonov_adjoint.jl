# ============================================================================
# Tikhonov on the global adjoint λ — decompose the real-gradient noise.
#
# Real adjoint gradient:  g_l = −λᵀ (dC/dl) u
#   (a) λ = C⁻ᵀ(∂J/∂u): for a flux observable ∂J/∂u is a derivative ⇒ high-freq
#       RHS ⇒ λ carries C⁻ᵀ-amplified high-frequency content.
#   (b) dC/dl = ∂W/∂l (Tony): intrinsically rough, independent of λ.
# Tikhonov/Helmholtz-smooth λ → λ_α attacks (a) only. Sweep the smoothing length
# and watch g_l's hi/low: how much of the real 3.8 is amplified-λ (removable)
# vs the stencil floor (b, irreducible here)?
#
#   λ_α = (I − (c·h)² L_lap)⁻¹ λ      (low-pass; c·h = smoothing length)
#   g_l(α) = −λ_αᵀ (dC/dl) u          (dC/dl·u via FD on C, fixed u)
#
# Run:  jlrun plate_with_hole/probe_tikhonov_adjoint.jl   (from examples/)
# ============================================================================
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Macchiato
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
const adjl = find_neighbors(vcat(interior,outer,hole0), sten)
const right_idx=[i for (kk,i) in enumerate(outer_idx) if tag[kk]===:xhi]
θ=[atan(hole0[j][2],hole0[j][1]) for j in 1:nθ]; r̂=[hole0[j]/hypot(hole0[j]...) for j in 1:nθ]
famp(g,m)= m==0 ? abs(sum(g)/nθ) :
    hypot((2/nθ)*sum(g[j]*cos(m*θ[j]) for j in 1:nθ), (2/nθ)*sum(g[j]*sin(m*θ[j]) for j in 1:nθ))
hiband(g)=sqrt(sum(famp(g,m)^2 for m in (nθ÷2-3):(nθ÷2))); loband(g)=sqrt(sum(famp(g,m)^2 for m in 2:5))
hw=(s=[hypot((hole0[mod1(j+1,nθ)]-hole0[j])...) for j in 1:nθ]; [(s[mod1(j-1,nθ)]+s[j])/2 for j in 1:nθ])

basis=PHS(3;poly_deg=3)
function build_C(pts)
    Wxx=_build_weights(Partial(2,1),pts,pts,adjl,basis); Wyy=_build_weights(Partial(2,2),pts,pts,adjl,basis)
    Wdx=_build_weights(Partial(1,1),pts,pts,adjl,basis); Wdy=_build_weights(Partial(1,2),pts,pts,adjl,basis)
    rowt=fill(:interior,N); nx=zeros(N); ny=zeros(N)
    for (kk,i) in enumerate(outer_idx)
        t=tag[kk]; n=outer_n(t); nx[i]=n[1]; ny[i]=n[2]; rowt[i]=(t===:xlo||t===:xhi) ? :dir : :neu
    end
    hn,_=polyline_normals(flat(pts), hole_idx, collect(1:nθ))
    for (kk,i) in enumerate(hole_idx); rowt[i]=:neu; nx[i]=hn[kk][1]; ny[i]=hn[kk][2]; end
    neu=[i for i in 1:N if rowt[i]===:neu]; dir=[i for i in 1:N if rowt[i]===:dir]
    Rint=sparse(interior_idx,interior_idx,ones(n_int),N,N)
    Rneu=sparse(neu,neu,ones(length(neu)),N,N); Rdir=sparse(dir,dir,ones(length(dir)),N,N)
    C=Rint*(Wxx+Wyy)+Rneu*(spdiagm(0=>nx)*Wdx+spdiagm(0=>ny)*Wdy)+Rdir
    b=zeros(N); for i in dir; b[i]=(pts[i][1]<0 ? 1.0 : -1.0); end
    return C, b, Wdx, (Wxx+Wyy)
end

pts0=vcat(interior,outer,hole0)
C, bvec, Wdx, Llap = build_C(pts0)
u = lu(C)\bvec
# ∂J/∂u for J = dx·Σ_right (Wdx u)
dJdu = zeros(N); for i in right_idx; dJdu .+= dx .* Vector(Wdx[i,:]); end
λ = lu(sparse(C'))\dJdu                       # global adjoint (true)

# (dC/dl)·u via FD on C, per hole node l
ε=1e-5; R=zeros(N,nθ)
for j in 1:nθ
    hp=copy(pts0); hp[hole_idx[j]]=hole0[j]+ε.*r̂[j]; Cp,_=build_C(hp)
    hm=copy(pts0); hm[hole_idx[j]]=hole0[j]-ε.*r̂[j]; Cm,_=build_C(hm)
    R[:,j] = (Cp*u - Cm*u)./(2ε)
end
grad(λv)= [ -dot(λv, R[:,j])/hw[j] for j in 1:nθ ]   # adjoint gradient density

# Tikhonov/Helmholtz smoothing of λ: (I - (c h)² Llap) λ_α = λ
h=dx
function smoothλ(c)
    c==0 && return λ
    M = sparse(1.0I, N, N) - (c*h)^2 .* Llap
    return lu(M)\λ
end

println("=== Tikhonov on global adjoint λ — noise decomposition (flux objective, s=$sten) ===")
println("N=$N nθ=$nθ.  g_l = −λᵀ(dC/dl)u (density).  Smoothing length = c·h, h=$h.\n")
g0=grad(λ)
@printf("  c=0 (true λ)     : ‖g‖=%.3e  hi/low=%.2f\n", norm(g0.-sum(g0)/nθ), hiband(g0)/max(loband(g0),1e-30))
for c in (1.0, 2.0, 4.0, 8.0, 16.0)
    gα=grad(smoothλ(c))
    @printf("  c=%4.0f (len=%.2f)  : ‖g‖=%.3e  hi/low=%.2f   (‖λ_α-λ‖/‖λ‖=%.2f)\n",
            c, c*h, norm(gα.-sum(gα)/nθ), hiband(gα)/max(loband(gα),1e-30),
            norm(smoothλ(c).-λ)/norm(λ))
end
println("\nReading:")
println("  • hi/low drops a lot as λ is smoothed ⇒ the real-gradient noise was")
println("    dominated by C⁻ᵀ-amplified λ (source a); Tikhonov-on-λ is a real lever.")
println("  • hi/low plateaus well above ~0.3 ⇒ residual is the stencil floor (b),")
println("    untouched by λ-smoothing ⇒ still need smooth design space for that part.")