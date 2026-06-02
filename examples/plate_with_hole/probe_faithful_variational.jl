# ============================================================================
# (a) faithfulness of the Tikhonov-smoothed gradient + (b) smooth-observable fix
#
# (a) The c=1 smoothing dropped hi/low 3.80→0.29 but changed λ by 141%. Is the
#     smoothed gradient FAITHFUL (low modes preserved) or smooth-but-wrong?
#     → cosine similarity of low-mode coeffs (m=2..6) of g_smoothed vs g_true.
#
# (b) Parameter-free version: use a SMOOTH observable J=∫_Ω T (∂J/∂u = const,
#     no derivative) instead of the flux (∂J/∂u = derivative). Then λ=C⁻ᵀ(smooth)
#     is not amplified. Does the gradient sit at the ~0.3 floor with NO smoothing?
#
# Run:  jlrun plate_with_hole/probe_faithful_variational.jl   (from examples/)
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
const interior_idx=collect(1:n_int); const outer_idx=collect((n_int+1):(n_int+n_out))
const hole_idx=collect((n_int+n_out+1):(n_int+n_out+nθ)); const N=n_int+n_out+nθ
outer_n(t)= t===:xhi ? SVector(1.0,0.0) : t===:xlo ? SVector(-1.0,0.0) : t===:yhi ? SVector(0.0,1.0) : SVector(0.0,-1.0)
flat(v)=reduce(vcat,[[p[1],p[2]] for p in v])
const adjl = find_neighbors(vcat(interior,outer,hole0), sten)
const right_idx=[i for (kk,i) in enumerate(outer_idx) if tag[kk]===:xhi]
θ=[atan(hole0[j][2],hole0[j][1]) for j in 1:nθ]; r̂=[hole0[j]/hypot(hole0[j]...) for j in 1:nθ]
hw=(s=[hypot((hole0[mod1(j+1,nθ)]-hole0[j])...) for j in 1:nθ]; [(s[mod1(j-1,nθ)]+s[j])/2 for j in 1:nθ])
fc(g,m)= (2/nθ)*sum(g[j]*cos(m*θ[j]) for j in 1:nθ); fs(g,m)=(2/nθ)*sum(g[j]*sin(m*θ[j]) for j in 1:nθ)
famp(g,m)= m==0 ? abs(sum(g)/nθ) : hypot(fc(g,m),fs(g,m))
hiband(g)=sqrt(sum(famp(g,m)^2 for m in (nθ÷2-3):(nθ÷2))); loband(g)=sqrt(sum(famp(g,m)^2 for m in 2:5))
lowcoef(g)=reduce(vcat,[[fc(g,m),fs(g,m)] for m in 2:6])
cossim(a,b)=dot(a,b)/(norm(a)*norm(b)+1e-300)

basis=PHS(3;poly_deg=3)
function build_C(pts)
    Wxx=_build_weights(Partial(2,1),pts,pts,adjl,basis); Wyy=_build_weights(Partial(2,2),pts,pts,adjl,basis)
    Wdx=_build_weights(Partial(1,1),pts,pts,adjl,basis); Wdy=_build_weights(Partial(1,2),pts,pts,adjl,basis)
    rowt=fill(:interior,N); nx=zeros(N); ny=zeros(N)
    for (kk,i) in enumerate(outer_idx); t=tag[kk]; n=outer_n(t); nx[i]=n[1]; ny[i]=n[2]; rowt[i]=(t===:xlo||t===:xhi) ? :dir : :neu; end
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
C,bvec,Wdx,Llap = build_C(pts0); u=lu(C)\bvec; CT=lu(sparse(C'))
ε=1e-5; R=zeros(N,nθ)
for j in 1:nθ
    hp=copy(pts0); hp[hole_idx[j]]=hole0[j]+ε.*r̂[j]; Cp,_=build_C(hp)
    hm=copy(pts0); hm[hole_idx[j]]=hole0[j]-ε.*r̂[j]; Cm,_=build_C(hm)
    R[:,j]=(Cp*u-Cm*u)./(2ε)
end
grad(λ)=[ -dot(λ,R[:,j])/hw[j] for j in 1:nθ ]
h=dx

# ---- flux observable + Tikhonov(c=1) ----
dJ_flux=zeros(N); for i in right_idx; dJ_flux .+= dx.*Vector(Wdx[i,:]); end
λ_flux=CT\dJ_flux; g_flux=grad(λ_flux)
λ_c1 = lu(sparse(1.0I,N,N) - (1.0*h)^2 .* Llap)\λ_flux; g_c1=grad(λ_c1)

println("=== (a) FAITHFULNESS of Tikhonov-smoothed gradient (flux objective) ===")
@printf("  g_true (c=0): hi/low=%.2f    g_smoothed (c=1): hi/low=%.2f\n",
        hiband(g_flux)/loband(g_flux), hiband(g_c1)/loband(g_c1))
@printf("  low-mode (m2..6) cosine(g_smoothed, g_true) = %.4f\n", cossim(lowcoef(g_c1),lowcoef(g_flux)))
@printf("  ‖low(g_c1)‖/‖low(g_flux)‖ = %.3f  (≈1 ⇒ low-mode amplitude preserved)\n",
        norm(lowcoef(g_c1))/norm(lowcoef(g_flux)))
println("  low-mode coeffs:  m  a(true)     a(smooth)    b(true)     b(smooth)")
for m in 2:6
    @printf("                  %2d  %+.3e  %+.3e  %+.3e  %+.3e\n", m, fc(g_flux,m),fc(g_c1,m),fs(g_flux,m),fs(g_c1,m))
end

# ---- (b) smooth observable J=∫T (∂J/∂u = interior indicator) ----
println("\n=== (b) SMOOTH OBSERVABLE J=∫T (∂J/∂u=const, no derivative; no smoothing) ===")
dJ_smooth=zeros(N); for i in interior_idx; dJ_smooth[i]=1.0; end
λ_s=CT\dJ_smooth; g_s=grad(λ_s)
@printf("  flux observable    : hi/low=%.2f\n", hiband(g_flux)/loband(g_flux))
@printf("  smooth observable  : hi/low=%.2f   (parameter-free)\n", hiband(g_s)/loband(g_s))
@printf("  ‖λ_smoothobs high-freq‖ vs flux: ‖λ_s‖=%.2e  ‖λ_flux‖=%.2e\n", norm(λ_s), norm(λ_flux))
println("\nReading:")
println("  (a) cosine→1 ⇒ smoothing kept the physical low-mode signal (faithful).")
println("  (b) smooth-observable hi/low ≈ stencil floor (~0.3) with NO smoothing")
println("      ⇒ a smooth observable formulation removes the adjoint amplification")
println("      parameter-free — the principled fix.")