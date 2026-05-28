# Ground-truth well-posedness test: scan compliance over hole aspect ratio at
# FIXED AREA. Bypasses the noisy nodal gradient entirely (each point is a full
# forward solve). If C is minimized at the circle (ρ=1), the objective is correct
# and the problem is well-posed; the only issue is the noisy gradient.
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Macchiato
import RadialBasisFunctions: _build_weights, Partial, MixedPartial, find_neighbors
using RadialBasisFunctions: PHS
using StaticArrays, LinearAlgebra, Printf

const Lx=4.0; const Ly=4.0; const a0=0.40; const b0=0.20; const dx=0.05; const σ∞=1.0
model = LinearElasticity(E=1.0e7, ν=0.3); μ,λstar = lame_parameters(model)
basis = PHS(3; poly_deg=3); const k=35
const margin = 1 + 1.2*dx/min(a0,b0)
const xs_grid = collect((-Lx/2+dx):dx:(Lx/2-dx/2))
const ys_grid = collect((-Ly/2+dx):dx:(Ly/2-dx/2))
const outer_pts=SVector{2,Float64}[]; const outer_tag=Symbol[]
for y in (-Ly/2):dx:(Ly/2)
    push!(outer_pts,SVector(-Lx/2,y));push!(outer_tag,:xlo)
    push!(outer_pts,SVector(Lx/2,y)); push!(outer_tag,:xhi)
end
for x in (-Lx/2+dx):dx:(Lx/2-dx)
    push!(outer_pts,SVector(x,Ly/2)); push!(outer_tag,:yhi)
    push!(outer_pts,SVector(x,-Ly/2));push!(outer_tag,:ylo)
end
const n_outer=length(outer_pts)
outer_normal(i)= outer_tag[i]===:xhi ? SVector(1.0,0.0) : outer_tag[i]===:xlo ? SVector(-1.0,0.0) :
                 outer_tag[i]===:yhi ? SVector(0.0,1.0) : SVector(0.0,-1.0)

function build_cloud(hole)
    interior_pts=SVector{2,Float64}[]
    for x in xs_grid, y in ys_grid
        p=SVector(x,y); inside=false; n=length(hole)
        for j in 1:n
            a=hole[j]; b=hole[mod1(j+1,n)]
            if (a[2]>p[2]) != (b[2]>p[2])
                xint=a[1]+(p[2]-a[2])*(b[1]-a[1])/(b[2]-a[2]); if p[1]<xint; inside=!inside; end
            end
        end
        inside && continue
        minimum(hypot(p[1]-h[1],p[2]-h[2]) for h in hole) < 1.2*dx && continue
        push!(interior_pts,p)
    end
    pts=vcat(interior_pts,outer_pts,hole); n_int=length(interior_pts)
    interior_idx=collect(1:n_int); outer_idx=collect((n_int+1):(n_int+n_outer))
    hole_idx=collect((n_int+n_outer+1):length(pts)); neumann_idx=vcat(outer_idx,hole_idx)
    nearest(p0,pool)=begin bj=pool[1];bd=Inf
        for j in pool; d=hypot(pts[j][1]-p0[1],pts[j][2]-p0[2]); if d<bd;bd=d;bj=j;end;end;bj end
    pin_ux1=nearest((0.0,0.8),interior_idx);pin_ux2=nearest((0.0,-0.8),interior_idx)
    pin_uy1=nearest((0.8,0.0),interior_idx);N=length(pts)
    (pts=pts,N=N,interior_idx=interior_idx,outer_idx=outer_idx,hole_idx=hole_idx,
     neumann_idx=neumann_idx,dirichlet_dofs=[pin_ux1,pin_ux2,pin_uy1+N])
end

function compliance(a,b,nθ)
    hole=[SVector(a*cos(2π*j/nθ), b*sin(2π*j/nθ)) for j in 0:(nθ-1)]
    c=build_cloud(hole); (;pts,N,outer_idx,neumann_idx,dirichlet_dofs)=c
    adjl=find_neighbors(pts,k); neu=pts[neumann_idx]; nadjl=adjl[neumann_idx]
    hpos=collect(1:length(hole))
    hn,_=polyline_normals(reduce(vcat,[[p[1],p[2]] for p in pts]), c.hole_idx, hpos)
    normals=SVector{2,Float64}[]; tract=SVector{2,Float64}[]
    for i in 1:length(outer_idx); nn=outer_normal(i);push!(normals,nn);push!(tract,σ∞.*nn);end
    append!(normals,hn); append!(tract,fill(SVector(0.0,0.0),length(hole)))
    layout=build_traction_layout(neumann_idx,nadjl,normals,tract,λstar,μ,N)
    Wx=_build_weights(Partial(2,1),pts,pts,adjl,basis); Wy=_build_weights(Partial(2,2),pts,pts,adjl,basis)
    Wxy=_build_weights(MixedPartial(1,2),pts,pts,adjl,basis)
    Wdx=_build_weights(Partial(1,1),pts,neu,nadjl,basis); Wdy=_build_weights(Partial(1,2),pts,neu,nadjl,basis)
    A=assemble_elasticity_from_weights(Wx,Wy,Wxy,N,λstar,μ); b=zeros(2N)
    apply_dirichlet!(A,b,dirichlet_dofs,zeros(length(dirichlet_dofs)))
    apply_traction!(A,b,layout,Wdx,Wdy)
    u=lu(A)\b; return dot(b,u)
end

# ---- AREA-MODE ground truth: uniformly scale the hole (same shape), FD dC/dscale.
# Physics: bigger hole = less material = softer = HIGHER compliance => dC/dscale > 0.
let nθ = max(48, round(Int, 2π*sqrt((a0^2+b0^2)/2)/dx))
    Cm = compliance(a0*0.97, b0*0.97, nθ)
    C0 = compliance(a0,      b0,      nθ)
    Cp = compliance(a0*1.03, b0*1.03, nθ)
    @printf("AREA-MODE FD (uniform hole scale ±3%%):  C(0.97)=%.5e  C(1.0)=%.5e  C(1.03)=%.5e\n", Cm,C0,Cp)
    @printf("  dC/d(scale) ≈ %+.4e   (MUST be > 0: bigger hole -> softer -> higher compliance)\n\n",
            (Cp-Cm)/0.06)
end

const Area = π*a0*b0
println("Fixed hole area = $(round(Area,digits=5))   (circle radius = $(round(sqrt(Area/π),digits=4)))")
println("\n  aspect a/b      a        b      nθ        C")
for ρ in [2.0, 1.7, 1.4, 1.2, 1.0, 0.83, 0.71, 0.5]
    a = sqrt(Area/π*ρ); b = sqrt(Area/π/ρ)
    nθ = max(48, round(Int, 2π*sqrt((a^2+b^2)/2)/dx))
    C = compliance(a,b,nθ)
    @printf("   %5.2f     %.4f   %.4f   %4d   %.6e\n", ρ, a, b, nθ, C)
end
