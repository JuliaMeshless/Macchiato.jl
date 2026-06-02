# ============================================================================
# probe_fd_smoothed_pullback.jl
#
# PROTOTYPE: FD-smoothed shape gradient.  Instead of the analytic ∂W/∂r
# through the ill-conditioned local RBF adjoint, we finite-difference the
# compliance w.r.t. each hole node's radial position at a step h comparable
# to the STENCIL RADIUS (~4.5 dx).  This mollifies the staircase ∂W/∂r
# artifact that causes the Nyquist noise.
#
# Sweeps fd_scale = h / local_dx to find the smoothing sweet spot.
# Measures hi/lo of the resulting radial gradient density at each scale.
# Compares against the analytic (discrete adjoint) gradient.
#
# Run:  jlrun plate_with_hole/probe_fd_smoothed_pullback.jl   (from examples/)
# ============================================================================
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Macchiato
import RadialBasisFunctions: _build_weights, Partial, MixedPartial, find_neighbors
using RadialBasisFunctions: PHS
using StaticArrays
using LinearAlgebra
using Printf

# ---- problem setup ----------------------------------------------------------
const Lx=4.0; const Ly=4.0; const a0=0.40; const b0=0.20
const dx=0.05; const σ∞=1.0
model=LinearElasticity(E=1.0e7,ν=0.3); μ,λstar=lame_parameters(model)
basis=PHS(3;poly_deg=3); const k=35

ellipse_val(p,a,b)=(p[1]/a)^2+(p[2]/b)^2
const margin=1+1.2*dx/min(a0,b0)
const nθ=max(48,round(Int,2π*sqrt((a0^2+b0^2)/2)/dx))

ref_interior=SVector{2,Float64}[]
let xs=(-Lx/2+dx):dx:(Lx/2-dx/2),ys=(-Ly/2+dx):dx:(Ly/2-dx/2)
    for x in xs,y in ys
        ellipse_val(SVector(x,y),a0,b0)>margin^2||continue
        push!(ref_interior,SVector(x,y))
    end
end
const n_int=length(ref_interior)
outer_pts=SVector{2,Float64}[];outer_tag=Symbol[]
for y in (-Ly/2):dx:(Ly/2)
    push!(outer_pts,SVector(-Lx/2,y));push!(outer_tag,:xlo)
    push!(outer_pts,SVector(Lx/2,y));push!(outer_tag,:xhi)
end
for x in (-Lx/2+dx):dx:(Lx/2-dx)
    push!(outer_pts,SVector(x,Ly/2));push!(outer_tag,:yhi)
    push!(outer_pts,SVector(x,-Ly/2));push!(outer_tag,:ylo)
end
const n_outer=length(outer_pts)
θ_vals=[2π*(j-1)/nθ for j in 1:nθ]
hole0=[SVector(a0*cos(t),b0*sin(t)) for t in θ_vals]
ref_pts=vcat(ref_interior,outer_pts,hole0)
const N=length(ref_pts)
const interior_idx=collect(1:n_int)
const outer_idx=collect((n_int+1):(n_int+n_outer))
const hole_idx=collect((n_int+n_outer+1):N)
const neumann_idx=vcat(outer_idx,hole_idx)
const hole_pos=collect(1:nθ)
nearest(p0,pool)=pool[argmin([hypot(ref_pts[i][1]-p0[1],ref_pts[i][2]-p0[2]) for i in pool])]
pin_ux1=nearest((0.0,0.8),interior_idx);pin_ux2=nearest((0.0,-0.8),interior_idx)
pin_uy1=nearest((0.8,0.0),interior_idx)
const dirichlet_dofs=[pin_ux1,pin_ux2,pin_uy1+N]
const active=let a=trues(2N);a[pin_ux1]=a[pin_ux2]=a[pin_uy1+N]=false;a end
const interior_rows=let r=falses(N);for i in interior_idx;r[i]=true;end;r end
flat(v)=reduce(vcat,[[p[1],p[2]] for p in v])
function outer_normal(i)
    outer_tag[i]===:xhi ? SVector(1.0,0.0) :
    outer_tag[i]===:xlo ? SVector(-1.0,0.0) :
    outer_tag[i]===:yhi ? SVector(0.0,1.0) : SVector(0.0,-1.0)
end
_ZJV=SVector(0.0,0.0);zero_njac(g)=NormalJacobian(g,g,_ZJV,_ZJV,_ZJV,_ZJV)

const adjl_ref=find_neighbors(ref_pts,k)
const pts=ref_pts
const stencil_radius = let rmax=0.0
    for i in 1:N, j in adjl_ref[i]; d=hypot(pts[i][1]-ref_pts[j][1],pts[i][2]-ref_pts[j][2]); d>rmax&&(rmax=d); end
    rmax
end
@printf("stencil radius ≈ %.3f  (≈ %.1f dx)\n",stencil_radius,stencil_radius/dx)

# ---- spectral analysis tools ------------------------------------------------
function famp(g,m)
    m == 0 && return abs(sum(g)/nθ)
    return hypot((2/nθ)*sum(g[j]*cos(m*θ_vals[j]) for j in 1:nθ),
                 (2/nθ)*sum(g[j]*sin(m*θ_vals[j]) for j in 1:nθ))
end
function hi_lo(g)
    lo=sqrt(sum(famp(g,m)^2 for m in 2:5))
    hi=sqrt(sum(famp(g,m)^2 for m in (nθ÷2-3):(nθ÷2)))
    return hi/lo
end
totnorm(g)=sqrt(sum(famp(g,m)^2 for m in 0:(nθ÷2)))

# ---- compute fast compliance (fixed topology, rebuild weights only) ----------
# Rebuild all operators and solve.  Keeps the SAME stencil topology (adjl_ref)
# so remeshing noise is avoided — the compliance is a SMOOTH function of node
# positions, and the FD gradient with a LARGE step will be smooth.
function solve_compliance(pts_use)
    Wx =_build_weights(Partial(2,1),     pts_use,pts_use,adjl_ref,basis)
    Wy =_build_weights(Partial(2,2),     pts_use,pts_use,adjl_ref,basis)
    Wxy=_build_weights(MixedPartial(1,2),pts_use,pts_use,adjl_ref,basis)
    A=assemble_elasticity_from_weights(Wx,Wy,Wxy,N,λstar,μ)
    b=zeros(2N)
    apply_dirichlet!(A,b,dirichlet_dofs,zeros(length(dirichlet_dofs)))
    n_pts=pts_use[neumann_idx]
    nadjl=adjl_ref[neumann_idx]
    hn,_=polyline_normals(flat(pts_use),hole_idx,hole_pos)
    norms=SVector{2,Float64}[];tracts=SVector{2,Float64}[]
    for ii in 1:n_outer;push!(norms,outer_normal(ii));push!(tracts,σ∞.*outer_normal(ii));end
    append!(norms,hn);append!(tracts,fill(SVector(0.0,0.0),nθ))
    layout=build_traction_layout(neumann_idx,nadjl,norms,tracts,λstar,μ,N)
    Wdx=_build_weights(Partial(1,1),pts_use,n_pts,nadjl,basis)
    Wdy=_build_weights(Partial(1,2),pts_use,n_pts,nadjl,basis)
    apply_traction!(A,b,layout,Wdx,Wdy)
    u=A\b
    return dot(b,u)
end

# ---- baseline: analytic gradient --------------------------------------------
println("Computing analytic gradient...")
res=shape_gradient(flat(pts),model,N,adjl_ref,basis,active,
    dirichlet_dofs,zeros(3),_->zeros(2N);
    interior_rows=interior_rows,
    neumann_ids=neumann_idx,neumann_adjl=adjl_ref[neumann_idx])
# We need the actual gradient, so let's do it properly with the layout
neumann_adjl_ref=adjl_ref[neumann_idx]
hn,hjacs=polyline_normals(flat(pts),hole_idx,hole_pos)
njacs=vcat(NormalJacobian[zero_njac(i) for i in outer_idx],hjacs)
normals=SVector{2,Float64}[];tractions=SVector{2,Float64}[]
for i in 1:n_outer;nn=outer_normal(i);push!(normals,nn);push!(tractions,σ∞.*nn);end
append!(normals,hn);append!(tractions,fill(SVector(0.0,0.0),nθ))
layout=build_traction_layout(neumann_idx,neumann_adjl_ref,normals,tractions,λstar,μ,N)
W_d2x=_build_weights(Partial(2,1),pts,pts,adjl_ref,basis)
W_d2y=_build_weights(Partial(2,2),pts,pts,adjl_ref,basis)
W_d2xy=_build_weights(MixedPartial(1,2),pts,pts,adjl_ref,basis)
W_dx=_build_weights(Partial(1,1),pts,pts[neumann_idx],neumann_adjl_ref,basis)
W_dy=_build_weights(Partial(1,2),pts,pts[neumann_idx],neumann_adjl_ref,basis)
A_full=assemble_elasticity_from_weights(W_d2x,W_d2y,W_d2xy,N,λstar,μ)
b_full=zeros(2N);apply_dirichlet!(A_full,b_full,dirichlet_dofs,zeros(3))
apply_traction!(A_full,b_full,layout,W_dx,W_dy)

res_full=shape_gradient(flat(pts),model,N,adjl_ref,basis,active,
    dirichlet_dofs,zeros(3),_->b_full;
    interior_rows=interior_rows,traction_layout=layout,
    neumann_ids=neumann_idx,neumann_adjl=neumann_adjl_ref,
    traction_jacobians=nothing,normal_jacobians=njacs)
Δpts_anal=res_full.Δpts
g_anal=[(Δpts_anal[2*i-1]*cos(θ_vals[j])+Δpts_anal[2*i]*sin(θ_vals[j]))
        for (j,i) in enumerate(hole_idx)]
C_base=solve_compliance(pts)

# ---- FD-smoothed gradient at varying scales ---------------------------------
println("\n=== FD-smoothed gradient sweep ===")
@printf("  %8s  %12s  %14s  %14s  %14s\n","fd_scale","h_actual","‖grad‖","hi/lo","cos w/ anal")
@printf("  %8s  %12s  %14s  %14s  %14s\n","-"^8,"-"^12,"-"^14,"-"^14,"-"^14)

# Only use scales where fixed adjacency is safe (≤ 2.5dx typically)
for fd_scale in [0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.5]
    h_fd=fd_scale*dx
    g_fd=zeros(nθ)
    for j in 1:nθ
        i_global=hole_idx[j]
        r̂=SVector(cos(θ_vals[j]),sin(θ_vals[j]))
        pts_p=copy(pts);pts_p[i_global]+=h_fd*r̂
        pts_m=copy(pts);pts_m[i_global]-=h_fd*r̂
        Cp=solve_compliance(pts_p)
        Cm=solve_compliance(pts_m)
        g_fd[j]=(Cp-Cm)/(2h_fd)
    end
    hl=hi_lo(g_fd)
    nr=totnorm(g_fd)
    cs=dot(g_fd,g_anal)/(norm(g_fd)*norm(g_anal))
    @printf("  %8.2f  %12.3e  %14.4e  %14.3f  %14.4f\n",fd_scale,h_fd,nr,hl,cs)
end

# ---- detailed comparison at best scale (h = 1.0 dx) -------------------------
println("\n=== Detailed spectrum at fd_scale=1.0 (h=dx) ===")
h_best=1.0*dx
g_best=zeros(nθ)
for j in 1:nθ
    i_global=hole_idx[j];r̂=SVector(cos(θ_vals[j]),sin(θ_vals[j]))
    pts_p=copy(pts);pts_p[i_global]+=h_best*r̂
    pts_m=copy(pts);pts_m[i_global]-=h_best*r̂
    g_best[j]=(solve_compliance(pts_p)-solve_compliance(pts_m))/(2h_best)
end

@printf("  %3s  %14s  %14s  %14s\n","m","analytic","FD(h=dx)","ratio")
for m in [0,2,4,6,8,10,12,16,20,24]
    fa=famp(g_anal,m);ff=famp(g_best,m)
    @printf("  %3d  %14.4e  %14.4e  %14.3f\n",m,fa,ff,fa>1e-30 ? ff/fa : NaN)
end
println("  ...")
for m in (nθ÷2-3):(nθ÷2)
    fa=famp(g_anal,m);ff=famp(g_best,m)
    @printf("  %3d  %14.4e  %14.4e  %14.3f\n",m,fa,ff,fa>1e-30 ? ff/fa : NaN)
end

# ---- mode-projected gradient: does it point in the right direction? ----------
# Project onto Fourier modes m=2,3,4
project_mode(g,m)=sum(g[j]*cos(m*θ_vals[j]) for j in 1:nθ)*(2/nθ)
# The compliance gradient w.r.t. a_m is: dC/da_m = Σ g_rad,j * cos(m*θ_j)
# For an ellipse (a0=0.40, b0=0.20), the circle has a=b=const, so:
#   a_2 = (r_major - r_minor)/2 * cos(2θ) [approximately]
#   Positive dC/da_2 means compliance INCREASES with a_2 (more elliptical)
#   So gradient descent should DECREASE a_2 → circle

println("\n=== Mode-projected design gradient ===")
@printf("  %6s  %14s  %14s  %14s\n","mode","anal dC/da_m","FD dC/da_m","ratio")
for m in [2,3,4,6,8]
    pa=project_mode(g_anal,m); pf=project_mode(g_best,m)
    @printf("  m=%-3d  %+14.4e  %+14.4e  %14.3f\n",m,pa,pf,abs(pa)>1e-30 ? pf/pa : NaN)
end

# FD-validate the mode-projected gradient by perturbing the Fourier coefficient
println("\n=== FD validation of mode gradient (a_2) ===")
ε_a=1e-4
# a_2 perturbation: r(θ) = r₀ + (a₂±ε)cos(2θ) on an otherwise-circular hole
r₀_c=sqrt(0.40*0.20)  # area-equivalent circle radius
A_target=π*0.40*0.20
a2_0=0.0  # start from circle
function C_from_a2(a2)
    r0=sqrt(max(0.0,(A_target-π/2*a2^2)/π))
    hole=[SVector((r0+a2*cos(2*t))*cos(t),(r0+a2*cos(2*t))*sin(t)) for t in θ_vals]
    return solve_compliance(vcat(ref_interior,outer_pts,hole))
end
C0=C_from_a2(a2_0)
Cp=C_from_a2(a2_0+ε_a)
Cm=C_from_a2(a2_0-ε_a)
fd_dCda2=(Cp-Cm)/(2ε_a)
# The mode gradient from the nodal gradient at a_2=0:
hole_c=[SVector(r₀_c*cos(t),r₀_c*sin(t)) for t in θ_vals]
pts_c=vcat(ref_interior,outer_pts,hole_c)
# Compute FD gradient at each hole node, then project
g_c=zeros(nθ)
for j in 1:nθ
    i=hole_idx[j];r̂=SVector(cos(θ_vals[j]),sin(θ_vals[j]))
    pts_p=copy(pts_c);pts_p[i]+=h_best*r̂
    pts_m=copy(pts_c);pts_m[i]-=h_best*r̂
    g_c[j]=(solve_compliance(pts_p)-solve_compliance(pts_m))/(2h_best)
end
dC_da2_fd=sum(g_c[j]*cos(2*θ_vals[j]) for j in 1:nθ)
@printf("  dC/da₂ (full FD, Δa₂=%.0e): %+.4e\n",ε_a,fd_dCda2)
@printf("  dC/da₂ (nodal FD→project):   %+.4e\n",dC_da2_fd)
@printf("  ratio: %.3f\n",dC_da2_fd/fd_dCda2)
@printf("  sign correct (should be + for ellipse→circle)? %s\n",
    fd_dCda2>0 ? "YES (dC/da₂>0 ⇒ descent reduces a₂)" : "NO")

println("\n======== VERDICT ========")
hl_anal=hi_lo(g_anal);hl_fd=hi_lo(g_best)
@printf("  Analytic gradient:      ‖g‖=%.4e  hi/lo=%.2f\n",totnorm(g_anal),hl_anal)
@printf("  FD-smoothed (h=dx):     ‖g‖=%.4e  hi/lo=%.2f\n",totnorm(g_best),hl_fd)
@printf("  Improvement factor:     %.1f×\n",hl_anal/max(hl_fd,1e-30))
fa2_anal=famp(g_anal,2);fa2_fd=famp(g_best,2)
@printf("  m=2 amplitude:  anal=%.4e  FD=%.4e  ratio=%.3f\n",fa2_anal,fa2_fd,fa2_fd/max(fa2_anal,1e-30))
