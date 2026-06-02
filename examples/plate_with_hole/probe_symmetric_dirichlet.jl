# ============================================================================
# probe_symmetric_dirichlet.jl
#
# Tests symmetric Dirichlet BC elimination.  With zero prescribed values
# (the common case), symmetric elimination just zeroes columns as well as rows.
# This makes A = Aᵀ, so η = A⁻¹b = u — inheriting u's smoothness (hi/lo=0.01).
#
# Questions:
#   Q1: Does symmetric elimination preserve the forward solution u?
#   Q2: Does it make η = u (up to machine precision)?
#   Q3: Does the shape gradient change?  (FD-validate against full FD)
#
# Run:  jlrun plate_with_hole/probe_symmetric_dirichlet.jl   (from examples/)
# ============================================================================
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Macchiato
import RadialBasisFunctions
import RadialBasisFunctions: _build_weights, Partial, MixedPartial, find_neighbors
using RadialBasisFunctions: PHS
using StaticArrays
using LinearAlgebra
using SparseArrays
using Printf

# ---- problem setup (same as fourier_opt) ------------------------------------
const Lx=4.0; const Ly=4.0; const a0=0.40; const b0=0.20
const dx=0.05; const σ∞=1.0
model = LinearElasticity(E=1.0e7, ν=0.3)
μ, λstar = lame_parameters(model)
basis = PHS(3; poly_deg=3); const k=35

ellipse_val(p,a,b)=(p[1]/a)^2+(p[2]/b)^2
const margin=1+1.2*dx/min(a0,b0)
const nθ=max(48,round(Int,2π*sqrt((a0^2+b0^2)/2)/dx))

ref_interior=SVector{2,Float64}[]
let xs=(-Lx/2+dx):dx:(Lx/2-dx/2), ys=(-Ly/2+dx):dx:(Ly/2-dx/2)
    for x in xs, y in ys
        ellipse_val(SVector(x,y),a0,b0)>margin^2 || continue
        push!(ref_interior,SVector(x,y))
    end
end
const n_int=length(ref_interior)
outer_pts=SVector{2,Float64}[]; outer_tag=Symbol[]
for y in (-Ly/2):dx:(Ly/2)
    push!(outer_pts,SVector(-Lx/2,y)); push!(outer_tag,:xlo)
    push!(outer_pts,SVector(Lx/2,y));  push!(outer_tag,:xhi)
end
for x in (-Lx/2+dx):dx:(Lx/2-dx)
    push!(outer_pts,SVector(x,Ly/2)); push!(outer_tag,:yhi)
    push!(outer_pts,SVector(x,-Ly/2)); push!(outer_tag,:ylo)
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
pin_ux1=nearest((0.0,0.8),interior_idx); pin_ux2=nearest((0.0,-0.8),interior_idx)
pin_uy1=nearest((0.8,0.0),interior_idx)
const dirichlet_dofs=[pin_ux1,pin_ux2,pin_uy1+N]
const active=let a=trues(2N); a[pin_ux1]=a[pin_ux2]=a[pin_uy1+N]=false; a end
const interior_rows=let r=falses(N); for i in interior_idx; r[i]=true; end; r end
flat(v)=reduce(vcat,[[p[1],p[2]] for p in v])
outer_normal(i)=outer_tag[i]===:xhi ? SVector(1.0,0.0) :
                outer_tag[i]===:xlo ? SVector(-1.0,0.0) :
                outer_tag[i]===:yhi ? SVector(0.0,1.0) : SVector(0.0,-1.0)
_ZJV=SVector(0.0,0.0); zero_njac(g)=NormalJacobian(g,g,_ZJV,_ZJV,_ZJV,_ZJV)

adjl_ref=find_neighbors(ref_pts,k); pts=ref_pts

# ---- build system -----------------------------------------------------------
neumann_adjl_ref=adjl_ref[neumann_idx]
hn,hjacs=polyline_normals(flat(pts),hole_idx,hole_pos)
njacs=vcat(NormalJacobian[zero_njac(i) for i in outer_idx],hjacs)
normals=SVector{2,Float64}[]; tractions=SVector{2,Float64}[]
for i in 1:n_outer
    nn=outer_normal(i); push!(normals,nn); push!(tractions,σ∞.*nn)
end
append!(normals,hn); append!(tractions,fill(SVector(0.0,0.0),nθ))
layout=build_traction_layout(neumann_idx,neumann_adjl_ref,normals,tractions,λstar,μ,N)

W_d2x=_build_weights(Partial(2,1),pts,pts,adjl_ref,basis)
W_d2y=_build_weights(Partial(2,2),pts,pts,adjl_ref,basis)
W_d2xy=_build_weights(MixedPartial(1,2),pts,pts,adjl_ref,basis)
neu_pts=pts[neumann_idx]
W_dx=_build_weights(Partial(1,1),pts,neu_pts,neumann_adjl_ref,basis)
W_dy=_build_weights(Partial(1,2),pts,neu_pts,neumann_adjl_ref,basis)

# ---- Method 1: current (row-only) Dirichlet --------------------------------
A1=assemble_elasticity_from_weights(W_d2x,W_d2y,W_d2xy,N,λstar,μ)
b1=zeros(2N)
apply_dirichlet!(A1,b1,dirichlet_dofs,zeros(length(dirichlet_dofs)))
apply_traction!(A1,b1,layout,W_dx,W_dy)
u1=A1\b1
η1=A1'\b1
C1=dot(b1,u1)

# ---- Method 2: symmetric Dirichlet (zero rows AND columns) ------------------
# CORRECT sparse implementation: cannot use A[i,j]=0.0 on structural nonzeros
# because it inserts new entries.  Instead, work with SparseMatrixCSC internals.
A2=assemble_elasticity_from_weights(W_d2x,W_d2y,W_d2xy,N,λstar,μ)
b2=zeros(2N)
let dir_set = BitVector(zeros(Bool, 2N))
    for i in dirichlet_dofs; dir_set[i] = true; end
    @inbounds for j in 1:size(A2, 2)
        in_dir_col = dir_set[j]
        for idx in nzrange(A2, j)
            i = A2.rowval[idx]
            if (dir_set[i] && i != j) || (in_dir_col && i != j)
                A2.nzval[idx] = 0.0
            end
        end
        if in_dir_col
            # Set diagonal to 1. It must exist in the pattern (it's a stencil diagonal).
            found = false
            for idx in nzrange(A2, j)
                if A2.rowval[idx] == j
                    A2.nzval[idx] = 1.0
                    found = true; break
                end
            end
            found || (A2[j,j] = 1.0)   # fallback: insert if missing
        end
    end
end
for (k,i) in enumerate(dirichlet_dofs)
    b2[i] = 0.0   # g = 0
end
dropzeros!(A2)   # remove structural zeros we just created
# Apply traction (uses same layout, same weights)
apply_traction!(A2,b2,layout,W_dx,W_dy)
u2=A2\b2
η2=A2'\b2   # should equal u2 if A2 is symmetric
C2=dot(b2,u2)

# ---- comparisons -------------------------------------------------------------
println("=== Q1: Forward solution preserved? ===")
@printf("  ‖u₁ - u₂‖∞  = %.2e\n", norm(u1-u2,Inf))
@printf("  |C₁ - C₂|/C₁ = %.2e\n", abs(C1-C2)/abs(C1))
@printf("  ‖u₁‖₂ = %.4e   ‖u₂‖₂ = %.4e\n", norm(u1), norm(u2))

println("\n=== Q2: Is A₂ symmetric?  Does η₂ = u₂? ===")
@printf("  ‖A₂ - A₂ᵀ‖∞ = %.2e\n", norm(A2-A2',Inf))
@printf("  ‖η₂ - u₂‖∞  = %.2e\n", norm(η2-u2,Inf))
@printf("  ‖η₁ - u₁‖∞  = %.2e  (current, for comparison)\n", norm(η1-u1,Inf))

# spectral content on the hole boundary
famp(g,m)=m==0 ? abs(sum(g)/nθ) :
    hypot((2/nθ)*sum(g[j]*cos(m*θ_vals[j]) for j in 1:nθ),
          (2/nθ)*sum(g[j]*sin(m*θ_vals[j]) for j in 1:nθ))
function hi_lo_ratio(g)
    lo=sqrt(sum(famp(g,m)^2 for m in 2:5))
    hi=sqrt(sum(famp(g,m)^2 for m in (nθ÷2-3):(nθ÷2)))
    return hi/lo
end

# Extract radial components of u and η at hole nodes
function radial_at_hole(v)
    [v[i]*cos(θ_vals[j])+v[i+N]*sin(θ_vals[j]) for (j,i) in enumerate(hole_idx)]
end
function tang_at_hole(v)
    [-v[i]*sin(θ_vals[j])+v[i+N]*cos(θ_vals[j]) for (j,i) in enumerate(hole_idx)]
end

println("\n=== Q3: Smoothness of u and η ===")
for (label,v) in [("u₁ (current fwd)",u1),("η₁ (current adj)",η1),
                   ("u₂ (sym fwd)",u2),("η₂ (sym adj)",η2)]
    r_hl=hi_lo_ratio(radial_at_hole(v))
    t_hl=hi_lo_ratio(tang_at_hole(v))
    @printf("  %-20s  hi/lo(rad)=%.3f  hi/lo(tan)=%.3f\n",label,r_hl,t_hl)
end

# ---- Q4: shape gradient comparison ------------------------------------------
println("\n=== Q4: Shape gradient comparison ===")
# We compare TWO gradient computations:
#   (a) Current: η = K⁻ᵀb (rough), u = K⁻¹b (smooth)
#   (b) Symmetric Dirichlet: η = u = K⁻¹b (both smooth)
# In both cases we keep the Neumann ΔW extraction identical (it uses the
# same traction_layout, which is the same in both methods).

# Helper: assemble gradient from ΔW and weight operators
function assemble_gradient(η_use, u_use, label)
    # Allocate and extract second-derivative ΔW
    dWx,dWy,dWxy=Macchiato.allocate_weight_gradients(W_d2x,W_d2x,W_d2x)
    Macchiato.extract_weight_sensitivities_elasticity!(
        dWx,dWy,dWxy,W_d2x,η_use,u_use,active,λstar,μ; interior_rows=interior_rows)
    # Allocate and extract first-derivative (Neumann) ΔW
    dWdx,dWdy=Macchiato.allocate_weight_gradients(W_dx,W_dy)
    Macchiato.extract_neumann_sensitivities!(dWdx,dWdy,layout,η_use,u_use,active)

    # Build caches (needed by _propagate_weight_gradient!)
    _,cache_x  = RadialBasisFunctions._build_weights_and_cache(Partial(2,1),pts,pts,adjl_ref,basis)
    _,cache_y  = RadialBasisFunctions._build_weights_and_cache(Partial(2,2),pts,pts,adjl_ref,basis)
    _,cache_xy = RadialBasisFunctions._build_weights_and_cache(MixedPartial(1,2),pts,pts,adjl_ref,basis)
    _,cache_dx = RadialBasisFunctions._build_weights_and_cache(Partial(1,1),pts,neu_pts,neumann_adjl_ref,basis)
    _,cache_dy = RadialBasisFunctions._build_weights_and_cache(Partial(1,2),pts,neu_pts,neumann_adjl_ref,basis)

    Δp=zeros(Float64,2N)
    Macchiato._propagate_weight_gradient!(Δp,dWx, W_d2x, cache_x,  pts,pts,adjl_ref,basis,Partial(2,1))
    Macchiato._propagate_weight_gradient!(Δp,dWy, W_d2y, cache_y,  pts,pts,adjl_ref,basis,Partial(2,2))
    Macchiato._propagate_weight_gradient!(Δp,dWxy,W_d2xy,cache_xy, pts,pts,adjl_ref,basis,MixedPartial(1,2))
    Macchiato._propagate_weight_gradient!(Δp,dWdx,W_dx,  cache_dx, pts,neu_pts,neumann_adjl_ref,basis,Partial(1,1);eval_offset=neumann_idx)
    Macchiato._propagate_weight_gradient!(Δp,dWdy,W_dy,  cache_dy, pts,neu_pts,neumann_adjl_ref,basis,Partial(1,2);eval_offset=neumann_idx)
    # Note: omitting traction_jacobians and normal_jacobians (frozen load case)

    grad_rad=[(Δp[2*i-1]*cos(θ_vals[j])+Δp[2*i]*sin(θ_vals[j]))
              for (j,i) in enumerate(hole_idx)]
    hl=hi_lo_ratio(grad_rad)
    nrm=norm(grad_rad)
    @printf("  %-12s  ‖grad‖=%.4e  hi/lo=%.3f\n",label,nrm,hl)
    return grad_rad, Δp
end

g1_rad_full, Δpts1_full = assemble_gradient(η1, u1, "η₁ (rough)")
g2_rad_sym,  Δpts2_sym  = assemble_gradient(u2, u2, "η₂=u₂(smooth)")

# ---- Q5: FD validation of symmetric gradient ------------------------------------
println("\n=== Q5: FD validation (symmetric gradient vs full FD) ===")
εfd=1e-4
n_test=min(6,nθ)
for j in 1:n_test
    i=hole_idx[j]; r̂=SVector(cos(θ_vals[j]),sin(θ_vals[j]))
    pts_p=copy(pts); pts_p[i]+=εfd*r̂
    pts_m=copy(pts); pts_m[i]-=εfd*r̂
    function solve_C_sym(pts_use)
        W2x=_build_weights(Partial(2,1),pts_use,pts_use,adjl_ref,basis)
        W2y=_build_weights(Partial(2,2),pts_use,pts_use,adjl_ref,basis)
        W2xy=_build_weights(MixedPartial(1,2),pts_use,pts_use,adjl_ref,basis)
        A_use=assemble_elasticity_from_weights(W2x,W2y,W2xy,N,λstar,μ)
        b_use=zeros(2N)
        # symmetric Dirichlet: zero rows AND columns
        dir_set=BitVector(zeros(Bool,2N))
        for ii in dirichlet_dofs; dir_set[ii]=true; end
        @inbounds for col in 1:size(A_use,2)
            in_dir=dir_set[col]
            for idx in nzrange(A_use,col)
                r=A_use.rowval[idx]
                if (dir_set[r] && r!=col) || (in_dir && r!=col)
                    A_use.nzval[idx]=0.0
                end
            end
            if in_dir
                for idx in nzrange(A_use,col)
                    A_use.rowval[idx]==col && (A_use.nzval[idx]=1.0)
                end
            end
        end
        dropzeros!(A_use)
        for (kk,ii) in enumerate(dirichlet_dofs); b_use[ii]=0.0; end
        # Neumann BC (same as before)
        n_adjl=find_neighbors(pts_use,neumann_idx,k)
        hn_use,_=polyline_normals(flat(pts_use),hole_idx,hole_pos)
        norms_use=SVector{2,Float64}[]
        for ii in 1:n_outer; push!(norms_use,outer_normal(ii)); end
        append!(norms_use,hn_use)
        tracts_use=SVector{2,Float64}[]
        for ii in 1:n_outer; push!(tracts_use,σ∞.*outer_normal(ii)); end
        append!(tracts_use,fill(SVector(0.0,0.0),nθ))
        lay_use=build_traction_layout(neumann_idx,n_adjl,norms_use,tracts_use,λstar,μ,N)
        Wdx_use=_build_weights(Partial(1,1),pts_use,pts_use[neumann_idx],n_adjl,basis)
        Wdy_use=_build_weights(Partial(1,2),pts_use,pts_use[neumann_idx],n_adjl,basis)
        apply_traction!(A_use,b_use,lay_use,Wdx_use,Wdy_use)
        u_use=A_use\b_use
        return dot(b_use,u_use)
    end
    Cp=solve_C_sym(pts_p); Cm=solve_C_sym(pts_m)
    fd=(Cp-Cm)/(2εfd)
    adj_rad=g2_rad_sym[j]
    @printf("  node %2d:  adj= %+.4e  FD= %+.4e  rel.err= %.2e\n",
        j,adj_rad,fd,abs(adj_rad-fd)/max(abs(fd),1e-30))
end

println("\n======== SUMMARY ========")
println("  Symmetric Dirichlet: forward u unchanged (‖Δu‖∞ = $(round(norm(u1-u2,Inf),sigdigits=2)))")
println("  A₂ symmetric? ‖A₂-A₂ᵀ‖∞ = $(round(norm(A2-A2',Inf),sigdigits=2))")
println("  η₂ = u₂ to within $(round(norm(η2-u2,Inf),sigdigits=2))  (vs ‖η₁-u₁‖∞ = $(round(norm(η1-u1,Inf),sigdigits=2)) for current)")
println("  Gradient hi/lo:  η₁=$(round(hi_lo_ratio(g1_rad_full),digits=2))  →  η₂=u₂: $(round(hi_lo_ratio(g2_rad_sym),digits=2))")
