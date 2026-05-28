# Probe: is the iter-0 hole gradient dominated by the uniform "area" mode?
# Decompose the raw radial gradient into its mean (area mode) and its
# deviation (shape mode), and check whether top vs right separate in sign
# AFTER the mean is removed. Standalone replica of the iter-0 setup.
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Macchiato
import RadialBasisFunctions: _build_weights, Partial, MixedPartial, find_neighbors
using RadialBasisFunctions: PHS
using StaticArrays, LinearAlgebra, Printf

const Lx = 4.0; const Ly = 4.0
const a0 = 0.40; const b0 = 0.20
const dx = 0.05
const σ∞ = 1.0
const nθ = max(48, round(Int, 2π * sqrt((a0^2 + b0^2) / 2) / dx))
model = LinearElasticity(E = 1.0e7, ν = 0.3)
μ, λstar = lame_parameters(model)
basis = PHS(3; poly_deg = 3); const k = 35

const margin = 1 + 1.2 * dx / min(a0, b0)
const xs_grid = collect((-Lx/2+dx):dx:(Lx/2-dx/2))
const ys_grid = collect((-Ly/2+dx):dx:(Ly/2-dx/2))
const outer_pts = SVector{2,Float64}[]; const outer_tag = Symbol[]
for y in (-Ly/2):dx:(Ly/2)
    push!(outer_pts, SVector(-Lx/2, y)); push!(outer_tag, :xlo)
    push!(outer_pts, SVector(Lx/2, y));  push!(outer_tag, :xhi)
end
for x in (-Lx/2+dx):dx:(Lx/2-dx)
    push!(outer_pts, SVector(x, Ly/2));  push!(outer_tag, :yhi)
    push!(outer_pts, SVector(x, -Ly/2)); push!(outer_tag, :ylo)
end
const n_outer = length(outer_pts)
outer_normal(i) = outer_tag[i]===:xhi ? SVector(1.0,0.0) :
                  outer_tag[i]===:xlo ? SVector(-1.0,0.0) :
                  outer_tag[i]===:yhi ? SVector(0.0,1.0) : SVector(0.0,-1.0)

function build_cloud(hole)
    interior_pts = SVector{2,Float64}[]
    for x in xs_grid, y in ys_grid
        p = SVector(x,y); inside=false; n=length(hole)
        for j in 1:n
            a=hole[j]; b=hole[mod1(j+1,n)]
            if (a[2]>p[2]) != (b[2]>p[2])
                xint = a[1] + (p[2]-a[2])*(b[1]-a[1])/(b[2]-a[2])
                if p[1]<xint; inside=!inside; end
            end
        end
        inside && continue
        dmin = minimum(hypot(p[1]-h[1], p[2]-h[2]) for h in hole)
        dmin < 1.2*dx && continue
        push!(interior_pts, p)
    end
    pts = vcat(interior_pts, outer_pts, hole)
    n_int = length(interior_pts)
    interior_idx = collect(1:n_int)
    outer_idx = collect((n_int+1):(n_int+n_outer))
    hole_idx = collect((n_int+n_outer+1):length(pts))
    neumann_idx = vcat(outer_idx, hole_idx)
    nearest(p0,pool)= begin bj=pool[1];bd=Inf
        for j in pool; d=hypot(pts[j][1]-p0[1],pts[j][2]-p0[2]); if d<bd;bd=d;bj=j;end;end; bj end
    pin_ux1=nearest((0.0,0.8),interior_idx); pin_ux2=nearest((0.0,-0.8),interior_idx)
    pin_uy1=nearest((0.8,0.0),interior_idx); N=length(pts)
    dirichlet_dofs=[pin_ux1,pin_ux2,pin_uy1+N]
    return (pts=pts,N=N,interior_idx=interior_idx,outer_idx=outer_idx,
            hole_idx=hole_idx,neumann_idx=neumann_idx,dirichlet_dofs=dirichlet_dofs)
end

hole0 = [SVector(a0*cos(2π*j/nθ), b0*sin(2π*j/nθ)) for j in 0:(nθ-1)]
cloud = build_cloud(hole0)
adjl = find_neighbors(cloud.pts, k)
neumann_adjl = adjl[cloud.neumann_idx]
(; pts,N,interior_idx,outer_idx,hole_idx,neumann_idx,dirichlet_dofs)=cloud

hole_pos_local = collect(1:length(hole0))
hn,hjacs = polyline_normals(reduce(vcat,[[p[1],p[2]] for p in pts]), hole_idx, hole_pos_local)
_ZJV=SVector(0.0,0.0); zero_njac(g)=NormalJacobian(g,g,_ZJV,_ZJV,_ZJV,_ZJV)
njacs = vcat(NormalJacobian[zero_njac(i) for i in outer_idx], hjacs)

normals=SVector{2,Float64}[]; tractions=SVector{2,Float64}[]
for i in 1:length(outer_idx); nn=outer_normal(i); push!(normals,nn); push!(tractions,σ∞.*nn); end
append!(normals,hn); append!(tractions, fill(SVector(0.0,0.0),length(hole0)))
layout = build_traction_layout(neumann_idx,neumann_adjl,normals,tractions,λstar,μ,N)
b=zeros(2N); for kk in eachindex(layout.rows); b[layout.rows[kk]]=layout.b_vals[kk]; end
active=trues(2N); for d in dirichlet_dofs; active[d]=false; end
interior_rows=falses(N); for i in interior_idx; interior_rows[i]=true; end

res = shape_gradient(reduce(vcat,[[p[1],p[2]] for p in pts]), model,N,adjl,basis,active,
    dirichlet_dofs, zeros(length(dirichlet_dofs)), _->b;
    interior_rows=interior_rows, traction_layout=layout,
    neumann_ids=neumann_idx, neumann_adjl=neumann_adjl,
    traction_jacobians=nothing, normal_jacobians=njacs)

C = dot(b,res.u)
g_hole = [SVector(res.Δpts[2*gi-1], res.Δpts[2*gi]) for gi in hole_idx]

# Radial (outward) component of the raw gradient at each hole node
rhat = [hole0[j]/hypot(hole0[j]...) for j in eachindex(hole0)]
g_rad = [dot(g_hole[j], rhat[j]) for j in eachindex(hole0)]
mean_rad = sum(g_rad)/length(g_rad)
dev_rad = g_rad .- mean_rad

# node indices closest to θ=0 (right tip) and θ=π/2 (top)
θ = [atan(hole0[j][2], hole0[j][1]) for j in eachindex(hole0)]
i_right = argmin(abs.(θ));            # θ≈0
i_top   = argmin(abs.(θ .- π/2));     # θ≈π/2

@printf("C = %.5e\n", C)
@printf("raw radial gradient: mean = %.4e   std = %.4e   (std/|mean| = %.3f)\n",
        mean_rad, sqrt(sum(abs2,dev_rad)/length(dev_rad)), sqrt(sum(abs2,dev_rad)/length(dev_rad))/abs(mean_rad))
@printf("\n              raw g_rad     dev (g_rad - mean)\n")
@printf("right tip  : %+.4e   %+.4e\n", g_rad[i_right], dev_rad[i_right])
@printf("top        : %+.4e   %+.4e\n", g_rad[i_top],   dev_rad[i_top])
println("\nDescent direction = -g.  'In' = toward center (shrink), 'Out' = away.")
@printf("RAW step:  right %s,  top %s\n",
        g_rad[i_right]>0 ? "IN" : "OUT", g_rad[i_top]>0 ? "IN" : "OUT")
@printf("AREA-PROJECTED step:  right %s,  top %s\n",
        dev_rad[i_right]>0 ? "IN" : "OUT", dev_rad[i_top]>0 ? "IN" : "OUT")
println("\nTarget (ellipse a=0.4>b=0.2 -> circle at fixed area): right should go IN (a down), top should go OUT (b up).")

# ---- Full angular profile + Fourier decomposition --------------------------
println("\n--- raw radial gradient around the loop (θ in degrees) ---")
order = sortperm(θ)
for j in order
    @printf("  θ=%6.1f°  g_rad=%+.4e\n", rad2deg(θ[j]), g_rad[j])
end

# Project onto low-frequency modes:  g_rad(θ) ≈ a0 + a2 cos2θ + b2 sin2θ + a4 cos4θ ...
n = length(θ)
proj(f) = sum(g_rad[j]*f(θ[j]) for j in 1:n)
norm2(f) = sum(f(θ[j])^2 for j in 1:n)
a0c = sum(g_rad)/n
a2 = proj(t->cos(2t))/norm2(t->cos(2t))
b2 = proj(t->sin(2t))/norm2(t->sin(2t))
a4 = proj(t->cos(4t))/norm2(t->cos(4t))
@printf("\nFourier of raw g_rad:  mean=%+.3e  cos2θ=%+.3e  sin2θ=%+.3e  cos4θ=%+.3e\n", a0c,a2,b2,a4)
# Energy in low modes vs total
recon = [a0c + a2*cos(2t)+b2*sin(2t)+a4*cos(4t) for t in θ]
res_hi = g_rad .- recon
@printf("‖low-mode (mean+2θ+4θ)‖ = %.3e   ‖high-freq residual‖ = %.3e   (resid/low = %.2f)\n",
        sqrt(sum(abs2,recon)), sqrt(sum(abs2,res_hi)), sqrt(sum(abs2,res_hi))/sqrt(sum(abs2,recon)))
println("\nFor ellipse(a>b)->circle the descent needs g_rad ~ +cos2θ  (>0 at right θ=0, <0 at top θ=90).")
@printf("Observed cos2θ coefficient sign: %s  (need POSITIVE for circle-seeking)\n", a2>0 ? "POSITIVE ✓" : "NEGATIVE ✗")

# ---- Contract the NOISY nodal gradient against the smooth (a,b) ellipse modes
# mode-a displacement at node j: ∂x_j/∂a = (cosφ_j, 0);  mode-b: (0, sinφ_j)
φ = [2π*(j-1)/length(hole0) for j in eachindex(hole0)]
dCda_adj = sum(g_hole[j][1]*cos(φ[j]) for j in eachindex(hole0))
dCdb_adj = sum(g_hole[j][2]*sin(φ[j]) for j in eachindex(hole0))
@printf("\nADJOINT gradient contracted onto smooth modes:  dC/da = %+.4e   dC/db = %+.4e\n", dCda_adj, dCdb_adj)
println("(Ground-truth from the well-posedness scan: dC/da > 0 [shrink a], dC/db < 0 [grow b].)")
@printf("Mode-projected adjoint says:  a should go %s,  b should go %s\n",
        dCda_adj>0 ? "DOWN ✓" : "UP ✗", dCdb_adj<0 ? "UP ✓" : "DOWN ✗")
