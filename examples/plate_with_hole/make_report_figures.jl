# ============================================================================
# make_report_figures.jl — generate the plate-with-hole figures for
# RBF_FD_shape_optimization.tex.  Writes PNGs into ../../figures/.
#
#   fig_setup.png    — point cloud, boundary classes, BCs, biaxial load
#   fig_noise.png    — per-node radial gradient (sawtooth) + Fourier spectrum
#   fig_morph.png    — Laplace morph: interior slaved to a moving boundary
#   fig_bias.png     — stale-cloud objective bias bracket (fixed/morph/two-front)
#
# Run:  jlrun plate_with_hole/make_report_figures.jl   (from examples/)
# ============================================================================
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Macchiato
import RadialBasisFunctions: _build_weights, Partial, MixedPartial, find_neighbors
using RadialBasisFunctions: PHS
using StaticArrays, LinearAlgebra, Printf
using CairoMakie

const FIGDIR = normpath(joinpath(@__DIR__, "..", "..", "figures"))
mkpath(FIGDIR)

# ---- shared setup (the validated plate-with-hole cloud) --------------------
const Lx=4.0; const Ly=4.0; const a0=0.40; const b0=0.20; const dx=0.05; const σ∞=1.0
model=LinearElasticity(E=1.0e7,ν=0.3); μ,λstar=lame_parameters(model)
basis=PHS(3;poly_deg=3); const k=35
const r_circle=sqrt(a0*b0)
const nθ=max(48,round(Int,2π*sqrt((a0^2+b0^2)/2)/dx))
const margin=1+1.2*dx/min(a0,b0)
flat(v)=reduce(vcat,[[p[1],p[2]] for p in v])
const _ZJV=SVector(0.0,0.0); zero_njac(i)=NormalJacobian(i,i,_ZJV,_ZJV,_ZJV,_ZJV)

interior=SVector{2,Float64}[]
for x in (-Lx/2+dx):dx:(Lx/2-dx/2), y in (-Ly/2+dx):dx:(Ly/2-dx/2)
    (x/a0)^2+(y/b0)^2 > margin^2 || continue
    push!(interior,SVector(x,y))
end
outer=SVector{2,Float64}[]; otag=Symbol[]
for y in (-Ly/2):dx:(Ly/2); push!(outer,SVector(-Lx/2,y)); push!(otag,:xlo); push!(outer,SVector(Lx/2,y)); push!(otag,:xhi); end
for x in (-Lx/2+dx):dx:(Lx/2-dx); push!(outer,SVector(x,Ly/2)); push!(otag,:yhi); push!(outer,SVector(x,-Ly/2)); push!(otag,:ylo); end
const n_int=length(interior); const n_outer=length(outer)
θ_vals=[2π*(j-1)/nθ for j in 1:nθ]
hole0=[SVector(a0*cos(t),b0*sin(t)) for t in θ_vals]
pts0=vcat(interior,outer,hole0); const N=length(pts0)
const interior_idx=collect(1:n_int)
const outer_idx=collect(n_int+1:n_int+n_outer)
const hole_idx=collect(n_int+n_outer+1:N)
const boundary_idx=vcat(outer_idx,hole_idx)
const neumann_idx=vcat(outer_idx,hole_idx)
const adjl0=find_neighbors(pts0,k)
nearest(p0)=interior_idx[argmin([hypot(pts0[i][1]-p0[1],pts0[i][2]-p0[2]) for i in interior_idx])]
const pin_ux1=nearest((0.0,0.8)); const pin_ux2=nearest((0.0,-0.8)); const pin_uy1=nearest((0.8,0.0))
const dirichlet_dofs=[pin_ux1,pin_ux2,pin_uy1+N]
const active=let a=trues(2N); a[pin_ux1]=a[pin_ux2]=a[pin_uy1+N]=false; a end
const interior_rows=let r=falses(N); for i in interior_idx; r[i]=true; end; r end
outer_normal(t)= t===:xhi ? SVector(1.0,0.0) : t===:xlo ? SVector(-1.0,0.0) : t===:yhi ? SVector(0.0,1.0) : SVector(0.0,-1.0)

# ============================================================================
# FIG 1 — problem setup
# ============================================================================
println("fig_setup ...")
let
    fig=Figure(size=(640,620))
    ax=Axis(fig[1,1]; title="Plate with a hole — discretization & boundary conditions",
            aspect=DataAspect(), xlabel="x", ylabel="y")
    scatter!(ax, getindex.(interior,1), getindex.(interior,2); color=(:gray70,0.7), markersize=3, label="interior")
    scatter!(ax, getindex.(outer,1), getindex.(outer,2); color=:black, markersize=5, label="outer (Neumann)")
    scatter!(ax, getindex.(hole0,1), getindex.(hole0,2); color=:firebrick, markersize=6, label="hole (free)")
    # biaxial traction arrows on the outer frame
    for (s,(x,y,dx2,dy2)) in enumerate([(2.2,0,0.5,0),(-2.2,0,-0.5,0),(0,2.2,0,0.5),(0,-2.2,0,-0.5)])
        arrows!(ax,[x-dx2],[y-dy2],[dx2],[dy2]; color=:steelblue, linewidth=2, arrowsize=12)
    end
    text!(ax, 2.35, 0.15; text="σ∞", color=:steelblue, fontsize=16)
    # Dirichlet pins
    pins=[pts0[pin_ux1],pts0[pin_ux2],pts0[pin_uy1]]
    scatter!(ax, getindex.(pins,1), getindex.(pins,2); marker=:xcross, color=:green, markersize=16, label="Dirichlet pins")
    axislegend(ax; position=:rt, framevisible=true, backgroundcolor=(:white,0.85))
    save(joinpath(FIGDIR,"fig_setup.png"), fig)
end

# ============================================================================
# adjoint nodal gradient at the hole (for the noise figure)
# ============================================================================
function adjoint_hole_grad(hole_pts)
    pts=vcat(interior,outer,hole_pts)
    hn,hjacs=polyline_normals(flat(pts),hole_idx,collect(1:nθ))
    njacs=vcat([zero_njac(i) for i in outer_idx],hjacs)
    normals=SVector{2,Float64}[]; tractions=SVector{2,Float64}[]
    for i in 1:n_outer; nn=outer_normal(otag[i]); push!(normals,nn); push!(tractions,σ∞.*nn); end
    append!(normals,hn); append!(tractions,fill(SVector(0.0,0.0),nθ))
    layout=build_traction_layout(neumann_idx,adjl0[neumann_idx],normals,tractions,λstar,μ,N)
    b=zeros(2N); for kk in eachindex(layout.rows); b[layout.rows[kk]]=layout.b_vals[kk]; end
    res=shape_gradient(flat(pts),model,N,adjl0,basis,active,dirichlet_dofs,zeros(3),_->b;
        interior_rows=interior_rows,traction_layout=layout,
        neumann_ids=neumann_idx,neumann_adjl=adjl0[neumann_idx],
        traction_jacobians=nothing,normal_jacobians=njacs)
    [SVector(res.Δpts[2*i-1],res.Δpts[2*i]) for i in hole_idx]
end

# ============================================================================
# FIG 2 — the noise: radial gradient sawtooth + rising Fourier spectrum
# ============================================================================
println("fig_noise ...")
let
    g_hole=adjoint_hole_grad(hole0)
    θ=[atan(hole0[j][2],hole0[j][1]) for j in 1:nθ]
    g_rad=[g_hole[j][1]*cos(θ[j])+g_hole[j][2]*sin(θ[j]) for j in 1:nθ]
    famp(m)= m==0 ? abs(sum(g_rad)/nθ) : hypot((2/nθ)*sum(g_rad[j]*cos(m*θ[j]) for j in 1:nθ),
                                                (2/nθ)*sum(g_rad[j]*sin(m*θ[j]) for j in 1:nθ))
    ms=0:(nθ÷2); amps=[famp(m) for m in ms]
    order=sortperm(θ)
    fig=Figure(size=(1100,420))
    ax1=Axis(fig[1,1]; title="Per-node radial shape gradient (boundary walk)",
             xlabel="boundary node angle θ [rad]", ylabel="∂C/∂r_j")
    lines!(ax1, θ[order], g_rad[order]; color=:firebrick, linewidth=1.5)
    scatter!(ax1, θ[order], g_rad[order]; color=:firebrick, markersize=4)
    ax2=Axis(fig[1,2]; title="Fourier amplitude spectrum (rises toward Nyquist)",
             xlabel="mode m", ylabel="|∂C/∂a_m|", yscale=log10)
    barplot!(ax2, collect(2:length(amps)-1), max.(amps[3:end],1e-12); color=:gray60)
    barplot!(ax2, [2], [max(amps[3],1e-12)]; color=:seagreen)
    text!(ax2, 2, amps[3]*1.3; text="m=2\n(circle-seeking)", color=:seagreen, fontsize=11, align=(:left,:bottom))
    save(joinpath(FIGDIR,"fig_noise.png"), fig)
end

# ============================================================================
# FIG 3 — Front 1 morph: interior slaved to a moving boundary
# ============================================================================
println("fig_morph ...")
let
    Wxx=_build_weights(Partial(2,1),pts0,pts0,adjl0,basis)
    Wyy=_build_weights(Partial(2,2),pts0,pts0,adjl0,basis)
    Wlap=Wxx+Wyy
    L_int=Wlap[interior_idx,interior_idx]+1e-8*I
    L_int_bnd=Wlap[interior_idx,boundary_idx]
    fac=lu(L_int)
    # move the hole from the ellipse toward the circle (a visible deformation)
    hole_new=[SVector(r_circle*cos(t),r_circle*sin(t)) for t in θ_vals]
    Δbx=vcat(zeros(n_outer),[hole_new[j][1]-hole0[j][1] for j in 1:nθ])
    Δby=vcat(zeros(n_outer),[hole_new[j][2]-hole0[j][2] for j in 1:nθ])
    Δix=fac\(-L_int_bnd*Δbx); Δiy=fac\(-L_int_bnd*Δby)
    moved=[interior[kk]+SVector(Δix[kk],Δiy[kk]) for kk in 1:n_int]
    fig=Figure(size=(640,620))
    ax=Axis(fig[1,1]; title="Front 1 — Laplace morph: interior follows the boundary",
            aspect=DataAspect(), xlabel="x", ylabel="y")
    scatter!(ax, getindex.(interior,1), getindex.(interior,2); color=(:gray75,0.6), markersize=3, label="reference interior")
    scatter!(ax, getindex.(moved,1), getindex.(moved,2); color=(:steelblue,0.8), markersize=3, label="morphed interior")
    # displacement arrows (subsampled, scaled ×3 for visibility)
    sub=1:7:n_int
    arrows!(ax, getindex.(interior[sub],1), getindex.(interior[sub],2),
            3 .*Δix[sub], 3 .*Δiy[sub]; color=(:black,0.5), linewidth=0.8, arrowsize=5)
    lines!(ax, vcat(getindex.(hole0,1),hole0[1][1]), vcat(getindex.(hole0,2),hole0[1][2]); color=:firebrick, linewidth=2, label="hole start")
    lines!(ax, vcat(getindex.(hole_new,1),hole_new[1][1]), vcat(getindex.(hole_new,2),hole_new[1][2]); color=:seagreen, linewidth=2, label="hole moved")
    axislegend(ax; position=:rt, framevisible=true, backgroundcolor=(:white,0.85))
    text!(ax, -1.9, -1.9; text="arrows ×3", color=:gray40, fontsize=11)
    save(joinpath(FIGDIR,"fig_morph.png"), fig)
end

# ============================================================================
# FIG 4 — the stale-cloud objective-bias bracket
# (measured discrete optima a₂*; the true circle is a₂=0)
# ============================================================================
println("fig_bias ...")
let
    regimes=["fixed cloud\n(stale interior)","Laplace morph\n(no remesh)","two-front\n(re-anchored remesh)"]
    a2star=[0.078, -0.065, 0.015]; cols=[:firebrick,:orange,:seagreen]
    fig=Figure(size=(760,360))
    ax=Axis(fig[1,1]; title="Discrete-compliance optimum a₂* vs the true circle",
            xlabel="a₂*  (true circle = 0)", yticksvisible=false, yticklabelsvisible=false)
    vlines!(ax, [0.0]; color=:blue, linestyle=:dash, linewidth=2)
    text!(ax, 0.002, 2.7; text="true circle", color=:blue, fontsize=12)
    for (i,(a,c,name)) in enumerate(zip(a2star,cols,regimes))
        scatter!(ax, [a],[Float64(i)]; color=c, markersize=18)
        text!(ax, a, i+0.18; text=@sprintf("%s  (a₂*=%+.3f)",name,a), color=c, fontsize=11, align=(:center,:bottom))
    end
    xlims!(ax,-0.12,0.12); ylims!(ax,0.3,3.5)
    save(joinpath(FIGDIR,"fig_bias.png"), fig)
end

println("\nSaved figures to: ", FIGDIR)
foreach(f->println("  ",f), filter(f->endswith(f,".png"), readdir(FIGDIR)))
