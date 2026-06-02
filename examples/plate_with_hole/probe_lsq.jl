# ============================================================================
# LSQ test — does least-squares (over-determined) differentiation smooth ∂W/∂x?
#
# Hypothesis (user): an over-determined weight construction makes W(X) a smoother
# function of node position (each weight averages more nodes ⇒ moving one node
# perturbs it less) ⇒ ∂W/∂x is smoother. The PRINCIPLED version of the k-sweep
# that partly helped (Test 1).
#
# Compare the q=2 Laplacian artifact for:
#   • PHS3+poly3 exact collocation, k=35  (our baseline; global coords)
#   • GMLS poly_deg-3 least-squares, stencil n=M..large  (local coords, uniform
#     weights; oversampling ratio n/M is the LSQ dial, M=10 monomials)
# The oversampling TREND (n=12→25→50) isolates the LSQ effect (local coords held
# fixed across the GMLS rows). Also report forward accuracy so we see LSQ doesn't
# wreck the solve.
#
# Run:  jlrun plate_with_hole/probe_lsq.jl   (from examples/)
# ============================================================================
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
import RadialBasisFunctions: _build_weights, Partial, find_neighbors
using RadialBasisFunctions: PHS
using StaticArrays, LinearAlgebra, SparseArrays, Printf

const Lx=4.0; const Ly=4.0; const a0=0.40; const b0=0.20; const dx=0.05
const nθ = max(48, round(Int, 2π*sqrt((a0^2+b0^2)/2)/dx))
const margin = 1 + 1.2*dx/min(a0,b0)
ellipse_val(p,a,b)=(p[1]/a)^2+(p[2]/b)^2

interior = SVector{2,Float64}[]
for x in (-Lx/2+dx):dx:(Lx/2-dx/2), y in (-Ly/2+dx):dx:(Ly/2-dx/2)
    ellipse_val(SVector(x,y),a0,b0) > margin^2 || continue
    push!(interior, SVector(x,y))
end
outer = SVector{2,Float64}[]
for y in (-Ly/2):dx:(Ly/2); push!(outer,SVector(-Lx/2,y)); push!(outer,SVector(Lx/2,y)); end
for x in (-Lx/2+dx):dx:(Lx/2-dx); push!(outer,SVector(x,Ly/2)); push!(outer,SVector(x,-Ly/2)); end
const n_int=length(interior); const n_out=length(outer)
hole0 = [SVector(a0*cos(2π*j/nθ), b0*sin(2π*j/nθ)) for j in 0:(nθ-1)]
const pts0 = vcat(interior, outer, hole0); const N=length(pts0)
const interior_idx = collect(1:n_int)
const hole_idx     = collect((n_int+n_out+1):N)

sf(p)=exp(0.30*p[1])*cos(0.40*p[2]); sη(p)=exp(0.20*p[2])*sin(0.30*p[1])
const fvec=[sf(p) for p in pts0]; const ηvec=[sη(p) for p in pts0]
const lap_exact=[(0.30^2-0.40^2)*sf(p) for p in pts0]
θ=[atan(hole0[j][2],hole0[j][1]) for j in 1:nθ]; r̂=[hole0[j]/hypot(hole0[j]...) for j in 1:nθ]

famp(f,m)= m==0 ? abs(sum(f)/nθ) :
    hypot((2/nθ)*sum(f[j]*cos(m*θ[j]) for j in 1:nθ), (2/nθ)*sum(f[j]*sin(m*θ[j]) for j in 1:nθ))
hiband(f)=sqrt(sum(famp(f,m)^2 for m in (nθ÷2-3):(nθ÷2)))
loband(f)=sqrt(sum(famp(f,m)^2 for m in 2:5))
totnorm(f)=norm(f .- sum(f)/nθ)

# ---- GMLS (poly_deg 3) least-squares Laplacian, local coords, uniform weights -
monos(ξ)=(x=ξ[1];y=ξ[2]; @SVector[1.0,x,y,x^2,x*y,y^2,x^3,x^2*y,x*y^2,y^3])
const M=10
const lp = @SVector[0.0,0,0,2,0,2,0,0,0,0]   # Laplacian of monomials at ξ=0
function gmls_L(pts, adjl_n)
    I=Int[]; J=Int[]; V=Float64[]
    for i in 1:N
        nbr = adjl_n[i]; n=length(nbr); xe=pts[i]
        P = Matrix{Float64}(undef, n, M)
        for a in 1:n; P[a,:]=monos(pts[nbr[a]]-xe); end
        # GMLS: ℒf ≈ lpᵀ (PᵀP)⁻¹ Pᵀ f  ⇒ row weights = P (PᵀP)⁻¹ lp
        wrow = P * ((P'P) \ lp)   # length n

        append!(I, fill(i,n)); append!(J, nbr); append!(V, wrow)
    end
    return sparse(I,J,V,N,N)
end

# ---- PHS3+poly3 exact collocation Laplacian (baseline) ---------------------
function phs_L(pts, adjl, basis)
    _build_weights(Partial(2,1),pts,pts,adjl,basis) + _build_weights(Partial(2,2),pts,pts,adjl,basis)
end

# ---- q=2 artifact + accuracy for a given Laplacian builder -----------------
function assess(name, Lbuild)
    L0 = Lbuild(pts0)
    lapf = L0*fvec
    acc = norm(lapf[interior_idx]-lap_exact[interior_idx])/norm(lap_exact[interior_idx])
    ε=1e-5; g=zeros(nθ)
    for j in 1:nθ
        hp=copy(pts0); hp[hole_idx[j]]=hole0[j]+ε.*r̂[j]
        hm=copy(pts0); hm[hole_idx[j]]=hole0[j]-ε.*r̂[j]
        g[j]=(dot(ηvec, Lbuild(hp)*fvec) - dot(ηvec, Lbuild(hm)*fvec))/(2ε)
    end
    @printf("  %-22s | acc(int)=%.2e | ‖g‖=%.3e  low=%.3e  hi=%.3e  hi/low=%.2f\n",
            name, acc, totnorm(g), loband(g), hiband(g), hiband(g)/max(loband(g),1e-30))
    return (totnorm(g), hiband(g)/max(loband(g),1e-30))
end

println("=== LSQ (GMLS) vs exact collocation — q=2 ∂W/∂x artifact ===")
println("N=$N  nθ=$nθ.  M=$M monomials (poly_deg 3).\n")

basis = PHS(3; poly_deg=3)
adjl35 = find_neighbors(pts0, 35)
rphs = assess("PHS3+poly3 colloc k=35", p->phs_L(p, adjl35, basis))

results = Tuple{Int,Float64,Float64}[]
for n in (12, 25, 50)
    adjn = find_neighbors(pts0, n)
    r = assess("GMLS poly3 n=$n (osr=$(round(n/M,digits=1)))", p->gmls_L(p, adjn))
    push!(results, (n, r[1], r[2]))
end

println("\nGMLS oversampling trend (artifact ‖g‖ and hi/low vs n/M):")
for (n,g,hl) in results
    @printf("  n=%2d  n/M=%.1f : ‖g‖=%.3e  hi/low=%.2f\n", n, n/M, g, hl)
end
@printf("\nBaseline PHS3 colloc: ‖g‖=%.3e  hi/low=%.2f\n", rphs[1], rphs[2])
println("\nReading:")
println("  • If ‖g‖ and hi/low DROP as oversampling n/M grows ⇒ LSQ smooths ∂W/∂x")
println("    (raises the clean-DOF floor), validating the least-squares route.")
println("  • If flat/rising ⇒ over-determination doesn't help the geometric")
println("    sensitivity either, and the artifact is deeper than conditioning.")