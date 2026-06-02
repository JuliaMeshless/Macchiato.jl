# ============================================================================
# test_shape_gradient_3d_traction.jl — FD-validate the 3D discrete adjoint with
# mixed Dirichlet + traction BCs (frozen normals; the Phase-B analogue).
#
# Uniaxial tension of a 3D block:
#   - face x=-Lx/2 : clamped (Dirichlet u=v=w=0)
#   - face x=+Lx/2 : loaded  (traction t = σ∞·(1,0,0))
#   - other 4 faces: traction-free (t = 0)
# Every boundary node carries a real BC ⇒ well-posed.  Cloud from WhatsThePoint
# (irregular volume fill; poly_deg=3 needs an irregular cloud — see
# test_shape_gradient_3d.jl).  Frozen normals: the traction layout is built ONCE
# from the reference cloud and held fixed across FD perturbations; only the RBF
# weight matrices recompute, so the adjoint must capture ∂W/∂pts in the Neumann
# rows (extract_neumann_sensitivities_3d!) plus the interior extraction.
#
# Run:  jlrun plate_with_hole/test_shape_gradient_3d_traction.jl   (from examples/)
# ============================================================================
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Macchiato
using WhatsThePoint
import WhatsThePoint as WTP
import RadialBasisFunctions: find_neighbors
using RadialBasisFunctions: PHS
using StaticArrays, LinearAlgebra, Printf
using Unitful: m, ustrip

const Lx, Ly, Lz = 0.6, 0.4, 0.3
const Δ  = 0.1
const k  = 50
const σ∞ = 1.0e4                    # traction magnitude (E=1e7 ⇒ u ~ 1e-3)

model = LinearElasticity3D(E = 1.0e7, ν = 0.3)
μ, λ  = lame_parameters_3d(model)
basis = PHS(3; poly_deg = 3)

# ---- WTP cloud: box surface → irregular volume fill -------------------------
function box_surface(hx, hy, hz, Δ)
    pts = WTP.Point[]; nrm = SVector{3,Float64}[]
    seen = Set{NTuple{3,Int}}()
    key(x, y, z) = (round(Int, x/Δ*1e3), round(Int, y/Δ*1e3), round(Int, z/Δ*1e3))
    add(x, y, z, n) = (k0 = key(x,y,z); k0 in seen || (push!(seen, k0);
                       push!(pts, WTP.Point(x*m, y*m, z*m)); push!(nrm, n)))
    rng(h) = range(-h, h; length = max(2, round(Int, 2h/Δ) + 1))
    for x in rng(hx), y in rng(hy)
        add(x, y, +hz, SVector(0.0,0.0,1.0)); add(x, y, -hz, SVector(0.0,0.0,-1.0))
    end
    for x in rng(hx), z in rng(hz)
        add(x, +hy, z, SVector(0.0,1.0,0.0)); add(x, -hy, z, SVector(0.0,-1.0,0.0))
    end
    for y in rng(hy), z in rng(hz)
        add(+hx, y, z, SVector(1.0,0.0,0.0)); add(-hx, y, z, SVector(-1.0,0.0,0.0))
    end
    return PointBoundary(identity.(pts), nrm, fill(Δ^2 * m^2, length(pts))), length(pts)
end

bnd_surf, _ = box_surface(Lx/2, Ly/2, Lz/2, Δ)
cloud = WTP.discretize(bnd_surf, ConstantSpacing(Δ * m); alg = SlakKosec())
svec(p) = (c = WTP.coords(p); SVector{3,Float64}(ustrip(c.x), ustrip(c.y), ustrip(c.z)))
vol_pts = svec.(WTP.points(cloud.volume))
bnd_pts = svec.(WTP.points(cloud.boundary))
pts = vcat(vol_pts, bnd_pts)
N = length(pts)
n_int = length(vol_pts)
interior_idx = collect(1:n_int)
boundary_idx = collect((n_int+1):N)
adjl = find_neighbors(pts, k)
@printf("WTP cloud: %d volume + %d boundary = %d total\n", n_int, length(bnd_pts), N)

# ---- classify boundary nodes by dominant face ------------------------------
half = (Lx/2, Ly/2, Lz/2)
function face_of(p)                       # → (axis ∈ 1:3, sign ∈ ±1)
    r = ntuple(d -> abs(p[d]) / half[d], 3)
    ax = argmax(r)
    return ax, sign(p[ax])
end

dirichlet_dofs = Int[]; dirichlet_vals = Float64[]
neumann_ids = Int[]; normals = SVector{3,Float64}[]; tractions = SVector{3,Float64}[]
for i in boundary_idx
    ax, sg = face_of(pts[i])
    if ax == 1 && sg < 0                  # clamped face x=-Lx/2
        for c in 0:2; push!(dirichlet_dofs, i + c*N); push!(dirichlet_vals, 0.0); end
    else                                  # Neumann face
        nrm = SVector{3,Float64}(ntuple(d -> d == ax ? sg : 0.0, 3))
        push!(neumann_ids, i); push!(normals, nrm)
        push!(tractions, (ax == 1 && sg > 0) ? σ∞ .* nrm : SVector(0.0,0.0,0.0))
    end
end
neumann_adjl = adjl[neumann_ids]

active = let a = trues(3N); for d in dirichlet_dofs; a[d] = false; end; a end
interior_rows = let r = falses(N); for i in interior_idx; r[i] = true; end; r end
@printf("BCs: %d Dirichlet dofs (clamped face) · %d Neumann nodes (1 loaded face + 4 free)\n",
        length(dirichlet_dofs), length(neumann_ids))

# Frozen normals: build the traction layout ONCE from the reference cloud.
layout = build_traction_layout_3d(neumann_ids, neumann_adjl, normals, tractions, λ, μ, N)

# ---- forward + gradient helpers (loss L = uᵀu/2) ----------------------------
flat3(P) = reduce(vcat, [[p[1], p[2], p[3]] for p in P])
solve3(pf, dLdu) = shape_gradient_3d(pf, model, N, adjl, basis, active,
                                     dirichlet_dofs, dirichlet_vals, dLdu;
                                     interior_rows = interior_rows,
                                     traction_layout = layout,
                                     neumann_ids = neumann_ids, neumann_adjl = neumann_adjl)
compliance(pf) = (u = solve3(pf, identity).u; dot(u, u) / 2)
compliance_grad(pf) = solve3(pf, u -> u).Δpts

# ---- FD validation ----------------------------------------------------------
g_ad = compliance_grad(flat3(pts))
@printf("‖u‖ = %.4e   (sanity: nonzero, finite)\n", norm(solve3(flat3(pts), identity).u))
fd_at(i, c, hh) = (δ = zeros(3N); δ[3i - 3 + c] = hh;
                   (compliance(flat3(pts) + δ) - compliance(flat3(pts) - δ)) / (2hh))
probe = vcat(1:5, (n_int÷2):(n_int÷2+2), neumann_ids[1:3], boundary_idx[end-2:end])
ratios = Float64[]
# Two step sizes: if a mismatch at small |FD| shrinks toward AD as hh grows,
# it's FD noise in a flat direction (ill-conditioned Neumann system), not a bug.
println("\nFD validation — shape_gradient_3d (mixed BC, frozen normals) vs central FD:")
@printf("  %4s %3s  %13s %13s %13s %10s\n", "pt", "c", "AD", "FD(1e-6)", "FD(3e-5)", "AD/FD(3e-5)")
for i in probe, c in 1:3
    fd1 = fd_at(i, c, 1.0e-6)
    fd2 = fd_at(i, c, 3.0e-5)
    ad = g_ad[3i - 3 + c]
    abs(fd2) > 1e-8 && push!(ratios, abs(ad) / abs(fd2))
    @printf("  %4d  %1d   %+13.6e %+13.6e %+13.6e %10.5f\n", i, c, ad, fd1, fd2, abs(ad)/max(abs(fd2),1e-30))
end
med = sort(ratios)[max(1, length(ratios) ÷ 2)]
n_good = count(r -> abs(r - 1.0) < 0.05, ratios)
println("\n--- Summary (ratio vs FD(3e-5)) ---")
@printf("  median |AD|/|FD| = %.6f   (over %d entries, |FD|>1e-8)\n", med, length(ratios))
@printf("  within 5%% of 1.0: %d / %d\n", n_good, length(ratios))
println(abs(med - 1.0) < 0.02 ?
    "  ✓ shape_gradient_3d VALIDATED (mixed Dirichlet + traction, frozen normals)" :
    "  ✗ DISCREPANCY — inspect")
