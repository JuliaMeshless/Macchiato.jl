# ============================================================================
# test_sph_design_space.jl — self-contained validation of the SH 3D design space
# (no PDE).  Checks the reusable core before it is wired into the optimizer:
#   1. Icosphere template sanity (counts, unit verts, outward faces).
#   2. Real-SH orthonormality via the icosphere quadrature (Yᵀ diag(w) Y ≈ I).
#   3. contract_gradient == Bᵀ : FD of the linear functional ⟨g, p(coeffs)⟩.
#   4. volume_gradient == ∂V/∂c : central FD of cavity_volume.
#   5. fit_ellipsoid_sh reproduces the ellipsoid radial field; pure Y₀₀ ⇒ sphere.
#
# Run:  jlrun cavity_3d/test_sph_design_space.jl   (from examples/)
# ============================================================================
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Macchiato
using StaticArrays, LinearAlgebra, Printf

const L     = 4
const NSUB  = 3
ds0 = SphericalHarmonicModes(L, NSUB)
nv  = length(ds0.dirs)
@printf("icosphere(%d): %d vertices, %d faces · SH L=%d ⇒ %d coeffs\n",
        NSUB, nv, length(ds0.faces), L, n_design_vars(ds0))

# ---- 1. template sanity -----------------------------------------------------
unit_ok = all(abs(norm(d) - 1) < 1e-12 for d in ds0.dirs)
out_ok = all(let (a,b,c)=f; dot(cross(ds0.dirs[b]-ds0.dirs[a], ds0.dirs[c]-ds0.dirs[a]),
                                ds0.dirs[a]+ds0.dirs[b]+ds0.dirs[c]) > 0 end
             for f in ds0.faces)
@printf("1. template: unit verts %s · outward faces %s · Σw=%.6f (4π=%.6f)\n",
        unit_ok ? "✓" : "✗", out_ok ? "✓" : "✗", sum(ds0.quad_w), 4π)

# ---- 2. orthonormality ------------------------------------------------------
G = ds0.Ymat' * (ds0.quad_w .* ds0.Ymat)        # ≈ ∮ Y_k Y_k' dΩ = δ
ortho_err = maximum(abs.(G - I))
@printf("2. SH orthonormality: max|YᵀWY - I| = %.3e   %s\n",
        ortho_err, ortho_err < 5e-3 ? "✓" : "✗ (raise NSUB for finer quadrature)")

# ---- 3. contract_gradient == Bᵀ --------------------------------------------
# Random coeffs (nonzero base radius) + random surface gradient g.
import Random; Random.seed!(1)
c_rand = 0.05 .* randn(n_design_vars(ds0)); c_rand[1] += 0.30 / ds0.Ymat[1,1]  # base radius ~0.3
ds = with_coeffs(ds0, c_rand)
g_surf = [SVector{3,Float64}(randn(), randn(), randn()) for _ in 1:nv]

g_c_ana = contract_gradient(ds, g_surf)
# Linear functional Φ(c) = Σ_j g_j · p_j(c);  dΦ/dc_k must equal g_c_ana[k].
Φ(c) = (p = boundary_points(with_coeffs(ds, c)); sum(dot(g_surf[j], p[j]) for j in 1:nv))
h = 1e-6
g_c_fd = similar(g_c_ana)
for k in eachindex(g_c_ana)
    δ = zeros(length(c_rand)); δ[k] = h
    g_c_fd[k] = (Φ(c_rand + δ) - Φ(c_rand - δ)) / (2h)
end
contract_err = maximum(abs.(g_c_ana - g_c_fd)) / max(1, maximum(abs.(g_c_fd)))
@printf("3. contract == Bᵀ: max rel err vs FD = %.3e   %s\n",
        contract_err, contract_err < 1e-7 ? "✓" : "✗")

# ---- 4. volume_gradient -----------------------------------------------------
gV_ana = volume_gradient(ds)
gV_fd = similar(gV_ana)
for k in eachindex(gV_ana)
    δ = zeros(length(c_rand)); δ[k] = h
    gV_fd[k] = (cavity_volume(with_coeffs(ds, c_rand + δ)) -
                cavity_volume(with_coeffs(ds, c_rand - δ))) / (2h)
end
vol_err = maximum(abs.(gV_ana - gV_fd)) / max(1, maximum(abs.(gV_fd)))
@printf("4. volume_gradient: max rel err vs FD = %.3e   %s\n",
        vol_err, vol_err < 1e-6 ? "✓" : "✗")

# ---- 5. ellipsoid fit + sphere ----------------------------------------------
ax, ay, az = 0.4, 0.2, 0.3
ds_e = fit_ellipsoid_sh(ds0, ax, ay, az)
r_fit = radii(ds_e)
r_true = [1/sqrt((d[1]/ax)^2 + (d[2]/ay)^2 + (d[3]/az)^2) for d in ds0.dirs]
fit_err = norm(r_fit - r_true) / norm(r_true)
@printf("5. ellipsoid fit (%.1f,%.1f,%.1f): rel radial err = %.3e   %s\n",
        ax, ay, az, fit_err, fit_err < 0.05 ? "✓ (low-L truncation expected)" : "✗")
# Pure Y₀₀ ⇒ constant radius (a sphere).
c_sphere = zeros(n_design_vars(ds0)); c_sphere[1] = 0.3 / ds0.Ymat[1,1]
r_sphere = radii(with_coeffs(ds0, c_sphere))
@printf("   pure Y₀₀: radius spread = %.3e (should be ~0, sphere)   %s\n",
        maximum(r_sphere) - minimum(r_sphere),
        (maximum(r_sphere) - minimum(r_sphere)) < 1e-12 ? "✓" : "✗")

println()
all_ok = unit_ok && out_ok && ortho_err < 5e-3 && contract_err < 1e-7 &&
         vol_err < 1e-6 && fit_err < 0.05
println(all_ok ? "ALL SH DESIGN-SPACE CORE TESTS PASS" : "TESTS FAILED")
