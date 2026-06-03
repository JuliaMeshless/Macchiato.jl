# ============================================================================
# design_space_3d.jl — 3D surface design parameterizations.
#
# Same seam as the 2D `AbstractDesignSpace` (design_space.jl): a design space
# maps a few smooth design variables `l` to boundary-surface points, and
# contracts the nodal gradient ∂L/∂pts onto the design via (∂pts/∂l)ᵀ.  Every
# parametrization is, at the current surface, a LINEAR sensitivity matrix
# `B = ∂(surface pts)/∂l`; `boundary_points` applies `B` (about the reference),
# `contract_gradient` applies `Bᵀ`.  The optimizer is identical across them.
#
# Concrete 3D design spaces:
#   - SphericalHarmonicModes  (radial r(θ,φ)=Σ c_lm Y_lm on a star-shaped cavity)
#
# Biancolini RBF control points drop in as another subtype (same interface).
# ============================================================================

using LinearAlgebra: normalize, dot, norm

# ============================================================================
# Icosphere — fixed triangulated template for the cavity surface.
# ============================================================================
# Vertices double as the FIXED unit directions d_j (the SH is sampled at them);
# faces double as the FIXED triangle connectivity fed to `triangle_normals`.
# Only the per-vertex radius moves under the design, so connectivity is constant
# (the discrete-adjoint contract).

const _ICOSA_FACES = NTuple{3,Int}[
    (1,12,6),(1,6,2),(1,2,8),(1,8,11),(1,11,12),
    (2,6,10),(6,12,5),(12,11,3),(11,8,7),(8,2,9),
    (4,10,5),(4,5,3),(4,3,7),(4,7,9),(4,9,10),
    (5,10,6),(3,5,12),(7,3,11),(9,7,8),(10,9,2),
]

function _icosahedron()
    t = (1 + sqrt(5)) / 2
    raw = [(-1,t,0),(1,t,0),(-1,-t,0),(1,-t,0),
           (0,-1,t),(0,1,t),(0,-1,-t),(0,1,-t),
           (t,0,-1),(t,0,1),(-t,0,-1),(-t,0,1)]
    verts = [normalize(SVector{3,Float64}(Float64(v[1]),Float64(v[2]),Float64(v[3]))) for v in raw]
    return verts, copy(_ICOSA_FACES)
end

function _subdivide(verts::Vector{SVector{3,Float64}}, faces::Vector{NTuple{3,Int}})
    cache = Dict{Tuple{Int,Int},Int}()
    newverts = copy(verts)
    mid(a, b) = begin
        key = a < b ? (a, b) : (b, a)
        haskey(cache, key) && return cache[key]
        v = normalize((newverts[a] + newverts[b]) / 2)
        push!(newverts, v); cache[key] = length(newverts); return length(newverts)
    end
    newfaces = NTuple{3,Int}[]
    for (a, b, c) in faces
        ab = mid(a, b); bc = mid(b, c); ca = mid(c, a)
        push!(newfaces, (a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca))
    end
    return newverts, newfaces
end

"""
    icosphere(n_sub) -> (dirs::Vector{SVector{3,Float64}}, faces::Vector{NTuple{3,Int}})

Unit-sphere geodesic mesh: an icosahedron subdivided `n_sub` times, vertices
projected to the unit sphere.  `n_sub=0` ⇒ 12 verts / 20 faces; each level ×4
faces (≈ ×4 verts).  Faces are wound CCW as seen from OUTSIDE (outward normal),
verified/flipped by `_orient_outward!`.
"""
function icosphere(n_sub::Int)
    verts, faces = _icosahedron()
    for _ in 1:n_sub
        verts, faces = _subdivide(verts, faces)
    end
    _orient_outward!(verts, faces)
    return verts, faces
end

# Ensure each face's (b-a)×(c-a) points away from the origin (outward on the
# unit sphere); flip winding otherwise.
function _orient_outward!(verts, faces)
    @inbounds for i in eachindex(faces)
        a, b, c = faces[i]
        nrm = cross(verts[b] - verts[a], verts[c] - verts[a])
        if dot(nrm, verts[a] + verts[b] + verts[c]) < 0
            faces[i] = (a, c, b)
        end
    end
    return faces
end

# Per-vertex solid-angle quadrature weights on the unit sphere (each triangle's
# area split 1/3 to its vertices, rescaled so Σ w = 4π).  Used for the volume
# integral and SH orthonormality checks.
function _vertex_quad_weights(verts, faces)
    w = zeros(length(verts))
    for (a, b, c) in faces
        area = norm(cross(verts[b] - verts[a], verts[c] - verts[a])) / 2
        w[a] += area / 3; w[b] += area / 3; w[c] += area / 3
    end
    w .*= (4π / sum(w))
    return w
end

# ============================================================================
# Real spherical harmonics (orthonormal, no Condon–Shortley phase).
# ============================================================================

# Associated Legendre P_l^m(x), m ≥ 0, standard recurrence (no CS phase).
function _assoc_legendre(l::Int, m::Int, x::Float64)
    pmm = 1.0
    if m > 0
        somx2 = sqrt(max(0.0, 1 - x^2))
        fact = 1.0
        for _ in 1:m
            pmm *= fact * somx2     # ∏ (2i-1) · (1-x²)^{m/2}
            fact += 2.0
        end
    end
    l == m && return pmm
    pmmp1 = x * (2m + 1) * pmm
    l == m + 1 && return pmmp1
    pll = 0.0
    for ll in (m + 2):l
        pll = ((2ll - 1) * x * pmmp1 - (ll + m - 1) * pmm) / (ll - m)
        pmm = pmmp1; pmmp1 = pll
    end
    return pll
end

# √((2l+1)/4π · (l-|m|)!/(l+|m|)!) — ratio form, overflow-free for any l.
function _sph_norm(l::Int, m::Int)
    am = abs(m)
    ratio = 1.0
    for k in (l - am + 1):(l + am)
        ratio /= k
    end
    return sqrt((2l + 1) / (4π) * ratio)
end

"""
    real_sph_harm(l, m, θ, φ) -> Float64

Orthonormal real spherical harmonic `Y_l^m(θ, φ)` (`θ` = colatitude from +z,
`φ` = azimuth), normalized so `∮ Y_l^m Y_{l'}^{m'} dΩ = δ_{ll'} δ_{mm'}`.
"""
function real_sph_harm(l::Int, m::Int, θ::Real, φ::Real)
    am = abs(m)
    K  = _sph_norm(l, m)
    P  = _assoc_legendre(l, am, cos(θ))
    m > 0 && return sqrt(2) * K * cos(am * φ) * P
    m < 0 && return sqrt(2) * K * sin(am * φ) * P
    return K * P
end

# Y_l^m at a unit direction (θ, φ derived from the direction).
function _sph_harm_dir(l::Int, m::Int, d::SVector{3,Float64})
    θ = acos(clamp(d[3], -1.0, 1.0))
    φ = atan(d[2], d[1])
    return real_sph_harm(l, m, θ, φ)
end

# ============================================================================
# SphericalHarmonicModes — radial star-shaped cavity design space.
# ============================================================================

"""
    SphericalHarmonicModes <: AbstractDesignSpace

Radial parameterization of a star-shaped cavity surface:

    r(d_j) = Σ_k c_k · Y_{l_k}^{m_k}(d_j),   surface point  p_j = r(d_j) · d_j

on a FIXED icosphere template (`dirs` = unit vertex directions, `faces` =
triangle connectivity for `triangle_normals`).  The sphere is pure `Y_0^0`, so
this recovers a spherical cavity with a single active coefficient.

Fixed (shared, set once): `dirs`, `faces`, `quad_w`, `lm`, `Ymat`.
Design: `coeffs` (one per `(l,m)` in `lm`).

Fields:
- `lm::Vector{Tuple{Int,Int}}` — active `(l,m)` pairs, fixed order.
- `coeffs::Vector{Float64}` — SH coefficients (the design variables).
- `dirs::Vector{SVector{3,Float64}}` — icosphere vertex directions.
- `faces::Vector{NTuple{3,Int}}` — icosphere triangle connectivity.
- `quad_w::Vector{Float64}` — per-vertex solid-angle weights (Σ = 4π).
- `Ymat::Matrix{Float64}` — `Ymat[j,k] = Y_{lm[k]}(dirs[j])` (the radial B).
- `sob_l::Float64` — Sobolev length (dimensionless ρ/r_ref); `NaN` until calibrated.
- `SOB_P::Int` — Sobolev order.
"""
struct SphericalHarmonicModes <: AbstractDesignSpace
    lm::Vector{Tuple{Int,Int}}
    coeffs::Vector{Float64}
    dirs::Vector{SVector{3,Float64}}
    faces::Vector{NTuple{3,Int}}
    quad_w::Vector{Float64}
    Ymat::Matrix{Float64}
    sob_l::Float64
    SOB_P::Int
end

n_design_vars(ds::SphericalHarmonicModes) = length(ds.lm)

"""
    sph_lm_list(L; include_l0=true) -> Vector{Tuple{Int,Int}}

All `(l,m)` pairs with `0 ≤ l ≤ L`, `-l ≤ m ≤ l`, in `(l, m)` order.
"""
function sph_lm_list(L::Int; include_l0::Bool = true)
    lm = Tuple{Int,Int}[]
    for l in (include_l0 ? 0 : 1):L, m in -l:l
        push!(lm, (l, m))
    end
    return lm
end

"""
    sph_lm_list(degrees) -> Vector{Tuple{Int,Int}}

`(l,m)` pairs for an explicit set of degrees (e.g. `[0, 2]` to keep only the
base radius and the quadrupole — the ellipsoid↔sphere subspace, excluding the
center-shifting `l=1` modes and high modes above the stencil-Nyquist budget).
"""
function sph_lm_list(degrees::AbstractVector{<:Integer})
    lm = Tuple{Int,Int}[]
    for l in degrees, m in -l:l
        push!(lm, (l, m))
    end
    return lm
end

"""
    SphericalHarmonicModes(L, n_sub; sob_l=NaN, SOB_P=2) -> SphericalHarmonicModes

Build a degree-`L` SH design space on an `icosphere(n_sub)` template, with zero
coefficients (degenerate radius 0 — set a base radius via `coeffs[1]` or
`fit_ellipsoid_sh`).  Precomputes `Ymat`.
"""
function SphericalHarmonicModes(L::Int, n_sub::Int; sob_l::Float64 = NaN, SOB_P::Int = 2)
    return SphericalHarmonicModes(sph_lm_list(L), n_sub; sob_l = sob_l, SOB_P = SOB_P)
end

"""
    SphericalHarmonicModes(degrees::AbstractVector{<:Integer}, n_sub; ...) -> SphericalHarmonicModes

Build a design space on an explicit degree set (e.g. `[0, 2]`).
"""
function SphericalHarmonicModes(degrees::AbstractVector{<:Integer}, n_sub::Int;
                               sob_l::Float64 = NaN, SOB_P::Int = 2)
    return SphericalHarmonicModes(sph_lm_list(degrees), n_sub; sob_l = sob_l, SOB_P = SOB_P)
end

# Core constructor from an explicit (l,m) list.
function SphericalHarmonicModes(lm::Vector{Tuple{Int,Int}}, n_sub::Int;
                               sob_l::Float64 = NaN, SOB_P::Int = 2)
    dirs, faces = icosphere(n_sub)
    quad_w = _vertex_quad_weights(dirs, faces)
    Ymat = [_sph_harm_dir(l, m, dirs[j]) for j in eachindex(dirs), (l, m) in lm]
    coeffs = zeros(length(lm))
    return SphericalHarmonicModes(lm, coeffs, dirs, faces, quad_w, Ymat, sob_l, SOB_P)
end

# Functional update: same template, new coefficients.
with_coeffs(ds::SphericalHarmonicModes, c::AbstractVector{Float64}) =
    SphericalHarmonicModes(ds.lm, collect(c), ds.dirs, ds.faces, ds.quad_w,
                           ds.Ymat, ds.sob_l, ds.SOB_P)

set_sobolev(ds::SphericalHarmonicModes, sob_l::Float64, SOB_P::Int) =
    SphericalHarmonicModes(ds.lm, ds.coeffs, ds.dirs, ds.faces, ds.quad_w,
                           ds.Ymat, sob_l, SOB_P)

# ---------- geometry ----------------------------------------------------------

"""
    radii(ds::SphericalHarmonicModes) -> Vector{Float64}

Per-vertex radius `r(d_j) = Σ_k c_k Y_k(d_j) = Ymat · coeffs`.
"""
radii(ds::SphericalHarmonicModes) = ds.Ymat * ds.coeffs

"""
    boundary_points(ds::SphericalHarmonicModes) -> Vector{SVector{3,Float64}}

Cavity surface points `p_j = r(d_j) · d_j` (centered at the origin).
"""
function boundary_points(ds::SphericalHarmonicModes)
    r = radii(ds)
    return [r[j] * ds.dirs[j] for j in eachindex(ds.dirs)]
end

surface_faces(ds::SphericalHarmonicModes) = ds.faces
directions(ds::SphericalHarmonicModes) = ds.dirs

# ---------- volume ------------------------------------------------------------

"""
    cavity_volume(ds::SphericalHarmonicModes) -> Float64

Star-shaped volume `V = (1/3) ∮ r³ dΩ ≈ (1/3) Σ_j r_j³ w_j`.
"""
function cavity_volume(ds::SphericalHarmonicModes)
    r = radii(ds)
    return sum(ds.quad_w[j] * r[j]^3 for j in eachindex(r)) / 3
end

"""
    volume_gradient(ds::SphericalHarmonicModes) -> Vector{Float64}

`∂V/∂c_k = ∮ r² Y_k dΩ ≈ Σ_j r_j² w_j Ymat[j,k]`.
"""
function volume_gradient(ds::SphericalHarmonicModes)
    r = radii(ds)
    rw = [ds.quad_w[j] * r[j]^2 for j in eachindex(r)]
    return ds.Ymat' * rw
end

# ---------- gradient contraction ---------------------------------------------

"""
    contract_gradient(ds::SphericalHarmonicModes, g_surf) -> Vector{Float64}

Contract the (morph-transposed) nodal gradient at the cavity surface onto the
SH coefficients: `g_c = Bᵀ g_surf` where `B[3j-.., k] = Y_k(d_j) · d_j`.  Since
the radius scales along the fixed direction `d_j`, this is the radial projection

    g_c_k = Σ_j (g_surf_j · d_j) · Y_k(d_j) = Ymatᵀ · g_rad.

`g_surf` is one `SVector{3}` per cavity vertex (already morph-transposed).
Returns the raw coefficient gradient; the caller applies the volume projection
(`project_volume`) and Sobolev preconditioning.
"""
function contract_gradient(
    ds::SphericalHarmonicModes,
    g_surf::AbstractVector{<:AbstractVector{Float64}},
)
    g_rad = [dot(g_surf[j], ds.dirs[j]) for j in eachindex(ds.dirs)]
    return ds.Ymat' * g_rad
end

"""
    project_volume(ds::SphericalHarmonicModes, g_c) -> Vector{Float64}

Project a coefficient gradient onto the fixed-volume manifold: remove the
component along `n = ∂V/∂c`, `g ← g − (g·n / ‖n‖²) n`.
"""
function project_volume(ds::SphericalHarmonicModes, g_c::AbstractVector{Float64})
    n = volume_gradient(ds)
    n2 = dot(n, n)
    n2 < 1e-30 && return collect(g_c)
    return g_c .- (dot(g_c, n) / n2) .* n
end

"""
    sph_sob_weight(l, sob_l, SOB_P) -> Float64

Sobolev preconditioner weight for SH degree `l`: `(1 + (sob_l·l)²)ᴾ`.
"""
sph_sob_weight(l::Int, sob_l::Real, SOB_P::Int) = (1.0 + (sob_l * l)^2)^SOB_P

# ---------- initialization ----------------------------------------------------

"""
    fit_ellipsoid_sh(ds, ax, ay, az) -> SphericalHarmonicModes

Fit a centered ellipsoid (semi-axes `ax, ay, az`) onto the SH basis by
quadrature projection of its radial function
`r(d) = 1/√((d_x/ax)² + (d_y/ay)² + (d_z/az)²)`.  Returns a copy of `ds` with
the fitted coefficients.  (`Y` is orthonormal ⇒ `c_k = ∮ r Y_k dΩ ≈ Σ r_j Y_k(d_j) w_j`.)
"""
function fit_ellipsoid_sh(ds::SphericalHarmonicModes, ax::Real, ay::Real, az::Real)
    r = [1 / sqrt((d[1]/ax)^2 + (d[2]/ay)^2 + (d[3]/az)^2) for d in ds.dirs]
    rw = r .* ds.quad_w
    c = ds.Ymat' * rw
    return with_coeffs(ds, c)
end

"""
    calibrate_sph(ds, ρ, r_ref) -> SphericalHarmonicModes

Set the Sobolev length from cloud geometry (`sob_l = ρ / r_ref`, the stencil
radius over the cavity radius), mirroring `calibrate_fourier`.  The SH degree
cap is the caller's choice (`L`); the stencil-Nyquist budget is
`L ≤ ⌊π r_ref / ρ⌋`.
"""
calibrate_sph(ds::SphericalHarmonicModes, ρ::Real, r_ref::Real) =
    set_sobolev(ds, ρ / r_ref, ds.SOB_P)
