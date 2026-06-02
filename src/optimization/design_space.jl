using Statistics: mean

# ============================================================================
# design_space.jl — Pluggable design parameterizations for shape optimization.
#
# A design space maps a small number of smooth design variables `l` to boundary
# geometry points. The discrete adjoint delivers the nodal gradient ∂L/∂pts;
# the design space contracts it onto the design via (∂pts/∂l)ᵀ · ∂L/∂pts.
# This contraction is what removes mesh-scale noise — the design variables
# carry only the smooth modes that the stencil-resolved DOF budget can trust.
#
# Concrete design spaces shipped here:
#   - FourierModes  (2D, radius Fourier series on a closed loop)
#
# 3D additions (surface harmonics, Biancolini RBF control points) drop in as
# new subtypes of AbstractDesignSpace — see design_space_3d.jl (Phase 2).
# ============================================================================

"""
    AbstractDesignSpace

Abstract type for smooth design parameterizations.

Interface (documented; dispatch is duck-typed):
- `boundary_points(ds, param_coords) -> Vector{SVector}` — generate boundary
  point positions from the current design parameters.
- `n_design_vars(ds) -> Int` — total number of independent design variables.
- `contract_gradient(ds, g_boundary, boundary_pts, param_coords)` — contract
  the nodal gradient at boundary points onto the design variables, returning
  the design gradient (format is concrete-type-specific).
"""
abstract type AbstractDesignSpace end

# ============================================================================
# FourierModes — 2D radius Fourier series on a closed boundary loop.
# ============================================================================

"""
    FourierModes <: AbstractDesignSpace

Fourier radius parameterization of a closed hole boundary:

    r(θ) = r₀ + Σₘ aₘ·cos(m·θ) + Σₘ bₘ·sin(m·θ)

with the area held fixed (r₀ is determined by the oscillatory coefficients).
Calibrated from the cloud geometry: the mode cap is set by the
stencil-Nyquist criterion ``m ≤ ⌊π r / ρ⌋``, and the Sobolev length
``ℓ_sob = ρ / r`` sets the residual smoother scale.

Fields:
- `r0::Float64` — area-constrained base radius.
- `a::Vector{Float64}` — cos coefficients (one per mode).
- `b::Vector{Float64}` — sin coefficients (one per mode).
- `modes::Vector{Int}` — which Fourier mode numbers are active (e.g. [2, 3, 4]).
- `sob_r::Float64` — Sobolev length scale (dimensionless, ρ/r_ref).
- `SOB_P::Int` — Sobolev order (default 2).
"""
struct FourierModes <: AbstractDesignSpace
    r0::Float64
    a::Vector{Float64}
    b::Vector{Float64}
    modes::Vector{Int}
    sob_r::Float64
    SOB_P::Int
end

n_design_vars(ds::FourierModes) = 1 + 2 * length(ds.modes)  # r0 + a + b

# ---------- geometry ----------------------------------------------------------

"""
    radius_at(ds::FourierModes, θ::Real) -> Float64

Scalar hole radius at polar angle `θ` according to the Fourier series.
"""
function radius_at(ds::FourierModes, θ::Real)
    return ds.r0 + sum(ds.a[i] * cos(m * θ) + ds.b[i] * sin(m * θ)
                       for (i, m) in enumerate(ds.modes); init = 0.0)
end

"""
    boundary_points(ds::FourierModes, θvals) -> Vector{SVector{2,Float64}}

Generate hole boundary points at each angle in `θvals`.
"""
function boundary_points(ds::FourierModes, θvals::AbstractVector{<:Real})
    return [let r = radius_at(ds, θ); SVector{2,Float64}(r * cos(θ), r * sin(θ)) end
            for θ in θvals]
end

# ---------- area constraint ---------------------------------------------------

"""
    r0_for_area(A_target::Real, a::AbstractVector, b::AbstractVector) -> Float64

Area-constrained base radius. The area of
``r(θ) = r₀ + Σ aₘ cos(mθ) + Σ bₘ sin(mθ)`` over [0, 2π) is

    A = π r₀² + (π/2) Σ (aₘ² + bₘ²)

so ``r₀ = √(max(0, (A_target - (π/2)(‖a‖² + ‖b‖²)) / π))``.
"""
function r0_for_area(A_target::Real, a::AbstractVector, b::AbstractVector)
    return sqrt(max(0.0, (A_target - (π / 2) * (sum(abs2, a) + sum(abs2, b))) / π))
end

# ---------- Sobolev preconditioner --------------------------------------------

"""
    sob_weight(m::Int, sob_r::Real, SOB_P::Int) -> Float64

Sobolev preconditioner weight for Fourier mode `m`:
``w_m = (1 + (sob_r · m)²)ᴾ``.
"""
sob_weight(m::Int, sob_r::Real, SOB_P::Int) = (1.0 + (sob_r * m)^2)^SOB_P

# ---------- gradient contraction ----------------------------------------------

"""
    contract_gradient(ds::FourierModes, g_hole, hole, θvals) -> (dC_da, dC_db)

Contract the morph-transposed nodal gradient at the hole boundary onto the
Fourier design modes, with area-constraint projection.

Arguments:
- `ds::FourierModes` — current design.
- `g_hole` — boundary gradient vector (one `SVector{2}` per hole node),
  *already morph-transposed* (caller runs `morph_transpose` first).
- `hole` — current hole boundary point positions (length nθ).
- `θvals` — polar angles of the hole nodes.

Returns `(dC_da, dC_db)` — vectors of length `length(ds.modes)`, the
area-constrained design gradient. The caller applies Sobolev preconditioning
externally (division by `sob_weight(m, sob_r, SOB_P)`) so the descent step
is scale-aware.
"""
function contract_gradient(
    ds::FourierModes,
    g_hole::AbstractVector{<:AbstractVector{Float64}},
    hole::AbstractVector{<:AbstractVector{Float64}},
    θvals::AbstractVector{<:Real},
)
    nθ = length(θvals)
    # Radial component of the boundary gradient
    g_rad = [(g_hole[j][1] * hole[j][1] + g_hole[j][2] * hole[j][2]) /
             hypot(hole[j]...) for j in 1:nθ]

    # Fourier projection
    dC_dr0 = sum(g_rad)
    dC_da = [sum(g_rad[j] * cos(m * θvals[j]) for j in 1:nθ) for m in ds.modes]
    dC_db = [sum(g_rad[j] * sin(m * θvals[j]) for j in 1:nθ) for m in ds.modes]

    # Project out the area direction: r₀ = r₀(a,b) ⇒ ∂r₀/∂aₖ = -aₖ/(2r₀)
    f = dC_dr0 / (2 * ds.r0)
    return dC_da .- f .* ds.a, dC_db .- f .* ds.b
end

# ---------- calibration -------------------------------------------------------

"""
    calibrate_fourier(pts, adjl, r_ref; SOB_P::Int = 2)
        -> (FourierModes, ρ)

Auto-calibrate a `FourierModes` design space from the cloud geometry.

Arguments:
- `pts` — reference point cloud (e.g. around the unit circle).
- `adjl` — adjacency list for the cloud.
- `r_ref` — reference radius of the hole (area-equivalent circle radius).
- `SOB_P` — Sobolev order (default 2).

Returns `(ds, ρ)` where `ds` is a `FourierModes` with zero oscillatory
coefficients (unit circle) and calibrated mode cap + Sobolev length,
and `ρ` is the stencil radius for diagnostics.

Calibration formulas:
- Stencil radius ``ρ = max_{i,j∈adjl[i]} |p_i - p_j|``
- Mode cap ``m_cap = max(2, ⌊π r_ref / ρ⌋)``  (stencil-Nyquist)
- Sobolev length ``ℓ_sob = ρ / r_ref``
"""
function calibrate_fourier(
    pts::AbstractVector{<:AbstractVector{Float64}},
    adjl::AbstractVector,
    r_ref::Real;
    SOB_P::Int = 2,
)
    ρ = maximum(hypot(pts[i][1] - pts[j][1], pts[i][2] - pts[j][2])
                for i in 1:length(pts) for j in adjl[i])
    m_cap = max(2, floor(Int, π * r_ref / ρ))
    modes = collect(2:m_cap)
    sob_r = ρ / r_ref
    r0 = r_ref  # circle bootstrap: zero oscillatory coefs ⇒ r₀²π = A_target
    ds = FourierModes(r0, Float64[], Float64[], modes, sob_r, SOB_P)
    return ds, ρ
end

# ---------- initialization ----------------------------------------------------

"""
    fit_start_fourier(a0, b0, θvals, nθ, modes, A_target) -> FourierModes

Fit the starting ellipse (semi-axes `a0`, `b0`) onto the Fourier mode basis.
Returns a `FourierModes` with `a, b` coefficients initialized from the ellipse.

Uses the discrete Fourier transform over `nθ` uniformly-spaced samples.
"""
function fit_start_fourier(
    a0::Real,
    b0::Real,
    θvals::AbstractVector{<:Real},
    nθ::Int,
    modes::AbstractVector{<:Integer},
    A_target::Real,
)
    rvals = [hypot(a0 * cos(t), b0 * sin(t)) for t in θvals]
    r0fit = mean(rvals)
    a = [(2 / nθ) * sum((rvals[j] - r0fit) * cos(m * θvals[j]) for j in 1:nθ) for m in modes]
    b = [(2 / nθ) * sum((rvals[j] - r0fit) * sin(m * θvals[j]) for j in 1:nθ) for m in modes]
    r0 = r0_for_area(A_target, a, b)
    return FourierModes(r0, a, b, collect(Int, modes), NaN, 2)  # sob_r, SOB_P set by caller after calibrate
end
