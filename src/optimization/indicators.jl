using Statistics: mean, std

# ============================================================================
# indicators.jl — Closed-loop cloud-quality indicators for shape optimization.
#
# Each indicator is a cheap scalar measured every iteration. When an enabled
# indicator trips (crosses its threshold), the two-front loop triggers a remesh
# (re-anchored re-initialization of the cloud). Indicators are independently
# toggleable — watch the log and prune to the minimal set.
#
# The six measures shipped here are generic over any problem whose cloud state
# exposes the expected fields (interior_idx, hole_idx, adjl, ref_radius,
# ref_interior, n_int, N). Add problem-specific measures by following the same
# `(pts, st) -> Float64` signature.
# ============================================================================

"""
    Indicator

Cloud-quality indicator for closed-loop remesh control.
Each indicator measures a scalar from `(pts, cloud_state)` every iteration;
when the value exceeds (or falls below) `threshold`, a remesh is triggered.

Fields:
- `name::Symbol` — short identifier for logging
- `measure::Function` — `(pts, cloud_state) -> Float64`
- `threshold::Float64` — trip threshold in problem units
- `trip_when::Symbol` — `:above` or `:below`
- `enabled::Bool` — toggle false to suppress this indicator
"""
Base.@kwdef struct Indicator
    name::Symbol
    measure::Function          # (pts, state) -> Float64
    threshold::Float64
    trip_when::Symbol          # :above or :below
    enabled::Bool = true
end

"""
    trips(ind::Indicator, v::Real) -> Bool

Return `true` if value `v` trips indicator `ind`.
"""
trips(ind::Indicator, v::Real) = ind.trip_when === :above ? v > ind.threshold : v < ind.threshold

# ---------- helper: nearest-neighbour distance for point i --------------------
_nn_dist(pts, adjl, i) = minimum(hypot(pts[i][1] - pts[j][1], pts[i][2] - pts[j][2])
                                  for j in adjl[i] if j != i)

# ---------- quality measures --------------------------------------------------

"""
    measure_morph_drift(pts, st) -> Float64

Maximum interior-node displacement from the anchored reference, in units of `dx`
(= background lattice spacing). High drift means the frozen stencils are operating
far from where they were built. Conservative — fires before the operator degrades
on gentle Laplace morphs.
"""
function measure_morph_drift(pts, st)
    maximum(hypot(pts[st.interior_idx[k]][1] - st.ref_interior[k][1],
                  pts[st.interior_idx[k]][2] - st.ref_interior[k][2])
            for k in 1:st.n_int) / st.dx
end

"""
    measure_min_gap(pts, st) -> Float64

Minimum distance between any interior point and any hole boundary point, in units
of `dx`. A collision guard — low values mean the interior is about to cross the
hole boundary.
"""
function measure_min_gap(pts, st)
    minimum(hypot(pts[i][1] - pts[j][1], pts[i][2] - pts[j][2])
            for i in st.interior_idx, j in st.hole_idx) / st.dx
end

"""
    measure_spacing_cv(pts, st) -> Float64

Coefficient of variation (std/mean) of nearest-neighbour distances over interior
nodes. Captures interior-cloud non-uniformity — a local clustering or spreading
that would ill-condition the RBF stencils.
"""
function measure_spacing_cv(pts, st)
    d = [_nn_dist(pts, st.adjl, i) for i in st.interior_idx]
    return std(d) / mean(d)
end

"""
    measure_boundary_cv(pts, st) -> Float64

Coefficient of variation of consecutive hole-node spacing along the boundary
loop. Captures boundary-node bunching, which concentrates resolution and can
worsen stencil conditioning locally.
"""
function measure_boundary_cv(pts, st)
    nθ = length(st.hole_idx)
    s = [hypot(pts[st.hole_idx[mod1(j + 1, nθ)]][1] - pts[st.hole_idx[j]][1],
               pts[st.hole_idx[mod1(j + 1, nθ)]][2] - pts[st.hole_idx[j]][2]) for j in 1:nθ]
    return std(s) / mean(s)
end

"""
    measure_min_sep(pts, st) -> Float64

Global minimum nearest-neighbour separation (any node), in units of `dx`.
Near-collisions ill-condition the local RBF saddle systems — the most direct
conditioning proxy.
"""
function measure_min_sep(pts, st)
    minimum(_nn_dist(pts, st.adjl, i) for i in 1:st.N) / st.dx
end

"""
    measure_stencil_growth(pts, st) -> Float64

Worst-case ratio of current frozen-stencil radius to its radius at anchor.
As the morph drifts nodes, frozen neighbours separate (radius grows) and the
fixed adjacency goes geometrically stale — the operator's actual support no
longer matches what the weights assume. More directly tied to operator validity
than absolute drift (a uniform translation grows no stencil).
"""
function measure_stencil_growth(pts, st)
    maximum(maximum(hypot(pts[i][1] - pts[j][1], pts[i][2] - pts[j][2])
                    for j in st.adjl[i] if j != i) / st.ref_radius[i]
            for i in 1:st.N)
end

# ---------- batch assessment --------------------------------------------------

"""
    assess(indicators::AbstractVector{Indicator}, pts, st)
        -> (vals::Vector{Pair{Symbol,Float64}}, tripped::Vector{Symbol})

Evaluate every enabled indicator on `(pts, st)`. Returns named values (for
logging) and the list of tripped indicator names (for triggering a remesh).
"""
function assess(indicators::AbstractVector{Indicator}, pts, st)
    vals = Pair{Symbol,Float64}[]
    tripped = Symbol[]
    for ind in indicators
        ind.enabled || continue
        v = ind.measure(pts, st)
        push!(vals, ind.name => v)
        trips(ind, v) && push!(tripped, ind.name)
    end
    return vals, tripped
end
