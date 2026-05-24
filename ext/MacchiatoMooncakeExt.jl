module MacchiatoMooncakeExt

import Macchiato
using Macchiato:
    apply_dirichlet!,
    extract_weight_sensitivities_elasticity!,
    extract_neumann_sensitivities!,
    extract_load_sensitivities!,
    extract_normal_sensitivities!,
    allocate_weight_gradients,
    assemble_elasticity_from_weights,
    apply_traction!,
    TractionLayout,
    NormalJacobian,
    LinearElasticity, lame_parameters,
    _propagate_weight_gradient!
using Mooncake
using RadialBasisFunctions: Partial, MixedPartial,
    _build_weights_and_cache, _pullback_weights!
using SparseArrays
using LinearAlgebra
using StaticArrays: SVector

# ============================================================================
# shape_gradient — manual adjoint with direct RBF backward (no Mooncake in Step 4)
# ============================================================================
# Step 4 uses RadialBasisFunctions._pullback_weights! directly — pure Julia
# linear algebra, sub-millisecond per operator. No build_rrule, no LLVM compile.

"""
    shape_gradient(
        pts_flat, model, N, adjl, basis, active,
        dirichlet_dofs, dirichlet_vals, ∂L_∂u;
        interior_rows, traction_layout, neumann_ids, neumann_adjl,
    )

Manual adjoint for 2D linear elasticity. Implements Steps 1–5 from
`plan_manual_adjoint.md`. Works for both Dirichlet-only and mixed
Dirichlet + traction BCs:

- **Dirichlet-only**: omit `traction_layout` / `neumann_ids` / `neumann_adjl`.
- **Mixed BC**: provide all three Neumann kwargs.

Step 4 uses `_propagate_weight_gradient!` which calls RBF's `_pullback_weights!`
directly — no Mooncake tracing, sub-millisecond per operator.

Returns `(u, Δpts)`.
"""
function Macchiato.shape_gradient(
    pts_flat::Vector{Float64},
    model::LinearElasticity,
    N::Int,
    adjl,
    basis,
    active::AbstractVector{Bool},
    dirichlet_dofs::Vector{Int},
    dirichlet_vals::Vector{Float64},
    ∂L_∂u::Function;
    interior_rows::Union{Nothing, BitVector} = nothing,
    traction_layout::Union{Nothing, TractionLayout} = nothing,
    neumann_ids::Union{Nothing, Vector{Int}} = nothing,
    neumann_adjl::Union{Nothing, Vector{Vector{Int}}} = nothing,
    traction_jacobians::Union{Nothing, AbstractVector{<:AbstractMatrix{Float64}}} = nothing,
    normal_jacobians::Union{Nothing, AbstractVector{NormalJacobian}} = nothing,
)
    has_neumann = traction_layout !== nothing
    μ, λstar = lame_parameters(model)

    # --- Step 1: Forward --------------------------------------------------
    pts = Macchiato._pts_from_flat(pts_flat)
    W_d2x,  cache_d2x  = _build_weights_and_cache(Partial(2, 1),      pts, pts, adjl, basis)
    W_d2y,  cache_d2y  = _build_weights_and_cache(Partial(2, 2),      pts, pts, adjl, basis)
    W_d2xy, cache_d2xy = _build_weights_and_cache(MixedPartial(1, 2), pts, pts, adjl, basis)

    if has_neumann
        neumann_pts = pts[neumann_ids]
        W_dx, cache_dx = _build_weights_and_cache(Partial(1, 1), pts, neumann_pts, neumann_adjl, basis)
        W_dy, cache_dy = _build_weights_and_cache(Partial(1, 2), pts, neumann_pts, neumann_adjl, basis)
    end

    A = assemble_elasticity_from_weights(W_d2x, W_d2y, W_d2xy, N, λstar, μ)
    b = zeros(2N)
    apply_dirichlet!(A, b, dirichlet_dofs, dirichlet_vals)
    if has_neumann
        apply_traction!(A, b, traction_layout, W_dx, W_dy)
    end

    F = lu(A)
    u = F \ b

    # --- Step 2: Adjoint solve -------------------------------------------
    η = F' \ ∂L_∂u(u)

    # --- Step 3: Extract ΔW ----------------------------------------------
    ΔW_d2x, ΔW_d2y, ΔW_d2xy = allocate_weight_gradients(W_d2x, W_d2x, W_d2x)
    extract_weight_sensitivities_elasticity!(
        ΔW_d2x, ΔW_d2y, ΔW_d2xy,
        W_d2x, η, u, active, λstar, μ;
        interior_rows = interior_rows,
    )
    if has_neumann
        ΔW_dx, ΔW_dy = allocate_weight_gradients(W_dx, W_dy)
        extract_neumann_sensitivities!(ΔW_dx, ΔW_dy, traction_layout, η, u, active)
    end

    # --- Step 4: Direct gradient propagation (no Mooncake) ----------------
    Δpts = zeros(Float64, 2N)
    _propagate_weight_gradient!(Δpts, ΔW_d2x,  W_d2x,  cache_d2x,  pts, pts, adjl, basis, Partial(2, 1))
    _propagate_weight_gradient!(Δpts, ΔW_d2y,  W_d2y,  cache_d2y,  pts, pts, adjl, basis, Partial(2, 2))
    _propagate_weight_gradient!(Δpts, ΔW_d2xy, W_d2xy, cache_d2xy, pts, pts, adjl, basis, MixedPartial(1, 2))
    if has_neumann
        _propagate_weight_gradient!(Δpts, ΔW_dx,   W_dx,   cache_dx,
                                    pts, neumann_pts, neumann_adjl, basis, Partial(1, 1);
                                    eval_offset = neumann_ids)
        _propagate_weight_gradient!(Δpts, ΔW_dy,   W_dy,   cache_dy,
                                    pts, neumann_pts, neumann_adjl, basis, Partial(1, 2);
                                    eval_offset = neumann_ids)
    end

    # --- Step 4b: dead-load b-side contribution (ηᵀ · ∂b/∂pts) ------------
    # No-op unless caller supplied per-Neumann-point traction Jacobians; safe
    # for the frozen-load case used by Phase A/B and the original Phase 3 run.
    if has_neumann && traction_jacobians !== nothing
        extract_load_sensitivities!(Δpts, traction_layout, η, neumann_ids, traction_jacobians)
    end

    # --- Step 4c: Phase D L2 — differentiable normals ---------------------
    # -ηᵀ · (∂A/∂n · ∂n/∂pts) · u, distributed via the sparse NormalJacobians.
    # No-op unless caller supplied normal Jacobians and the layout's coeffs
    # were updated to match the live normals (see `update_traction_coeffs!`).
    if has_neumann && normal_jacobians !== nothing
        extract_normal_sensitivities!(Δpts, traction_layout, η, u,
                                      W_dx, W_dy, normal_jacobians, λstar, μ;
                                      active = active)
    end

    return (u = u, Δpts = Δpts, η = η)
end

end # module
