"""Method type hierarchy for computing boundary derivative weights."""
abstract type DerivativeMethod end

abstract type StandardDerivative <: DerivativeMethod end
abstract type ShadowPointsMethod <: DerivativeMethod end
abstract type ShadowPointsFirstOrder <: ShadowPointsMethod end
abstract type ShadowPointsSecondOrder <: ShadowPointsMethod end

"""Map shadow operator to derivative method type for dispatch."""
derivative_method(::Nothing) = StandardDerivative
derivative_method(::WhatsThePoint.ShadowPoints{1}) = ShadowPointsFirstOrder
derivative_method(::WhatsThePoint.ShadowPoints{2}) = ShadowPointsSecondOrder

"""
Compute derivative weights for a boundary point. Dispatches to method-specific implementation.
Returns (neighbor_indices, weights).
"""
function compute_local_derivative_weights(
        surf, domain, shadow_op, A, global_i, local_i, normals;
        kwargs...)::Tuple{Vector{Int}, Vector{Float64}}
    return compute_local_derivative_weights(derivative_method(shadow_op),
        surf, domain, shadow_op, A, global_i, local_i, normals; kwargs...)
end

"""Standard directional derivative using direct finite difference."""
function compute_local_derivative_weights(
        ::Type{StandardDerivative}, surf, domain, shadow_op, A, global_i, local_i, normals; kwargs...)

    # Get neighbors from A (assuming structural symmetry)
    nbs = view(A.rowval, A.colptr[global_i]:(A.colptr[global_i + 1] - 1))

    # Get coords
    surf_pt = get_node_coords(surf, local_i)
    nbs_coords = [get_node_coords(domain.cloud, nb) for nb in nbs]

    # Compute weights
    n = ustrip(normals[local_i])
    d = directional(nbs_coords, [surf_pt], n)

    # d.weights is (1, length(nbs))
    return nbs, d.weights[1, :]
end

"""First-order shadow points: ∂u/∂n ≈ (u_surface - u_shadow)/Δ."""
function compute_local_derivative_weights(
        ::Type{ShadowPointsFirstOrder}, surf, domain, shadow_op, A, global_i, local_i, normals; kwargs...)
    nbs = view(A.rowval, A.colptr[global_i]:(A.colptr[global_i + 1] - 1))

    surf_pt = get_node_coords(surf, local_i)
    nbs_coords = [get_node_coords(domain.cloud, nb) for nb in nbs]

    n = ustrip(normals[local_i])
    d = ustrip(shadow_op.Δ(surf_pt))

    # Shadow point
    shadow_pt = surf_pt .- n .* d

    # Interpolation weights for surface point (regrid)
    op_surf = regrid(nbs_coords, [surf_pt]; kwargs...)
    update_weights!(op_surf)
    w_surf = op_surf.weights[1, :]

    # Interpolation weights for shadow point
    op_shadow = regrid(nbs_coords, [shadow_pt]; kwargs...)
    update_weights!(op_shadow)
    w_shadow = op_shadow.weights[1, :]

    # Derivative weights: (w_surf - w_shadow) / d
    w_deriv = (w_surf .- w_shadow) ./ d

    return nbs, w_deriv
end

"""Second-order shadow points: ∂u/∂n ≈ (3·u_surface - 4·u_shadow1 + u_shadow2)/(2·Δ)."""
function compute_local_derivative_weights(
        ::Type{ShadowPointsSecondOrder}, surf, domain, shadow_op, A, global_i, local_i, normals; kwargs...)
    nbs = view(A.rowval, A.colptr[global_i]:(A.colptr[global_i + 1] - 1))

    surf_pt = get_node_coords(surf, local_i)
    nbs_coords = [get_node_coords(domain.cloud, nb) for nb in nbs]

    n = ustrip(normals[local_i])
    d = ustrip(shadow_op.Δ(surf_pt))

    shadow_pt1 = surf_pt .- n .* d
    shadow_pt2 = surf_pt .- n .* (2 * d)

    op_surf = regrid(nbs_coords, [surf_pt]; kwargs...)
    update_weights!(op_surf)
    w_surf = op_surf.weights[1, :]

    op_shadow1 = regrid(nbs_coords, [shadow_pt1]; kwargs...)
    update_weights!(op_shadow1)
    w_shadow1 = op_shadow1.weights[1, :]

    op_shadow2 = regrid(nbs_coords, [shadow_pt2]; kwargs...)
    update_weights!(op_shadow2)
    w_shadow2 = op_shadow2.weights[1, :]

    # (3*u_s - 4*u_sh1 + u_sh2) / 2d
    w_deriv = (3 .* w_surf .- 4 .* w_shadow1 .+ w_shadow2) ./ (2 * d)

    return nbs, w_deriv
end
