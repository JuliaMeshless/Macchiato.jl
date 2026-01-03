"""
Compute interpolation weights at a point, returning dense vector.
"""
@inline function interpolation_weights(nbs_coords, pt; kwargs...)
    op = regrid(nbs_coords, [pt]; kwargs...)
    update_weights!(op)
    return collect(op.weights[1, :])
end

"""
Compute derivative weights for a boundary point using standard directional derivative.
No shadow points - uses direct finite difference.
Returns (neighbor_indices, weights).
"""
function compute_local_derivative_weights(
        surf, domain, shadow_op::Nothing, A, global_i, local_i, normals; kwargs...)

    # Get neighbors from A (assuming structural symmetry)
    nbs = A.rowval[A.colptr[global_i]:(A.colptr[global_i + 1] - 1)]

    # Get coords
    surf_pt = get_node_coords(surf, local_i)
    nbs_coords = [get_node_coords(domain.cloud, nb) for nb in nbs]

    # Compute weights
    n = ustrip(normals[local_i])
    d = directional(nbs_coords, [surf_pt], n)

    # Collect to dense to preserve precision
    w = collect(d.weights[1, :])
    return nbs, w
end

"""
Compute derivative weights for a boundary point using 1st order shadow points.
First-order shadow points: ∂u/∂n ≈ (u_surface - u_shadow)/Δ.
Returns (neighbor_indices, weights).
"""
function compute_local_derivative_weights(
        surf, domain, shadow_op::WhatsThePoint.ShadowPoints{1},
        A, global_i, local_i, normals; kwargs...)
    nbs = A.rowval[A.colptr[global_i]:(A.colptr[global_i + 1] - 1)]

    surf_pt = get_node_coords(surf, local_i)
    nbs_coords = [get_node_coords(domain.cloud, nb) for nb in nbs]

    n = ustrip(normals[local_i])
    d = ustrip(shadow_op.Δ(surf_pt))

    # Compute derivative using shadow point: (u_surf - u_shadow)/Δ
    shadow_pt = surf_pt .- n .* d
    w_surf = interpolation_weights(nbs_coords, surf_pt; kwargs...)
    w_shadow = interpolation_weights(nbs_coords, shadow_pt; kwargs...)
    w_deriv = @. (w_surf - w_shadow) / d
    return nbs, w_deriv
end

"""
Compute derivative weights for a boundary point using 2nd order shadow points.
Second-order shadow points: ∂u/∂n ≈ (3·u_surface - 4·u_shadow1 + u_shadow2)/(2·Δ).
Returns (neighbor_indices, weights).
"""
function compute_local_derivative_weights(
        surf, domain, shadow_op::WhatsThePoint.ShadowPoints{2},
        A, global_i, local_i, normals; kwargs...)
    nbs = A.rowval[A.colptr[global_i]:(A.colptr[global_i + 1] - 1)]

    surf_pt = get_node_coords(surf, local_i)
    nbs_coords = [get_node_coords(domain.cloud, nb) for nb in nbs]

    n = ustrip(normals[local_i])
    d = ustrip(shadow_op.Δ(surf_pt))

    # Compute derivative using 2nd order shadow points
    shadow_pt1 = surf_pt .- n .* d
    shadow_pt2 = surf_pt .- n .* (2 * d)
    w_surf = interpolation_weights(nbs_coords, surf_pt; kwargs...)
    w_shadow1 = interpolation_weights(nbs_coords, shadow_pt1; kwargs...)
    w_shadow2 = interpolation_weights(nbs_coords, shadow_pt2; kwargs...)
    w_deriv = @. (3 * w_surf - 4 * w_shadow1 + w_shadow2) / (2 * d)
    return nbs, w_deriv
end
