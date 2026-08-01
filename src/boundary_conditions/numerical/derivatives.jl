"""
Compute interpolation weights at a point, returning dense vector.
"""
@inline function interpolation_weights(nbs_coords, pt; kwargs...)
    W = weights(regrid(nbs_coords, [pt]; kwargs...))
    return W isa AbstractVector ? collect(W) : collect(W[1, :])
end

"""
    derivative_rows(surf, domain, scheme, ids, normals; adjl=nothing, kwargs...)

Weights of `∂/∂n` at the boundary nodes `ids`, as a
`length(ids) × length(cloud)` sparse block whose row `r` corresponds to node
`ids[r]`.

`adjl`, when given, lists every cloud node's stencil by global index; otherwise
stencils are the `DEFAULT_STENCIL_SIZE` nearest neighbours. `basis` must match
the one the interior operator was built with.
"""
function derivative_rows(
        surf, domain, ::Nothing, ids, normals;
        adjl = nothing, basis = PHS(3; poly_deg = 2), kwargs...
    )
    coords = _ustrip(_coords(domain.cloud))
    eval_points = coords[ids]

    # Stencils may reach interior nodes, so the data set is the whole cloud and
    # only the evaluation set is restricted to this surface.
    stencils = isnothing(adjl) ?
        find_neighbors(coords, eval_points, DEFAULT_STENCIL_SIZE) : adjl[ids]

    # ∂/∂n = Σ_d n_d ∂/∂x_d, combined from one batched Jacobian.
    G = gradient(coords; eval_points = eval_points, adjl = stencils, basis = basis)
    n = ustrip.(normals)
    return mapreduce(+, enumerate(weights(G))) do (d, W)
        Diagonal(getindex.(n, d)) * W
    end
end

# Shadow-point schemes have no batched form; build the same block node by node.
function derivative_rows(surf, domain, scheme, ids, normals; kwargs...)
    I, J, V = Int[], Int[], Float64[]
    for (local_i, global_i) in enumerate(ids)
        nbs, w = compute_local_derivative_weights(
            surf, domain, scheme, nothing, global_i, local_i, normals; kwargs...
        )
        append!(I, fill(local_i, length(nbs)))
        append!(J, nbs)
        append!(V, w)
    end
    return sparse(I, J, V, length(ids), length(domain.cloud))
end

"""
Compute derivative weights for a boundary point using 1st order shadow points.
First-order shadow points: ∂u/∂n ≈ (u_surface - u_shadow)/Δ.
Returns (neighbor_indices, weights).
"""
function compute_local_derivative_weights(
        surf, domain, shadow_op::WhatsThePoint.ShadowPoints{1},
        A, global_i, local_i, normals; kwargs...
    )
    nbs = A.rowval[A.colptr[global_i]:(A.colptr[global_i + 1] - 1)]

    surf_pt = get_node_coords(surf, local_i)
    nbs_coords = [get_node_coords(domain.cloud, nb) for nb in nbs]

    n = ustrip.(normals[local_i])
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
        A, global_i, local_i, normals; kwargs...
    )
    nbs = A.rowval[A.colptr[global_i]:(A.colptr[global_i + 1] - 1)]

    surf_pt = get_node_coords(surf, local_i)
    nbs_coords = [get_node_coords(domain.cloud, nb) for nb in nbs]

    n = ustrip.(normals[local_i])
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
