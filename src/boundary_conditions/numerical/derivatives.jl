"""
Compute interpolation weights at a point, returning dense vector.
"""
@inline function interpolation_weights(nbs_coords, pt; kwargs...)
    # The shadow scheme calls this on a PER-NODE stencil (nbs_coords), so the
    # cloud-wide `adjl`/`k` in kwargs must NOT be forwarded to regrid — the local
    # interpolation uses every supplied neighbour and no external adjacency.
    k_local = length(nbs_coords)
    W = weights(regrid(nbs_coords, [pt]; k = k_local))
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

# Shadow-point schemes have no per-node batched RBF solve built into the RBF
# operator API, but the whole surface block IS expressible as batched sparse
# regrid operators:  W_surf = regrid(data, surf_pts),  W_shadow = regrid(data,
# shadow_pts)  each build a single sparse (n_boundary × n_data) weight matrix in
# the parallel CPU kernel. The derivative block is then just the sparse
# difference of the two. This replaces the old node-by-node loop which did two
# dense local solves PER boundary node and never finished at scale.
function derivative_rows(surf, domain, scheme, ids, normals; adjl = nothing,
                         basis = PHS(3; poly_deg = 2), A = nothing, kwargs...)
    coords = _ustrip(_coords(domain.cloud))
    surf_pts = coords[ids]                 # n_boundary eval points (the nodes)
    n = ustrip.(normals)
    d = map(eachindex(ids)) do li
        ustrip(shadow_op_value(scheme, get_node_coords(surf, li)))
    end

    # Stencils are per eval point and come from the same cloud adjacency the
    # interior operator uses (material-aware when provided).
    stencils = isnothing(adjl) ?
        find_neighbors(coords, surf_pts, DEFAULT_STENCIL_SIZE) : adjl[ids]

    # shadow points, one step (and, for order 2, two steps) inward along +n
    sh1 = [surf_pts[i] .- n[i] .* d[i] for i in eachindex(ids)]

    W_surf = weights(regrid(coords, surf_pts; adjl = stencils, basis = basis))
    W_sh1 = weights(regrid(coords, sh1; adjl = stencils, basis = basis))

    if scheme isa WhatsThePoint.ShadowPoints{1}
        # ∂u/∂n ≈ (u_surf - u_shadow)/Δ
        return Diagonal(1.0 ./ d) * (W_surf - W_sh1)
    else
        # ∂u/∂n ≈ (3 u_surf - 4 u_sh1 + u_sh2) / (2Δ)
        sh2 = [surf_pts[i] .- n[i] .* (2 .* d[i]) for i in eachindex(ids)]
        W_sh2 = weights(regrid(coords, sh2; adjl = stencils, basis = basis))
        return Diagonal(3.0 ./ (2.0 .* d)) * W_surf .-
               Diagonal(2.0 ./ d) * W_sh1 .+
               Diagonal(1.0 ./ (2.0 .* d)) * W_sh2
    end
end

# Δ value at a boundary point: ShadowPoints stores a constant (wrapped in a
# closure) or a function of position; both are callable.
shadow_op_value(s, p) = s(p)
striptype(x) = x

"""
Compute derivative weights for a boundary point using 1st order shadow points.
First-order shadow points: ∂u/∂n ≈ (u_surface - u_shadow)/Δ.
Returns (neighbor_indices, weights).

(Kept for the older node-by-node API; the batched `derivative_rows` above is
what the steady solve actually uses.)
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
