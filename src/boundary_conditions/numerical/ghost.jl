# ============================================================================
# Ghost points for transient derivative (Neumann/Robin) boundary conditions
# ============================================================================
#
# For a transient problem a Neumann/Robin boundary node must stay a *dynamic*
# unknown governed by the diffusion equation, with the flux folded into the
# discrete operator via a fictitious "ghost" node one spacing OUTSIDE the domain
# along +n. The ghost's value is eliminated algebraically from the wall-centered
# flux condition `u_ghost = u(mirror) + dist·∂ₙu`, with `u` at the mirror point one
# spacing inside interpolated (RBF `regrid` weights) over nearby volume nodes, and
# its stencil column tied back onto that interpolation row. This yields a stable operator
# `u' = W u + b(t)` — the scattered-node analogue of the finite-difference
# ghost-cell method for a no-flux wall. Unlike a penalty on the derivative, it
# does not depend on the sign of the RBF-FD self-weight, so it does not blow up.
#
# Dirichlet surfaces are ignored here; they are pinned separately in the BC
# closure. The relocation of the inward `ShadowPoints` derivative machinery from
# WhatsThePoint alongside this file is a planned follow-up.

# Type-stable per-surface source filler: `bc` is concrete, so the inner call
# specializes; the outer loop over these (one per Neumann/Robin surface) is short.
function _flux_source_applier(bc, idx, coef, xs)
    return function (cbuf, t)
        for m in eachindex(idx)
            cbuf[idx[m]] = coef[m] * bc(xs[m], t)
        end
        return nothing
    end
end

"""
    build_neumann_diffusion(domain; k=DEFAULT_STENCIL_SIZE, operator=...)

Assemble the transient scalar-diffusion operator for a `Domain` with all
Neumann/Robin flux conditions folded in via ghost nodes.

Returns `(W, G, source_appliers)`:
- `W::SparseMatrixCSC` (N×N): the solution-dependent diffusion operator — interior
  operator rows (default: RBF-FD Laplacian) plus ghost-folded boundary rows
  (including the Robin `α·u` coupling). Not scaled by the diffusivity.
- `G::SparseMatrixCSC` (N×n_ghost): scatters the per-ghost flux source `c(t)` into
  the right-hand side. Not scaled by the diffusivity.
- `source_appliers::Vector`: one closure per Neumann/Robin surface; calling
  `applier(cbuf, t)` fills that surface's entries of the length-`n_ghost` source
  buffer with `coef * bc(x, t)`.

`operator` is a builder callable `(data_pts; eval_points, k) -> RadialBasisOperator`
whose weights supply the interior operator rows; it defaults to the isotropic RBF-FD
`laplacian`. Pass a closure to fold the flux conditions into an anisotropic or
otherwise custom second-order operator — the ghost elimination is operator-agnostic
as long as the operator annihilates constants (so the ghost-folded rows keep zero
row-sum). For example, fiber-aligned anisotropic diffusion:

    D = SVector(D_L, D_T, D_T)   # fibers along x
    build_neumann_diffusion(
        domain; k = 60,
        operator = (data; eval_points, k) -> custom(
            data, @operator(∇ ⋅ (D * ∇));
            eval_points = eval_points, k = k, basis = PHS(3; poly_deg = 2),
        ),
    )

The boundary-first node ordering (`[boundary; volume]`) of the point cloud is
assumed, so a mirror-interpolation node's global index is
`n_boundary + its_volume_index`.
"""
function build_neumann_diffusion(
        domain; k = DEFAULT_STENCIL_SIZE,
        operator = (data; eval_points, k) -> laplacian(data; eval_points = eval_points, k = k),
    )
    all_pts = _ustrip(_coords(domain.cloud))
    vol_pts = _ustrip(_coords(domain.cloud.volume))
    N = length(all_pts)
    Nb = N - length(vol_pts)

    tree_all = KDTree(all_pts)

    ghost_pts = eltype(all_pts)[]
    mirror_pts = eltype(all_pts)[]
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    source_appliers = Function[]

    j = 0  # running ghost index (column of G / entry of cbuf)
    for (name, (ids, bc)) in domain.boundaries
        bc_family(typeof(bc)) === DerivativeBoundaryCondition || continue
        normals = normal(domain.cloud[name])
        is_robin = bc_type(typeof(bc)) === Robin

        surf_idx = Int[]
        surf_coef = Float64[]
        surf_x = eltype(all_pts)[]
        for (local_i, gi) in enumerate(ids)
            x = all_pts[gi]
            n = ustrip.(normals[local_i])
            _, nn_dist = knn(tree_all, x, 2, true)
            Δ = nn_dist[2]
            ghost = x .+ Δ .* n
            dist = 2 * Δ  # ghost-to-mirror gap, centered on the wall node

            j += 1
            push!(ghost_pts, ghost)
            push!(mirror_pts, x .- Δ .* n)
            if is_robin
                push!(rows, j); push!(cols, gi); push!(vals, -dist * α(bc) / β(bc))
            end
            push!(surf_idx, j)
            push!(surf_coef, is_robin ? dist / β(bc) : dist)
            push!(surf_x, x)
        end
        push!(source_appliers, _flux_source_applier(bc, surf_idx, surf_coef, surf_x))
    end

    n_ghost = j
    # Eliminate each ghost via the wall-centered difference `u_ghost = u(mirror) +
    # dist·∂ₙu`, with `u(mirror)` expressed as an RBF interpolation over nearby
    # volume nodes. On a scattered cloud no node sits at the mirror point, and
    # snapping to the nearest one instead injects a tangential `∇u ⋅ offset` error
    # into the eliminated ghost value that never converges away; the poly-augmented
    # interpolation row is exact for linear fields, so the folded operator stays
    # consistent.
    if n_ghost > 0
        M = sparse(regrid(vol_pts, mirror_pts; k = min(k, length(vol_pts))))
        Mi, Mj, Mv = findnz(M)
        append!(rows, Mi)
        append!(cols, Nb .+ Mj)
        append!(vals, Mv)
    end
    data_pts = vcat(all_pts, ghost_pts)
    W_ext = sparse(operator(data_pts; eval_points = all_pts, k = k))
    W_real = W_ext[:, 1:N]
    W_ghost = W_ext[:, (N + 1):(N + n_ghost)]
    T = sparse(rows, cols, vals, n_ghost, N)

    W = W_real + W_ghost * T
    return W, W_ghost, source_appliers
end
