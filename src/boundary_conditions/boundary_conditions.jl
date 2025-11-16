"""
    boundary_conditions.jl

Generic boundary condition framework for MeshlessMultiphysics.jl.

Imports the boundary condition system from RadialBasisFunctions.jl:
    Robin BC: α*u + β*∂ₙu = g

Special cases:
- Dirichlet: α=1, β=0  →  u = g
- Neumann:   α=0, β=1  →  ∂ₙu = g
- Robin:     α≠0, β≠0  →  α*u + β*∂ₙu = g
"""

using RadialBasisFunctions: BoundaryCondition, Dirichlet, Neumann, Robin, Internal
using RadialBasisFunctions: α, β, is_dirichlet, is_neumann, is_robin, is_internal

export BoundaryCondition, Dirichlet, Neumann, Robin, Internal
export α, β, is_dirichlet, is_neumann, is_robin, is_internal

# ============================================================================
# Default BC Implementation
# ============================================================================

"""
    make_bc!(A, b, boundary, surf, domain, ids; kwargs...)

Apply boundary condition to linear system. Dispatches based on BC type.

Concrete BC types should define:
- bc_type(::MyBC) → BoundaryCondition (Dirichlet/Neumann/Robin)
- bc_value(bc::MyBC) → prescribed value

Special cases can override for custom behavior.
"""
function make_bc!(A, b, boundary, surf, domain, ids; kwargs...)
    bc = bc_type(boundary)
    value = bc_value(boundary)

    if is_dirichlet(bc)
        make_bc_dirichlet!(A, b, ids, value)
    elseif is_neumann(bc)
        make_bc_neumann!(A, b, surf, domain, ids, value; kwargs...)
    else  # Robin
        make_bc_robin!(A, b, α(bc), β(bc), surf, domain, ids, value; kwargs...)
    end

    return A
end

# ============================================================================
# BC Type Implementations
# ============================================================================

function make_bc_dirichlet!(A::AbstractMatrix{TA}, b::AbstractVector{TB},
        ids, value) where {TA, TB}
    for i in ids
        A[i, :] .= zero(TA)
        A[i, i] = one(TA)
        b[i] = value isa AbstractVector ? value[i - first(ids) + 1] : convert(TB, value)
    end
end

function make_bc_neumann!(A::AbstractMatrix{TA}, b::AbstractVector{TB},
        surf, domain, ids, flux_value; shadow_op = nothing, kwargs...) where {TA, TB}
    if shadow_op !== nothing
        # Use shadow points method for derivative approximation
        weights = compute_normal_derivative_weights(surf, domain, shadow_op; kwargs...)
    else
        # Use standard directional derivative
        d = directional(coordinates(domain.cloud), coordinates(surf), normals(surf);
            k = get(kwargs, :k, 40))
        update_weights!(d)
        weights = d.weights
    end

    for i in ids
        A[i, :] .= weights[i, :]
        b[i] = convert(TB, flux_value)
    end
end

function make_bc_robin!(A::AbstractMatrix{TA}, b::AbstractVector{TB},
        α_val, β_val, surf, domain, ids, value; shadow_op = nothing, kwargs...) where {
        TA, TB}
    if shadow_op !== nothing
        # Use shadow points method for derivative approximation
        weights = compute_normal_derivative_weights(surf, domain, shadow_op; kwargs...)
    else
        # Use standard directional derivative
        d = directional(coordinates(domain.cloud), coordinates(surf), normals(surf);
            k = get(kwargs, :k, 40))
        update_weights!(d)
        weights = d.weights
    end

    for i in ids
        A[i, :] .= convert(TA, β_val) .* weights[i, :]
        A[i, i] += convert(TA, α_val)
        b[i] = convert(TB, value)
    end
end

# ============================================================================
# Shadow Points Support (for high-accuracy normal derivatives)
# ============================================================================

"""
    compute_normal_derivative_weights(surf, domain, shadow_op; kwargs...)

Compute weights for normal derivative approximation using shadow points method.

Shadow points are virtual points placed outside the boundary along the normal direction.
They enable high-accuracy finite difference approximations of normal derivatives without
requiring directional derivative operators.

# Arguments
- `surf`: Surface where derivative is computed
- `domain`: Problem domain
- `shadow_op`: Shadow points operator (e.g., ShadowPoints(Δ, order))

# Returns
- Sparse matrix of weights for computing ∂u/∂n

# Shadow Point Orders
- Order 1: `(u_surf - u_shadow) / Δ` - first order accurate
- Order 2: `(3*u_surf - 4*u_shadow1 + u_shadow2) / (2*Δ)` - second order accurate

# Example
```julia
# Adiabatic BC with 2nd order shadow points
struct Adiabatic{T} <: EnergyBoundaryCondition
    shadow_op::T
end
Adiabatic() = Adiabatic(nothing)  # Standard Neumann
Adiabatic(Δ, order=1) = Adiabatic(ShadowPoints(Δ, order))  # Shadow points

bc_type(::Adiabatic) = Neumann()
bc_value(::Adiabatic) = 0.0

# Pass shadow_op through to make_bc!
function make_bc!(A, b, boundary::Adiabatic, surf, domain, ids; kwargs...)
    make_bc_neumann!(A, b, surf, domain, ids, 0.0; shadow_op=boundary.shadow_op, kwargs...)
end
```
"""
function compute_normal_derivative_weights(surf, domain, shadow_op; kwargs...)
    coords = _coords(domain.cloud)

    # Determine order from shadow_op type
    order = shadow_op isa ShadowPoints{2} ? 2 : 1

    # Generate shadow points
    shadow_points1 = generate_shadows(surf, shadow_op)

    # Build surface weights
    surf_weights = regrid(coords, _coords(surf); kwargs...)
    update_weights!(surf_weights)

    if order == 1
        # First order: (u_surf - u_shadow) / Δ = ∂u/∂n
        shadow1 = regrid(coords, _coords(shadow_points1); kwargs...)
        update_weights!(shadow1)

        weights = columnwise_div(
            surf_weights.weights .- shadow1.weights,
            ustrip(shadow_op.Δ(1))
        )
    else  # order == 2
        # Second order: (3*u_surf - 4*u_shadow1 + u_shadow2) / (2*Δ) = ∂u/∂n
        shadow_points2 = generate_shadows(surf,
            ShadowPoints(ConstantSpacing(shadow_op.Δ.Δx * 2)))

        shadow1 = regrid(coords, shadow_points1; kwargs...)
        update_weights!(shadow1)

        shadow2 = regrid(coords, shadow_points2; kwargs...)
        update_weights!(shadow2)

        weights = columnwise_div(
            3 * surf_weights.weights .- 4 * shadow1.weights .+ shadow2.weights,
            2 * shadow_op.Δ.Δx
        )
    end

    return weights
end

"""
    add_shadows!(cloud, surf, shadow)

Add shadow points to a cloud for shadow point boundary conditions.

This modifies the cloud by adding virtual shadow points offset from the boundary
along the normal direction. These points are used for finite difference approximations
of normal derivatives.

# Arguments
- `cloud`: Point cloud to modify
- `surf`: Surface symbol (e.g., :surface1)
- `shadow`: Shadow points operator (e.g., ShadowPoints(Δ, order))

# Returns
- Indices of the newly added shadow points

# Example
```julia
shadow_ids = add_shadows!(cloud, :surface1, ShadowPoints(ConstantSpacing(0.01m), 1))
```
"""
function add_shadows!(cloud, surf, shadow)
    s = cloud[surf]
    delete!(cloud.surfaces, surf)
    cloud[surf] = s(shadow)

    shadow_points = generate_shadows(cloud[surf], shadow)

    # add shadow points to the cloud
    append!(cloud.points, shadow_points)
    cloud_length = length(cloud.points)
    shadow_ids = (cloud_length - length(shadow_points) + 1):cloud_length
    vol_ids = only(cloud.volume.points.indices)
    new_ids = first(vol_ids):(last(vol_ids) + length(shadow_points))
    cloud.volume = PointVolume(view(cloud.points, new_ids))

    return shadow_ids
end

"""
    cone(cloud, surf, k)

Create a cone-shaped neighborhood search for directional derivative calculations.

Filters neighbors to only include points within a cone aligned with the surface normal.
This improves accuracy of directional derivatives by excluding points that are not
in the direction of the normal vector. The cone half-angle is 56°.

# Arguments
- `cloud`: Point cloud
- `surf`: Surface points
- `k`: Number of neighbors to find

# Returns
- Adjacency list with cone-filtered neighbors

# Example
```julia
# Use cone search for improved directional derivatives
adjl = cone(domain.cloud, surf, 40)
d = directional(_coords(domain.cloud), _coords(surf), normals(surf); adjl=adjl)
```
"""
function cone(cloud, surf, k)
    all_points = _coords(cloud)
    surf_points = _coords(surf)
    normal = normals(surf)
    offset = first(only(surf.points.indices))

    tree = KDTree(all_points)
    adjl, _ = knn(tree, surf_points, k, true)

    for (i, neighbors) in enumerate(adjl)
        O = all_points[first(neighbors)]
        n = -normal[first(neighbors) - offset + 1]
        L = 0
        new_k = k
        local new_neighbors
        while L < k
            a, _ = knn(tree, O, new_k, true)
            new_neighbors = filter(a) do i
                v = all_points[i] - O
                abs(∠(v, n)) < (56 * π / 180)
            end
            L = length(new_neighbors)
            new_k += 10
        end
        adjl[i] = new_neighbors
    end
    return adjl
end

# Helper functions for shadow points
function columnwise_div(A::SparseMatrixCSC, B::AbstractVector)
    I, J, V = findnz(A)
    for idx in eachindex(V)
        V[idx] /= B[I[idx]]
    end
    return sparse(I, J, V)
end

columnwise_div(A::SparseMatrixCSC, B::Number) = A ./ B

function replace_rows(A, weights, ids, offset)
    I, J, V = findnz(A)
    I2, J2, V2 = findnz(weights)
    i = findall(i -> i ∈ ids, I)
    i2 = findall(i -> i ∈ (ids .- offset), I2)
    deleteat!(I, i)
    deleteat!(J, i)
    deleteat!(V, i)
    append!(I, I2[i2] .+ offset)
    append!(J, J2[i2])
    append!(V, V2[i2])
    return sparse(I, J, V)
end
