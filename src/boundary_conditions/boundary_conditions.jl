abstract type AbstractBoundaryCondition end

abstract type Dirichlet <: AbstractBoundaryCondition end
abstract type Neumann <: AbstractBoundaryCondition end
abstract type Robin <: AbstractBoundaryCondition end

bc_type(::Type{<:Dirichlet}) = Dirichlet
bc_type(::Type{<:Neumann}) = Neumann
bc_type(::Type{<:Robin}) = Robin

function make_bc!(A, b, boundary::T, surf, domain, ids; kwargs...) where {T}
    return make_bc!(bc_type(T), A, b, boundary, surf, domain, ids; kwargs...)
end

function make_bc!(::Type{Dirichlet}, A, b, boundary, surf, domain, ids; kwargs...)
    make_bc_dirichlet!(A, b, ids, boundary())
    return A
end

function make_bc!(::Type{Neumann}, A, b, boundary, surf, domain, ids; kwargs...)
    shadow_op = hasproperty(boundary, :shadow_op) ? boundary.shadow_op : nothing
    make_bc_neumann!(A, b, surf, domain, ids, boundary(); shadow_op=shadow_op, kwargs...)
    return A
end

function make_bc!(::Type{Robin}, A, b, boundary, surf, domain, ids; kwargs...)
    shadow_op = hasproperty(boundary, :shadow_op) ? boundary.shadow_op : nothing
    make_bc_robin!(A, b, α(boundary), β(boundary), surf, domain, ids, boundary();
                   shadow_op=shadow_op, kwargs...)
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

# ============================================================================
# Derivative Method Traits (Single Layer)
# ============================================================================

abstract type DerivativeMethod end

abstract type StandardDerivative <: DerivativeMethod end
abstract type ShadowPointsFirstOrder <: DerivativeMethod end
abstract type ShadowPointsSecondOrder <: DerivativeMethod end

# Trait accessor
derivative_method(::Nothing) = StandardDerivative
derivative_method(::ShadowPoints{1}) = ShadowPointsFirstOrder
derivative_method(::ShadowPoints{2}) = ShadowPointsSecondOrder

# ============================================================================
# Derivative Weight Computation (Trait-Based Dispatch)
# ============================================================================

function compute_derivative_weights(surf, domain, shadow_op; kwargs...)
    return compute_derivative_weights(derivative_method(shadow_op),
                                     surf, domain, shadow_op; kwargs...)
end

# Standard directional derivative
function compute_derivative_weights(::Type{StandardDerivative}, surf, domain, ::Nothing; kwargs...)
    d = directional(coordinates(domain.cloud), coordinates(surf), normals(surf);
                   k = get(kwargs, :k, 40))
    update_weights!(d)
    return d.weights
end

# First order shadow points: (u_surf - u_shadow) / Δ = ∂u/∂n
function compute_derivative_weights(::Type{ShadowPointsFirstOrder}, surf, domain, shadow_op; kwargs...)
    coords = _coords(domain.cloud)

    # Generate shadow points
    shadow_points1 = generate_shadows(surf, shadow_op)

    # Build interpolation weights
    surf_weights = regrid(coords, _coords(surf); kwargs...)
    update_weights!(surf_weights)

    shadow1 = regrid(coords, _coords(shadow_points1); kwargs...)
    update_weights!(shadow1)

    # First-order finite difference
    return columnwise_div(
        surf_weights.weights .- shadow1.weights,
        get_spacing(shadow_op)
    )
end

# Second order shadow points: (3·u_surf - 4·u_shadow1 + u_shadow2) / (2·Δ) = ∂u/∂n
function compute_derivative_weights(::Type{ShadowPointsSecondOrder}, surf, domain, shadow_op; kwargs...)
    coords = _coords(domain.cloud)
    Δx = get_spacing(shadow_op)

    # Generate both shadow layers
    shadow_points1 = generate_shadows(surf, shadow_op)
    shadow_points2 = generate_shadows(surf,
                                     ShadowPoints(ConstantSpacing(2 * Δx), 2))

    # Build interpolation weights
    surf_weights = regrid(coords, _coords(surf); kwargs...)
    update_weights!(surf_weights)

    shadow1 = regrid(coords, _coords(shadow_points1); kwargs...)
    update_weights!(shadow1)

    shadow2 = regrid(coords, _coords(shadow_points2); kwargs...)
    update_weights!(shadow2)

    # Second-order finite difference
    return columnwise_div(
        3 * surf_weights.weights .- 4 * shadow1.weights .+ shadow2.weights,
        2 * Δx
    )
end

# Helper to extract spacing from shadow operator
get_spacing(shadow_op) = ustrip(shadow_op.Δ.Δx)

# ============================================================================
# BC Implementations (Simplified with Holy Traits)
# ============================================================================

function make_bc_neumann!(A::AbstractMatrix{TA}, b::AbstractVector{TB},
        surf, domain, ids, flux_value; shadow_op = nothing, kwargs...) where {TA, TB}
    weights = compute_derivative_weights(surf, domain, shadow_op; kwargs...)

    for i in ids
        A[i, :] .= weights[i, :]
        b[i] = convert(TB, flux_value)
    end
end

function make_bc_robin!(A::AbstractMatrix{TA}, b::AbstractVector{TB},
        α_val, β_val, surf, domain, ids, value; shadow_op = nothing, kwargs...) where {
        TA, TB}
    weights = compute_derivative_weights(surf, domain, shadow_op; kwargs...)

    for i in ids
        A[i, :] .= convert(TA, β_val) .* weights[i, :]
        A[i, i] += convert(TA, α_val)
        b[i] = convert(TB, value)
    end
end

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
