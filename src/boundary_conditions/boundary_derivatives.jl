abstract type DerivativeMethod end

abstract type StandardDerivative <: DerivativeMethod end
abstract type ShadowPointsFirstOrder <: DerivativeMethod end
abstract type ShadowPointsSecondOrder <: DerivativeMethod end

# Trait accessor
derivative_method(::Nothing) = StandardDerivative
derivative_method(::ShadowPoints{1}) = ShadowPointsFirstOrder
derivative_method(::ShadowPoints{2}) = ShadowPointsSecondOrder

#This is tremendously inefficient because we are allocating I think
#Normals should come from WhatsThePoint directly
function surface_normals(surf)
    normals_vec = ustrip.(normal(surf))
    @assert all_equals(normals_vec) "All normals must be equal for the same surface"
    return first(normals_vec)
end

function all_equals(v::Vector{<:AbstractVector})
    length(v) == 1 && return true
    first_elem = v[1]
    @inbounds for elem in view(v, 2:length(v))
        if elem != first_elem
            return false
        end
    end
    return true
end

# Extract spacing value from shadow operator
# Note: shadow_op.Δ is a function that takes a point and returns spacing
# We evaluate it at a representative point (first coordinate of the surface)
function get_spacing(shadow_op, surf)
    # Get first surface point as representative location
    first_point = first(_coords(surf))
    return ustrip(shadow_op.Δ(first_point))
end

function compute_derivative_weights(surf, domain, shadow_op; kwargs...)
    return compute_derivative_weights(derivative_method(shadow_op),
        surf, domain, shadow_op; kwargs...)
end

# Standard directional derivative
function compute_derivative_weights(
        ::Type{StandardDerivative}, surf, domain, ::Nothing; kwargs...)
    domain_coords = ustrip.(_coords(domain.cloud))
    # println("domain_coords: ", domain_coords)
    surf_coords = ustrip.(_coords(surf))
    # println("surf_coords: ", surf_coords)
    normals_vec = surface_normals(surf)
    # println("normals_vec: ", normals_vec)
    d = directional(domain_coords, surf_coords, normals_vec)
    return d.weights
end

# First order shadow points: (u_surf - u_shadow) / Δ = ∂u/∂n
function compute_derivative_weights(
        ::Type{ShadowPointsFirstOrder}, surf, domain, shadow_op; kwargs...)
    # TODO:Strip units from all coordinates (temporary?)
    coords = ustrip.(_coords(domain.cloud))
    surf_coords = ustrip.(_coords(surf))

    # TODO: Get spacing and normal (dimensionless)
    Δ = get_spacing(shadow_op, surf)
    normal = surface_normals(surf)

    # TODO: Manually compute shadow points: shadow = surface - normal * Δ
    # (WhatsThePoint's generate_shadows giving issues)
    shadow_coords1 = map(surf_coords) do pt
        pt .- normal .* Δ
    end

    # Build interpolation weights (all coordinates are now dimensionless)
    surf_weights = regrid(coords, surf_coords; kwargs...)
    update_weights!(surf_weights)

    shadow1 = regrid(coords, shadow_coords1; kwargs...)
    update_weights!(shadow1)

    # First-order finite difference
    return columnwise_div(
        surf_weights.weights .- shadow1.weights,
        Δ
    )
end

# Second order shadow points: (3·u_surf - 4·u_shadow1 + u_shadow2) / (2·Δ) = ∂u/∂n
function compute_derivative_weights(
        ::Type{ShadowPointsSecondOrder}, surf, domain, shadow_op; kwargs...)
    # again needed to strip units (temporary?)
    coords = ustrip.(_coords(domain.cloud))
    surf_coords = ustrip.(_coords(surf))

    # Get spacing and normal (dimensionless)
    Δ_val = get_spacing(shadow_op, surf)
    normal = surface_normals(surf)

    # Manually compute shadow points for both layers 
    # First layer: shadow1 = surface - normal * Δ
    shadow_coords1 = map(surf_coords) do pt
        pt .- normal .* Δ_val
    end

    # Second layer: shadow2 = surface - normal * 2Δ
    shadow_coords2 = map(surf_coords) do pt
        pt .- normal .* (2 * Δ_val)
    end

    # Build interpolation weights (all coordinates are now dimensionless)
    surf_weights = regrid(coords, surf_coords; kwargs...)
    update_weights!(surf_weights)

    shadow1 = regrid(coords, shadow_coords1; kwargs...)
    update_weights!(shadow1)

    shadow2 = regrid(coords, shadow_coords2; kwargs...)
    update_weights!(shadow2)

    # Second-order finite difference
    return columnwise_div(
        3 * surf_weights.weights .- 4 * shadow1.weights .+ shadow2.weights,
        2 * Δ_val
    )
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

### These are not used currently but may be useful in future (TODO)
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

function cone(cloud, surf, k)
    all_points = _coords(cloud)
    surf_points = _coords(surf)
    normal = surface_normals(surf)
    offset = first(only(surf.points.indices))

    tree = KDTree(all_points)
    adjl, _ = knn(tree, surf_points, k, true)

    for (i, neighbors) in enumerate(adjl)
        O = all_points[first(neighbors)]
        n = -normal  # Use the single normal vector directly (all normals are equal)
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
