## Unused utility functions that might come handy in the future

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
    normals = normal(surf)
    offset = first(only(surf.points.indices))

    tree = KDTree(all_points)
    adjl, _ = knn(tree, surf_points, k, true)

    for (i, neighbors) in enumerate(adjl)
        O = all_points[first(neighbors)]
        n = -ustrip(normals[i])
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
