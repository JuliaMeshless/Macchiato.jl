exact_solution(x, y) = x^2 + y^2 + 3

function exact_flux(x, y, nx, ny)
    d_sol_dx = 2 * x
    d_sol_dy = 2 * y
    gradT_dot_n = d_sol_dx * nx + d_sol_dy * ny
    return gradT_dot_n
end

function exact_laplacian(x, y)
    return 4.0
end

function calculate_source_term(cloud)
    # Calculate source term for each point in the volume
    source_term = zeros(length(cloud.volume))
    for (i, coord) in enumerate(MM._coords(cloud.volume))
        x = ustrip(coord[1])
        y = ustrip(coord[2])
        source_term[i] = exact_laplacian(x, y)
    end
    return source_term
end
