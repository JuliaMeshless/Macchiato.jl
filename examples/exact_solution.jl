
exact_solution(x, y) = x^2 + y^2 + 3

# For Neumann: q = -k * grad(T) ⋅ n, grad(T) = (2x, 2y)
function exact_flux(x, y, nx, ny)
    d_sol_dx = 2 * x
    d_sol_dy = 2 * y
    gradT_dot_n = d_sol_dx * nx + d_sol_dy * ny
    return gradT_dot_n
end

function exact_laplacian(x, y)
    return 4.0
end

function calculate_bcs(cloud, bcs_dict)
    all_surfaces = cloud.boundary.surfaces
    surf_names = all_surfaces.keys
    bcs = Dict{Symbol, Any}()
    for sname in surf_names
        surf = all_surfaces[sname]
        surf_elements = surf.geoms
        bc_values = zeros(Float64, length(surf_elements))
        for (i, elem) in enumerate(surf_elements)
            x = ustrip(elem.point.coords.x)
            y = ustrip(elem.point.coords.y)
            if bcs_dict[sname] == :Neumann
                nx = Float64(elem.normal[1])
                ny = Float64(elem.normal[2])
                q = ustrip(exact_flux(x, y, nx, ny))
                bc_values[i] = q
            else
                Tval = ustrip(exact_solution(x, y))
                bc_values[i] = Tval
            end
        end
        bcs[sname] = bc_values
    end

    return bcs
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
