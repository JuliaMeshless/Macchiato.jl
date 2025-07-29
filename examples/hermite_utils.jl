struct BoundaryCondition
    type::Symbol  # :Dirichlet or :Neumann
    value::Function  # function to compute the value at (x, y, nx, ny)
end

"""
make_bcs(bcs_type_dict)
    Returns a Dict of EnergyBoundaryCondition objects for each boundary surface.
"""
function make_bcs(bcs_type_dict)
    bcs = Dict{Symbol, EnergyBoundaryCondition}()
    for (k, v) in bcs_type_dict
        if v.type == :Dirichlet
            bcs[k] = Temperature(v.value)
        elseif v.type == :Neumann
            bcs[k] = HeatFlux(v.value)
        else
            error("Unsupported boundary condition type: $(v.type)")
        end
    end
    return bcs
end

"""
LinearProblemHermite(domain, bcs_type_dict, source_term=nothing)
    Returns a LinearProblem and boundary values for the Hermite system.
"""
function LinearProblemHermite(domain, bcs_type_dict, source_term = nothing)
    A, b, boundary_values = make_system_Hermite(domain, bcs_type_dict, source_term)
    return LinearSolve.LinearProblem(dropzeros(A), b), boundary_values
end
"""
make_system_Hermite(domain, bcs_type_dict, source_term = nothing; kwargs...)
    Assembles the Hermite system using shared boundary logic and domain info.
"""
function make_system_Hermite(domain, bcs_type_dict, source_term = nothing; kwargs...)
    all_coords, all_normals, is_boundary, is_Neumann, surface_name = get_new_domain_info(
        domain.cloud, bcs_type_dict)
    rbf_basis = RBF.PHS(3; poly_deg = 2)
    mon = RBF.MonomialBasis(2, rbf_basis.poly_deg)
    Lrbf = RBF.∇²(rbf_basis)
    Lmon = RBF.∇²(mon)
    k_stencil = RBF.autoselect_k(all_coords, rbf_basis)
    adjl = RBF.find_neighbors(all_coords, k_stencil)
    A, rhs = RBF._build_weights(
        all_coords, all_normals, is_boundary, is_Neumann, adjl, rbf_basis, Lrbf, Lmon, mon)
    b, boundary_values = get_b(
        rhs, domain, is_boundary, all_coords, all_normals, bcs_type_dict)
    if source_term == nothing
        b = -b
    else
        b = source_term - b
    end
    return A, b, boundary_values
end
"""
get_b(rhs, domain, is_boundary, all_coords, all_normals, bcs_type_dict)
    Returns boundary values for Dirichlet and Neumann BCs, using per-node logic and the provided coordinate and normal arrays.
"""
function get_b(rhs, domain, is_boundary, all_coords, all_normals, bcs_type_dict)
    boundary_values = zeros(sum(is_boundary))
    global2boundary = cumsum(is_boundary)
    for b in domain.boundaries
        surface = b.first
        ids, bc = b.second
        local_ids = global2boundary[ids]
        bc_type = bcs_type_dict[surface].type
        if bc_type == :Dirichlet
            val = bc isa Temperature ? bc.temperature : nothing
            for (i, idx) in enumerate(ids)
                cx, cy = all_coords[idx]
                boundary_values[local_ids[i]] = val(cx, cy)
            end
        elseif bc_type == :Neumann
            val = bc isa HeatFlux ? bc.heat_flux : nothing
            for (i, idx) in enumerate(ids)
                cx, cy = all_coords[idx]
                nx, ny = all_normals[idx]
                boundary_values[local_ids[i]] = val(cx, cy, nx, ny)
            end
        else
            error("Unsupported boundary condition type in get_b: $(bc_type)")
        end
    end
    return rhs * boundary_values, boundary_values
end
using NearestNeighbors
using Unitful: ustrip

"""
get_new_domain_info(cloud, bcs_dict)
    Returns (all_coords, all_normals, is_boundary, is_Neumann, surface_name) for the given cloud and boundary condition dictionary.
    bcs_dict should be Dict{Symbol, BoundaryCondition}.
"""
function get_new_domain_info(cloud, bcs_dict)
    surfaces = cloud.boundary.surfaces
    all_coords = ustrip.(MeshlessMultiphysics._coords(cloud))
    all_normals = zeros(typeof(all_coords[1]), size(all_coords))
    is_boundary = falses(length(all_coords))
    is_Neumann = falses(length(all_coords))
    surface_name = Vector{Symbol}(undef, length(all_coords))

    all_coords_tree = KDTree(all_coords)
    boundary_coordinates = zeros(typeof(all_coords[1]))
    for key in keys(surfaces)
        bc_type = get(bcs_dict, key, nothing)
        for bnd_geom in surfaces[key].geoms
            @reset boundary_coordinates[1] = ustrip(bnd_geom.point.coords.x)
            @reset boundary_coordinates[2] = ustrip(bnd_geom.point.coords.y)
            idx, dist = knn(all_coords_tree, boundary_coordinates, 1)
            if dist[1] > 1e-5
                @warn "Boundary point $(boundary_coordinates) is too far from the nearest point in the cloud"
            else
                global_index = idx[1]
                @reset all_normals[global_index] .= ustrip(bnd_geom.normal)
                is_boundary[global_index] = true
                is_Neumann[global_index] = (bc_type !== nothing && bc_type.type == :Neumann)
                surface_name[global_index] = key
            end
        end
    end
    return all_coords, all_normals, is_boundary, is_Neumann, surface_name
end
