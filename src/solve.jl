abstract type AbstractProblem end

function MultiphysicsProblem(
        domain::Domain{Dim}, u0, tspan; kwargs...) where {Dim}
    # create initial system matrix and rhs based on physics model
    # current setup only works when you have one physics model
    A, b = make_system(only(domain.models), domain; kwargs...)

    # Apply Neumann/Robin BCs to the system matrix A and vector b
    # We do NOT apply Dirichlet BCs here, as they are handled explicitly in the ODE function
    for boundary in domain.boundaries
        ids, bc = boundary.second
        if !(bc isa Dirichlet)
            surf = domain.cloud[boundary.first]
            make_bc!(A, b, bc, surf, domain, ids; kwargs...)
        end
    end

    # Create functions for Dirichlet BCs to override du/dt
    boundary_funcs = []
    for boundary in domain.boundaries
        ids, bc = boundary.second
        if bc isa Dirichlet
            # Zero out the rows in the system matrix and vector for Dirichlet nodes
            # This ensures they don't contribute to the physics calculation
            # and avoids wasted computation.
            A[ids, :] .= 0
            b[ids] .= 0

            surf = domain.cloud[boundary.first]
            push!(boundary_funcs, make_bc(bc, surf, domain, ids; kwargs...))
        end
    end

    function f(du, u, p, t)
        # 1. Compute physics (including Neumann/Robin BCs via modified A)
        mul!(du, A, u)
        du .+= b

        # 2. Apply Dirichlet BCs (override du/dt)
        for bc_func in boundary_funcs
            bc_func(du, u, p, t)
        end
        return nothing
    end

    num_vars = _num_vars(domain.models, Dim)

    return ODEProblem(f, repeat(u0, num_vars), tspan)
end

function LinearProblem(domain::Domain;
        scheme = nothing,
        kwargs...)
    # create initial system matrix and rhs based on physics model
    # current setup only works when you have one physics model
    println("Creating linear problem")
    @time A, b = make_system(only(domain.models), domain; kwargs...)
    #return dropzeros(A), b

    for boundary in domain.boundaries
        ids, bc = boundary.second
        println("Applying boundary condition: ", boundary.first)
        surf = domain.cloud[boundary.first]
        @time make_bc!(A, b, bc, surf, domain, ids; scheme = scheme, kwargs...)
    end

    println("Done creating linear problem")
    return LinearSolve.LinearProblem(dropzeros(A), b)
end

function _num_vars(models::AbstractVector{<:AbstractModel}, Dim)
    sum(m -> _num_vars(m, Dim), models)
end
