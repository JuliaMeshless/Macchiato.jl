abstract type AbstractProblem end

function MultiphysicsProblem(
        domain::Domain{Dim}, u0, tspan; kwargs...) where {Dim}
    boundary_funcs = mapreduce(vcat, domain.boundaries) do b
        ids, bc = b.second
        surf = domain.cloud[b.first]
        make_bc(bc, surf, domain, ids; kwargs...)
    end
    model_funcs = mapreduce(
        m -> make_f(m, domain; kwargs...), vcat, domain.models)

    model_funcs = model_funcs isa Vector ? model_funcs : [model_funcs]

    function f(du, u, p, t)
        for model in model_funcs
            model(du, u, p, t)
        end
        for bc in boundary_funcs
            bc(du, u, p, t)
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
        @time make_bc!(A, b, bc, surf, domain, ids; scheme=scheme, kwargs...)
    end

    println("Done creating linear problem")
    return LinearSolve.LinearProblem(dropzeros(A), b)
end

function _num_vars(models::AbstractVector{<:AbstractModel}, Dim)
    sum(m -> _num_vars(m, Dim), models)
end
