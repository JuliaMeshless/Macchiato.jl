abstract type AbstractProblem end

"""
    MultiphysicsProblem(domain, u0, tspan; kwargs...)

Construct an `ODEProblem` for transient simulation from a `Domain`, an initial condition
vector `u0`, and a time span `tspan = (t_start, t_stop)`.

Assembles all model physics functions and boundary condition functions from the domain,
then composes them into a single ODE right-hand side `f(du, u, p, t)`.
"""
function MultiphysicsProblem(
        domain::Domain{Dim}, u0, tspan; kwargs...
    ) where {Dim}
    boundary_funcs = mapreduce(vcat, domain.boundaries) do b
        ids, bc = b.second
        surf = domain.cloud[b.first]
        make_bc(bc, surf, domain, ids; kwargs...)
    end
    model_funcs = mapreduce(
        m -> make_f(m, domain; kwargs...), vcat, domain.models
    )

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

"""
    LinearSolve.LinearProblem(domain::Domain; scheme=nothing, kwargs...)

Construct a `LinearProblem` for steady-state simulation from a `Domain`.

Assembles the system matrix `A` and right-hand side `b` from the physics model,
then applies boundary conditions by modifying the appropriate rows of `A` and `b`.
The resulting system `Ax = b` is solved with `LinearSolve.solve`.
"""
function LinearSolve.LinearProblem(
        domain::Domain;
        scheme = nothing,
        kwargs...
    )
    # create initial system matrix and rhs based on physics model
    # current setup only works when you have one physics model
    println("Creating linear problem")
    model = only(domain.models)
    @time A, b = make_system(model, domain; kwargs...)
    bc_kw = _bc_kwargs(model)

    for boundary in domain.boundaries
        ids, bc = boundary.second
        println("Applying boundary condition: ", boundary.first)
        surf = domain.cloud[boundary.first]
        @time make_bc!(A, b, bc, surf, domain, ids; scheme = scheme, bc_kw..., kwargs...)
    end

    println("Done creating linear problem")
    return LinearSolve.LinearProblem(dropzeros(A), b)
end

_bc_kwargs(::AbstractModel) = NamedTuple()

function _bc_kwargs(model::LinearElasticity)
    μ, λstar = lame_parameters(model)
    return (; λstar = λstar, μ_lame = μ)
end

function _num_vars(models::AbstractVector{<:AbstractModel}, Dim)
    return sum(m -> _num_vars(m, Dim), models)
end
