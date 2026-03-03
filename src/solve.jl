abstract type AbstractProblem end

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
