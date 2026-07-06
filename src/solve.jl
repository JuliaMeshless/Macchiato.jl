"""
    LinearSolve.LinearProblem(domain::Domain; scheme=nothing, verbose=false, kwargs...)

Construct a `LinearProblem` for steady-state simulation from a `Domain`.

Assembles the system matrix `A` and right-hand side `b` from the physics model,
then applies boundary conditions by modifying the appropriate rows of `A` and `b`.
The resulting system `Ax = b` is solved with `LinearSolve.solve`. Pass `verbose=true`
to print assembly progress.

Steady-state solving currently supports a single physics model per domain.
"""
function LinearSolve.LinearProblem(
        domain::Domain;
        scheme = nothing,
        verbose = false,
        kwargs...
    )
    length(domain.models) == 1 || throw(
        ArgumentError(
            "Steady-state solving currently supports a single physics model per domain; got $(length(domain.models))."
        )
    )
    model = only(domain.models)

    verbose && println("Creating linear problem")
    A, b = make_system(model, domain; kwargs...)
    bc_kw = _bc_kwargs(model)

    for boundary in domain.boundaries
        ids, bc = boundary.second
        verbose && println("Applying boundary condition: ", boundary.first)
        surf = domain.cloud[boundary.first]
        make_bc!(A, b, bc, surf, domain, ids; scheme = scheme, bc_kw..., kwargs...)
    end

    verbose && println("Done creating linear problem")
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
