using Test
using Aqua
using Macchiato

@testset "Aqua quality checks" begin
    # upwind(data, eval_points, dim, basis=...) and upwind(data, dim, basis=...) in
    # src/upwinding.jl are genuinely ambiguous for upwind(vec, vec, basis); needs a src fix
    Aqua.test_ambiguities(Macchiato; exclude = [Macchiato.upwind])
    Aqua.test_unbound_args(Macchiato)
    Aqua.test_undefined_exports(Macchiato)
    Aqua.test_project_extras(Macchiato)
    # Statistics is required by the docs/examples environments
    Aqua.test_stale_deps(Macchiato; ignore = [:Statistics])
    Aqua.test_deps_compat(Macchiato)
    Aqua.test_piracies(Macchiato)
end
