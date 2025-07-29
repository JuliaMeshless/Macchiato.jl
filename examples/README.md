# New additions (written by human)

`heat-equation-2d-Hermite.jl` solves the 2d heat conduction problem with Hermite scheme with Dirichlet and Neumann conditions present at the same time
`manufactured_solution_check.jl` implements a test where the correctness of the implementation is checked using the method of manufactured solutions (might be moved to a test file in the future)

I was able to simplify the code significantly, however, I still haven't found a way to avoid the creation of the dictionary called `bcs_type_dict`, I think the domain.jl file should be edited and didn't want to do it yet.
This forced me to introduce the struct BoundaryCondition at the beginning of `hermite_utils.jl`.

...Below follows the output of chatGPT when asked to propose further abstracions for solving this issue.

# Boundary Condition Abstraction in Examples (written by AI)

## Current Approach

Boundary conditions are defined using a dictionary (`bcs_type_dict`) mapping surface names to a unified `BoundaryCondition` abstraction. This abstraction supports both Dirichlet and Neumann types, with values provided as functions (or constants).

```julia
bcs_type_dict = Dict(
    :surface1 => BoundaryCondition(:Dirichlet, (x, y) -> 10.0),
    :surface2 => BoundaryCondition(:Neumann, (x, y, nx, ny) -> 0.0),
    :surface3 => BoundaryCondition(:Dirichlet, (x, y) -> 5.0)
)
```

Boundary condition objects for the solver are constructed via `make_bcs(bcs_type_dict)`, and passed to the domain:

```julia
bcs = make_bcs(bcs_type_dict)
domain = MM.Domain(cloud, bcs, SolidEnergy(...))
```

The boundary condition type dictionary (`bcs_type_dict`) is kept separate from the domain object. Utilities such as `get_new_domain_info`, `get_b`, and `make_system_Hermite` require both the domain and the type dictionary as arguments.

### Sub-optimality

- The separation of boundary condition types from the domain object means that all utilities must be passed both the domain and the type dictionary.
- This can lead to errors, mismatches, and less maintainable code.

### Recommendation for Further Abstraction

- Integrate the boundary condition type abstraction directly into the domain struct (e.g., as a field in `MM.Domain` or a wrapper).
- This would allow all utilities to query the domain for both the boundary condition object and its type, eliminating the need for a separate dictionary.
- Benefits include a cleaner API, reduced risk of mismatches, and easier extension for more complex boundary conditions.

#### Example (Current Usage)

```julia
bcs_type_dict = Dict(...)
bcs = make_bcs(bcs_type_dict)
domain = MM.Domain(cloud, bcs, ...)
prob, boundary_values = LinearProblemHermite(domain, bcs_type_dict)
```

#### Example (Recommended Future Usage)

```julia
# Integrate bcs_type_dict into domain
struct MyDomain
    cloud
    bcs
    bcs_type_dict
    ...
end
# Utilities would then use domain.bcs_type_dict
```

For now, keep `bcs_type_dict` and `bcs` as separate objects and pass both to utilities. For deeper abstraction, consider integrating the type dictionary into the domain struct.

## Further Abstraction Suggestions

Here are additional ideas for making boundary condition handling more robust and extensible:

### 1. Boundary Condition Polymorphism
- Define an abstract interface for boundary conditions (e.g., `AbstractBoundaryCondition`).
- Implement methods like `apply!`, `value_at`, or `enforce!` for each BC type (Dirichlet, Neumann, Robin, Mixed).
- Utilities can then call these methods generically, without needing to check types or dictionary keys.

### 2. Boundary Condition Registry
- Use a registry or struct that encapsulates all boundary condition logic, rather than a flat dictionary.
- This registry can provide methods for querying, updating, and applying BCs, making the code more modular.

### 3. Unified Domain API
- Extend `MM.Domain` to expose a unified API for boundary queries:
  - `domain.get_boundary_type(surface)`
  - `domain.get_boundary_value(surface, x, y, nx, ny)`
- This hides implementation details and makes utilities agnostic to how BCs are stored.

### 4. Boundary Condition Composition
- Allow composition of boundary conditions (e.g., time-dependent, spatially-varying, or mixed BCs).
- Use function composition or callable structs to build complex BCs from simple ones.

### 5. Automatic Boundary Detection
- Automate the assignment of boundary types and values based on geometry, tags, or metadata.
- This reduces manual dictionary setup and makes the code more robust to changes in geometry.

### 6. Boundary Condition Validation
- Add validation routines that check for consistency between the domain, BC registry, and solver setup.
- Warn or error if there are mismatches, missing BCs, or unsupported combinations.

### 7. Solver-agnostic Boundary Handling
- Abstract BC enforcement so that it works for different solvers (steady, transient, nonlinear, etc.) without rewriting logic.
- Use traits or dispatch to select the correct enforcement strategy.

### 8. Documentation and Examples
- Document common BC patterns and provide example scripts for each abstraction.
- This helps users understand and extend the framework for their own problems.

---

If you want to pursue any of these ideas, you can start by refactoring your code or utilities to follow these patterns, or reach out for more design help!
