# Package Design

This page describes the architecture of Macchiato.jl for users who want to understand the internals or extend the package with new physics.

## Simulation Pipeline

Every simulation follows the same pipeline:

```
Geometry → Model → BCs → Domain → Simulation → Results
```

1. **Geometry** — WhatsThePoint.jl creates a `PointCloud` with boundary surfaces and interior points
2. **Model** — Choose a physics model (`SolidEnergy`, `LinearElasticity`, etc.)
3. **Boundary Conditions** — Assign BCs to each named surface
4. **Domain** — `Domain(cloud, bcs, model)` validates and binds everything together
5. **Simulation** — `Simulation(domain)` auto-detects steady/transient mode; `run!` executes
6. **Results** — Extract fields with `temperature`, `displacement`, etc.; export with `exportvtk`

## Type Hierarchy

### Models

```
AbstractModel
├── Solid
│   ├── SolidEnergy         # Heat equation
│   └── LinearElasticity    # Navier-Cauchy equations
├── Fluid
│   └── IncompressibleNavierStokes
└── Time
    ├── Steady
    └── Unsteady
```

`AbstractModel` represents **any PDE** — the built-in subtypes (`SolidEnergy`, `LinearElasticity`, etc.) are convenience models that ship with the package. You can define your own model for any equation; see [Custom PDEs](@ref) for a complete walkthrough.

Every model must implement:
- `_num_vars(::MyModel, dim)` — number of solution variables (1 for scalar, `dim` for vector, `dim+1` for velocity+pressure)
- `make_system(::MyModel, domain)` — assemble `(A, b)` for steady-state
- `make_f(::MyModel, domain)` — return `f(du, u, p, t)` for transient ODE integration

Optionally, models can implement `equation_set(::Type{<:MyModel})` to return an [`EquationSet`](@ref) trait for BC compatibility checking. If not defined, it defaults to `GenericEquations()` which is compatible with everything.

### Boundary Conditions

```
AbstractBoundaryCondition
├── Dirichlet                    # u = g
│   ├── PrescribedValue{P, F}   # Generic: parameterized by EquationSet P
│   └── Displacement             # Mechanics-specific (vector-valued)
└── DerivativeBoundaryCondition
    ├── Neumann                  # ∂u/∂n = q
    │   ├── PrescribedFlux{P, F}
    │   ├── ZeroFlux{P}
    │   └── Traction             # Mechanics-specific (vector-valued)
    └── Robin                    # α u + β ∂u/∂n = g
        └── Convection
```

## Boundary Condition System

The BC system separates **mathematical type** from **equation set** using a two-axis design:

### Mathematical Axis

The mathematical type determines *how* the BC modifies the linear system:
- **Dirichlet** → replace the row with identity: `A[i,:] = eᵢ`, `b[i] = g`
- **Neumann** → replace the row with normal derivative weights: `A[i,:] = ∂/∂n weights`, `b[i] = q`
- **Robin** → combine value and derivative: `A[i,:] = α·eᵢ + β·∂/∂n weights`, `b[i] = g`

### Equation Set Axis

The equation set determines *which models* a BC is compatible with:

```julia
abstract type EquationSet end
struct EnergyEquations      <: EquationSet end
struct FluidEquations       <: EquationSet end
struct MechanicsEquations   <: EquationSet end
struct UniversalEquations   <: EquationSet end  # compatible with all equation sets
struct GenericEquations     <: EquationSet end  # default for custom PDEs, compatible with all
```

Compatibility is checked at `Domain` construction via `is_compatible(bc_set, model_set)`. Same-type pairs are always compatible. `UniversalEquations` and `GenericEquations` are compatible with all other equation sets.

### Generic Types and Named Aliases

The generic BC types are parameterized by equation set:

```julia
struct PrescribedValue{P <: EquationSet, F <: Function} <: Dirichlet ... end
struct PrescribedFlux{P <: EquationSet, F <: Function} <: Neumann ... end
struct ZeroFlux{P <: EquationSet} <: Neumann end
```

For custom PDEs, unparameterized constructors default to `GenericEquations`:

```julia
PrescribedValue(0.0)    # equivalent to PrescribedValue{GenericEquations}(0.0)
PrescribedFlux(1.0)     # equivalent to PrescribedFlux{GenericEquations}(1.0)
ZeroFlux()              # equivalent to ZeroFlux{GenericEquations}()
```

Built-in physics aliases provide user-friendly names:

```julia
# Energy aliases
Temperature(value) = PrescribedValue{EnergyEquations}(value)
HeatFlux(flux)     = PrescribedFlux{EnergyEquations}(flux)
const Adiabatic    = ZeroFlux{EnergyEquations}

# Fluid aliases
VelocityInlet(v)     = PrescribedValue{FluidEquations}(v)
const VelocityOutlet = ZeroFlux{FluidEquations}
```

This design means adding a new equation set requires zero changes to the BC application machinery — you just define new aliases using the existing generic types.

## Steady-State vs Transient

The [`Simulation`](@ref) constructor auto-detects mode from keyword arguments:

```julia
Simulation(domain)                              # → SteadyState
Simulation(domain; Δt=0.001, stop_time=1.0)     # → Transient
```

### Steady-State Path

```
Domain → make_system(model, domain) → (A, b)
       → make_bc!(A, b, ...) for each BC
       → LinearSolve.solve(LinearProblem(A, b))
       → solution vector
```

The model builds the system matrix `A` and RHS `b` (e.g., for heat: `A = α∇²`, `b = source`). Each BC modifies the rows of `A` and `b` corresponding to its surface's point indices.

### Transient Path

```
Domain → make_f(model, domain) → f_model(du, u, p, t)
       → make_bc(bc, ...) for each BC → f_bc(du, u, p, t)
       → ODEProblem(f_combined, u0, tspan)
       → OrdinaryDiffEq.solve(prob, solver; dt=Δt)
```

The model builds an in-place ODE function that computes `du/dt`. Each BC also returns an in-place function. They are composed into a single `f(du, u, p, t)` that is passed to OrdinaryDiffEq.jl.

## Extending: Adding a New Built-in Physics

To add a new physics (e.g., electrostatics) as a built-in, follow these steps:

### 1. Define the equation set trait

In `src/boundary_conditions/core/equation_set.jl`:

```julia
struct ElectrostaticEquations <: EquationSet end
```

### 2. Create BC aliases

Create `src/boundary_conditions/electrostatics.jl`:

```julia
# Dirichlet: prescribed voltage
Voltage(value) = PrescribedValue{ElectrostaticEquations}(value)

# Neumann: prescribed surface charge density
SurfaceCharge(σ) = PrescribedFlux{ElectrostaticEquations}(σ)

# Neumann: zero normal electric field
const Insulating = ZeroFlux{ElectrostaticEquations}
```

### 3. Create the physics model

Create `src/models/electrostatics.jl`:

```julia
struct Electrostatics{E} <: Solid
    ε::E  # permittivity
end

equation_set(::Type{<:Electrostatics}) = ElectrostaticEquations()
_num_vars(::Electrostatics, _) = 1

function make_system(model::Electrostatics, domain; kwargs...)
    # Assemble ε∇²ϕ = -ρ_free
    ...
end
```

### 4. Register in the module

In `src/Macchiato.jl`:

```julia
export ElectrostaticEquations
include("boundary_conditions/electrostatics.jl")
export Voltage, SurfaceCharge, Insulating
include("models/electrostatics.jl")
export Electrostatics
```

The generic BC implementations (`PrescribedValue`, `PrescribedFlux`, `ZeroFlux`) handle all the matrix/ODE application logic automatically. No changes to the solver infrastructure are needed.

!!! tip "Custom PDEs don't need any of this"
    If you're defining a custom PDE outside the package, you don't need to define an equation set trait at all. Just create your model struct, implement `_num_vars` and `make_system`, and use the generic BC constructors (`PrescribedValue(0.0)`, etc.). See [Custom PDEs](@ref) for the minimal approach.
