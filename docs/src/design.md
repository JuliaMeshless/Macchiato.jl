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
├── SolidEnergy              # Heat equation
├── Solid
│   └── LinearElasticity     # Navier-Cauchy equations
└── Fluid
    └── IncompressibleNavierStokes

AbstractSimulationMode
├── Steady
└── Transient
```

`AbstractModel` represents **any PDE** — the built-in subtypes (`SolidEnergy`, `LinearElasticity`, etc.) are convenience models that ship with the package. You can define your own model for any equation; see [Custom PDEs](@ref) for a complete walkthrough.

Every model must implement:
- `_num_vars(::MyModel, dim)` — number of solution variables (1 for scalar, `dim` for vector, `dim+1` for velocity+pressure)
- `make_system(::MyModel, domain)` — assemble `(A, b)` for steady-state *(required for steady-state)*
- `make_f(::MyModel, domain)` — return `f(du, u, p, t)` for transient ODE integration *(required for transient)*

### Boundary Conditions

```
AbstractBoundaryCondition
├── Dirichlet                  # u = g
│   ├── PrescribedValue{F}    # Generic scalar BC
│   └── Displacement{F}       # Mechanics-specific (vector-valued)
└── DerivativeBoundaryCondition
    ├── Neumann                # ∂u/∂n = q
    │   ├── PrescribedFlux{F}
    │   ├── ZeroFlux
    │   └── Traction{F}       # Mechanics-specific (vector-valued)
    └── Robin                  # α u + β ∂u/∂n = g
        └── Convection
```

## Boundary Condition System

The BC system separates **mathematical type** from **physics naming** using a two-layer design:

### Mathematical Layer

The mathematical type determines *how* the BC modifies the linear system:
- **Dirichlet** → replace the row with identity: `A[i,:] = eᵢ`, `b[i] = g`
- **Neumann** → replace the row with normal derivative weights: `A[i,:] = ∂/∂n weights`, `b[i] = q`
- **Robin** → combine value and derivative: `A[i,:] = α·eᵢ + β·∂/∂n weights`, `b[i] = g`

### Generic Types and Named Aliases

The generic BC types carry a `name::Symbol` field used only for display:

```julia
struct PrescribedValue{F <: Function} <: Dirichlet
    f::F
    name::Symbol    # display label, e.g. :Temperature
end

struct PrescribedFlux{F <: Function} <: Neumann
    f::F
    name::Symbol
end

struct ZeroFlux <: Neumann
    name::Symbol
end
```

For custom PDEs, the generic constructors use a default name:

```julia
PrescribedValue(0.0)    # name = :PrescribedValue
PrescribedFlux(1.0)     # name = :PrescribedFlux
ZeroFlux()              # name = :ZeroFlux
```

Built-in physics aliases are constructor functions that set a physics-meaningful name:

```julia
# Energy aliases
Temperature(value) = PrescribedValue((x, t) -> value, :Temperature)
HeatFlux(flux)     = PrescribedFlux((x, t) -> flux, :HeatFlux)
Adiabatic()        = ZeroFlux(:Adiabatic)

# Fluid aliases
VelocityInlet(v)   = PrescribedValue((x, t) -> v, :VelocityInlet)
VelocityOutlet()   = ZeroFlux(:VelocityOutlet)

# Wall BC
Wall()             = PrescribedValue((x, t) -> 0.0, :Wall)
Wall(v)            = PrescribedValue((x, t) -> v, :Wall)

# Pressure outlet
PressureOutlet(p)  = PrescribedValue((x, t) -> p, :PressureOutlet)
```

The `name` field affects only `Base.show` — dispatch never inspects it. A `Temperature(300.0)` and a `PrescribedValue(300.0)` produce functionally identical objects. This means adding a new physics only requires defining new alias constructors — zero changes to the BC application machinery.

Mechanics BCs (`Displacement`, `Traction`) are standalone structs rather than aliases because they carry vector-valued output `(ux, uy)` / `(tx, ty)`.

## Steady-State vs Transient

The [`Simulation`](@ref) constructor takes an explicit mode argument:

```julia
Simulation(domain)                                          # → Steady (default)
Simulation(domain, Transient(Δt=0.001, stop_time=1.0))     # → Transient
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

### 1. Create BC aliases

Create `src/boundary_conditions/electrostatics.jl`:

```julia
# Dirichlet: prescribed voltage
Voltage(value::Number) = PrescribedValue((x, t) -> value, :Voltage)
Voltage(f::Function)   = PrescribedValue(f, :Voltage)

# Neumann: prescribed surface charge density
SurfaceCharge(σ::Number) = PrescribedFlux((x, t) -> σ, :SurfaceCharge)
SurfaceCharge(f::Function) = PrescribedFlux(f, :SurfaceCharge)

# Neumann: zero normal electric field
Insulating() = ZeroFlux(:Insulating)
```

### 2. Create the physics model

Create `src/models/electrostatics.jl`:

```julia
struct Electrostatics{E} <: Solid
    ε::E  # permittivity
end

_num_vars(::Electrostatics, _) = 1

function make_system(model::Electrostatics, domain; kwargs...)
    # Assemble ε∇²ϕ = -ρ_free
    ...
end
```

### 3. Register in the module

In `src/Macchiato.jl`:

```julia
include("boundary_conditions/electrostatics.jl")
export Voltage, SurfaceCharge, Insulating
include("models/electrostatics.jl")
export Electrostatics
```

The generic BC implementations (`PrescribedValue`, `PrescribedFlux`, `ZeroFlux`) handle all the matrix/ODE application logic automatically. No changes to the solver infrastructure are needed.

!!! tip "Custom PDEs don't need any of this"
    If you're defining a custom PDE outside the package, you don't need to create alias constructors at all. Just create your model struct, implement `_num_vars` and `make_system`, and use the generic BC constructors (`PrescribedValue(0.0)`, etc.). See [Custom PDEs](@ref) for the minimal approach.
