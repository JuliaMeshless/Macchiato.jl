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

Every model must implement:
- `physics_domain(::Type{<:MyModel})` — returns a `PhysicsDomain` trait for BC compatibility checking
- `_num_vars(::MyModel, dim)` — number of solution variables (1 for scalar, `dim` for vector, `dim+1` for velocity+pressure)
- `make_system(::MyModel, domain)` — assemble `(A, b)` for steady-state
- `make_f(::MyModel, domain)` — return `f(du, u, p, t)` for transient ODE integration

### Boundary Conditions

```
AbstractBoundaryCondition
├── Dirichlet                    # u = g
│   ├── PrescribedValue{P, F}   # Generic: parameterized by PhysicsDomain P
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

The BC system separates **mathematical type** from **physics domain** using a two-axis design:

### Mathematical Axis

The mathematical type determines *how* the BC modifies the linear system:
- **Dirichlet** → replace the row with identity: `A[i,:] = eᵢ`, `b[i] = g`
- **Neumann** → replace the row with normal derivative weights: `A[i,:] = ∂/∂n weights`, `b[i] = q`
- **Robin** → combine value and derivative: `A[i,:] = α·eᵢ + β·∂/∂n weights`, `b[i] = g`

### Physics Domain Axis

The physics domain determines *which models* a BC is compatible with:

```julia
abstract type PhysicsDomain end
struct EnergyPhysics    <: PhysicsDomain end
struct FluidPhysics     <: PhysicsDomain end
struct MechanicsPhysics <: PhysicsDomain end
struct WallPhysics      <: PhysicsDomain end  # compatible with Energy, Fluids, Mechanics
```

Compatibility is checked at `Domain` construction via `is_compatible(bc_domain, model_domain)`. Same-domain pairs are always compatible. `WallPhysics` is compatible with all other domains.

### Generic Types and Physics Aliases

The generic BC types are parameterized by physics domain:

```julia
struct PrescribedValue{P <: PhysicsDomain, F <: Function} <: Dirichlet ... end
struct PrescribedFlux{P <: PhysicsDomain, F <: Function} <: Neumann ... end
struct ZeroFlux{P <: PhysicsDomain} <: Neumann end
```

Physics-specific aliases provide user-friendly names:

```julia
# Energy aliases
Temperature(value) = PrescribedValue{EnergyPhysics}(value)
HeatFlux(flux)     = PrescribedFlux{EnergyPhysics}(flux)
const Adiabatic    = ZeroFlux{EnergyPhysics}

# Fluid aliases
VelocityInlet(v)     = PrescribedValue{FluidPhysics}(v)
const VelocityOutlet = ZeroFlux{FluidPhysics}
```

This design means adding a new physics domain requires zero changes to the BC application machinery — you just define new aliases using the existing generic types.

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

## Extending: Adding a New Physics Domain

To add a new physics (e.g., electrostatics), follow these steps:

### 1. Define the physics domain trait

In `src/boundary_conditions/core/physics_traits.jl`:

```julia
struct ElectrostaticPhysics <: PhysicsDomain end
is_compatible(::ElectrostaticPhysics, ::ElectrostaticPhysics) = true
```

### 2. Create BC aliases

Create `src/boundary_conditions/electrostatics.jl`:

```julia
# Dirichlet: prescribed voltage
Voltage(value) = PrescribedValue{ElectrostaticPhysics}(value)

# Neumann: prescribed surface charge density
SurfaceCharge(σ) = PrescribedFlux{ElectrostaticPhysics}(σ)

# Neumann: zero normal electric field
const Insulating = ZeroFlux{ElectrostaticPhysics}
```

### 3. Create the physics model

Create `src/models/electrostatics.jl`:

```julia
struct Electrostatics{E} <: Solid
    ε::E  # permittivity
end

physics_domain(::Type{<:Electrostatics}) = ElectrostaticPhysics()
_num_vars(::Electrostatics, _) = 1

function make_system(model::Electrostatics, domain; kwargs...)
    # Assemble ε∇²ϕ = -ρ_free
    ...
end
```

### 4. Register in the module

In `src/Macchiato.jl`:

```julia
export ElectrostaticPhysics
include("boundary_conditions/electrostatics.jl")
export Voltage, SurfaceCharge, Insulating
include("models/electrostatics.jl")
export Electrostatics
```

The generic BC implementations (`PrescribedValue`, `PrescribedFlux`, `ZeroFlux`) handle all the matrix/ODE application logic automatically. No changes to the solver infrastructure are needed.
