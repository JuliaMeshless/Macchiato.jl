# Getting Started

This tutorial walks through a complete 2D steady-state heat conduction simulation — from geometry to results. Along the way, it explains the key types and how they compose.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/JuliaMeshless/WhatsThePoint.jl")
Pkg.add(url="https://github.com/JuliaMeshless/RadialBasisFunctions.jl")
Pkg.add(url="https://github.com/JuliaMeshless/Macchiato.jl")
```

## Step 1: Define the Geometry

Every simulation starts with a point cloud — a set of scattered points that discretize the domain boundary and interior. WhatsThePoint.jl handles this.

```julia
using WhatsThePoint
using Unitful: m, °

# Create a 1m × 1m rectangle boundary with points and normals
part = PointBoundary(rectangle(1m, 1m)...)

# Split the continuous boundary into 4 named surfaces at 75° corners
split_surface!(part, 75°)
# This creates :surface1 (bottom), :surface2 (right), :surface3 (top), :surface4 (left)
```

Splitting at corners creates named surfaces so you can assign different boundary conditions to each edge. Now fill the interior:

```julia
# Discretize: place interior points at ~1/33 m spacing
dx = 1/33 * m
cloud = discretize(part, ConstantSpacing(dx), alg=VanDerSandeFornberg())

# Optional: repel points toward uniform spacing
cloud, _ = repel(cloud, ConstantSpacing(dx); α=dx/20, max_iters=500)
```

The resulting `PointCloud` contains both boundary points (organized by surface) and interior (volume) points.

## Step 2: Define the Physics Model

Physics models define the PDE being solved. For heat conduction, use [`SolidEnergy`](@ref):

```julia
using Macchiato

model = SolidEnergy(k=1.0, ρ=1.0, cₚ=1.0)
```

This defines the heat equation with thermal conductivity `k`, density `ρ`, and specific heat `cₚ`.

!!! tip "Solving your own PDE?"
    `SolidEnergy` is one of several built-in models, but you can define a model for **any** PDE. See the [Custom PDEs](@ref) tutorial to learn how.

## Step 3: Define Boundary Conditions

Boundary conditions are specified as a `Dict` mapping surface names to BC objects:

```julia
bcs = Dict(
    :surface1 => Temperature(0.0),    # bottom: T = 0
    :surface2 => Temperature(0.0),    # right:  T = 0
    :surface3 => Temperature(100.0),  # top:    T = 100
    :surface4 => Temperature(0.0)     # left:   T = 0
)
```

[`Temperature`](@ref) is a Dirichlet BC — it prescribes the value directly. The BC system is organized by mathematical type:

| Type | Meaning | Energy Examples |
|------|---------|-----------------|
| **Dirichlet** | Prescribes value: `u = g` | [`Temperature`](@ref) |
| **Neumann** | Prescribes flux: `∂u/∂n = q` | [`HeatFlux`](@ref), [`Adiabatic`](@ref) |
| **Robin** | Mixed: `α u + β ∂u/∂n = g` | [`Convection`](@ref) |

All BCs accept either a constant value or a function `(x, t) -> value` for spatially or temporally varying conditions:

```julia
# Spatially varying temperature
Temperature((x, t) -> 100.0 * sin(π * x[1]))

# Insulated boundary (zero heat flux)
Adiabatic()

# Convective cooling: h=10, k=1, T_ambient=25
Convection(10.0, 1.0, 25.0)
```

See the [API Reference](@ref) for the complete list of boundary condition types.

!!! tip "Named BCs are aliases for generic types"
    `Temperature`, `HeatFlux`, and `Adiabatic` are constructor functions that create [`PrescribedValue`](@ref), [`PrescribedFlux`](@ref), and [`ZeroFlux`](@ref) instances with a physics-meaningful display name. When defining a [custom PDE](@ref "Custom PDEs"), you can use the generic constructors directly — `PrescribedValue(0.0)`, `PrescribedFlux(1.0)`, `ZeroFlux()` — with no trait boilerplate required.

## Step 4: Create the Domain

The [`Domain`](@ref) ties geometry, boundary conditions, and model together:

```julia
domain = Domain(cloud, bcs, model)
```

The `Domain` validates that every BC key matches a surface in the point cloud.

## Step 5: Create and Run the Simulation

```julia
sim = Simulation(domain)
run!(sim)
```

[`Simulation`](@ref) automatically detects the simulation mode:
- **No `Δt` provided** → steady-state: assembles `Ax = b` and solves with LinearSolve.jl
- **`Δt` provided** → transient: builds an ODE right-hand side and integrates with OrdinaryDiffEq.jl

For steady-state, `run!` calls `LinearSolve.LinearProblem(domain)` internally, which:
1. Asks the model to build its system matrix and RHS via `make_system`
2. Applies each BC by modifying the appropriate matrix rows
3. Solves the sparse linear system

## Step 6: Extract and Visualize Results

```julia
# Extract the temperature field
T = temperature(sim)

# Export to VTK for ParaView visualization
using WhatsThePoint: points
exportvtk("heat_solution", points(cloud), [T], ["T"])
```

Each physics model has dedicated field extraction functions:
- `temperature(sim)` — for `SolidEnergy`
- `displacement(sim)` — for `LinearElasticity`, returns `(ux, uy)` or `(ux, uy, uz)`
- `velocity(sim)`, `pressure(sim)` — for `IncompressibleNavierStokes`

## Going Transient

Converting the same problem to a transient simulation requires minimal changes — add a time step and stop time, and set initial conditions:

```julia
# Same domain as before
sim = Simulation(domain; Δt=0.001, stop_time=1.0)

# Set initial temperature to 0 everywhere
set!(sim, T=0.0)

# Or use a function of position
set!(sim, T=x -> 50.0 * exp(-10 * ((x[1]-0.5)^2 + (x[2]-0.5)^2)))

# Run transient simulation (time-steps with Tsit5 by default)
run!(sim)

T_final = temperature(sim)
```

You can also attach callbacks and output writers for monitoring and saving intermediate results:

```julia
# Print progress every 100 iterations
sim.callbacks[:progress] = Callback(
    s -> println("t = $(s.time), iter = $(s.iteration)"),
    IterationInterval(100)
)

# Save VTK files every 0.1 time units
sim.output_writers[:vtk] = VTKOutputWriter("results/heat", schedule=TimeInterval(0.1))

run!(sim)
```

See [`Callback`](@ref), [`VTKOutputWriter`](@ref), and [`JLD2OutputWriter`](@ref) in the API reference for details.
