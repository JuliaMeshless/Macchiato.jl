# API Reference

```@meta
CurrentModule = Macchiato
```

## Domain

```@docs
Domain
add!
delete!
```

## Models

### Energy

```@docs
SolidEnergy
```

### Mechanics

```@docs
LinearElasticity
lame_parameters
```

### Fluids

```@docs
IncompressibleNavierStokes
AbstractViscosity
NewtonianViscosity
CarreauYasudaViscosity
```

### Simulation Modes

```@docs
AbstractSimulationMode
Steady
Transient
```

## Model Interface

These are the functions to implement when defining a [custom PDE](@ref "Custom PDEs").

```@docs
AbstractModel
_num_vars
make_system
make_f
```

## Boundary Conditions

### Core Types

```@docs
AbstractBoundaryCondition
Dirichlet
Neumann
Robin
DerivativeBoundaryCondition
```

### Generic BC Types

```@docs
PrescribedValue
PrescribedFlux
ZeroFlux
```

!!! note
    These generic types work with **any** physics model. For custom PDEs, use them directly: `PrescribedValue(0.0)`, `PrescribedFlux(1.0)`, `ZeroFlux()`. See [Custom PDEs](@ref) for a complete example.

### Energy BCs

```@docs
Temperature
HeatFlux
Adiabatic
Convection
```

### Mechanics BCs

```@docs
Displacement
Traction
TractionFree
```

### Fluid BCs

```@docs
VelocityInlet
PressureOutlet
VelocityOutlet
```

### Wall BC

```@docs
Wall
```

## Simulation

```@docs
Simulation
run!
set!
```

## Field Extraction

```@docs
temperature
velocity
pressure
displacement
```

## Solvers

```@docs
LinearSolve.LinearProblem
```

## I/O

```@docs
exportvtk
savevtk!
```

## Operators

```@docs
upwind
```
