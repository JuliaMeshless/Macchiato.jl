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

### Time

```@docs
Time
Steady
Unsteady
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

### Equation Set Traits

```@docs
EquationSet
EnergyEquations
FluidEquations
UniversalEquations
MechanicsEquations
GenericEquations
equation_set
is_compatible
```

### Generic BC Types

```@docs
PrescribedValue
PrescribedFlux
ZeroFlux
```

!!! note
    These generic types work with **any** equation set — not just the built-in ones. For custom PDEs, use the unparameterized constructors directly: `PrescribedValue(0.0)`, `PrescribedFlux(1.0)`, `ZeroFlux()`. See [Custom PDEs](@ref) for a complete example.

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

## Callbacks and Schedules

```@docs
Callback
AbstractSchedule
IterationInterval
TimeInterval
WallTimeInterval
SpecifiedTimes
```

## Output Writers

```@docs
AbstractOutputWriter
VTKOutputWriter
JLD2OutputWriter
```

## Solvers

```@docs
MultiphysicsProblem
LinearSolve.LinearProblem
```

## I/O

```@docs
exportvtk
savevtk!
```

## Operators

```@docs
Upwind
upwind
```
