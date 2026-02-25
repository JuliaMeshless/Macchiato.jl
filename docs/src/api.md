# API Reference

```@meta
CurrentModule = MeshlessMultiphysics
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

### Physics Domain Traits

```@docs
PhysicsDomain
EnergyPhysics
FluidPhysics
WallPhysics
MechanicsPhysics
physics_domain
is_compatible
```

### Generic BC Types

```@docs
PrescribedValue
PrescribedFlux
ZeroFlux
```

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
