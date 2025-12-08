# ============================================================================
# Mathematical Boundary Condition Type Hierarchy
# ============================================================================

"""
    AbstractBoundaryCondition

Base abstract type for all boundary conditions.

All boundary conditions must be a subtype of one of the mathematical BC types:
- `Dirichlet`: Essential boundary conditions (prescribe values)
- `Neumann`: Natural boundary conditions (prescribe fluxes/derivatives)
- `Robin`: Mixed boundary conditions (combination of value and derivative)

# Implementing a New BC
To create a new boundary condition:
1. Choose the appropriate mathematical type (Dirichlet, Neumann, or Robin)
2. Define a struct that subtypes it
3. Implement the BC value accessor: `(bc::YourBC)() = ...`
4. Define the physics domain trait: `physics_domain(::Type{<:YourBC}) = ...`

# Example
```julia
struct MyCustomBC{T} <: Dirichlet
    value::T
end

(bc::MyCustomBC)() = bc.value
physics_domain(::Type{<:MyCustomBC}) = EnergyPhysics()
```
"""
abstract type AbstractBoundaryCondition end

"""
    Dirichlet <: AbstractBoundaryCondition

Abstract type for Dirichlet (essential) boundary conditions.

Dirichlet BCs directly prescribe the value of the solution at the boundary.
Examples: prescribed temperature, velocity, displacement, etc.

# Implementation
Dirichlet BCs are applied by modifying the linear system:
- Set row i of matrix A to [0, ..., 0, 1, 0, ..., 0] (1 at position i)
- Set row i of vector b to the prescribed value

# Examples of Dirichlet BCs
- `Temperature`: Prescribed temperature in thermal problems
- `VelocityInlet`: Prescribed velocity at fluid inlet
- `Displacement`: Prescribed displacement in structural problems
"""
abstract type Dirichlet <: AbstractBoundaryCondition end

"""
    DerivativeBoundaryCondition <: AbstractBoundaryCondition

Abstract type for boundary conditions that involve derivatives (Neumann and Robin).

These BCs require computing derivatives normal to the boundary surface.
"""
abstract type DerivativeBoundaryCondition <: AbstractBoundaryCondition end

"""
    Neumann <: DerivativeBoundaryCondition

Abstract type for Neumann (natural) boundary conditions.

Neumann BCs prescribe the derivative (flux) normal to the boundary: ∂u/∂n = g

# Implementation
Neumann BCs are applied by replacing the row in the linear system with
derivative weights that approximate ∂u/∂n.

# Examples of Neumann BCs
- `HeatFlux`: Prescribed heat flux in thermal problems (∂T/∂n = q/k)
- `Adiabatic`: Zero heat flux (∂T/∂n = 0)
- `Traction`: Prescribed traction in structural problems
- `VelocityOutlet`: Zero-gradient velocity outlet (∂v/∂n = 0)
"""
abstract type Neumann <: DerivativeBoundaryCondition end

"""
    Robin <: DerivativeBoundaryCondition

Abstract type for Robin (mixed) boundary conditions.

Robin BCs are a linear combination of the value and its normal derivative:
    β·∂u/∂n + α·u = g

# Implementation
Robin BCs combine derivative weights with a diagonal contribution:
- Derivative weights are scaled by β
- A diagonal term α is added
- RHS is set to g

# Examples of Robin BCs
- `Convection`: Heat transfer at boundary (h·T + k·∂T/∂n = h·T∞)
"""
abstract type Robin <: DerivativeBoundaryCondition end

# ============================================================================
# Type Dispatch Helpers
# ============================================================================

"""
    bc_family(T::Type{<:AbstractBoundaryCondition})

Returns the family of a boundary condition type (Dirichlet or DerivativeBoundaryCondition).

This is used for dispatching to the appropriate implementation method.

# Examples
```julia
bc_family(Temperature{Float64})  # Returns Dirichlet
bc_family(Adiabatic)             # Returns DerivativeBoundaryCondition
bc_family(Convection{...})       # Returns DerivativeBoundaryCondition
```
"""
bc_family(::Type{<:Dirichlet}) = Dirichlet
bc_family(::Type{<:DerivativeBoundaryCondition}) = DerivativeBoundaryCondition

"""
    bc_type(T::Type{<:AbstractBoundaryCondition})

Returns the specific mathematical type of a boundary condition (Dirichlet, Neumann, or Robin).

This provides finer-grained type information for dispatch.

# Examples
```julia
bc_type(Temperature{Float64})  # Returns Dirichlet
bc_type(Adiabatic)             # Returns Neumann
bc_type(Convection{...})       # Returns Robin
```
"""
bc_type(::Type{<:Dirichlet}) = Dirichlet
bc_type(::Type{<:Neumann}) = Neumann
bc_type(::Type{<:Robin}) = Robin
