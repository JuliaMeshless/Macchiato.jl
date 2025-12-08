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
"""
abstract type AbstractBoundaryCondition end

"""
    Dirichlet <: AbstractBoundaryCondition

Abstract type for Dirichlet (essential) boundary conditions.
Dirichlet BCs directly prescribe the value of the solution at the boundary.
"""
abstract type Dirichlet <: AbstractBoundaryCondition end

"""
    DerivativeBoundaryCondition <: AbstractBoundaryCondition

Abstract type for boundary conditions that involve derivatives (Neumann and Robin).
"""
abstract type DerivativeBoundaryCondition <: AbstractBoundaryCondition end

"""
    Neumann <: DerivativeBoundaryCondition

Abstract type for Neumann (natural) boundary conditions.
Neumann BCs prescribe the derivative (flux) normal to the boundary: ∂u/∂n = g
"""
abstract type Neumann <: DerivativeBoundaryCondition end

"""
    Robin <: DerivativeBoundaryCondition

Abstract type for Robin (mixed) boundary conditions.
Robin BCs are a linear combination of the value and its normal derivative:
    β·∂u/∂n + α·u = g
"""
abstract type Robin <: DerivativeBoundaryCondition end

# ============================================================================
# Type Dispatch Helpers
# ============================================================================

"""
    bc_family(T::Type{<:AbstractBoundaryCondition})

Returns the family of a boundary condition type (Dirichlet or DerivativeBoundaryCondition).
"""
bc_family(::Type{<:Dirichlet}) = Dirichlet
bc_family(::Type{<:DerivativeBoundaryCondition}) = DerivativeBoundaryCondition

"""
    bc_type(T::Type{<:AbstractBoundaryCondition})

Returns the specific mathematical type of a boundary condition (Dirichlet, Neumann, or Robin).
"""
bc_type(::Type{<:Dirichlet}) = Dirichlet
bc_type(::Type{<:Neumann}) = Neumann
bc_type(::Type{<:Robin}) = Robin
