# ============================================================================
# Mathematical Boundary Condition Type Hierarchy
# ============================================================================

"""
    AbstractBoundaryCondition

Base abstract type for all boundary conditions.
All BCs must subtype one of: `Dirichlet`, `Neumann`, or `Robin`.
"""
abstract type AbstractBoundaryCondition end

"""
    Dirichlet <: AbstractBoundaryCondition

Essential boundary conditions that prescribe values at the boundary.
Examples: `Temperature`, `VelocityInlet`, `Displacement`.
"""
abstract type Dirichlet <: AbstractBoundaryCondition end

"""
    DerivativeBoundaryCondition <: AbstractBoundaryCondition

Abstract type for BCs involving derivatives (Neumann and Robin).
"""
abstract type DerivativeBoundaryCondition <: AbstractBoundaryCondition end

"""
    Neumann <: DerivativeBoundaryCondition

Natural boundary conditions that prescribe normal derivatives: ∂u/∂n = g.
Examples: `HeatFlux`, `Adiabatic`, `VelocityOutlet`.
"""
abstract type Neumann <: DerivativeBoundaryCondition end

"""
    Robin <: DerivativeBoundaryCondition

Mixed boundary conditions combining value and derivative: β·∂u/∂n + α·u = g.
Example: `Convection`.
"""
abstract type Robin <: DerivativeBoundaryCondition end

# ============================================================================
# Type Dispatch Helpers
# ============================================================================

"""
    bc_family(T::Type{<:AbstractBoundaryCondition})

Returns the BC family (Dirichlet or DerivativeBoundaryCondition) for dispatching.
"""
bc_family(::Type{<:Dirichlet}) = Dirichlet
bc_family(::Type{<:DerivativeBoundaryCondition}) = DerivativeBoundaryCondition

"""
    bc_type(T::Type{<:AbstractBoundaryCondition})

Returns the specific mathematical type (Dirichlet, Neumann, or Robin).
"""
bc_type(::Type{<:Dirichlet}) = Dirichlet
bc_type(::Type{<:Neumann}) = Neumann
bc_type(::Type{<:Robin}) = Robin
