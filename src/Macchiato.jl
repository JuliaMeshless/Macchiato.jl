module Macchiato

using CUDA
using LinearAlgebra
using LoopVectorization
using FileIO
using Meshes: 𝔼, Manifold, ∠
using WhatsThePoint
using CoordRefSystems
using Accessors
using ProgressMeter
using Unitful
using StaticArrays
using RadialBasisFunctions
using SparseArrays
using NearestNeighbors: KDTree, knn
using OrdinaryDiffEq
using OhMyThreads
using WriteVTK
using JLD2

import LinearSolve

include("utils.jl")

# Default RBF-FD stencil size (nearest neighbors per node).
const DEFAULT_STENCIL_SIZE = 40

#################### Abstract Types ####################

"""
    AbstractModel

Abstract supertype for all PDE models. Subtype this to define a custom PDE.

See [Custom PDEs](@ref) for a complete walkthrough.
"""
abstract type AbstractModel end

#################### Boundary Conditions ####################
include("boundary_conditions/numerical/derivatives.jl")
include("boundary_conditions/boundary_conditions.jl")

export AbstractBoundaryCondition
export Dirichlet, DerivativeBoundaryCondition, Neumann, Robin
# Generic BC types
export PrescribedValue, PrescribedFlux, ZeroFlux

#################### Domains ####################
include("domain.jl")
export Domain
export add!, delete!

include("boundary_conditions/walls.jl")
export Wall

include("boundary_conditions/fluids.jl")
export VelocityInlet, PressureOutlet, VelocityOutlet

include("boundary_conditions/energy.jl")
export Adiabatic, Temperature, HeatFlux, Convection

include("boundary_conditions/mechanics.jl")
export Displacement, Traction, TractionFree

# Ghost-node machinery for transient Neumann/Robin BCs (needs α/β from energy.jl above)
include("boundary_conditions/numerical/ghost.jl")

export make_bc, make_bc!

#################### Models ####################
abstract type Fluid <: AbstractModel end
abstract type Solid <: AbstractModel end

export AbstractModel, Fluid, Solid
export AbstractViscosity, NewtonianViscosity, CarreauYasudaViscosity

include("models/time.jl")
export AbstractSimulationMode, Steady, Transient

include("models/fluids.jl")
export IncompressibleNavierStokes

include("models/energy.jl")
export SolidEnergy

include("models/mechanics.jl")
export LinearElasticity, lame_parameters

"""
    _num_vars(model::AbstractModel, dim) -> Int

Return the number of solution variables per point for `model` in `dim` dimensions.

Examples: 1 for scalar PDEs, `dim` for vector PDEs, `dim + 1` for velocity + pressure.
"""
function _num_vars end

"""
    make_f(model::AbstractModel, domain; kwargs...) -> f

Return an in-place ODE function `f(du, u, p, t)` for transient integration.

Required for transient simulations. Macchiato passes the returned function to OrdinaryDiffEq.jl.
"""
function make_f end

"""
    make_system(model::AbstractModel, domain; kwargs...) -> (A, b)

Assemble the system matrix `A` and right-hand side `b` for steady-state solving.

Required for steady-state simulations. Macchiato applies boundary conditions and solves `Ax = b`.
"""
function make_system end

export make_f, make_system
export _num_vars

#################### Solvers ####################
include("solve.jl")

#################### Operators ####################
include("upwinding.jl")
export upwind

#################### IO ####################
include("io.jl")
export exportvtk, savevtk!

#################### Simulation API ####################
include("set.jl")
include("simulation.jl")

export Simulation, run!, set!
export temperature, velocity, pressure, displacement

function __init__()
    threads = Threads.nthreads()
    if threads > 1
        @info "Macchiato will use $threads threads"
    end

    return if CUDA.has_cuda()
        @info "CUDA-enabled GPU(s) detected:"
        for dev in CUDA.devices()
            @info "$dev: $(CUDA.name(dev))"
        end

        CUDA.allowscalar(false)
    end
end

end
