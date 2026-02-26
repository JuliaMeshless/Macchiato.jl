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
using Unitful: ustrip
using StaticArrays
using LinearAlgebra
using RadialBasisFunctions
using SparseArrays
using OrdinaryDiffEq
using OhMyThreads
using WriteVTK
using JLD2

import LinearSolve

include("utils.jl")

#################### Abstract Types ####################
abstract type AbstractModel end

#################### Boundary Conditions ####################
include("boundary_conditions/numerical/derivatives.jl")
include("boundary_conditions/boundary_conditions.jl")

export AbstractBoundaryCondition
# Physics domain traits
export PhysicsDomain, EnergyPhysics, FluidPhysics, WallPhysics, MechanicsPhysics
export physics_domain, is_compatible
# Generic BC types
export FixedValue, Flux, ZeroGradient

#################### Domains ####################
include("domain.jl")
export Domain

include("boundary_conditions/walls.jl")
export Wall

include("boundary_conditions/fluids.jl")
export VelocityInlet, PressureOutlet, VelocityOutlet

include("boundary_conditions/energy.jl")
export Adiabatic, Temperature, HeatFlux, Convection

include("boundary_conditions/mechanics.jl")
export Displacement, Traction, TractionFree

export make_bc, make_bc!

#################### Models ####################
abstract type Fluid <: AbstractModel end
abstract type Solid <: AbstractModel end

export AbstractModel
export AbstractViscosity, NewtonianViscosity, CarreauYasudaViscosity

include("models/time.jl")

include("models/fluids.jl")
export IncompressibleNavierStokes

include("models/energy.jl")
export SolidEnergy

include("models/mechanics.jl")
export LinearElasticity, lame_parameters

export make_f, make_system
export _num_vars

#################### Solvers ####################
include("solve.jl")
export MultiphysicsProblem

#################### Operators ####################
abstract type AbstractOperator end
export AbstractOperator

include("upwinding.jl")
export upwind

#################### IO ####################
include("io.jl")
export exportvtk, savevtk!, save

#################### Simulation API ####################
include("callbacks.jl")
include("output_writers.jl")
include("set.jl")
include("simulation.jl")

export Simulation, run!, set!
export Callback, AbstractSchedule, IterationInterval, TimeInterval, WallTimeInterval, SpecifiedTimes
export AbstractOutputWriter, VTKOutputWriter, JLD2OutputWriter
export temperature, velocity, pressure, displacement

# test funcs
export node_drop, findmin_turbo
export cov, make_memory_contiguous, ranges_from_permutation, permute!

# utils
export findmin_turbo

function __init__()
    threads = Threads.nthreads()
    if threads > 1
        @info "Macchiato will use $threads threads"
    end

    if CUDA.has_cuda()
        @info "CUDA-enabled GPU(s) detected:"
        for dev in CUDA.devices()
            @info "$dev: $(CUDA.name(dev))"
        end

        CUDA.allowscalar(false)
    end
end

end
