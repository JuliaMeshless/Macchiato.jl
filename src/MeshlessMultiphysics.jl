module MeshlessMultiphysics

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
using LinearAlgebra
using RadialBasisFunctions
using SparseArrays
using OrdinaryDiffEq
using LinearSolve
using OhMyThreads
using WriteVTK

include("utils.jl")

#################### Abstract Types ####################
abstract type AbstractModel end

#################### Boundary Conditions ####################
include("boundary_conditions/boundary_derivatives.jl")
include("boundary_conditions/boundary_conditions.jl")

export AbstractBoundaryCondition

#################### Domains ####################
include("domain.jl")
export Domain

include("boundary_conditions/walls.jl")
export Wall

include("boundary_conditions/fluids.jl")
export VelocityInlet, PressureOutlet

include("boundary_conditions/energy.jl")
export Adiabatic, Temperature, HeatFlux, Convection

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

# test funcs
export node_drop, findmin_turbo
export cov, make_memory_contiguous, ranges_from_permutation, permute!

# utils
export findmin_turbo

function __init__()
    threads = Threads.nthreads()
    if threads > 1
        @info "MeshlessMultiphysics will use $threads threads"
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
