module MeshlessMultiphysics

using CUDA
using LinearAlgebra
using LoopVectorization
using FileIO
using PointClouds

# define abstract types
abstract type AbstractBoundaryCondition end
abstract type AbstractModel end

export AbstractBoundaryCondition, AbstractModel

# include submodules
include("utils.jl")
include("Domains/Domains.jl")
include("BoundaryConditions/BoundaryConditions.jl")
include("Models/Models.jl")
include("Solvers/Solvers.jl")
include("Operators/Operators.jl")
#include("io.jl")

using .BoundaryConditions
using .Models
using .Domains

# io
export exportvtk, savevtk!, save

# test funcs
export node_drop, findmin_turbo
export cov, make_memory_contiguous, ranges_from_permutation, permute!

# Boundary Conditions
export FluidBoundaryCondition, EnergyBoundaryCondition
export Wall
export Temperature, HeatFlux, Convection, Adiabatic
export VelocityInlet, PressureOutlet

# Models
export AbstractViscosity, NewtonianViscosity, CarreauYasudaViscosity
export IncompressibleNavierStokes, SolidEnergy

# Domains
export Domain

# Operators
export AbstractOperator, upwind

# Solvers
export MultiphysicsProblem

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
