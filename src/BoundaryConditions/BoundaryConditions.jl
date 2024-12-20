module BoundaryConditions

using MeshlessMultiphysics
using ..Domains

using PointClouds
using RadialBasisFunctions
using StaticArrays
using LinearAlgebra
using SparseArrays
using OhMyThreads
using LoopVectorization
using Meshes: ∠, 𝔼

include("../utils.jl")
include("walls.jl")
include("fluids.jl")
include("energy.jl")

# abstract and supertypes
export FluidBoundaryCondition, EnergyBoundaryCondition

# walls
export Wall

# fluids
export VelocityInlet, PressureOutlet

# energy
export Adiabatic, Temperature, HeatFlux, Convection

export make_bc, make_bc!

end # module
