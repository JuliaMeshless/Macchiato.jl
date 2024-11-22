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
using Meshes: ∠

include("walls.jl")
include("fluids.jl")
include("energy.jl")

# abstract and supertypes
export FluidBoundary, EnergyBoundary

# walls
export Wall

# fluids
export VelocityInlet, PressureOutlet

# energy
export Adiabatic, Temperature, HeatFlux, Convection

export make_bc, make_bc!

end # module
