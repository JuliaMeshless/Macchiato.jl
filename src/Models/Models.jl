module Models

using MeshlessMultiphysics
using ..Domains

using PointClouds
using RadialBasisFunctions
using LinearAlgebra

abstract type Fluid <: AbstractModel end
abstract type Solid <: AbstractModel end

include("time.jl")
include("fluids.jl")
include("energy.jl")

export AbstractViscosity, NewtonianViscosity, CarreauYasudaViscosity

export AbstractModel
export IncompressibleNavierStokes
export SolidEnergy
export make_f, make_system
export _num_vars

end # module
