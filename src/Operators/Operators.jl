module Operators

using MeshlessMultiphysics
using ..Domains

using PointClouds
using RadialBasisFunctions
using LinearAlgebra

abstract type AbstractOperator end

include("upwinding.jl")

export AbstractOperator, upwind

end # module
