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
include("models/mechanics_3d.jl")
export LinearElasticity, lame_parameters
export LinearElasticity3D, lame_parameters_3d
export assemble_elasticity_3d_from_weights

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

#################### Optimization / AD ####################
include("optimization/solve_ift.jl")
include("optimization/manual_adjoint.jl")
include("optimization/manual_adjoint_3d.jl")
include("optimization/design_space.jl")
include("optimization/design_space_3d.jl")
include("optimization/morph_extension.jl")
include("optimization/indicators.jl")
export apply_dirichlet!
export active_dofs, build_dirichlet_info
export extract_weight_sensitivities_elasticity!, allocate_weight_gradients
export assemble_elasticity_from_weights
export shape_gradient
export TractionLayout, build_traction_layout, apply_traction!
export extract_neumann_sensitivities!
export extract_load_sensitivities!
export NormalJacobian, polyline_normals
export update_traction_coeffs!, extract_normal_sensitivities!

# 3D adjoint
export shape_gradient_3d
export rigid_body_modes_3d
export extract_weight_sensitivities_elasticity_3d!
export TractionLayout3D, build_traction_layout_3d, apply_traction_3d!
export extract_neumann_sensitivities_3d!
export NormalJacobian3D, triangle_normals
export update_traction_coeffs_3d!, extract_normal_sensitivities_3d!

# Design space
export AbstractDesignSpace, FourierModes
export boundary_points, radius_at, r0_for_area
export contract_gradient, sob_weight
export calibrate_fourier, fit_start_fourier

# Design space (3D)
export SphericalHarmonicModes, icosphere, real_sph_harm, sph_lm_list
export n_design_vars, radii, surface_faces, directions
export cavity_volume, volume_gradient, project_volume, sph_sob_weight
export fit_ellipsoid_sh, calibrate_sph, with_coeffs, set_sobolev

# Extension (morph)
export AbstractExtension, LaplaceExtension, build_laplace_extension
export morph, morph_transpose

# Indicator registry
export Indicator, trips, assess
export measure_morph_drift, measure_min_gap
export measure_spacing_cv, measure_boundary_cv
export measure_min_sep, measure_stencil_growth

#################### Operators ####################
abstract type AbstractOperator end
export AbstractOperator

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

# utils
export findmin_turbo

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
