# ============================================================================
# VelocityInlet (Dirichlet)
# ============================================================================

"""
    VelocityInlet(velocity)

Prescribed velocity at inlet. Value can be a Number or Function `(x, t) -> velocity`.
"""
VelocityInlet(velocity::Number) = PrescribedValue{FluidPhysics}(velocity)
VelocityInlet(f::Function) = PrescribedValue{FluidPhysics}(f)

# Note: show method removed - can't distinguish VelocityInlet from PressureOutlet
# since both are PrescribedValue{FluidPhysics}

# ============================================================================
# PressureOutlet (Dirichlet)
# ============================================================================

"""
    PressureOutlet(pressure)

Prescribed pressure at outlet. Value can be a Number or Function `(x, t) -> pressure`.
"""
PressureOutlet(pressure::Number) = PrescribedValue{FluidPhysics}(pressure)
PressureOutlet(f::Function) = PrescribedValue{FluidPhysics}(f)

# Note: show method removed - can't distinguish VelocityInlet from PressureOutlet
# since both are PrescribedValue{FluidPhysics}

# ============================================================================
# VelocityOutlet (Neumann with zero gradient)
# ============================================================================

"""
    VelocityOutlet()

Zero-gradient velocity outlet: ∂v/∂n = 0. Used for fully developed outflow.
"""
const VelocityOutlet = ZeroFlux{FluidPhysics}

Base.show(io::IO, ::ZeroFlux{FluidPhysics}) = print(io, "VelocityOutlet")
