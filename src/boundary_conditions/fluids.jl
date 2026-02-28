# ============================================================================
# VelocityInlet (Dirichlet)
# ============================================================================

"""
    VelocityInlet(velocity)

Prescribed velocity at inlet. Value can be a Number or Function `(x, t) -> velocity`.
"""
VelocityInlet(v::Number) = PrescribedValue((x, t) -> v, :VelocityInlet)
VelocityInlet(f::Function) = PrescribedValue{typeof(f)}(f, :VelocityInlet)

# ============================================================================
# PressureOutlet (Dirichlet)
# ============================================================================

"""
    PressureOutlet(pressure)

Prescribed pressure at outlet. Value can be a Number or Function `(x, t) -> pressure`.
"""
PressureOutlet(v::Number) = PrescribedValue((x, t) -> v, :PressureOutlet)
PressureOutlet(f::Function) = PrescribedValue{typeof(f)}(f, :PressureOutlet)

# ============================================================================
# VelocityOutlet (Neumann with zero gradient)
# ============================================================================

"""
    VelocityOutlet()

Zero-gradient velocity outlet: ∂v/∂n = 0. Used for fully developed outflow.
"""
VelocityOutlet() = ZeroFlux(:VelocityOutlet)
