import RadialBasisFunctions: _build_weights, Partial, MixedPartial

"""
    extract_weight_sensitivities_elasticity!(
        ΔW_d2x, ΔW_d2y, ΔW_d2xy,
        W_template, η, u, active, λstar, μ;
        interior_rows = nothing,
    )

Step 3 of the manual adjoint for 2D linear elasticity. Given the global adjoint
vector `η = Aᵀ \\ ∂L/∂u` and forward solution `u`, populate the gradients of
the three weight matrices that assemble A:

    A11 = c1·W_d2x + c2·W_d2y     (c1 = λstar + 2μ, c2 = μ)
    A12 = c3·W_d2xy               (c3 = λstar + μ)
    A22 = c2·W_d2x + c1·W_d2y

The IFT gives `ΔA = -η · uᵀ` at active rows. Per-DOF gating (not per-point AND
— see plan_manual_adjoint.md §Corrections §1):

    ΔA11[i,j] = active[i]   ? -η[i]   · u[j]   : 0
    ΔA22[i,j] = active[i+N] ? -η[i+N] · u[j+N] : 0
    ΔA12[i,j] = active[i]   ? -η[i]   · u[j+N] : 0
    ΔA21[i,j] = active[i+N] ? -η[i+N] · u[j]   : 0

`W_template` provides the shared sparsity pattern; the three ΔW outputs must
be pre-allocated with that exact pattern (see `allocate_weight_gradients`).

`interior_rows` is an optional `BitVector` of length N to exclude Neumann rows
from the main extraction loop (Phase B). Default `nothing` ⇒ visit all rows.
"""
function extract_weight_sensitivities_elasticity!(
    ΔW_d2x::SparseMatrixCSC{Float64, Int},
    ΔW_d2y::SparseMatrixCSC{Float64, Int},
    ΔW_d2xy::SparseMatrixCSC{Float64, Int},
    W_template::SparseMatrixCSC{Float64, Int},
    η::AbstractVector{Float64},
    u::AbstractVector{Float64},
    active::AbstractVector{Bool},
    λstar::Real,
    μ::Real;
    interior_rows::Union{Nothing, BitVector} = nothing,
)
    N = size(W_template, 1)
    @assert length(η) == 2N
    @assert length(u) == 2N
    @assert length(active) == 2N

    c1 = λstar + 2μ
    c2 = μ
    c3 = λstar + μ

    fill!(ΔW_d2x.nzval, 0.0)
    fill!(ΔW_d2y.nzval, 0.0)
    fill!(ΔW_d2xy.nzval, 0.0)

    @inbounds for j in 1:N
        for idx in nzrange(W_template, j)
            i = W_template.rowval[idx]

            if interior_rows !== nothing && !interior_rows[i]
                continue
            end

            t11 = active[i]      ? -η[i]     * u[j]     : 0.0
            t22 = active[i + N]  ? -η[i + N] * u[j + N] : 0.0
            t12 = active[i]      ? -η[i]     * u[j + N] : 0.0
            t21 = active[i + N]  ? -η[i + N] * u[j]     : 0.0

            ΔW_d2x.nzval[idx]  = c1 * t11 + c2 * t22
            ΔW_d2y.nzval[idx]  = c2 * t11 + c1 * t22
            ΔW_d2xy.nzval[idx] = c3 * (t12 + t21)
        end
    end

    return nothing
end

"""
    allocate_weight_gradients(W_template)

Allocate `(ΔW_d2x, ΔW_d2y, ΔW_d2xy)` as zero `SparseMatrixCSC` matrices sharing
`W_template`'s sparsity pattern (independent colptr/rowval copies, so mutating
one ΔW won't affect another or the template).
"""
function allocate_weight_gradients(W_template::SparseMatrixCSC{Float64, Int})
    m, n = size(W_template)
    nnz = length(W_template.nzval)
    mk() = SparseMatrixCSC(m, n, copy(W_template.colptr),
                                  copy(W_template.rowval),
                                  zeros(nnz))
    return mk(), mk(), mk()
end

"""
    assemble_elasticity_from_weights(W_d2x, W_d2y, W_d2xy, N, λstar, μ)

Build the 2N×2N plane-stress Navier system from pre-computed weight matrices.
Mirrors `make_system_differentiable` but takes the already-evaluated `W`
matrices, so the forward pass of the manual adjoint can reuse them in Step 3
without rebuilding. Outside Mooncake's trace ⇒ we can use sparse-sparse `+`
and `hvcat` freely.
"""
function assemble_elasticity_from_weights(
    W_d2x::SparseMatrixCSC{Float64, Int},
    W_d2y::SparseMatrixCSC{Float64, Int},
    W_d2xy::SparseMatrixCSC{Float64, Int},
    N::Int,
    λstar::Real,
    μ::Real,
)
    c1 = λstar + 2μ
    c2 = μ
    c3 = λstar + μ

    A11 = c1 * W_d2x + c2 * W_d2y
    A12 = c3 * W_d2xy
    A22 = c2 * W_d2x + c1 * W_d2y

    return [A11 A12; A12 A22]
end

# ============================================================================
# Mooncake-extension stubs (real implementations in MacchiatoMooncakeExt)
# ============================================================================

function build_weight_rule(args...; kwargs...)
    return error("Mooncake.jl must be loaded to use `build_weight_rule`. " *
                 "Run `using Mooncake` first.")
end

function apply_weight_rule(args...; kwargs...)
    return error("Mooncake.jl must be loaded to use `apply_weight_rule`. " *
                 "Run `using Mooncake` first.")
end

function shape_gradient_dirichlet(args...; kwargs...)
    return error("Mooncake.jl must be loaded to use `shape_gradient_dirichlet`. " *
                 "Run `using Mooncake` first.")
end
