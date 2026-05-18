module MacchiatoMooncakeExt

import Macchiato
using Macchiato:
    PDESolveIFT, apply_dirichlet!,
    batch_overwrite_sparse_rows!,
    active_dofs, build_dirichlet_info,
    extract_weight_sensitivities_elasticity!,
    allocate_weight_gradients,
    assemble_elasticity_from_weights,
    LinearElasticity, lame_parameters,
    Simulation
using Mooncake
using RadialBasisFunctions: PHS, find_neighbors, _build_weights,
    Partial, MixedPartial
using SparseArrays
using LinearAlgebra
using StaticArrays: SVector

# apply_dirichlet! modifies A and b in-place with constant values that don't
# depend on any traced inputs. Its backward zeroes out the gradient for BC rows
# (which PDESolveIFT already excludes via active_dofs, but Mooncake needs to
# trace through sparse-matrix setindex! which may not have built-in rules).
Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{
    typeof(apply_dirichlet!),
    SparseMatrixCSC{Float64, Int},
    Vector{Float64},
    AbstractVector{Int},
    AbstractVector{<:Real},
}

function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(apply_dirichlet!)},
    A_cd::Mooncake.CoDual{SparseMatrixCSC{Float64, Int}},
    b_cd::Mooncake.CoDual{Vector{Float64}},
    dofs_cd,
    vals_cd,
)
    A    = Mooncake.primal(A_cd)
    b    = Mooncake.primal(b_cd)
    dofs = Mooncake.primal(dofs_cd)
    vals = Mooncake.primal(vals_cd)

    apply_dirichlet!(A, b, dofs, vals)
    dofs_set = Set(dofs)

    function apply_dirichlet_pb!!(::Mooncake.NoRData)
        # BC values are constants w.r.t. any traced input:
        # zero out gradient contributions at BC rows so they don't corrupt ΔA.
        ΔA_nzval = Mooncake.tangent(A_cd).data.nzval
        rows = rowvals(A)
        for j in 1:size(A, 2)
            for idx in nzrange(A, j)
                if rows[idx] in dofs_set
                    ΔA_nzval[idx] = 0.0
                end
            end
        end
        Δb = Mooncake.tangent(b_cd)
        for i in dofs
            Δb[i] = 0.0
        end
        return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData()
    end

    return Mooncake.zero_fcodual(nothing), apply_dirichlet_pb!!
end

Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{PDESolveIFT, SparseMatrixCSC{Float64, Int}, Vector{Float64}}

function Mooncake.rrule!!(
    solver_cd::Mooncake.CoDual{PDESolveIFT},
    A_cd::Mooncake.CoDual{SparseMatrixCSC{Float64, Int}},
    b_cd::Mooncake.CoDual{Vector{Float64}},
)
    solver = Mooncake.primal(solver_cd)
    A = Mooncake.primal(A_cd)
    b = Mooncake.primal(b_cd)

    F = lu(A)
    u = F \ b
    u_cd = Mooncake.zero_fcodual(u)

    function pde_solve_pb!!(::Mooncake.NoRData)
        Δu = Mooncake.tangent(u_cd)
        η = F' \ Δu

        Mooncake.tangent(b_cd) .+= η

        ΔA_nzval = Mooncake.tangent(A_cd).data.nzval
        rows = rowvals(A)
        for j in 1:size(A, 2)
            for idx in nzrange(A, j)
                i = rows[idx]
                if solver.active_dofs[i]
                    ΔA_nzval[idx] -= η[i] * u[j]
                end
            end
        end

        return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData()
    end

    return u_cd, pde_solve_pb!!
end

# ============================================================================
# batch_overwrite_sparse_rows! — generic sparse row assembly primitive
# ============================================================================
# Model-agnostic: overwrites specified rows of A with linear combinations of
# weight-matrix entries. The physics lives in `coeffs`, computed by the model
# callback outside this primitive. The backward pass redirects ΔA at the
# overwritten entries to Δweights, then zeros ΔA/Δb at those rows.

Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{
    typeof(batch_overwrite_sparse_rows!),
    SparseMatrixCSC{Float64, Int},
    Vector{Float64},
    Vector{Int},
    Vector{Int},
    Vector{Int},
    Vector{Int},
    Vector{SparseMatrixCSC{Float64, Int}},
    Vector{Float64},
    Vector{Int},
    Vector{Float64},
}

function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(batch_overwrite_sparse_rows!)},
    A_cd::Mooncake.CoDual{SparseMatrixCSC{Float64, Int}},
    b_cd::Mooncake.CoDual{Vector{Float64}},
    rows_cd,
    col_ptr_cd,
    a_cols_cd,
    w_cols_cd,
    weights_cd,
    coeffs_cd,
    weight_rows_cd,
    b_vals_cd,
)
    A  = Mooncake.primal(A_cd)
    b  = Mooncake.primal(b_cd)
    rows_vals  = Mooncake.primal(rows_cd)
    col_ptr_vals  = Mooncake.primal(col_ptr_cd)
    a_cols_vals  = Mooncake.primal(a_cols_cd)
    w_cols_vals  = Mooncake.primal(w_cols_cd)
    weights_vals = Mooncake.primal(weights_cd)
    coeffs_vals  = Mooncake.primal(coeffs_cd)
    weight_rows_vals = Mooncake.primal(weight_rows_cd)
    b_vals_vals = Mooncake.primal(b_vals_cd)

    batch_overwrite_sparse_rows!(
        A, b, rows_vals, col_ptr_vals, a_cols_vals, w_cols_vals,
        weights_vals, coeffs_vals, weight_rows_vals, b_vals_vals,
    )

    M = length(weights_vals)
    row_set = Set{Int}(rows_vals)

    function batch_overwrite_pb!!(::Mooncake.NoRData)
        ΔA_nzval = Mooncake.tangent(A_cd).data.nzval
        Δb = Mooncake.tangent(b_cd)
        rows_A = rowvals(A)

        # Collect Δweight nzval vectors
        ΔW_nzvals = [Mooncake.tangent(weights_cd[m]).data.nzval for m in 1:M]

        # Redirect ΔA to Δweights using transpose of coefficients
        for k in 1:length(rows_vals)
            global_row = rows_vals[k]
            wr = weight_rows_vals[k]
            for p in col_ptr_vals[k]:(col_ptr_vals[k + 1] - 1)
                a_col = a_cols_vals[p]
                w_col = w_cols_vals[p]
                idx_a = Macchiato._find_nzval(A, global_row, a_col)
                Δa = idx_a > 0 ? ΔA_nzval[idx_a] : 0.0
                for m in 1:M
                    c = coeffs_vals[(p - 1) * M + m]
                    idx_w = Macchiato._find_nzval(weights_vals[m], wr, w_col)
                    if idx_w > 0
                        ΔW_nzvals[m][idx_w] += c * Δa
                    end
                end
            end
        end

        # Zero ΔA and Δb at overwritten rows
        for j in 1:size(A, 2)
            for idx in nzrange(A, j)
                if rows_A[idx] in row_set
                    ΔA_nzval[idx] = 0.0
                end
            end
        end
        for i in row_set
            Δb[i] = 0.0
        end

        return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(),
               Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(),
               Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(),
               Mooncake.NoRData(), Mooncake.NoRData()
    end

    return Mooncake.zero_fcodual(nothing), batch_overwrite_pb!!
end

# ============================================================================
# Phase 2: gradient(sim, loss; wrt=:pts) user-facing API
# ============================================================================

function Macchiato.gradient(
    sim::Simulation,
    loss_function;
    wrt::Symbol = :pts,
    k::Int = 35,
    basis = PHS(3; poly_deg = 3),
)
    @assert wrt == :pts "Only wrt=:pts is currently supported"

    domain = sim.domain
    coords = Macchiato._ustrip(Macchiato._coords(domain.cloud))
    N = length(coords)
    pts_flat = vcat([collect(p) for p in coords]...)

    adjl = find_neighbors(coords, k)
    active = active_dofs(domain)
    dirichlet_dofs, dirichlet_vals = build_dirichlet_info(domain)
    solver = PDESolveIFT(active)

    setup = (;
        sim,
        adjl,
        active,
        dirichlet_dofs,
        dirichlet_vals,
        basis,
        solver,
    )

    function wrapped_loss(pts_in)
        return loss_function(pts_in, setup)
    end

    rule = Mooncake.build_rrule(wrapped_loss, pts_flat)
    _, (_, grad_flat) = Mooncake.value_and_gradient!!(rule, wrapped_loss, pts_flat)

    return [SVector{2,Float64}(grad_flat[2i - 1], grad_flat[2i]) for i in 1:N]
end

# ============================================================================
# Manual adjoint — Step 4 helpers and Phase-A orchestrator
# ============================================================================
# Closures bind `q = pts_from_flat(p)` ONCE so Mooncake's _build_weights rrule
# sees a single SVector array (matches the rrule's tested code path; see
# plan_manual_adjoint.md §Corrections §3).

function _pts_from_flat(p::AbstractVector{<:Real})
    N = length(p) ÷ 2
    return [SVector{2}(p[2i - 1], p[2i]) for i in 1:N]
end

function _make_weight_closure(op, adjl, basis)
    let op = op, adjl = adjl, basis = basis
        function (p::Vector{Float64})
            q = _pts_from_flat(p)
            return _build_weights(op, q, q, adjl, basis)
        end
    end
end

"""
    build_weight_rule(op, pts_flat, adjl, basis)

Build a Mooncake rrule for a single-operator `_build_weights` call. Returns a
`(closure, rule)` NamedTuple to be passed to `apply_weight_rule`. Rule build
compiles the rrule for one primitive call — fast (~seconds) because the trace
is shallow (one primitive + N SVector constructions).
"""
function Macchiato.build_weight_rule(
    op,
    pts_flat::Vector{Float64},
    adjl,
    basis,
)
    closure = _make_weight_closure(op, adjl, basis)
    rule = Mooncake.build_rrule(closure, pts_flat)
    return (closure = closure, rule = rule)
end

"""
    apply_weight_rule(wr, pts_flat, ΔW)

Run the pre-built rule with `ΔW` seeded as the output tangent. Returns Δpts as
a `Vector{Float64}` of the same length as `pts_flat`.

Uses `value_and_pullback!!` with `friendly_tangents=true`, so `ΔW` is passed as
a primal-typed `SparseMatrixCSC` (we put gradient values in its `nzval`) and
the returned gradient is a primal `Vector{Float64}`.
"""
function Macchiato.apply_weight_rule(
    wr::NamedTuple,
    pts_flat::Vector{Float64},
    ΔW::SparseMatrixCSC{Float64, Int},
)
    _val, grads = Mooncake.value_and_pullback!!(
        wr.rule, ΔW, wr.closure, pts_flat; friendly_tangents = true,
    )
    return grads[2]
end

"""
    shape_gradient_dirichlet(
        pts_flat, model, N, adjl, basis, active,
        dirichlet_dofs, dirichlet_vals, ∂L_∂u, weight_rules,
    )

Phase-A manual adjoint for 2D linear elasticity with Dirichlet BCs only.
Implements Steps 1–5 from `plan_manual_adjoint.md`:

  1. Forward: build W_d2x/y/xy, assemble A, apply Dirichlet, solve A u = b.
  2. Adjoint: η = Aᵀ \\ ∂L/∂u  (reuses the LU factor from forward).
  3. Extract: ΔW_d2x/y/xy from (η, u, active) via the closed-form coefficient
     map (per-DOF gated).
  4. Local sensitivity: each ΔW_k → Δpts_k via the per-operator rule.
  5. Accumulate: Δpts = Δpts_d2x + Δpts_d2y + Δpts_d2xy.

Arguments
- `∂L_∂u::Function`: callback `u::Vector{Float64} -> Vector{Float64}` (length 2N).
- `weight_rules::NamedTuple`: must have `d2x`, `d2y`, `d2xy` fields — each a
  `(closure, rule)` from `build_weight_rule`.

Returns `(L_value=nothing, u, Δpts)`. The caller computes any user-facing loss
from `u`; we just need `∂L/∂u` for the adjoint.
"""
function Macchiato.shape_gradient_dirichlet(
    pts_flat::Vector{Float64},
    model::LinearElasticity,
    N::Int,
    adjl,
    basis,
    active::AbstractVector{Bool},
    dirichlet_dofs::Vector{Int},
    dirichlet_vals::Vector{Float64},
    ∂L_∂u::Function,
    weight_rules::NamedTuple,
)
    μ, λstar = lame_parameters(model)

    # Step 1: Forward
    pts = _pts_from_flat(pts_flat)
    W_d2x  = _build_weights(Partial(2, 1),      pts, pts, adjl, basis)
    W_d2y  = _build_weights(Partial(2, 2),      pts, pts, adjl, basis)
    W_d2xy = _build_weights(MixedPartial(1, 2), pts, pts, adjl, basis)

    A = assemble_elasticity_from_weights(W_d2x, W_d2y, W_d2xy, N, λstar, μ)
    b = zeros(2N)
    apply_dirichlet!(A, b, dirichlet_dofs, dirichlet_vals)

    F = lu(A)
    u = F \ b

    # Step 2: Adjoint solve (reuse F)
    η = F' \ ∂L_∂u(u)

    # Step 3: Extract ΔW (per-DOF gated)
    ΔW_d2x, ΔW_d2y, ΔW_d2xy = allocate_weight_gradients(W_d2x)
    extract_weight_sensitivities_elasticity!(
        ΔW_d2x, ΔW_d2y, ΔW_d2xy,
        W_d2x, η, u, active, λstar, μ,
    )

    # Step 4: Local sensitivity per operator
    Δpts_d2x  = Macchiato.apply_weight_rule(weight_rules.d2x,  pts_flat, ΔW_d2x)
    Δpts_d2y  = Macchiato.apply_weight_rule(weight_rules.d2y,  pts_flat, ΔW_d2y)
    Δpts_d2xy = Macchiato.apply_weight_rule(weight_rules.d2xy, pts_flat, ΔW_d2xy)

    # Step 5: Accumulate
    Δpts = Δpts_d2x .+ Δpts_d2y .+ Δpts_d2xy

    return (u = u, Δpts = Δpts)
end

end # module
