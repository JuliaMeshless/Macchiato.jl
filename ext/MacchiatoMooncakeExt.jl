module MacchiatoMooncakeExt

import Macchiato
using Macchiato:
    PDESolveIFT, apply_dirichlet!,
    batch_overwrite_sparse_rows!,
    active_dofs, build_dirichlet_info,
    Simulation
using Mooncake
using RadialBasisFunctions: PHS, find_neighbors
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

end # module
