module MacchiatoMooncakeExt

using Macchiato: PDESolveIFT, apply_dirichlet!
using Mooncake
using SparseArrays
using LinearAlgebra

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

end # module
