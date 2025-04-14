function _build_weights(
        data, normals, boundary_flag, is_Neumann, adjl, basis, ℒrbf, ℒmon, mon
)
    nchunks = Threads.nthreads()
    TD = eltype(first(data))
    dim = length(first(data))
    nmon = binomial(dim + basis.poly_deg, basis.poly_deg)
    k = length(first(adjl))

    (I_lhs, J_lhs, V_lhs), (I_rhs, J_rhs, V_rhs) = _preallocate_IJV_matrices(
        adjl, data, boundary_flag, ℒrbf
    )

    stencil_data = [StencilData(TD, dim, k + nmon, k, _num_ops(ℒrbf)) for _ in 1:nchunks]

    lhs_offsets, rhs_offsets = _calculate_thread_offsets(adjl, boundary_flag, nchunks)

    # Build stencil for each data point and store in global weight matrices
    Threads.@threads for (ichunk, xrange) in enumerate(index_chunks(adjl; n = nchunks))
        lhs_idx = lhs_offsets[ichunk] + 1
        rhs_idx = rhs_offsets[ichunk] + 1

        for i in xrange
            if !boundary_flag[i]

                # Update stencil and compute weights in a single call
                _update_stencil!(
                    stencil_data[ichunk],
                    adjl[i],
                    data,
                    boundary_flag,
                    is_Neumann,
                    normals,
                    ℒrbf,
                    ℒmon,
                    data[i],
                    basis,
                    mon,
                    k
                )

                # Copy results from stencil_data to global matrices
                lhs_idx, rhs_idx = _write_coefficients_to_global_matrices!(
                    V_lhs,
                    V_rhs,
                    stencil_data[ichunk],
                    adjl[i],
                    boundary_flag,
                    lhs_idx,
                    rhs_idx
                )
            end
        end
    end

    return _return_global_matrices(I_lhs, J_lhs, V_lhs, I_rhs, J_rhs, V_rhs, boundary_flag)
end

_num_ops(_) = 1
_num_ops(ℒ::Tuple) = length(ℒ)

function _preallocate_IJV_matrices(adjl, data, boundary_flag, ℒrbf)
    TD = eltype(first(data))
    Na = length(adjl)
    num_ops = _num_ops(ℒrbf)

    # Count entries in one pass while also collecting the I, J pairs
    lhs_pairs = Tuple{Int, Int}[]
    rhs_pairs = Tuple{Int, Int}[]

    for i in 1:Na
        if !boundary_flag[i]  # internal node
            for j in adjl[i]
                if !boundary_flag[j]  # internal neighbor
                    push!(lhs_pairs, (i, j))
                else  # boundary neighbor
                    push!(rhs_pairs, (i, j))
                end
            end
        end
    end

    # Create arrays with exact sizes
    lhs_count = length(lhs_pairs)
    rhs_count = length(rhs_pairs)

    I_lhs = zeros(Int, lhs_count)
    J_lhs = zeros(Int, lhs_count)
    V_lhs = zeros(TD, lhs_count, num_ops)

    I_rhs = zeros(Int, rhs_count)
    J_rhs = zeros(Int, rhs_count)
    V_rhs = zeros(TD, rhs_count, num_ops)

    # Create mapping from global indices to internal/boundary-specific indices
    internal_idx = zeros(Int, length(boundary_flag))
    boundary_idx = zeros(Int, length(boundary_flag))
    int_count = 1
    bnd_count = 1

    for i in eachindex(boundary_flag)
        if !boundary_flag[i]
            internal_idx[i] = int_count
            int_count += 1
        else
            boundary_idx[i] = bnd_count
            bnd_count += 1
        end
    end

    # Fill indices
    lhs_idx = 1
    rhs_idx = 1

    for i in eachindex(adjl)
        if !boundary_flag[i]  # If node i is internal
            i_internal = internal_idx[i]  # Remap to internal index

            for j in adjl[i]  # For each neighbor
                if !boundary_flag[j]  # Internal neighbor
                    j_internal = internal_idx[j]  # Remap to internal index
                    I_lhs[lhs_idx] = i_internal
                    J_lhs[lhs_idx] = j_internal
                    lhs_idx += 1
                else  # Boundary neighbor
                    j_boundary = boundary_idx[j]  # Remap to boundary index
                    I_rhs[rhs_idx] = i_internal
                    J_rhs[rhs_idx] = j_boundary
                    rhs_idx += 1
                end
            end
        end
    end

    return (I_lhs, J_lhs, V_lhs), (I_rhs, J_rhs, V_rhs)
end

"""
    _calculate_thread_offsets(adjl, boundary_flag, nchunks)

Calculate the starting offsets for each thread when filling LHS and RHS matrices.
- lhs_offsets: Starting indices for internal-to-internal connections
- rhs_offsets: Starting indices for internal-to-boundary connections

Returns a tuple of (lhs_offsets, rhs_offsets).
"""
function _calculate_thread_offsets(adjl, boundary_flag, nchunks)
    thread_lhs_counts = zeros(Int, nchunks)
    thread_rhs_counts = zeros(Int, nchunks)

    # Count elements per thread
    for (ichunk, xrange) in enumerate(index_chunks(adjl; n = nchunks))
        for i in xrange
            if !boundary_flag[i]  # Only internal nodes generate equations
                local_adjl = adjl[i]
                for j_global in local_adjl
                    if !boundary_flag[j_global]  # Internal neighbor -> LHS
                        thread_lhs_counts[ichunk] += 1
                    else  # Boundary neighbor -> RHS
                        thread_rhs_counts[ichunk] += 1
                    end
                end
            end
        end
    end

    # Calculate starting indices for each thread
    lhs_offsets = cumsum([0; thread_lhs_counts[1:(end - 1)]])
    rhs_offsets = cumsum([0; thread_rhs_counts[1:(end - 1)]])

    return lhs_offsets, rhs_offsets
end

function _write_coefficients_to_global_matrices!(
        V_lhs, V_rhs, stencil, local_adjl, boundary_flag, lhs_idx, rhs_idx
)
    for (j_local, j_global) in enumerate(local_adjl)
        if !boundary_flag[j_global]  # Internal node -> goes to LHS
            V_lhs[lhs_idx, :] = stencil.lhs_v[j_local, :]
            lhs_idx += 1
        else  # Boundary node -> goes to RHS
            V_rhs[rhs_idx, :] = stencil.rhs_v[j_local, :]
            rhs_idx += 1
        end
    end

    return lhs_idx, rhs_idx
end

"""
    _return_global_matrices(I_lhs, J_lhs, V_lhs, I_rhs, J_rhs, V_rhs, boundary_flag)

Constructs sparse matrix representation of the global linear system.

# Arguments
- `I_lhs`, `J_lhs`, `V_lhs`: COO format components for LHS matrix
- `I_rhs`, `J_rhs`, `V_rhs`: COO format components for RHS matrix
- `boundary_flag`: Boolean array indicating boundary nodes

# Returns
- For single operators: tuple of (lhs_matrix, rhs_matrix)
- For multiple operators: tuple of (lhs_matrices, rhs_matrices)
"""
function _return_global_matrices(I_lhs, J_lhs, V_lhs, I_rhs, J_rhs, V_rhs, boundary_flag)
    nrows = count(.!boundary_flag)
    ncols_lhs = count(.!boundary_flag)
    ncols_rhs = count(boundary_flag)

    if size(V_lhs, 2) == 1
        lhs_matrix = sparse(I_lhs, J_lhs, V_lhs[:, 1], nrows, ncols_lhs)
        rhs_matrix = sparse(I_rhs, J_rhs, V_rhs[:, 1], nrows, ncols_rhs)
        return lhs_matrix, rhs_matrix
    else
        lhs_matrices = ntuple(
            i -> sparse(I_lhs, J_lhs, V_lhs[:, i], nrows, ncols_lhs), size(V_lhs, 2)
        )
        rhs_matrices = ntuple(
            i -> sparse(I_rhs, J_rhs, V_rhs[:, i], nrows, ncols_rhs), size(V_rhs, 2)
        )
        return lhs_matrices, rhs_matrices
    end
end

struct StencilData{T}
    A::AbstractMatrix{T}            # Local system matrix
    b::AbstractMatrix{T}                          # Local RHS matrix (one column per operator)
    d::Vector{AbstractVector{T}}                  # Local data points (now using the same type T)
    is_boundary::AbstractVector{Bool}             # Whether nodes are on boundary
    is_Neumann::AbstractVector{Bool}              # Whether nodes have Neumann boundary conditions
    normal::Vector{AbstractVector{T}}             # Normal vectors (meaningful for Neumann points)
    lhs_v::AbstractMatrix{T}                      # Local coefficients for internal nodes
    rhs_v::AbstractMatrix{T}                      # Local coefficients for boundary nodes
    weights::AbstractMatrix{T}                    # Weights for the local system

    function StencilData{T}(n::Int, k::Int, num_ops::Int, dim::Int) where {T}
        A = Symmetric(zeros(T, n, n), :U)
        b = zeros(T, n, num_ops)
        d = [zeros(T, dim) for _ in 1:k]
        is_boundary = zeros(Bool, k)
        is_Neumann = zeros(Bool, k)
        normal = [zeros(T, dim) for _ in 1:k]
        lhs_v = zeros(T, k, num_ops)
        rhs_v = zeros(T, k, num_ops)
        weights = zeros(T, n, num_ops)

        return new(A, b, d, is_boundary, is_Neumann, normal, lhs_v, rhs_v, weights)
    end
end

#convenience constructor
function StencilData(T::Type, data_dim::Int, n::Int, k::Int, num_ops::Int)
    return StencilData{T}(n, k, num_ops, data_dim)
end

function _update_stencil!(
        stencil::StencilData{T},
        local_adjl,
        data,
        boundary_flag,
        is_Neumann,
        normals,
        ℒrbf,
        ℒmon,
        eval_point,
        basis::B,
        mon::MonomialBasis{Dim, Deg},
        k::Int
) where {T, B <: AbstractRadialBasis, Dim, Deg}
    fill!(stencil.lhs_v, 0)
    fill!(stencil.rhs_v, 0)
    fill!(parent(stencil.A), 0)
    fill!(stencil.b, 0)

    for (idx, j) in enumerate(local_adjl)
        stencil.d[idx] = data[j]
    end

    for (j_local, j_global) in enumerate(local_adjl)
        stencil.is_boundary[j_local] = boundary_flag[j_global]
        stencil.is_Neumann[j_local] = is_Neumann[j_global]
        if is_Neumann[j_global]
            copyto!(stencil.normal[j_local], convert.(T, normals[j_global]))
        else
            stencil.normal[j_local] .= zero(T)
        end
    end

    _build_collocation_matrix_Hermite!(stencil, basis, mon, k)
    _build_rhs!(stencil, ℒrbf, ℒmon, eval_point, basis, k)

    fill!(stencil.weights, 0)
    stencil.weights .= bunchkaufman!(stencil.A) \ stencil.b

    # Store weights in appropriate matrices
    for j in 1:k
        if !stencil.is_boundary[j]
            stencil.lhs_v[j, :] .= view(stencil.weights, j, :)
        else
            stencil.rhs_v[j, :] .= view(stencil.weights, j, :)
        end
    end

    return nothing
end

function _build_collocation_matrix_Hermite!(
        stencil::StencilData{T}, basis::B, mon::MonomialBasis{Dim, Deg},
        k::K
) where {T, B <: AbstractRadialBasis, K <: Int, Dim, Deg}
    A = parent(stencil.A)
    n = size(A, 2)
    @inbounds for j in 1:k, i in 1:j
        _calculate_matrix_entry_RBF!(i, j, stencil, basis)
    end

    if Deg > -1
        @inbounds for i in 1:k
            _calculate_matrix_entry_poly!(
                A, i, k + 1, n, stencil.d[i], stencil.is_Neumann[i], stencil.normal[i], mon
            )
        end
    end

    return nothing
end

function _calculate_matrix_entry_RBF!(i, j, stencil::StencilData, basis)
    A = parent(stencil.A)
    data = stencil.d
    is_Neumann_i = stencil.is_Neumann[i]
    is_Neumann_j = stencil.is_Neumann[j]
    if !is_Neumann_i && !is_Neumann_j
        A[i, j] = basis(data[i], data[j])
    elseif is_Neumann_i && !is_Neumann_j
        n = stencil.normal[i]
        A[i, j] = LinearAlgebra.dot(n, ∇(basis)(data[i], data[j]))
    elseif !is_Neumann_i && is_Neumann_j
        n = stencil.normal[j]
        A[i, j] = LinearAlgebra.dot(n, -∇(basis)(data[i], data[j]))
    elseif is_Neumann_i && is_Neumann_j
        ni = stencil.normal[i]
        nj = stencil.normal[j]
        A[i, j] = directional∂²(basis, ni, nj)(data[i], data[j])
    end
    return nothing
end

function _calculate_matrix_entry_poly!(
        A, row, col_start, col_end, data_point, is_Neumann, normal, mon
)
    a = view(A, row, col_start:col_end)
    if is_Neumann
        ∂_normal(mon, normal)(a, data_point)
    else
        mon(a, data_point)
    end

    return nothing
end

function _build_rhs!(
        stencil::StencilData{T}, ℒrbf::Tuple, ℒmon::Tuple, eval_point, basis::B,
        k::Int
) where {T, B <: AbstractRadialBasis}
    b = stencil.b
    data = stencil.d

    @assert size(b, 2)==length(ℒrbf)==length(ℒmon) "b, ℒrbf, and ℒmon must have the same length"

    # radial basis section
    for (j, ℒ) in enumerate(ℒrbf)
        @inbounds for i in eachindex(data)
            if stencil.is_Neumann[i]
                b[i, j] = ℒ(eval_point, data[i], stencil.normal[i])
            else
                b[i, j] = ℒ(eval_point, data[i])
            end
        end
    end

    # monomial augmentation
    if basis.poly_deg > -1
        N = size(b, 1)
        for (j, ℒ) in enumerate(ℒmon)
            bmono = view(b, (k + 1):N, j)
            ℒ(bmono, eval_point)
        end
    end

    return nothing
end

# Handle the case when ℒrbf and ℒmon are single operators (not tuples)
function _build_rhs!(
        stencil::StencilData{T}, ℒrbf, ℒmon, eval_point, basis::B, k::Int
) where {T, B <: AbstractRadialBasis}
    return _build_rhs!(stencil, (ℒrbf,), (ℒmon,), eval_point, basis, k)
end
