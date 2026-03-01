# ============================================================================
# 2D Cantilever Beam - Linear Elasticity (Ferrite.jl FEM)
# ============================================================================
# Ferrite.jl finite-element solution for the same Timoshenko cantilever beam
# problem solved by examples/cantilever_beam_2d.jl (meshless method).
#
# Geometry: L × 2D beam, x ∈ [0, L], y ∈ [-D, D]
# Left end (x=0): Clamped (prescribed displacement from exact solution)
# Right end (x=L): Parabolic shear traction
# Top/bottom (y=±D): Traction-free (natural BC)
#
# Timoshenko beam solution (plane stress):
#   u(x,y) = -P/(6EI) [y((6L-3x)x + (2+ν)(y²-D²))]
#   v(x,y) = P/(6EI) [3νy²(L-x) + (4+5ν)D²x + (3L-x)x²]
#
# where I = 2D³/3 is the second moment of area.
#
# Element type: QuadraticQuadrilateral (9-node Q2)
# Mesh: 80×20 elements (dx ≈ 0.1, comparable to meshless spacing)
# ============================================================================
using Pkg
Pkg.activate(@__DIR__)

using Ferrite
using SparseArrays: nonzeros, nnz
using Statistics: mean
using CairoMakie

# ============================================================================
# Problem Parameters
# ============================================================================

L = 8.0    # Beam length
D = 1.0    # Half-height (beam goes from y=-D to y=D)
P = 1000.0 # Applied load (total shear force)
E_val = 1.0e7
ν_val = 0.3
I_val = 2D^3 / 3  # Second moment of area

# ============================================================================
# Timoshenko Analytical Solution
# ============================================================================

function u_exact(x, y)
    return -P / (6E_val * I_val) * y * ((6L - 3x) * x + (2 + ν_val) * (y^2 - D^2))
end

function v_exact(x, y)
    return P / (6E_val * I_val) *
        (3ν_val * y^2 * (L - x) + (4 + 5ν_val) * D^2 * x + (3L - x) * x^2)
end

# ============================================================================
# Mesh Setup
# ============================================================================

grid = generate_grid(QuadraticQuadrilateral, (80, 20), Tensors.Vec((0.0, -D)), Tensors.Vec((L, D)))

# ============================================================================
# FE Setup
# ============================================================================

ip = Lagrange{RefQuadrilateral, 2}()^2            # Quadratic vector interpolation (2D)
ip_geo = Lagrange{RefQuadrilateral, 2}()           # Quadratic geometry interpolation
qr = QuadratureRule{RefQuadrilateral}(3)           # Cell quadrature (order 3)
qr_facet = FacetQuadratureRule{RefQuadrilateral}(3) # Facet quadrature (order 3)

cellvalues = CellValues(qr, ip, ip_geo)
facetvalues = FacetValues(qr_facet, ip, ip_geo)

# ============================================================================
# DofHandler
# ============================================================================

dh = DofHandler(grid)
add!(dh, :u, ip)
close!(dh)

# ============================================================================
# Boundary Conditions
# ============================================================================

# Left (clamped): exact displacement from Timoshenko solution
ch = ConstraintHandler(dh)
add!(
    ch, Dirichlet(
        :u, getfacetset(grid, "left"),
        (x, t) -> [u_exact(x[1], x[2]), v_exact(x[1], x[2])], [1, 2]
    )
)
close!(ch)
Ferrite.update!(ch, 0.0)

# ============================================================================
# Plane Stress Constitutive Tensor
# ============================================================================
#
# Voigt matrix (Tensors.jl convention — tensor strain, no factor of 2):
#   [σ₁₁]   E/(1-ν²) [1    ν    0    ] [ε₁₁]
#   [σ₂₂] =          [ν    1    0    ] [ε₂₂]
#   [σ₁₂]            [0    0  (1-ν)  ] [ε₁₂]

C_factor = E_val / (1 - ν_val^2)
C_voigt = C_factor * [
    1.0     ν_val   0.0
    ν_val   1.0     0.0
    0.0     0.0     (1.0 - ν_val)
]
C = fromvoigt(SymmetricTensor{4, 2}, C_voigt)

# ============================================================================
# Assembly: Stiffness Matrix
# ============================================================================

function assemble_cell!(ke, cellvalues, C)
    for qp in 1:getnquadpoints(cellvalues)
        dΩ = getdetJdV(cellvalues, qp)
        for i in 1:getnbasefunctions(cellvalues)
            ∇ˢʸᵐNᵢ = shape_symmetric_gradient(cellvalues, qp, i)
            for j in 1:getnbasefunctions(cellvalues)
                ∇ˢʸᵐNⱼ = shape_symmetric_gradient(cellvalues, qp, j)
                ke[i, j] += (∇ˢʸᵐNᵢ ⊡ C ⊡ ∇ˢʸᵐNⱼ) * dΩ
            end
        end
    end
    return ke
end

function assemble_global!(K, dh, cellvalues, C)
    n_basefuncs = getnbasefunctions(cellvalues)
    ke = zeros(n_basefuncs, n_basefuncs)
    assembler = start_assemble(K)
    for cell in CellIterator(dh)
        Ferrite.reinit!(cellvalues, cell)
        fill!(ke, 0.0)
        assemble_cell!(ke, cellvalues, C)
        assemble!(assembler, celldofs(cell), ke)
    end
    return K
end

function assemble_traction!(f_ext, dh, facetvalues, facetset)
    n_basefuncs = getnbasefunctions(facetvalues)
    fe = zeros(n_basefuncs)
    for facet in FacetIterator(dh, facetset)
        reinit!(facetvalues, facet)
        fill!(fe, 0.0)
        coords = getcoordinates(facet)
        for qp in 1:getnquadpoints(facetvalues)
            x = spatial_coordinate(facetvalues, qp, coords)
            t = Tensors.Vec{2}((0.0, P * (D^2 - x[2]^2) / (2I_val)))
            dΓ = getdetJdV(facetvalues, qp)
            for i in 1:n_basefuncs
                Nᵢ = shape_value(facetvalues, qp, i)
                fe[i] += (t ⋅ Nᵢ) * dΓ
            end
        end
        assemble!(f_ext, celldofs(facet), fe)
    end
    return f_ext
end

function ferrite_assemble!(K, f_ext, dh, cellvalues, facetvalues, C, ch)
    fill!(nonzeros(K), 0.0)
    assemble_global!(K, dh, cellvalues, C)
    fill!(f_ext, 0.0)
    assemble_traction!(f_ext, dh, facetvalues, getfacetset(grid, "right"))
    apply!(K, f_ext, ch)
    return nothing
end

# ============================================================================
# Warmup (JIT compilation)
# ============================================================================

K = allocate_matrix(dh)
f_ext = zeros(ndofs(dh))
ferrite_assemble!(K, f_ext, dh, cellvalues, facetvalues, C, ch)
u = K \ f_ext

# ============================================================================
# Timed Runs
# ============================================================================

t_assembly = @elapsed ferrite_assemble!(K, f_ext, dh, cellvalues, facetvalues, C, ch)
t_solve = @elapsed u = K \ f_ext

# ============================================================================
# Compare with Analytical Solution
# ============================================================================

u_nodes = evaluate_at_grid_nodes(dh, u, :u)
node_coords = [node.x for node in grid.nodes]

x = [c[1] for c in node_coords]
y = [c[2] for c in node_coords]
ux_fem = [u_nodes[i][1] for i in eachindex(u_nodes)]
uy_fem = [u_nodes[i][2] for i in eachindex(u_nodes)]

ux_ana = [u_exact(x[i], y[i]) for i in eachindex(x)]
uy_ana = [v_exact(x[i], y[i]) for i in eachindex(x)]

err_ux = ux_fem .- ux_ana
err_uy = uy_fem .- uy_ana

abs_err_ux = abs.(err_ux)
abs_err_uy = abs.(err_uy)

mean_abs_ux = mean(abs_err_ux)
mean_abs_uy = mean(abs_err_uy)
max_abs_ux = maximum(abs_err_ux)
max_abs_uy = maximum(abs_err_uy)

N = length(x)
println("\n========================================")
println("Performance Summary")
println("========================================")
println("Method:        Ferrite FEM (Q2)")
println("Nodes:         $N")
println("Elements:      $(getncells(grid))")
println("DOFs:          $(ndofs(dh))")
println("System nnz:    $(nnz(K))")
println("Assembly time: $(round(t_assembly; digits = 4)) s")
println("Solve time:    $(round(t_solve; digits = 4)) s")
println("Total time:    $(round(t_assembly + t_solve; digits = 4)) s")
println()
println("========================================")
println("Cantilever Beam Results (Ferrite FEM)")
println("========================================")
println("Beam: L=$L, D=$D, P=$P, E=$E_val, ν=$ν_val")
println("Nodes: $N, Elements: $(getncells(grid))")
println()
println("Absolute Error (vs Timoshenko):")
println("  ux: mean = $(round(mean_abs_ux; digits = 8)), max = $(round(max_abs_ux; digits = 8))")
println("  uy: mean = $(round(mean_abs_uy; digits = 8)), max = $(round(max_abs_uy; digits = 8))")
println()

# Max tip deflection (analytical at x=L, y=0):
v_tip_exact = v_exact(L, 0.0)
println("Tip deflection (analytical): $v_tip_exact")

# Find rightmost point for comparison
tip_indices = findall(i -> abs(x[i] - L) < 0.01 && abs(y[i]) < 0.01, eachindex(x))
if !isempty(tip_indices)
    v_tip_num = mean(uy_fem[tip_indices])
    tip_abs_err = abs(v_tip_num - v_tip_exact)
    println("Tip deflection (numerical): $v_tip_num")
    println("Tip deflection error: $(round(tip_abs_err; digits = 8))")
end

# ============================================================================
# Visualization
# ============================================================================

displacement_mag = sqrt.(ux_fem .^ 2 .+ uy_fem .^ 2)
ana_mag = sqrt.(ux_ana .^ 2 .+ uy_ana .^ 2)
error_mag = sqrt.(err_ux .^ 2 .+ err_uy .^ 2)

fig = Figure(; size = (1400, 1800));

# Row 1: analytical displacement components
ax1 = Axis(
    fig[1, 1]; title = "uₓ (analytical)", xlabel = "x", ylabel = "y",
    aspect = DataAspect()
)
sc1 = scatter!(ax1, x, y; color = ux_ana, colormap = :RdBu, markersize = 4)
Colorbar(fig[1, 2], sc1)

ax2 = Axis(
    fig[1, 3]; title = "uᵧ (analytical)", xlabel = "x", ylabel = "y",
    aspect = DataAspect()
)
sc2 = scatter!(ax2, x, y; color = uy_ana, colormap = :RdBu, markersize = 4)
Colorbar(fig[1, 4], sc2)

# Row 2: FEM displacement components
ax3 = Axis(
    fig[2, 1]; title = "uₓ (FEM)", xlabel = "x", ylabel = "y",
    aspect = DataAspect()
)
sc3 = scatter!(ax3, x, y; color = ux_fem, colormap = :RdBu, markersize = 4)
Colorbar(fig[2, 2], sc3)

ax4 = Axis(
    fig[2, 3]; title = "uᵧ (FEM)", xlabel = "x", ylabel = "y",
    aspect = DataAspect()
)
sc4 = scatter!(ax4, x, y; color = uy_fem, colormap = :RdBu, markersize = 4)
Colorbar(fig[2, 4], sc4)

# Row 3: displacement magnitude (analytical vs FEM)
ax5 = Axis(
    fig[3, 1]; title = "‖u‖ (analytical)", xlabel = "x", ylabel = "y",
    aspect = DataAspect()
)
sc5 = scatter!(ax5, x, y; color = ana_mag, colormap = :viridis, markersize = 4)
Colorbar(fig[3, 2], sc5)

ax6 = Axis(
    fig[3, 3]; title = "‖u‖ (FEM)", xlabel = "x", ylabel = "y",
    aspect = DataAspect()
)
sc6 = scatter!(ax6, x, y; color = displacement_mag, colormap = :viridis, markersize = 4)
Colorbar(fig[3, 4], sc6)

# Row 4: absolute error
ax7 = Axis(
    fig[4, 1]; title = "abs error uₓ", xlabel = "x", ylabel = "y",
    aspect = DataAspect()
)
sc7 = scatter!(ax7, x, y; color = abs_err_ux, colormap = :inferno, markersize = 4)
Colorbar(fig[4, 2], sc7)

ax8 = Axis(
    fig[4, 3]; title = "abs error uᵧ", xlabel = "x", ylabel = "y",
    aspect = DataAspect()
)
sc8 = scatter!(ax8, x, y; color = abs_err_uy, colormap = :inferno, markersize = 4)
Colorbar(fig[4, 4], sc8)

fig
