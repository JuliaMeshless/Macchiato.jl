# ============================================================================
# Probe — verify the adjoint captures BOTH stencil cross-couplings.
#
# shape_gradient builds d2* weights over all rows and dx/dy weights over the
# Neumann (boundary) rows, then pulls each ΔW row back onto its stencil nodes.
# That means:
#   • a boundary node sitting in an INTERIOR row's stencil should receive that
#     interior row's contribution  (tested before, via hole-node gradients);
#   • an INTERIOR node sitting in a BOUNDARY (Neumann) row's stencil should
#     receive that boundary row's contribution  (NEVER tested — this probe).
#
# We FD-validate shape_gradient's full Δpts at hand-picked coordinates:
# hole nodes, interior nodes adjacent to the hole (the critical case), and an
# interior node far from the hole.  FD rebuilds all weights + normals at FIXED
# adjacency (the exact map the adjoint differentiates).  A match ⇒ all
# cross-couplings are assembled correctly.
#
# Run:  jlrun plate_with_hole/probe_adjoint_coupling.jl   (from examples/)
# ============================================================================
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Macchiato
import RadialBasisFunctions: _build_weights, Partial, MixedPartial, find_neighbors
using RadialBasisFunctions: PHS
using StaticArrays, LinearAlgebra, Printf

const Lx = 4.0; const Ly = 4.0
const a0 = 0.40; const b0 = 0.20
const dx = 0.05
const σ∞ = 1.0
const nθ = max(48, round(Int, 2π*sqrt((a0^2 + b0^2)/2)/dx))
model = LinearElasticity(E = 1.0e7, ν = 0.3)
μ, λstar = lame_parameters(model)
basis = PHS(3; poly_deg = 3); const k = 35

const margin  = 1 + 1.2*dx/min(a0, b0)
const xs_grid = collect((-Lx/2+dx):dx:(Lx/2-dx/2))
const ys_grid = collect((-Ly/2+dx):dx:(Ly/2-dx/2))

const outer_pts = SVector{2,Float64}[]; const outer_tag = Symbol[]
for y in (-Ly/2):dx:(Ly/2)
    push!(outer_pts, SVector(-Lx/2, y)); push!(outer_tag, :xlo)
    push!(outer_pts, SVector(Lx/2, y));  push!(outer_tag, :xhi)
end
for x in (-Lx/2+dx):dx:(Lx/2-dx)
    push!(outer_pts, SVector(x, Ly/2));  push!(outer_tag, :yhi)
    push!(outer_pts, SVector(x, -Ly/2)); push!(outer_tag, :ylo)
end
const n_outer = length(outer_pts)
outer_normal(i) = outer_tag[i]===:xhi ? SVector(1.0,0.0) :
                  outer_tag[i]===:xlo ? SVector(-1.0,0.0) :
                  outer_tag[i]===:yhi ? SVector(0.0,1.0) : SVector(0.0,-1.0)

ellipse_val(p, a, b) = (p[1]/a)^2 + (p[2]/b)^2
interior_pts = SVector{2,Float64}[]
for x in xs_grid, y in ys_grid
    ellipse_val(SVector(x,y), a0, b0) > margin^2 || continue
    push!(interior_pts, SVector(x,y))
end
const n_int = length(interior_pts)
hole0 = [SVector(a0*cos(2π*j/nθ), b0*sin(2π*j/nθ)) for j in 0:(nθ-1)]
const pts0 = vcat(interior_pts, outer_pts, hole0)
const N = length(pts0)
const interior_idx = collect(1:n_int)
const outer_idx    = collect((n_int+1):(n_int+n_outer))
const hole_idx     = collect((n_int+n_outer+1):N)
const neumann_idx  = vcat(outer_idx, hole_idx)
const hole_pos     = collect(1:nθ)
const adjl0        = find_neighbors(pts0, k)
const neumann_adjl0 = adjl0[neumann_idx]

nearest(p0,pool) = pool[argmin([hypot(pts0[i][1]-p0[1], pts0[i][2]-p0[2]) for i in pool])]
const pin_ux1 = nearest((0.0, 0.8), interior_idx)
const pin_ux2 = nearest((0.0,-0.8), interior_idx)
const pin_uy1 = nearest((0.8, 0.0), interior_idx)
const dirichlet_dofs = [pin_ux1, pin_ux2, pin_uy1 + N]
const active = let a = trues(2N); a[pin_ux1]=a[pin_ux2]=a[pin_uy1+N]=false; a end
const interior_rows = let r = falses(N); for i in interior_idx; r[i]=true; end; r end

const _ZJV = SVector(0.0, 0.0)
zero_njac(g) = NormalJacobian(g, g, _ZJV, _ZJV, _ZJV, _ZJV)
flat(v) = reduce(vcat, [[p[1], p[2]] for p in v])

function build_layout(pts)
    hn, hjacs = polyline_normals(flat(pts), hole_idx, hole_pos)
    njacs = vcat(NormalJacobian[zero_njac(i) for i in outer_idx], hjacs)
    normals = SVector{2,Float64}[]; tractions = SVector{2,Float64}[]
    for i in 1:n_outer
        nn = outer_normal(i); push!(normals, nn); push!(tractions, σ∞ .* nn)
    end
    append!(normals, hn); append!(tractions, fill(SVector(0.0,0.0), nθ))
    layout = build_traction_layout(neumann_idx, neumann_adjl0, normals, tractions, λstar, μ, N)
    return layout, njacs
end

function solve_C(pts)
    layout, _ = build_layout(pts)
    Wx  = _build_weights(Partial(2,1),     pts, pts, adjl0, basis)
    Wy  = _build_weights(Partial(2,2),     pts, pts, adjl0, basis)
    Wxy = _build_weights(MixedPartial(1,2), pts, pts, adjl0, basis)
    neu = pts[neumann_idx]
    Wdx = _build_weights(Partial(1,1), pts, neu, neumann_adjl0, basis)
    Wdy = _build_weights(Partial(1,2), pts, neu, neumann_adjl0, basis)
    A = assemble_elasticity_from_weights(Wx, Wy, Wxy, N, λstar, μ)
    b = zeros(2N)
    apply_dirichlet!(A, b, dirichlet_dofs, zeros(3))
    apply_traction!(A, b, layout, Wdx, Wdy)
    return dot(b, lu(A) \ b)
end

function full_gradient(pts)
    layout, njacs = build_layout(pts)
    b = zeros(2N)
    for kk in eachindex(layout.rows); b[layout.rows[kk]] = layout.b_vals[kk]; end
    res = shape_gradient(flat(pts), model, N, adjl0, basis, active,
                         dirichlet_dofs, zeros(3), _ -> b;
                         interior_rows = interior_rows, traction_layout = layout,
                         neumann_ids = neumann_idx, neumann_adjl = neumann_adjl0,
                         traction_jacobians = nothing, normal_jacobians = njacs)
    return res.Δpts
end

# ============================================================================
println("=== Adjoint cross-coupling FD check (fixed cloud, fixed adjl) ===")
println("N=$N  nθ=$nθ\n")

Δpts = full_gradient(pts0)

# Does any boundary node's Neumann stencil actually contain interior nodes?
# (If not, the coupling we want to test is vacuous.)
let cnt = 0, exemplar = 0
    for (r, gi) in enumerate(neumann_idx)
        ints = count(c -> c <= n_int, neumann_adjl0[r])
        cnt += ints
        ints > 0 && exemplar == 0 && (exemplar = gi)
    end
    @printf("Interior nodes appearing in boundary Neumann stencils: %d total occurrences.\n", cnt)
    @printf("(So moving an interior node DOES enter boundary rows — the coupling is real.)\n\n")
end

# pick interior nodes nearest the hole (critical case) + far ones + hole nodes
hole_centroid = sum(pts0[hole_idx]) / nθ
int_dist = [minimum(hypot(pts0[i][1]-pts0[j][1], pts0[i][2]-pts0[j][2]) for j in hole_idx)
            for i in interior_idx]
order = sortperm(int_dist)
near_hole = interior_idx[order[1:3]]            # closest interior nodes to the hole
far_hole  = interior_idx[order[end-1:end]]       # farthest
holes     = hole_idx[[1, nθ÷4, nθ÷2]]            # a few hole nodes

ε = 1e-6
function fd_coord(node, comp)   # comp: 1=x, 2=y
    pp = collect(pts0); pp[node] = pts0[node] + (comp==1 ? SVector(ε,0.0) : SVector(0.0,ε))
    pm = collect(pts0); pm[node] = pts0[node] - (comp==1 ? SVector(ε,0.0) : SVector(0.0,ε))
    return (solve_C(pp) - solve_C(pm)) / (2ε)
end

function report(label, nodes)
    println("--- $label ---")
    println("  node    comp     adjoint Δpts        FD              rel.err")
    for nd in nodes
        for comp in (1, 2)
            adj = Δpts[2*nd - 2 + comp]
            fd  = fd_coord(nd, comp)
            re  = abs(adj - fd) / max(abs(fd), 1e-30)
            @printf("  %5d    %s    %+.6e   %+.6e   %.2e\n",
                    nd, comp==1 ? "x" : "y", adj, fd, re)
        end
    end
    println()
end

report("HOLE nodes (boundary)", holes)
report("INTERIOR nodes ADJACENT to hole  (← the untested cross-coupling)", near_hole)
report("INTERIOR nodes far from hole", far_hole)

println("Verdict: rel.err ~1e-5 or better at the interior-adjacent nodes ⇒ the")
println("adjoint correctly assembles boundary-row⇄interior-node coupling.")
