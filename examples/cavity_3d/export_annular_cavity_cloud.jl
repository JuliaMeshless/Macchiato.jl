# Generate the annular cavity cloud via WTP pipeline and export to VTK for ParaView.
# Run:  jlrun cavity_3d/export_annular_cavity_cloud.jl   (from examples/)
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

import WhatsThePoint as WTP
const Meshes = WTP.Meshes
using StaticArrays, LinearAlgebra, Printf
using Unitful: m, ustrip

const L_OUT = 1.0
const NSUB  = 2
const r_cav = 0.547
const Δ     = 0.08

function _make_sphere_mesh(R, nθ, nφ)
    pts = [WTP.Point(R * sin(π * i / nθ) * cos(2π * j / nφ) * m,
                     R * sin(π * i / nθ) * sin(2π * j / nφ) * m,
                     R * cos(π * i / nθ) * m)
           for j in 0:(nφ - 1), i in 0:nθ][:]
    conn = WTP.Connectivity{WTP.Triangle}[]
    for i in 0:(nθ - 1), j in 0:(nφ - 1)
        a = i * nφ + j + 1; b = i * nφ + (j + 1) % nφ + 1
        c = (i + 1) * nφ + j + 1; d = (i + 1) * nφ + (j + 1) % nφ + 1
        i > 0    && push!(conn, WTP.connect((a, c, b)))
        i < nθ-1 && push!(conn, WTP.connect((b, c, d)))
    end
    return WTP.SimpleMesh(pts, conn)
end

function _make_annular_cavity_mesh(R_outer, r_inner, nθ, nφ)
    outer = _make_sphere_mesh(R_outer, nθ, nφ)
    inner = _make_sphere_mesh(r_inner, nθ, nφ)
    outer_v = collect(Meshes.vertices(outer))
    inner_v = collect(Meshes.vertices(inner))
    n_outer = length(outer_v)
    all_conn = WTP.Connectivity{WTP.Triangle}[]
    for c in Meshes.topology(outer); push!(all_conn, c); end
    for c in Meshes.topology(inner)
        p = Meshes.indices(c)
        push!(all_conn, WTP.connect((p[1] + n_outer, p[3] + n_outer, p[2] + n_outer)))
    end
    return WTP.SimpleMesh(vcat(outer_v, inner_v), all_conn)
end

nθ = 2 * 2^NSUB; nφ = 2 * nθ
mesh = _make_annular_cavity_mesh(L_OUT, r_cav, nθ, nφ)
@printf("Mesh: %d vertices, %d triangles\n",
        length(Meshes.vertices(mesh)), Meshes.nelements(mesh))

spacing = WTP.ConstantSpacing(Δ * m)
bnd = WTP.PointBoundary(mesh, spacing)
alg = WTP.Octree(mesh; spacing, alpha = 1.0, placement = :bridson)
V_solid = (4 / 3) * π * (L_OUT^3 - r_cav^3)
n_target = round(Int, V_solid / Δ^3)
@printf("Boundary: %d points · target volume: %d nodes\n", length(WTP.points(bnd)), n_target)

cloud = WTP.discretize(bnd, spacing; alg, max_points = n_target)
nv = length(WTP.points(WTP.volume(cloud)))
nb = length(WTP.points(WTP.boundary(cloud)))
@printf("Cloud: %d volume + %d boundary = %d total\n", nv, nb, nv + nb)

vtk_path = joinpath(@__DIR__, "annular_cavity_cloud")
WTP.save(vtk_path, cloud; format = :vtk)
@printf("Saved to %s.vtu (open in ParaView)\n", vtk_path)
