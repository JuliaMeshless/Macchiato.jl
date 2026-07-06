using Pkg
Pkg.activate(joinpath(@__DIR__, "../"))

using WhatsThePoint
import Meshes
using Meshes: SimpleMesh, connect, Triangle, Point, coords, refine, TriRefinement
using Unitful: mm, ustrip
using StaticArrays
using GLMakie: Figure, Axis3, meshscatter!, display

Lx, Ly, Lz = 20.0, 7.0, 3.0

vertices = Point.([
    (0.0, 0.0, 0.0), (Lx, 0.0, 0.0), (Lx, Ly, 0.0), (0.0, Ly, 0.0),
    (0.0, 0.0, Lz),  (Lx, 0.0, Lz),  (Lx, Ly, Lz),  (0.0, Ly, Lz),
])

triangles = [
    connect((1, 3, 2), Triangle), connect((1, 4, 3), Triangle),
    connect((5, 6, 7), Triangle), connect((5, 7, 8), Triangle),
    connect((1, 2, 6), Triangle), connect((1, 6, 5), Triangle),
    connect((3, 4, 8), Triangle), connect((3, 8, 7), Triangle),
    connect((1, 5, 8), Triangle), connect((1, 8, 4), Triangle),
    connect((2, 3, 7), Triangle), connect((2, 7, 6), Triangle),
]

mesh = SimpleMesh(vertices, triangles)
max_edge = sqrt(Lx^2 + Ly^2 + Lz^2)
n_refine = ceil(Int, log2(max_edge / dx))
for _ in 1:n_refine
    mesh = refine(mesh, TriRefinement())
end
println("Refined mesh: $(length(collect(Meshes.elements(mesh)))) triangles")
boundary = PointBoundary(mesh)

dx = 0.5
spacing = ConstantSpacing(dx * mm)
alg = Octree(mesh; spacing)

println("node_min_ratio: $(alg.node_min_ratio)")
println("alpha: $(alg.alpha)")
println("placement: $(alg.placement)")

# Oversample 3× to compensate for points rejected by bounding-box filter (WhatsThePoint#76, #77)
target_vol_pts = round(Int, 3 * Lx * Ly * Lz / dx^3)
cloud = WhatsThePoint.discretize(boundary, spacing; alg, max_points = target_vol_pts)

# Filter volume points against the known domain bounding box
function extract_coords(ptcloud)
    map(WhatsThePoint.points(ptcloud)) do p
        c = coords(p)
        SVector(ustrip(c.x), ustrip(c.y), ustrip(c.z))
    end
end

vol_pts_raw = extract_coords(cloud.volume)
vol_pts = filter(p -> 0 ≤ p[1] ≤ Lx && 0 ≤ p[2] ≤ Ly && 0 ≤ p[3] ≤ Lz, vol_pts_raw)
bnd_pts = extract_coords(WhatsThePoint.boundary(cloud))

println("Raw volume: $(length(vol_pts_raw)), after filter: $(length(vol_pts)), removed: $(length(vol_pts_raw) - length(vol_pts))")
println("Boundary: $(length(bnd_pts))")
println("Total: $(length(bnd_pts) + length(vol_pts))")

# Visualize filtered points (not raw cloud, which contains out-of-domain points)
all_pts = reduce(hcat, vcat(bnd_pts, vol_pts))  # 3 × N matrix
fig = Figure(; size = (1000, 1000));
ax = Axis3(fig[1, 1]; aspect = :data, title = "Filtered point cloud ($(size(all_pts, 2)) pts)")
meshscatter!(ax, all_pts[1, :], all_pts[2, :], all_pts[3, :]; markersize = 0.15)
fig
