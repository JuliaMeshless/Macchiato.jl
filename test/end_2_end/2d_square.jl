using WhatsThePoint
import WhatsThePoint as WTP
using Unitful
using Unitful: m, °, ustrip

"""
    create_2d_square_domain(dx) -> PointBoundary

The unit square's boundary as a **single closed counter-clockwise loop**, with outward
normals and per-point areas.

The loop is deliberately left un-split. `WhatsThePoint.discretize` builds a
`SegmentQuadtree`, which treats every named surface of a `PointBoundary` as its own
closed loop and validates each one's shoelace signed area. Splitting the square into
its four edges first would hand it four *open*, collinear segments and trip that check.
Use [`create_2d_square_cloud`](@ref), which splits after discretizing.
"""
function create_2d_square_domain(dx::Unitful.Length = 1 / 129 * m)
    L = (1m, 1m)

    rx = dx:dx:(L[1] - dx)
    ry = dx:dx:(L[2] - dx)

    p_bot = map(i -> WTP.Point(i, 0m), rx)
    p_right = map(i -> WTP.Point(L[1], i), ry)
    p_top = map(i -> WTP.Point(i, L[2]), reverse(rx))
    p_left = map(i -> WTP.Point(0m, i), reverse(ry))

    n_bot = map(i -> WTP.Vec(0.0, -1.0), rx)
    n_right = map(i -> WTP.Vec(1.0, 0.0), ry)
    n_top = map(i -> WTP.Vec(0.0, 1.0), rx)
    n_left = map(i -> WTP.Vec(-1.0, 0.0), ry)

    p = vcat(p_bot, p_right, p_top, p_left) # points
    n = vcat(n_bot, n_right, n_top, n_left) # normals
    a = fill(dx, length(p))

    return PointBoundary(p, n, a)
end

"""
    create_2d_square_cloud(dx) -> PointCloud

Point cloud for the unit square, with its four edges labelled `:surface1` (bottom),
`:surface2` (right), `:surface3` (top), `:surface4` (left) so each can carry its own
boundary condition.

Discretizing and labelling are separate steps on purpose: the geometry must stay one
closed loop while the quadtree is built, and the edge labels are only needed afterwards,
to key the BC dictionary.
"""
function create_2d_square_cloud(dx::Unitful.Length = 1 / 129 * m)
    cloud = WTP.discretize(create_2d_square_domain(dx), ConstantSpacing(dx))
    split_surface!(cloud, 75°)  # label the four edges by normal discontinuity
    return cloud
end
