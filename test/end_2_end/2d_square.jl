using WhatsThePoint
import WhatsThePoint as WTP
using Unitful
using Unitful: m, °, ustrip

"""
Creates a 2D square domain for Method of Manufactured Solutions (MMS) tests.
returns a square part with points, normals, and areas defined.
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
    part = PointBoundary(p, n, a)
    split_surface!(part, 75°)  # Strip units - older WhatsThePoint doesn't accept Unitful angles

    return part
end
