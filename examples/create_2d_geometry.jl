using WhatsThePoint
const WTP = WhatsThePoint
using Unitful: m, °

function create_2d_geometry()
    L = (1m, 1m)
    dx = 1 / 129 * m # boundary point spacing
    S = ConstantSpacing(dx)
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
    p = vcat(p_bot, p_right, p_top, p_left)
    n = vcat(n_bot, n_right, n_top, n_left)
    a = fill(dx, length(p))
    part = PointBoundary(p, n, a)
    split_surface!(part, 75°)
    combine_surfaces!(part, :surface3, :surface4)
    Δ = dx
    cloud = WhatsThePoint.discretize(part, ConstantSpacing(Δ), alg = VanDerSandeFornberg())
    conv = repel!(cloud, ConstantSpacing(Δ); α = Δ / 20, max_iters = 1000)
    return part, cloud, Δ
end