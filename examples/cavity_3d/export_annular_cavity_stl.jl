using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

import WhatsThePoint as WTP
const _M = WTP.Meshes
const _GeoIO = WTP.GeoIO
using Unitful: m
using Printf

const L_OUT = 1.0
const NSUB  = 2
const r_cav = 0.547
const STL_PATH = joinpath(@__DIR__, "annular_cavity.stl")

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

nθ = 2 * 2^NSUB; nφ = 2 * nθ
outer = _make_sphere_mesh(L_OUT, nθ, nφ)
inner = _make_sphere_mesh(r_cav, nθ, nφ)
outer_v = collect(_M.vertices(outer))
inner_v = collect(_M.vertices(inner))
n_outer = length(outer_v)
all_conn = WTP.Connectivity{WTP.Triangle}[]
for c in _M.topology(outer); push!(all_conn, c); end
for c in _M.topology(inner)
    p = _M.indices(c)
    push!(all_conn, WTP.connect((p[1] + n_outer, p[3] + n_outer, p[2] + n_outer)))
end
mesh = WTP.SimpleMesh(vcat(outer_v, inner_v), all_conn)

isfile(STL_PATH) && rm(STL_PATH)
_GeoIO.save(STL_PATH, _GeoIO.georef(nothing, mesh))
@printf("Exported: %s  (%d vertices, %d triangles)\n",
    STL_PATH, length(_M.vertices(mesh)), _M.nelements(mesh))
