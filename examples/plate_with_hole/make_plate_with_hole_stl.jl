# Generate a watertight binary STL of a 3D rectangular plate (W×H×T, T small)
# with a centered elliptical through-hole. This is the *initial* geometry for the
# plate-with-hole shape-optimization problem (Stage 1: ellipse → circle), and the
# bridge to the STL-driven pipeline: this surface feeds WhatsThePoint point
# generation, then the hole wall is the design boundary.
#
# Self-contained (stdlib only). Run:  julia --startup-file=no make_plate_with_hole_stl.jl
#
# Geometry: an O-grid annulus (rectangle minus ellipse) on the top & bottom
# faces, plus the inner (hole) wall and outer (rectangle) wall through the
# thickness. Both boundaries are sampled along the SAME polar rays, so radial
# connectors never cross and the annulus is valid. Corner rays are inserted so
# the rectangle edges stay flat.

using Printf

# ---- parameters (set A == B for a circular hole) --------------------------
const W = 2.0      # plate width   (x)
const H = 2.0      # plate height  (y)
const T = 0.1      # plate thickness (z) — "small"
const A = 0.4      # hole semi-axis x
const B = 0.2      # hole semi-axis y  (A:B = 2:1 ellipse, optimizes toward circle)
const NPHI = 96    # angular samples around the hole
const NRAD = 8     # radial layers across the annulus (ellipse → rectangle)
const OUT  = joinpath(@__DIR__, "plate_with_hole.stl")

# ---- tiny 3-vector helpers (avoid any package dependency) -----------------
sub3(a, b) = (a[1] - b[1], a[2] - b[2], a[3] - b[3])
dot3(a, b) = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
cross3(a, b) = (a[2] * b[3] - a[3] * b[2], a[3] * b[1] - a[1] * b[3], a[1] * b[2] - a[2] * b[1])
nrm3(a) = sqrt(dot3(a, a))

# ---- angular samples: uniform + the 4 rectangle-corner rays ---------------
phis = Float64[2π * j / NPHI for j in 0:(NPHI - 1)]
for (sx, sy) in ((1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0))
    push!(phis, mod2pi(atan(sy * H / 2, sx * W / 2)))
end
sort!(phis)
let kept = Float64[phis[1]]                       # dedupe near-coincident rays
    for φ in phis[2:end]
        (φ - kept[end] > 1e-9) && push!(kept, φ)
    end
    global phis = kept
end
const M = length(phis)

# ray-to-rectangle radius along polar angle φ
function r_rect(φ)
    c, s = cos(φ), sin(φ)
    tx = abs(c) < 1e-15 ? Inf : (W / 2) / abs(c)
    ty = abs(s) < 1e-15 ? Inf : (H / 2) / abs(s)
    return min(tx, ty)
end

# ---- build the (NRAD+1) × M point grid on top and bottom faces ------------
const zt, zb = T / 2, -T / 2
top = Array{NTuple{3, Float64}}(undef, NRAD + 1, M)
bot = Array{NTuple{3, Float64}}(undef, NRAD + 1, M)
for j in 1:M
    φ = phis[j]; c, s = cos(φ), sin(φ)
    rin  = 1 / sqrt((c / A)^2 + (s / B)^2)        # ray-to-ellipse radius
    rout = r_rect(φ)
    for i in 0:NRAD
        rr = rin + (rout - rin) * (i / NRAD)
        x, y = rr * c, rr * s
        top[i + 1, j] = (x, y, zt)
        bot[i + 1, j] = (x, y, zb)
    end
end

# ---- triangle soup, oriented outward via an outward reference -------------
const tris = Vector{NTuple{4, NTuple{3, Float64}}}()   # (normal, v1, v2, v3)

function push_tri!(p1, p2, p3, outward)
    n = cross3(sub3(p2, p1), sub3(p3, p1))
    L = nrm3(n)
    L < 1e-20 && return                             # drop degenerate sliver
    n = (n[1] / L, n[2] / L, n[3] / L)
    if dot3(n, outward) < 0                         # flip to face outward
        p2, p3 = p3, p2
        n = (-n[1], -n[2], -n[3])
    end
    push!(tris, (n, p1, p2, p3))
    return
end
quad!(p1, p2, p3, p4, outward) =
    (push_tri!(p1, p2, p3, outward); push_tri!(p1, p3, p4, outward))

next(j) = j == M ? 1 : j + 1

for j in 1:M, i in 1:NRAD                            # top face (annulus), +z
    jn = next(j)
    quad!(top[i, j], top[i, jn], top[i + 1, jn], top[i + 1, j], (0.0, 0.0, 1.0))
end
for j in 1:M, i in 1:NRAD                            # bottom face (annulus), -z
    jn = next(j)
    quad!(bot[i, j], bot[i, jn], bot[i + 1, jn], bot[i + 1, j], (0.0, 0.0, -1.0))
end
for j in 1:M                                         # inner wall (hole), normal toward axis
    jn = next(j)
    a, b = top[1, j], top[1, jn]
    mx, my = (a[1] + b[1]) / 2, (a[2] + b[2]) / 2
    r = hypot(mx, my)
    quad!(a, b, bot[1, jn], bot[1, j], (-mx / r, -my / r, 0.0))
end
for j in 1:M                                         # outer wall (rectangle), normal away
    jn = next(j)
    a, b = top[NRAD + 1, j], top[NRAD + 1, jn]
    mx, my = (a[1] + b[1]) / 2, (a[2] + b[2]) / 2
    r = hypot(mx, my)
    quad!(a, b, bot[NRAD + 1, jn], bot[NRAD + 1, j], (mx / r, my / r, 0.0))
end

# ---- write binary STL -----------------------------------------------------
open(OUT, "w") do io
    write(io, zeros(UInt8, 80))                      # 80-byte header
    write(io, UInt32(length(tris)))                  # triangle count
    for (n, p1, p2, p3) in tris
        for v in (n, p1, p2, p3)
            write(io, Float32(v[1])); write(io, Float32(v[2])); write(io, Float32(v[3]))
        end
        write(io, UInt16(0))                         # attribute byte count
    end
end

# ---- validate: watertight (each edge shared exactly twice) + Euler χ ------
vkey(p) = (round(Int, p[1] * 1e6), round(Int, p[2] * 1e6), round(Int, p[3] * 1e6))
verts = Dict{NTuple{3, Int}, Int}()
vid(p) = get!(verts, vkey(p), length(verts) + 1)
edges = Dict{Tuple{Int, Int}, Int}()
addedge(u, v) = (e = u < v ? (u, v) : (v, u); edges[e] = get(edges, e, 0) + 1)
for (_, p1, p2, p3) in tris
    i1, i2, i3 = vid(p1), vid(p2), vid(p3)
    addedge(i1, i2); addedge(i2, i3); addedge(i3, i1)
end
V, E, F = length(verts), length(edges), length(tris)
nonmanifold = count(!=(2), values(edges))
χ = V - E + F

allv = [(p[1], p[2], p[3]) for (_, a, b, c) in tris for p in (a, b, c)]
bbmin = (minimum(p[1] for p in allv), minimum(p[2] for p in allv), minimum(p[3] for p in allv))
bbmax = (maximum(p[1] for p in allv), maximum(p[2] for p in allv), maximum(p[3] for p in allv))

println("Wrote: ", OUT)
@printf("  plate %.3g × %.3g × %.3g   hole ellipse a=%.3g b=%.3g  (%s)\n",
        W, H, T, A, B, A == B ? "circle" : "ellipse $(round(A/B,digits=2)):1")
@printf("  triangles F=%d   unique verts V=%d   edges E=%d\n", F, V, E)
@printf("  bbox  x[%.3f,%.3f] y[%.3f,%.3f] z[%.3f,%.3f]\n",
        bbmin[1], bbmax[1], bbmin[2], bbmax[2], bbmin[3], bbmax[3])
@printf("  non-manifold edges = %d   (watertight ⇔ 0)\n", nonmanifold)
@printf("  Euler χ = V−E+F = %d   (plate w/ through-hole = torus ⇒ expect 0)\n", χ)
if nonmanifold == 0 && χ == 0
    println("  ✓ closed, watertight, genus-1 surface")
else
    println("  ✗ topology check failed — inspect before using")
end
