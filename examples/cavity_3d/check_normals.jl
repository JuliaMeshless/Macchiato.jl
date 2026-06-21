using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

import WhatsThePoint as WTP
const _M = WTP.Meshes
const _GeoIO = WTP.GeoIO
using LinearAlgebra: norm, dot
using StaticArrays: SVector
using Printf
using Unitful: ustrip

function check_normals()
    STL_PATH = joinpath(@__DIR__, "annular_cavity.stl")
    mesh = _GeoIO.load(STL_PATH).geometry
    ntri = _M.nelements(mesh)
    @printf("Loaded %s: %d vertices, %d triangles\n", STL_PATH, length(collect(_M.vertices(mesh))), ntri)

    to_svec(v) = SVector{3,Float64}(ustrip(v[1]), ustrip(v[2]), ustrip(v[3]))
    elems = collect(_M.elements(mesh))

    mid = 0.77
    n_outer_ok = 0; n_outer_bad = 0
    n_cavity_ok = 0; n_cavity_bad = 0
    worst_outer = (0, -1.0)
    worst_cavity = (0, -1.0)

    for i in 1:ntri
        nv = to_svec(_M.normal(elems[i]))
        cv = to_svec(_M.to(_M.centroid(elems[i])))
        r = norm(cv)
        dp = dot(nv, cv / r)

        if r > mid
            if dp > 0
                n_outer_ok += 1
            else
                n_outer_bad += 1
                dp < worst_outer[2] && (worst_outer = (i, dp))
            end
        else
            if dp < 0
                n_cavity_ok += 1
            else
                n_cavity_bad += 1
                dp > worst_cavity[2] && (worst_cavity = (i, dp))
            end
        end
    end

    @printf("\n--- Normal orientation check ---\n")
    @printf("Outer sphere (r > %.2f):  %d correct (outward),  %d WRONG\n", mid, n_outer_ok, n_outer_bad)
    @printf("Cavity      (r < %.2f):  %d correct (inward),   %d WRONG\n", mid, n_cavity_ok, n_cavity_bad)
    n_outer_bad > 0 && @printf("  worst outer: tri %d, dot=%.4f\n", worst_outer...)
    n_cavity_bad > 0 && @printf("  worst cavity: tri %d, dot=%.4f\n", worst_cavity...)
    @printf("\nVERDICT: %s\n",
        (n_outer_bad == 0 && n_cavity_bad == 0) ?
        "ALL NORMALS CORRECT" :
        "$(n_outer_bad + n_cavity_bad) normals MISORIENTED")
end

check_normals()
