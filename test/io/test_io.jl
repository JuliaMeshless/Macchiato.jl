using Test
using Macchiato
import Macchiato as MM
using WhatsThePoint
import WhatsThePoint as WTP
using Unitful: m, °

include(joinpath(@__DIR__, "..", "end_2_end", "2d_square.jl"))

function create_io_test_points()
    dx = 1 / 9 * m
    part = create_2d_square_domain(dx)
    cloud = WTP.discretize(part, ConstantSpacing(dx), alg = VanDerSandeFornberg())
    return WTP.points(cloud)
end

@testset "VTK output" begin
    io_pts = create_io_test_points()
    io_N = length(io_pts)

    mktempdir() do dir
        @testset "exportvtk writes a vtu file" begin
            field = Float64.(1:io_N)
            ret = exportvtk(joinpath(dir, "out"), io_pts, [field], ["T"])
            @test ret === nothing
            outfile = joinpath(dir, "out.vtu")
            @test isfile(outfile)
            @test filesize(outfile) > 0
        end

        @testset "exportvtk embeds field names" begin
            T = Float64.(1:io_N)
            p = fill(2.5, io_N)
            exportvtk(joinpath(dir, "fields"), io_pts, [T, p], ["T", "p"])
            s = read(joinpath(dir, "fields.vtu"), String)
            @test occursin("Name=\"T\"", s)
            @test occursin("Name=\"p\"", s)
        end

        @testset "savevtk! finalizes a manually built grid" begin
            coords = reduce(hcat, MM._ustrip(MM._coords(io_pts)))
            cells = MM.createvtkcells(coords)
            vtk = MM.createvtkfile(joinpath(dir, "manual"), coords, cells)
            MM.addfieldvtk!(vtk, "T", Float64.(1:io_N))
            MM.savevtk!(vtk)
            @test isfile(joinpath(dir, "manual.vtu"))
        end

        @testset "pvd collection smoke" begin
            pvd = MM.createpvd(joinpath(dir, "series"))
            coords = reduce(hcat, MM._ustrip(MM._coords(io_pts)))
            cells = MM.createvtkcells(coords)
            vtk = MM.createvtkfile(joinpath(dir, "step0"), coords, cells)
            MM.pvdappend!(pvd, 0.0, vtk)
            MM.savevtk!(pvd)
            @test isfile(joinpath(dir, "series.pvd"))
        end
    end
end
