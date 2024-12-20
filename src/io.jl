#TODO
using Delaunay
function createvtkcells(coords, triangulate = true, nonconvex = false)
    if triangulate
        # compute delaunay triangulation so you can view in paraview as a surface
        p = Matrix(transpose(coords))
        conn = convert(Matrix{Int32}, delaunay(p).simplices)

        if nonconvex
            keep = Int32[]
            for i in axes(conn, 1)
                center = [mean(coords[1, conn[i, :]]), mean(coords[2, conn[i, :]])]
                if isinside2d(center, pointsboundary)
                    push!(keep, i)
                end
            end
            conn2 = zeros(Int32, 3, size(keep, 1))
            for i in axes(keep, 1)
                conn2[:, i] = conn[keep[i], :]
            end
            cells = MeshCell{VTKCellType, Vector{Int32}}[]
            for i in axes(conn2, 2)
                push!(cells, MeshCell(VTKCellTypes.VTK_TRIANGLE, conn2[:, i]))
            end
            #return cells, keep, del
            return cells
        end

        cells = MeshCell{VTKCellType, Vector{Int32}}[]
        for i in axes(conn, 1)
            push!(cells, MeshCell(VTKCellTypes.VTK_TRIANGLE, conn[i, :]))
        end
        return cells
    else
        # only save as points/vertexes
        cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in 1:size(coords, 2)]
        return cells
    end
end

function createvtkfile(filename::String, coords, cells)
    return vtk_grid(filename, coords, cells)
end

function createmultiblockvtk(filename::String)
    return vtk_multiblock(filename)
end

function addgrid!(vtmfile, grid)
    return vtk_grid(vtmfile, grid)
end

function addfieldvtk!(vtkfile, scalarname::String, data)
    return vtkfile[scalarname, VTKPointData()] = data
end

function savevtk!(vtkfile)
    return vtk_save(vtkfile)
end

function createpvd(filename::String; append = true)
    if isfile(filename * ".pvd")
        pvd = paraview_collection(filename; append = append)
    else
        pvd = paraview_collection(filename)
    end
    return pvd
end

function pvdappend!(pvd, time, vtkfile)
    return pvd[time] = vtkfile
end

function exportvtk(filename::String, points::Vector, data::Vector, names::Vector;
        createcells = false)
    p = reduce(hcat, Meshes._coords.(points))
    cells = createvtkcells(p, createcells)
    vtkfile = createvtkfile(filename, p, cells)
    for (name, field) in zip(names, data)
        addfieldvtk!(vtkfile, name, field)
    end
    savevtk!(vtkfile)
    return nothing
end
