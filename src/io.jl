#TODO
function createvtkcells(coords)
    # only save as points/vertexes
    return [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in 1:size(coords, 2)]
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

function exportvtk(filename::String, points::Vector, data::Vector, names::Vector)
    p = reduce(hcat, Meshes._coords.(points))
    cells = createvtkcells(p)
    vtkfile = createvtkfile(filename, p, cells)
    for (name, field) in zip(names, data)
        addfieldvtk!(vtkfile, name, field)
    end
    savevtk!(vtkfile)
    return nothing
end
