"""
    createvtkcells(coords)

Create VTK vertex cells for a point-based dataset. Each point becomes a single `VTK_VERTEX` cell.
"""
function createvtkcells(coords)
    return [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in 1:size(coords, 2)]
end

"""
    createvtkfile(filename, coords, cells)

Create a VTK grid file from coordinates and cell definitions. Returns a VTK file handle.
"""
function createvtkfile(filename::String, coords, cells)
    return vtk_grid(filename, coords, cells)
end

"""
    createmultiblockvtk(filename)

Create a VTK multiblock file for grouping multiple grids.
"""
function createmultiblockvtk(filename::String)
    return vtk_multiblock(filename)
end

"""
    addgrid!(vtmfile, grid)

Add a grid block to a VTK multiblock file.
"""
function addgrid!(vtmfile, grid)
    return vtk_grid(vtmfile, grid)
end

"""
    addfieldvtk!(vtkfile, scalarname, data)

Attach point data to a VTK file under the given field name.
"""
function addfieldvtk!(vtkfile, scalarname::String, data)
    return vtkfile[scalarname, VTKPointData()] = data
end

"""
    savevtk!(vtkfile)

Write the VTK file to disk.
"""
function savevtk!(vtkfile)
    return vtk_save(vtkfile)
end

"""
    createpvd(filename; append=true)

Create or open a ParaView Data (`.pvd`) collection file for time-series output.
"""
function createpvd(filename::String; append = true)
    if isfile(filename * ".pvd")
        pvd = paraview_collection(filename; append = append)
    else
        pvd = paraview_collection(filename)
    end
    return pvd
end

"""
    pvdappend!(pvd, time, vtkfile)

Append a VTK file to a PVD collection at the given simulation time.
"""
function pvdappend!(pvd, time, vtkfile)
    return pvd[time] = vtkfile
end

"""
    exportvtk(filename, points, data, names)

Export point-based simulation results to a VTK file.

# Arguments
- `filename::String`: Output file path (without `.vtu` extension)
- `points::AbstractVector`: Point cloud or coordinate vector
- `data::AbstractVector{<:AbstractVector}`: Field data arrays to export
- `names::AbstractVector`: Corresponding field names (e.g., `["T", "u"]`)

# Example
```julia
exportvtk("results/temperature", points(cloud), [T_values], ["T"])
```
"""
function exportvtk(
        filename::String, points::AbstractVector,
        data::AbstractVector{<:AbstractVector}, names::AbstractVector
    )
    p = reduce(hcat, _ustrip(_coords(points)))
    cells = createvtkcells(p)
    vtkfile = createvtkfile(filename, p, cells)
    for (name, field) in zip(names, data)
        addfieldvtk!(vtkfile, name, field)
    end
    savevtk!(vtkfile)
    return nothing
end
