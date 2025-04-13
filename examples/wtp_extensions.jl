using WhatsThePoint
import WhatsThePoint: _angle
using Meshes
using StaticArrays
using LinearAlgebra
using Unitful

# Extension for 2D Meshes.Vec with Unitful quantities
function _angle(u::Meshes.Vec{2, <:Unitful.Quantity}, v::Meshes.Vec{2, <:Unitful.Quantity})
    # Convert Meshes.Vec to SVector for compatibility
    u_sv = SVector{2}(u.coords)
    v_sv = SVector{2}(v.coords)
    # Use the existing implementation
    return _angle(u_sv, v_sv)
end

# Extension for 3D Meshes.Vec with Unitful quantities (just in case)
function _angle(u::Meshes.Vec{3, <:Unitful.Quantity}, v::Meshes.Vec{3, <:Unitful.Quantity})
    # Convert Meshes.Vec to SVector for compatibility
    u_sv = SVector{3}(u.coords)
    v_sv = SVector{3}(v.coords)
    # Use the existing implementation
    return _angle(u_sv, v_sv)
end