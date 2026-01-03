function findmin_turbo(x)
    indmin = 0
    minval = typemax(eltype(x))
    @turbo for i in eachindex(x)
        newmin = x[i] < minval
        minval = newmin ? x[i] : minval
        indmin = newmin ? i : indmin
    end
    return minval, indmin
end

function findmin_turbo(x, ids)
    indmin = 0
    minval = typemax(eltype(x))
    @turbo for i in eachindex(ids)
        id = ids[i]
        newmin = x[id] < minval
        minval = newmin ? x[id] : minval
        indmin = newmin ? id : indmin
    end
    return minval, indmin
end

function _coords(cloud::Union{PointCloud{𝔼{2}}, PointSurface{𝔼{2}}})
    map(points(cloud)) do p
        c = coords(p)
        SVector(c.x, c.y)
    end
end

function _coords(points::AbstractVector{<:Point{𝔼{2}}})
    map(points) do p
        c = coords(p)
        SVector(c.x, c.y)
    end
end

function _coords(cloud::Union{PointCloud{𝔼{3}}, PointSurface{𝔼{3}}})
    pts = points(cloud)
    map(pts) do p
        c = coords(p)
        SVector(c.x, c.y, c.z)
    end
end

function _coords(points::AbstractVector{<:Point{𝔼{3}}})
    map(points) do p
        c = coords(p)
        SVector(c.x, c.y, c.z)
    end
end

_coords(vol::PointVolume) = _coords(points(vol))

_ustrip(x::AbstractVector{<:AbstractVector}) = map(x -> ustrip.(x), x)

function get_node_coords(cloud::Union{PointCloud{𝔼{2}}, PointSurface{𝔼{2}}}, i)
    p = points(cloud)[i]
    c = coords(p)
    return ustrip.(SVector(c.x, c.y))
end

function get_node_coords(cloud::Union{PointCloud{𝔼{3}}, PointSurface{𝔼{3}}}, i)
    p = points(cloud)[i]
    c = coords(p)
    return ustrip.(SVector(c.x, c.y, c.z))
end
