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

function _coords(cloud::PointCloud{𝔼{3}})
    map(pointify(cloud)) do p
        c = coords(p)
        SVector(c.x, c.y, c.z)
    end
end
