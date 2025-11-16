function viz_2d(
        domain,
        labels;
        size = (1000, 1000),
        colorrange = WhatsThePoint._get_colorrange(labels),
        colormap = :Spectral,
        levels = 32,
        kwargs...
)
    fig = Figure(; size = size)
    ax = Axis(fig[1, 1]; aspect = DataAspect())

    cmap = Makie.cgrad(colormap, levels; categorical = true)

    #boundary
    for b in domain.boundaries
        ids = b.second[1]
        points = pointify(domain.cloud[b.first])
        c = coords.(points)
        x = map(c -> ustrip(c.x), c)
        y = map(c -> ustrip(c.y), c)
        meshscatter!(
            ax,
            ustrip.(x),
            ustrip.(y);
            color = labels[ids],
            shading = Makie.NoShading,
            colorrange = colorrange,
            colormap = cmap,
            kwargs...
        )
    end

    # volume
    start = maximum(domain.boundaries) do b
        b[2][1][end]
    end + 1
    ids = start:(start + length(domain.cloud.volume) - 1)
    c = coords.(domain.cloud.volume.points)
    x = map(c -> ustrip(c.x), c)
    y = map(c -> ustrip(c.y), c)
    meshscatter!(
        ax,
        ustrip.(x),
        ustrip.(y);
        color = labels[ids],
        shading = Makie.NoShading,
        colorrange = colorrange,
        colormap = cmap,
        kwargs...
    )
    Colorbar(fig[1, 2]; colorrange = colorrange, colormap = cmap)
    save("visualization.png", fig)
    return nothing
end

function viz_3d(
        domain,
        labels,
        f_filter = x -> true;
        size = (1000, 1000),
        colorrange = WhatsThePoint._get_colorrange(labels),
        azimuth = 1.275π,
        elevation = π / 8,
        colormap = :Spectral,
        levels = 32,
        kwargs...
)
    fig = Figure(; size = size)
    ax = Axis3(fig[1, 1]; azimuth = azimuth, elevation = elevation)
    ax.aspect = :data

    cmap = Makie.cgrad(colormap, levels; categorical = true)

    for b in domain.boundaries
        ids = b.second[1]
        points = pointify(domain.cloud[b.first])
        c = coords.(points)
        filtered_ids = findall(f_filter, c)
        c = c[filtered_ids]
        x = map(c -> ustrip(c.x), c)
        y = map(c -> ustrip(c.y), c)
        z = map(c -> ustrip(c.z), c)
        meshscatter!(
            ax,
            ustrip.(x),
            ustrip.(y),
            ustrip.(z);
            color = labels[ids][filtered_ids],
            colorrange = colorrange,
            colormap = cmap,
            kwargs...
        )
    end

    # volume
    start = maximum(domain.boundaries) do b
        b[2][1][end]
    end + 1
    ids = start:(start + length(domain.cloud.volume) - 1)
    c = coords.(domain.cloud.volume.points)
    filtered_ids = findall(f_filter, c)
    c = c[filtered_ids]
    x = map(c -> ustrip(c.x), c)
    y = map(c -> ustrip(c.y), c)
    z = map(c -> ustrip(c.z), c)
    meshscatter!(
        ax,
        ustrip.(x),
        ustrip.(y),
        ustrip.(z);
        color = labels[ids][filtered_ids],
        colorrange = colorrange,
        colormap = cmap,
        kwargs...
    )
    Colorbar(fig[1, 2]; colorrange = colorrange, colormap = cmap)
    return fig
end