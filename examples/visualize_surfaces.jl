function visualize_surfaces(part; surfaces_to_plot = (:surface1, :surface2, :surface3),
        colors = Dict(:surface1 => :red, :surface2 => :blue, :surface3 => :green),
        size = (600, 600))
    surfaces = part.surfaces
    fig = Figure(size = size)
    ax = Axis(fig[1, 1]; aspect = DataAspect())
    for s in surfaces_to_plot
        xs = [g.point.coords.x for g in surfaces[s].geoms]
        ys = [g.point.coords.y for g in surfaces[s].geoms]
        meshscatter!(ax, xs, ys; color = colors[s], markersize = 12, label = string(s))
    end
    Makie.axislegend(ax)
    return fig
end