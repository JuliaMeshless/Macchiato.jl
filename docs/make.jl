using MeshlessMultiphysics
using Documenter

DocMeta.setdocmeta!(
    MeshlessMultiphysics, :DocTestSetup, :(using MeshlessMultiphysics); recursive = true
)

makedocs(;
    modules = [MeshlessMultiphysics],
    authors = "Kyle Beggs",
    sitename = "MeshlessMultiphysics.jl",
    repo = Documenter.Remotes.GitHub("JuliaMeshless", "MeshlessMultiphysics.jl"),
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://JuliaMeshless.github.io/MeshlessMultiphysics.jl",
        edit_link = "main",
        assets = String[]
    ),
    warnonly = [:missing_docs],
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Examples" => "examples.md",
        "Package Design" => "design.md",
        "API Reference" => "api.md"
    ]
)

deploydocs(;
    repo = "github.com/JuliaMeshless/MeshlessMultiphysics.jl",
    devbranch = "main",
    versions = ["stable" => "v^", "dev" => "dev"]
)
