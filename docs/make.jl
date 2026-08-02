using Macchiato
using Documenter
using DocumenterVitepress

DocMeta.setdocmeta!(
    Macchiato, :DocTestSetup, :(using Macchiato); recursive = true
)

makedocs(;
    modules = [Macchiato],
    authors = "Kyle Beggs",
    sitename = "Macchiato.jl",
    repo = Documenter.Remotes.GitHub("JuliaMeshless", "Macchiato.jl"),
    format = DocumenterVitepress.MarkdownVitepress(;
        repo = "github.com/JuliaMeshless/Macchiato.jl",
        devbranch = "main",
        devurl = "dev",
        build_vitepress = (!haskey(ENV, "VITEPRESS_DEV"))
    ),
    warnonly = [:missing_docs],
    pages = [
        "Getting Started" => [
            "Quick Start" => "getting_started.md",
            "Custom PDEs" => "custom_pdes.md",
        ],
        "Manual" => [
            "Examples" => "examples.md",
            "Package Design" => "design.md",
        ],
        "API Reference" => "api.md",
    ],
    clean = false
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/JuliaMeshless/Macchiato.jl",
    devbranch = "main",
    push_preview = true
)
