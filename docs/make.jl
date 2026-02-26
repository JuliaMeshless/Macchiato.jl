using Macchiato
using Documenter

DocMeta.setdocmeta!(
    Macchiato, :DocTestSetup, :(using Macchiato); recursive = true
)

makedocs(;
    modules = [Macchiato],
    authors = "Kyle Beggs",
    sitename = "Macchiato.jl",
    repo = Documenter.Remotes.GitHub("JuliaMeshless", "Macchiato.jl"),
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://JuliaMeshless.github.io/Macchiato.jl",
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
    repo = "github.com/JuliaMeshless/Macchiato.jl",
    devbranch = "main",
    versions = ["stable" => "v^", "dev" => "dev"]
)
