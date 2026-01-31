using Quasar
using Documenter
using DocumenterVitepress

DocMeta.setdocmeta!(Quasar, :DocTestSetup, :(using Quasar); recursive=true)

makedocs(;
    modules=[Quasar],
    authors="Kareem Fareed",
    sitename="Quasar.jl",
    format=DocumenterVitepress.MarkdownVitepress(
        repo="https://github.com/KookiesNKareem/Quasar.jl",
        devbranch="main",
        build_vitepress=false,  # Don't use JLL Node.js - we'll build manually
    ),
    pages=[
        "Home" => "index.md",
        "Getting Started" => [
            "getting-started/installation.md",
            "getting-started/quickstart.md",
        ],
        "Manual" => [
            "manual/backends.md",
            "manual/montecarlo.md",
            "manual/optimization.md",
        ],
        "API Reference" => "api.md",
    ],
    warnonly=true,
)

# Note: deploydocs is not used - we deploy via CI workflow with peaceiris/actions-gh-pages
