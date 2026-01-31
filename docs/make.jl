using Quasar
using Documenter

DocMeta.setdocmeta!(Quasar, :DocTestSetup, :(using Quasar); recursive=true)

makedocs(;
    modules=[Quasar],
    authors="Kareem Fareed",
    sitename="Quasar.jl",
    format=Documenter.HTML(;
        canonical="https://KookiesNKareem.github.io/Quasar.jl",
        edit_link="main",
        assets=String[],
        prettyurls=get(ENV, "CI", "false") == "true",
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

deploydocs(;
    repo="github.com/KookiesNKareem/Quasar.jl",
    devbranch="main",
    push_preview=true,
)
