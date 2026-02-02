using QuantNova
using Documenter
using DocumenterVitepress

DocMeta.setdocmeta!(QuantNova, :DocTestSetup, :(using QuantNova); recursive=true)

makedocs(;
    modules=[QuantNova],
    authors="Kareem Fareed",
    sitename="QuantNova.jl",
    format=DocumenterVitepress.MarkdownVitepress(
        repo="github.com/KookiesNKareem/QuantNova.jl",
        devbranch="main",
        devurl="dev",
        build_vitepress=false,  # Don't use JLL Node.js - we'll build manually
    ),
    pages=[
        "Home" => "index.md",
        "Getting Started" => [
            "getting-started/installation.md",
            "getting-started/quickstart.md",
        ],
        "Examples" => [
            "examples/pricing-calibration-demo.md",
            "examples/backtesting-demo.md",
            "examples/optimization-demo.md",
            "examples/option-pricing.md",
            "examples/portfolio-risk.md",
            "examples/monte-carlo-exotic.md",
            "examples/yield-curve.md",
        ],
        "Manual" => [
            "manual/backends.md",
            "manual/montecarlo.md",
            "manual/optimization.md",
            "manual/interest-rates.md",
        ],
        "API Reference" => "api.md",
    ],
    warnonly=true,
)
