using OnlineLearningTools
using Documenter

DocMeta.setdocmeta!(OnlineLearningTools, :DocTestSetup, :(using OnlineLearningTools); recursive=true)

makedocs(;
    modules=[OnlineLearningTools],
    authors="Thomas Dubos <thomas.dubos@polytechnique.edu> and contributors",
    sitename="OnlineLearningTools.jl",
    format=Documenter.HTML(;
        canonical="https://dubosipsl.github.io/OnlineLearningTools.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/dubosipsl/OnlineLearningTools.jl",
    devbranch="main",
)
