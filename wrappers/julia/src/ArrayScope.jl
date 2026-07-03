# Package entry point: the implementation lives one directory up so it can
# also be `include`d directly without Pkg.
include(joinpath(@__DIR__, "..", "ArrayScope.jl"))
