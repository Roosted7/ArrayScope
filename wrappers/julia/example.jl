# ArrayScope from Julia — examples.
#
# Setup (one of):
#   include("ArrayScope.jl"); using .ArrayScope
#   # or, as a package: ] dev /path/to/ArrayScope/wrappers/julia ; using ArrayScope
#
# Requires the viewer itself: pip install arrayscope

include(joinpath(@__DIR__, "ArrayScope.jl"))
using .ArrayScope

# A synthetic complex 4-D "MRI-like" dataset: x, y, slice, coil.
x = range(-1, 1; length=192)
y = range(-1, 1; length=160)
vol = [exp(-4 * (xi^2 + yi^2)) * cis(6π * xi * s / 8 + c) for
       xi in x, yi in y, s in 1:12, c in 1:4]
vol = ComplexF32.(vol)

# Non-blocking (default): the REPL stays usable, the viewer opens detached.
arrayscope(vol; name="demo_kspace")

# Blocking, e.g. at the end of a batch script:
# arrayscope(abs.(vol); name="demo_magnitude", block=true)

# Large-array tips:
# - Dense `Array`s are written with one raw buffer write and memory-mapped by
#   the viewer; views/BitArrays are materialized once first.
# - On Linux: ENV["ARRAYSCOPE_HANDOFF_DIR"] = "/dev/shm" keeps the handoff in RAM.
