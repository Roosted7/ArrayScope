"""
    ArrayScope

Julia invocation wrapper for the ArrayScope viewer (a Python/Qt application).

The wrapper hands the array to a separate viewer process through a raw `.npy`
file and returns immediately, so the Julia session stays responsive and no Qt
event loop ever runs inside Julia.

Efficiency contract (see `docs/invocation.md` in the ArrayScope repository):

- The array is written with a single raw `write` of its memory buffer — no
  per-element conversion, no compression, no serialization framework. Julia
  arrays are column-major, which `.npy` expresses natively as
  `fortran_order: True`, so no transpose or reorder happens on either side.
- The viewer is launched with `--mmap`, so it memory-maps the file
  (copy-on-write) instead of reading it eagerly. Pages are shared with the OS
  page cache that the write just populated; for large arrays nothing is read
  back from disk while it is still cached.
- `--consume` makes the viewer unlink the handoff file as soon as it is
  mapped (best effort on Windows, where mapped files cannot be unlinked; a
  stale-file janitor in this wrapper cleans those up on the next call).
- On Linux, point `ARRAYSCOPE_HANDOFF_DIR` at `/dev/shm` to keep the handoff
  entirely in shared memory.

Usage:

    include("ArrayScope.jl"); using .ArrayScope
    arrayscope(rand(64, 64, 32))
    arrayscope(A; name="kspace", block=true)
"""
module ArrayScope

export arrayscope

const _SUPPORTED_ELTYPES = Union{
    Bool,
    Int8, Int16, Int32, Int64,
    UInt8, UInt16, UInt32, UInt64,
    Float16, Float32, Float64,
    ComplexF32, ComplexF64,
}

_npy_descr(::Type{Bool})       = "|b1"
_npy_descr(::Type{Int8})       = "|i1"
_npy_descr(::Type{UInt8})      = "|u1"
_npy_descr(::Type{Int16})      = "<i2"
_npy_descr(::Type{UInt16})     = "<u2"
_npy_descr(::Type{Int32})      = "<i4"
_npy_descr(::Type{UInt32})     = "<u4"
_npy_descr(::Type{Int64})      = "<i8"
_npy_descr(::Type{UInt64})     = "<u8"
_npy_descr(::Type{Float16})    = "<f2"
_npy_descr(::Type{Float32})    = "<f4"
_npy_descr(::Type{Float64})    = "<f8"
_npy_descr(::Type{ComplexF32}) = "<c8"
_npy_descr(::Type{ComplexF64}) = "<c16"
_npy_descr(::Type{T}) where {T} = throw(ArgumentError(
    "arrayscope: unsupported element type $T. Supported: Bool, " *
    "Int8..Int64, UInt8..UInt64, Float16/32/64, ComplexF32/64."))

function _npy_shape_str(dims::Dims)
    length(dims) == 1 && return "($(dims[1]),)"
    return "(" * join(dims, ", ") * ")"
end

"""
    write_npy(path, A::Array)

Write a dense array as an uncompressed NumPy `.npy` (format 1.0) file.

The data section is emitted with one raw `write(io, A)` of the array's
memory, declared as `fortran_order: True` — Julia's native column-major
layout — so this is a single sequential pass with zero element-wise work.
"""
function write_npy(path::AbstractString, A::Array{T}) where {T<:_SUPPORTED_ELTYPES}
    ENDIAN_BOM == 0x04030201 || error(
        "arrayscope: only little-endian hosts are supported by the .npy handoff.")

    dict = "{'descr': '$(_npy_descr(T))', 'fortran_order': True, " *
           "'shape': $(_npy_shape_str(size(A))), }"
    # magic(6) + version(2) + header-length field(2) + header, padded with
    # spaces so the data section starts on a 64-byte boundary; header ends '\n'.
    unpadded = 6 + 2 + 2 + length(dict) + 1
    pad = (64 - unpadded % 64) % 64
    header = dict * repeat(" ", pad) * "\n"
    length(header) <= typemax(UInt16) || error("arrayscope: .npy header too large")

    open(path, "w") do io
        write(io, UInt8[0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59])  # \x93NUMPY
        write(io, UInt8(1), UInt8(0))                          # format 1.0
        write(io, htol(UInt16(length(header))))
        write(io, header)
        write(io, A)                                           # raw buffer, one pass
    end
    return path
end

function _handoff_dir(dir::Union{Nothing,AbstractString})
    d = dir !== nothing ? String(dir) :
        get(ENV, "ARRAYSCOPE_HANDOFF_DIR", joinpath(tempdir(), "arrayscope-handoff"))
    mkpath(d)
    return d
end

"""Best-effort removal of handoff files older than `max_age_s` (default 24h).

Covers files the viewer could not `--consume` (e.g. Windows, or a viewer that
was closed before loading)."""
function _clean_stale_handoffs(dir::AbstractString; max_age_s::Real=86_400)
    now = time()
    for f in readdir(dir; join=true)
        endswith(f, ".npy") || continue
        try
            now - mtime(f) > max_age_s && rm(f; force=true)
        catch
        end
    end
    return nothing
end

_sanitize(name::AbstractString) =
    (s = replace(String(name), r"[^A-Za-z0-9_.-]" => "_"); isempty(s) ? "array" : s)

function _viewer_command(exe::Union{Nothing,Cmd,AbstractString})
    exe isa Cmd && return exe
    exe isa AbstractString && return Cmd([String(exe)])
    env_exe = get(ENV, "ARRAYSCOPE_EXE", nothing)
    env_exe !== nothing && !isempty(env_exe) && return Cmd([env_exe])
    Sys.which("arrayscope") !== nothing && return `arrayscope`
    for py in ("python3", "python")
        Sys.which(py) !== nothing && return `$py -m arrayscope`
    end
    error("arrayscope: viewer executable not found. Install it with " *
          "`pip install arrayscope`, or set ARRAYSCOPE_EXE / pass exe=... ")
end

"""
    arrayscope(A; name="A", block=false, mmap=true, keep=false, dir=nothing, exe=nothing)

View an array in ArrayScope. Non-blocking by default: writes a raw `.npy`
handoff file and launches a detached viewer process.

Keywords:

- `name`: window/handoff name (defaults to a generic name; pass the variable
  name for a recognizable title).
- `block`: wait for the viewer window to close before returning.
- `mmap`: let the viewer memory-map the file (copy-on-write) instead of an
  eager read. Recommended for large arrays; disable only when the handoff dir
  is on slow storage that should be released immediately.
- `keep`: do not ask the viewer to delete the handoff file after loading.
  Returns the path either way.
- `dir`: handoff directory (default `ARRAYSCOPE_HANDOFF_DIR` or
  `<tempdir>/arrayscope-handoff`; on Linux consider `/dev/shm`).
- `exe`: viewer command (`String` or `Cmd`). Default: `ARRAYSCOPE_EXE`, then
  `arrayscope` on PATH, then `python -m arrayscope`.

Returns the handoff file path.
"""
function arrayscope(A::AbstractArray; name::AbstractString="array",
                    block::Bool=false, mmap::Bool=true, keep::Bool=false,
                    dir::Union{Nothing,AbstractString}=nothing,
                    exe::Union{Nothing,Cmd,AbstractString}=nothing)
    ndims(A) >= 1 || throw(ArgumentError("arrayscope: data must have at least 1 dimension"))
    # Dense column-major Array writes zero-copy; anything else (views,
    # BitArray, ranges) is materialized once.
    data = A isa Array{<:_SUPPORTED_ELTYPES} ? A : Array(A)
    eltype(data) <: _SUPPORTED_ELTYPES || _npy_descr(eltype(data))

    d = _handoff_dir(dir)
    _clean_stale_handoffs(d)
    stamp = string(round(Int, time() * 1000), "-", rand(UInt16))
    path = joinpath(d, string(_sanitize(name), "-", stamp, ".npy"))
    write_npy(path, data)

    flags = String[]
    mmap && push!(flags, "--mmap")
    keep || push!(flags, "--consume")
    cmd = `$(_viewer_command(exe)) --title $(String(name)) $(flags) $(path)`

    if block
        run(cmd)
    else
        run(detach(cmd); wait=false)
    end
    return path
end

end # module
