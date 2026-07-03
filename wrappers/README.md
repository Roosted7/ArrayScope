# Language invocation wrappers

Thin, dependency-free adapters that open the ArrayScope viewer from other
languages. Each writes the array as a raw `.npy` handoff file and launches a
detached viewer process via the stable CLI (`arrayscope --mmap --consume`);
none of them re-implements any viewer behavior.

- [`julia/`](julia/ArrayScope.jl) — `arrayscope(A; name=..., block=...)`, plus
  a `Project.toml` so the folder can be `Pkg.develop`ed. Example:
  [`julia/example.jl`](julia/example.jl).
- [`matlab/`](matlab/arrayscope.m) — `arrayscope(A, 'name', ..., 'block', ...)`.
  Example: [`matlab/arrayscope_example.m`](matlab/arrayscope_example.m).

Both work on Linux, macOS, and Windows and require only `pip install
arrayscope` on the machine. Design, efficiency contract, and options are
documented in [`docs/invocation.md`](../docs/invocation.md); the Python side
of the handoff is pinned by `tests/io/test_language_handoff.py`.
