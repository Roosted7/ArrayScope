# Invocation

How to open the viewer from Python, the command line, Julia, and MATLAB, and
the efficiency contract those routes share. The roadmap's rule for invocation
adapters applies throughout: every host language calls the same stable entry
points (the `arrayscope()` Python API and the CLI); no adapter re-implements a
frontend or a state machine.

## Python

`arrayscope` is a callable module:

```python
import numpy as np
import arrayscope as asc

asc(np.random.rand(64, 64, 32), title="volume")   # non-blocking child process
asc(data, block=True)                              # wait for the window
```

From a plain Python process the non-blocking form spawns a separate viewer
process. When Qt already runs in the calling process (IPython/Jupyter with
`%gui qt`), the window opens inline; without a running Qt event loop the call
falls back to blocking mode with a warning.

## Command line

```bash
arrayscope data.npy                       # one file: blocking
arrayscope a.npy b.h5 c.mat               # several files: one shared event loop
arrayscope --title "kspace" data.npy      # window title override
arrayscope --mmap data.npy                # memory-map instead of eager read
arrayscope --mmap --consume handoff.npy   # wrapper handoff (see below)
```

`--mmap` opens `.npy` files copy-on-write (`numpy` `mmap_mode='c'`): reads are
lazy and shared with the OS page cache, and in-place edits stay private to the
viewer process without writing back to the file. `--consume` deletes the input
file once it is loaded (or mapped) — meant for temporary handoff files, best
effort on Windows where a mapped file cannot be unlinked.

## Julia and MATLAB wrappers

The wrappers live in [`wrappers/julia`](../wrappers/julia/ArrayScope.jl) and
[`wrappers/matlab`](../wrappers/matlab/arrayscope.m), with runnable examples
alongside them. Both are thin, dependency-free adapters over the CLI: they
write the array to a raw `.npy` handoff file and launch a detached viewer
process. The host session never runs Qt and stays responsive; `block=true`
waits instead.

Julia:

```julia
include("ArrayScope.jl"); using .ArrayScope    # or: ] dev wrappers/julia
arrayscope(vol; name="kspace")                  # non-blocking
arrayscope(abs.(vol); name="mag", block=true)   # blocking
```

MATLAB (add `wrappers/matlab` to the MATLAB path):

```matlab
arrayscope(vol, 'name', 'kspace');              % non-blocking
arrayscope(abs(vol), 'name', 'mag', 'block', true);
```

Both accept `mmap` (default on), `keep` (retain the handoff file), `dir`
(handoff directory), and `exe` (viewer executable) options. The viewer
executable resolves in this order: explicit `exe` option, `ARRAYSCOPE_EXE`
environment variable, `arrayscope` on `PATH` (Julia additionally falls back
to `python -m arrayscope`). Requirements are only a working `pip install
arrayscope` on the machine — no PyCall/PythonCall, and no MATLAB `pyenv`
configuration.

## The handoff contract and why it is fast

ArrayScope targets huge arrays, so the handoff avoids per-element work,
compression, and redundant copies:

1. **One raw write.** Julia (`write(io, A)`) and MATLAB (`fwrite`) emit the
   array's memory buffer directly into an uncompressed `.npy` file. Julia and
   MATLAB are column-major, which `.npy` expresses natively as
   `fortran_order: True`, so neither side transposes, reorders, or converts
   elements. The write lands in the OS page cache; for arrays that fit in
   RAM, "disk" is not on the critical path. (MATLAB complex arrays cost one
   extra interleave pass because `fwrite` cannot write complex buffers;
   everything else is a single sequential write. Contrast with
   `save('-v7')`, which zlib-compresses everything.)
2. **Memory-mapped load.** The wrappers pass `--mmap`, so the viewer maps the
   file copy-on-write instead of reading it. Pages are shared with the page
   cache the writer just populated — the viewer does not materialize a second
   in-RAM copy up front, and untouched regions are paged in only when
   rendering or inspection actually reads them. The header is padded so the
   data section starts 64-byte aligned.
3. **Immediate cleanup without a copy.** The wrappers pass `--consume`: the
   viewer unlinks the handoff file as soon as it is mapped. On POSIX the
   mapping keeps the data alive until the viewer exits, so the temp file
   costs no lasting disk space. On Windows the unlink is refused while
   mapped; the wrappers delete stale handoff files (older than a day) on
   their next call.
4. **Shared-memory option.** The handoff directory defaults to
   `<tempdir>/arrayscope-handoff` and can be overridden with
   `ARRAYSCOPE_HANDOFF_DIR` (or the `dir` option). On Linux, pointing it at
   `/dev/shm` makes the handoff literally a shared-memory segment.

The Python side of this contract (byte-level `.npy` acceptance,
copy-on-write mapping, consume semantics) is pinned by
`tests/io/test_language_handoff.py`.

### Element types

| Host type | `.npy` descr |
|---|---|
| `Bool` / `logical` | `\|b1` |
| signed ints 8–64 | `\|i1`, `<i2`, `<i4`, `<i8` |
| unsigned ints 8–64 | `\|u1`, `<u2`, `<u4`, `<u8` |
| `Float16/32/64` (`half`/`single`/`double`) | `<f2`, `<f4`, `<f8` |
| `ComplexF32/64` (complex `single`/`double`) | `<c8`, `<c16` |

Non-dense inputs (Julia views, `BitArray`s, ranges) are materialized once
before writing. Complex integer MATLAB arrays are rejected. Only
little-endian hosts are supported, which covers all current Windows, macOS,
and Linux targets.

### Cross-platform notes

- **Linux/macOS:** detached launch via the shell (`&`) or `detach` in Julia;
  consume-after-map works fully.
- **Windows:** detached launch via `start "" /b`; the consume unlink is
  deferred to the wrappers' stale-file cleanup. Everything else is identical.
- The wrappers never depend on in-process Python bridges (MATLAB `pyenv`
  in-process mode, PythonCall), so they cannot conflict with the host's own
  Qt/Java event loops and work on any MATLAB ≥ R2019b / Julia ≥ 1.6 without
  configuration.

### Choosing eager vs. mapped

`--mmap` is the wrapper default because it is strictly lazier. Disable it
(`mmap=false`) when the handoff directory sits on slow network storage whose
pages should not back a long-lived viewer session; the viewer then reads the
file once, eagerly, and `--consume` frees it everywhere including Windows.
