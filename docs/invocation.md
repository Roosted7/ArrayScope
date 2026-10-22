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

An inline call returns a viewer window. Closing a launched viewer cancels its
work and destroys its Qt widget tree; open a new viewer to inspect again, and
do not call Qt widget methods on the closed window. Direct `ArrayScopeWindow`
construction keeps caller-owned Qt lifetime for embedding.

## Command line

Starting `arrayscope` without arguments opens a launcher with **Open Files…**
and **Open DICOM Folder…** buttons. The standard Open shortcut (`Ctrl+O`, or
`Cmd+O` on macOS) opens the file picker. Files and DICOM folders can also be
dragged onto the launcher, which honors the saved appearance preference.
The home screen also offers locally generated volume, time-series, and complex
examples with named dimensions and interaction hints. Recent history resumes
loaded data files and browsed containers; missing paths remain visible with an
explanation. Failed/canceled asynchronous loads and consumed temporary wrapper
handoffs do not enter history.

```bash
arrayscope data.npy                       # one file: blocking
arrayscope a.npy b.h5 c.mat               # several files: one shared event loop
arrayscope --title "kspace" data.npy      # window title override
arrayscope --mmap data.npy                # memory-map instead of eager read
arrayscope --mmap --consume handoff.npy   # wrapper handoff (see below)
arrayscope measurement.h5 --dataset scan/image  # an explicit lazy HDF5 dataset
```

`--mmap` opens `.npy` files copy-on-write (`numpy` `mmap_mode='c'`): reads are
lazy and shared with the OS page cache, and in-place edits stay private to the
viewer process without writing back to the file. `--consume` deletes the input
file once it is loaded (or mapped) — meant for temporary handoff files, best
effort on Windows where a mapped file cannot be unlinked.

HDF5 selectors browse metadata and open supported datasets lazily, including
nested numeric/complex arrays. Compound subarray fields use an explicitly
labeled **Load all** fallback. `--dataset` selects an exact supported HDF5 key
without a browser; that source needs its file for later reads and cannot use
`--consume`. See [HDF5 loading](hdf5-data.md) for encoding and identity limits.
NPZ selectors inspect array headers to list available datasets;
they load the chosen array only, and ignore object metadata without unpickling
it. MAT selectors discover nested numeric fields alongside metadata. Search
matches names and full paths; the tree shows shape, type, and estimated memory.
Press Enter or use Open to load the selected array. Canceling a selector or
encountering a read failure closes its container.

Choose **File → Export Selected Array…**, or press **Ctrl+S** (**Cmd+S** on macOS),
to export a selected array as `.npy` or `.npz`.
Choose inclusive start/end indices, optional singleton squeezing, and a
destination before any export evaluation. The preview reports output shape,
dtype, and array bytes, with exact integer entry even for very large axes. The
selection uses derived array coordinates and native scientific values, including
complex values; display channels, colors, scaling, and flips are not applied.

Preflight, evaluation, and writing run on the window's execution kernel. Small
selections avoid full materialization when operation capabilities permit it;
FFTs, opaque stages, and unknown auxiliary-input costs require explicit
confirmation. Cancel keeps feedback visible until cooperative work stops, so a
single native/plugin operation may finish before cancellation takes effect.
Actively streaming loads must finish or stop before export starts.

An accepted export retains its document and source references through later
operation edits or source replacement. Notified mutations of shared primary or
auxiliary sources cancel the export. References do not copy source pixels:
unnotified external writes cannot provide snapshot isolation. Changed linked
definitions and file inputs are checked again before publication.

NumPy exports stage complete files beside their destinations before replacing
them. Failures and cancellation before publication preserve previous exports,
and uppercase `.NPY` or `.NPZ` names are used exactly as entered. Once atomic
publication completes, a late cancellation still reports success. Replacement
is atomic per file; a full-derived NPZ export keeps data and embedded recipes
in one file. Separate NPY recipe sidecars do not form a multi-file transaction.
On Windows, replacing an open memory-mapped source may be refused while
leaving the original intact. The existing full-derived recipe export remains
separate from the selected-region workflow. Selected NPZ export embeds the
original operation recipe and a separate selection/source manifest; applying
that recipe to the exported derived values would process them a second time.

Operation recipes can also travel through the clipboard: choose **Copy
Operation Recipe to Clipboard** or **Paste Operation Recipe (Replace Stack)**
from File, Operations → More, or the `Ctrl+K` command palette. JSON preserves
operation parameters and enabled/disabled state. Paste replaces the current
stack and uses the same shape, input-slot, and imported-command review rules
as loading a recipe file. Both installations need the referenced plugins.

**Edit → Undo/Redo**, the Operations dock buttons, and the command palette
restore operation edits: additions, deletions, parameter and axis changes,
ordering, enabled state, clearing, and recipe replacement. Text editors keep
local Undo/Redo shortcuts. Camera, display, ROI, and external file contents
are outside operation history.

The coordinator keeps up to 64 transitions for the current source. Replacing,
reloading, notifying an in-place source change, or materializing starts fresh
history. Extra auxiliary inputs retained solely by past/future edits have a
512 MiB logical-byte budget, counted by shared buffer owners without reading
data; unknown sizes are treated conservatively. History may be shortened, and
an oversized input's Undo can release its Redo snapshot. Disabled controls
explain memory pruning. This budget estimates retained input ownership, not
physical process RSS.

Same-source rank changes preserve operation history and source epoch in the
successor window. The existing rank handoff still resets view state, ROIs,
comparison, and linked-window sync participation; enable sync again in the new
window when needed. Same-rank operation Undo/Redo publishes through ordinary
linked-window sync, with each received stack recorded as one edit.

## Replay recipes without a display

```bash
arrayscope input.npy --export result.npz --recipe steps.json --dry-run
arrayscope input.npy --export result.npz --recipe steps.json --range 0=10:30
arrayscope measurement.h5 --dataset scan/image --export selected.npz --dry-run
```

Batch export opens one regular `.npy` input or an explicit HDF5 dataset
read-only, preserves its raw shape,
and reuses the exact region planner, evaluator, and atomic writer. It runs
before Qt startup, viewer supervision, or accelerator prewarming. Supported
built-in recipes retain authored disabled steps; unsupported plugin/user/command
IDs are rejected before lookup, and malformed enabled/version fields fail
clearly. Inputs and recipe files cannot alias the output.

`--dry-run` prints a versioned JSON plan without reading array payloads or
writing output. Actual exports print one JSON result to stdout and bounded
progress/errors to stderr. `--allow-expensive` accepts estimated-cost warnings;
it does not override hard source-read limits or impose a general peak-RAM cap.
Ctrl+C before publication preserves previous output. See the
[complete replay guide](recipe-replay.md) for range/squeeze semantics, FFT
conventions, embedded provenance, examples, and exit codes.

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
