# Replay an operation recipe without opening a window

Use ArrayScope to explore an array, save its operation recipe, and then apply
the same built-in operations from a terminal or script. Batch export opens a
single NumPy `.npy` input read-only and writes scientific array values to
`.npy` or `.npz`. It does not start Qt or apply display colors, display scaling,
or window levels.

## Start with an array

Create a small input to try the examples:

```python
import numpy as np

data = np.arange(8 * 6 * 4, dtype=np.float32).reshape(8, 6, 4)
np.save("input.npy", data)
```

Export it without opening the viewer:

```bash
arrayscope input.npy --export copy.npz
```

The input is opened lazily and read-only automatically; `--mmap` is a viewer
option and is not needed here. Input shapes retain scalar, empty, and trailing
singleton dimensions. By default, singleton dimensions are removed from the
exported selection; add `--no-squeeze` to preserve them.

An export command prints one JSON result with `schema_version: 1` to standard
output. Progress and errors go to standard error, so a script can capture the
result separately:

```bash
arrayscope input.npy --export copy.npz > result.json
```

Run the installed `arrayscope` command or use `python -m arrayscope` in its
place. From a source checkout, use `tools/env python -m arrayscope`.

## Bring a recipe from the viewer

In the viewer, add or edit operations in the Operations panel, then choose
**File → Save Operation Recipe**. **Copy Operation Recipe to Clipboard** gives
the same JSON representation for saving or inspecting in a text editor.
View recipes include display settings and are a different format.

This first batch interface supports built-in operations, including operations
over sets of axes. User-authored operations, command wrappers, optional packs,
and third-party plugin IDs are rejected, including unsupported steps marked
disabled. `--allow-expensive` does not enable them.

For the sample input, save the following as `steps.json`:

```json
{
  "version": 2,
  "operations": [
    {"id": "crop", "axis": 0, "parameters": {"start": 1, "stop": 7}, "enabled": true},
    {"id": "reverse", "axis": 1, "enabled": true},
    {"id": "scale", "parameters": {"factor": 2.0}, "enabled": true},
    {"id": "mean", "axis": 2, "enabled": false}
  ]
}
```

Each enabled step receives the preceding step's output. Here, crop changes the
shape from `(8, 6, 4)` to `(6, 6, 4)`, reverse changes sample order on axis 1,
and scale doubles the values. The disabled mean remains in the saved recipe
and does not run. The source file is not modified.

```bash
arrayscope input.npy --recipe steps.json --export derived.npz
```

## Export a selected region

Ranges use zero-based axes and half-open bounds: `AXIS=START:STOP` includes
`START` and excludes `STOP`. They select from the **derived output after the
recipe**, in array axis order. They do not refer to screen coordinates or to
the original axes before shape-changing operations.

```bash
arrayscope input.npy --recipe steps.json --export selected.npz \
  --range 0=1:4 --range 1=2:5 --range 2=1:2
```

This selects shape `(3, 3, 1)` from the derived array. Default squeezing writes
shape `(3, 3)`, containing 36 bytes of `float32` values. File headers and
embedded metadata add to the final file size. Preserve all selected axes with:

```bash
arrayscope input.npy --recipe steps.json --export selected-3d.npy \
  --range 0=1:4 --range 1=2:5 --range 2=1:2 --no-squeeze
```

An omitted axis selects its entire derived extent. Bounds must be nonnegative
integers within that extent; negative indices, strides, and repeated axes are
not accepted. Equal start and stop values, such as `--range 0=3:3`, make a valid
empty selection. Selecting one value on every axis produces a scalar when
squeezing is enabled.

Check the example against NumPy:

```python
import numpy as np

source = np.load("input.npy", allow_pickle=False)
expected = (source[1:7, ::-1, :] * 2)[1:4, 2:5, 1:2]
with np.load("selected.npz", allow_pickle=False) as saved:
    np.testing.assert_array_equal(saved["array"], expected.squeeze())
np.testing.assert_array_equal(np.load("selected-3d.npy"), expected)
```

## Inspect the work before running it

Add `--dry-run` to validate the request and print its plan without evaluating
array values or writing output:

```bash
arrayscope input.npy --recipe steps.json --export selected.npz \
  --range 0=1:4 --range 1=2:5 --range 2=1:2 --dry-run > plan.json
```

The plan describes output shape and dtype, logical output bytes, estimated
primary-source reads and peak working memory, and any warnings. These are
planning estimates, not measurements of disk traffic, process RSS, or final
file size. Unknown costs remain JSON `null`; unknown never means zero.
Reading an input or recipe header is distinct from evaluating array values.
Dry-run succeeds even when the plan has `needs_confirmation: true`, so you can
inspect a costly request before consenting to execution. Its `paths` list is
empty because no output was written.

A tiny output can require a much larger intermediate stage, for example when
the recipe contains an FFT. Export refuses costly or unknown work unless it
has explicit consent. Review the plan, then opt in when appropriate:

```bash
arrayscope input.npy --recipe steps.json --export derived.npz --allow-expensive
```

`--allow-expensive` accepts the cost warning for supported built-in operations,
including large output or intermediate allocations. Estimated peak memory is
an estimate for that decision, not a hard RAM or process-RSS cap. Hard
source-read limits still apply; a request whose required input region exceeds
that budget is refused even with consent. Shrinking the output along a full
transform axis may not shrink its required input.

## FFT conventions match the viewer

The named FFT operations preserve ArrayScope's MRI convention. `centered_fft`
applies `fftshift`, then an **inverse FFT** with `norm="ortho"`, then
`ifftshift`, along the selected axis. `centered_ifft` uses a forward FFT in
the same shift order. These names follow the viewer's established vocabulary;
they do not mean an unshifted `numpy.fft.fft` and `numpy.fft.ifft` pair.

`centered_fftn` and `centered_ifftn` apply the corresponding single-axis
operation successively over the normalized axis set. Odd-length axes make the
shift order significant. Use the viewer's convention when checking results
against another tool; the implementation is in
[`fft_backend.py`](../arrayscope/operations/fft_backend.py).

## Keep results and provenance together

Prefer `.npz` when carrying the processing record with the result. Open it
without pickle:

```python
import json
import numpy as np

with np.load("selected.npz", allow_pickle=False) as saved:
    values = saved["array"]
    original_recipe = json.loads(saved["recipe_json"].item())
    selection = json.loads(saved["selection_json"].item())
```

The recipe describes the pipeline applied to the **original input**. The
selection manifest separately records the derived shape, selected ranges,
squeezing, output axes, and dtype. Applying the original recipe to the
already-derived `array` would apply those operations a second time. Keep the
original input when you need to rerun the calculation; the export does not
embed a copy of that input or provide a durable content hash for it.
An `.npy` output contains the values alone, without recipe sidecars.

Batch input currently accepts one regular `.npy` file. Containers such as
HDF5, MAT, and NPZ are output or viewer workflows, not batch input selectors.
Object arrays requiring pickle are not supported. Keep input files unchanged
while an export runs: a read-only mapping prevents this process from writing
them, but does not freeze writes by other processes. The output must be a
different file from the input; paths that alias the input are rejected.

## Cancellation and exit codes

Press Ctrl+C to cancel. Before publication, cancellation or failure preserves
an existing destination and removes staged output. Once publication has
committed, a late interrupt cannot undo the saved result and success is
reported. An NPZ archive keeps values and metadata in one output file.

| Exit code | Meaning |
| --- | --- |
| `0` | Export completed, or dry-run completed successfully. |
| `1` | I/O or runtime failure. |
| `2` | Invalid command, recipe, range, or unsupported input. |
| `3` | Cost consent required or a hard source-read budget refused the request. |
| `130` | Interrupted before the output was committed. |

Batch-only options require `--export`; it also requires exactly one input and
an explicit `.npy` or `.npz` output suffix. Viewer options such as `--title`,
`--consume`, `--mmap`, and `--trace` cannot be combined with batch export.
