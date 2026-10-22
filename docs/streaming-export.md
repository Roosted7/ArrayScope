# Stream a large scientific array to NumPy

Add `--stream` to headless export when the selected output is too large to
evaluate as one in-memory array. ArrayScope plans bounded pieces, evaluates
them through the existing operation pipeline, and writes one `.npy` file with
the complete output shape. Streaming is explicit: ordinary batch export and
the viewer's export workflow keep their existing behavior.

```bash
arrayscope input.npy --export output.npy --stream --dry-run
```

Inspect the plan first, then remove `--dry-run` to run it. Streaming currently
accepts one numeric/boolean `.npy` input or one explicitly selected, supported
HDF5 dataset. It supports built-in operation recipes and writes `.npy` only.
`--stream` with `.npz` is a usage error; omit `--stream` for the existing
archive export workflow, which can require the complete selected array in
memory.

## Try an exact selection and recipe

Create small inputs whose results are easy to verify:

```python
import h5py
import numpy as np

data = np.arange(12 * 10 * 3, dtype=np.float32).reshape(12, 10, 3)
np.save("input.npy", data)
with h5py.File("input.h5", "w") as output:
    output.create_dataset("scan/image", data=data, chunks=(4, 5, 1), compression="gzip")
```

Save this as `steps.json`, or use **Save Operation Recipe** in the viewer:

```json
{
  "version": 2,
  "operations": [
    {"id": "crop", "axis": 0, "parameters": {"start": 1, "stop": 11}, "enabled": true},
    {"id": "scale", "parameters": {"factor": 2.0}, "enabled": true},
    {"id": "reverse", "axis": 1, "enabled": true},
    {"id": "mean", "axis": 2, "enabled": false}
  ]
}
```

The disabled mean remains disabled. The other steps produce a derived shape
of `(10, 10, 3)`. Select from that derived output using zero-based, half-open
bounds:

```bash
arrayscope input.npy --recipe steps.json --export selected.npy --stream \
  --range 0=2:8 --range 1=1:9 --range 2=1:2 --dry-run > plan.json
```

This selects `(6, 8, 1)` values before squeezing. The default output shape is
`(6, 8)`, containing 192 bytes of `float32` values. Run the request:

```bash
arrayscope input.npy --recipe steps.json --export selected.npy --stream \
  --range 0=2:8 --range 1=1:9 --range 2=1:2 > export.json
```

Read the completed file like any other NumPy array:

```python
import numpy as np

source = np.load("input.npy", mmap_mode="r", allow_pickle=False)
expected = (source[1:11, ::-1, :] * 2)[2:8, 1:9, 1:2].squeeze()
actual = np.load("selected.npy", mmap_mode="r", allow_pickle=False)
np.testing.assert_array_equal(actual, expected)
```

The small tutorial may fit in one chunk. Larger eligible requests are split
without changing those array values or coordinates.

## Select an HDF5 dataset

The same request can read the HDF5 input without materializing the full dataset:

```bash
arrayscope input.h5 --dataset scan/image --recipe steps.json \
  --export selected-hdf5.npy --stream \
  --range 0=2:8 --range 1=1:9 --range 2=1:2 --dry-run
```

Dataset selection is exact and required for HDF5. Its raw shape and dtype are
preserved; no complex interpretation is guessed from its name. Compressed
chunks can require more decoding memory than the selected logical values.
An oversized storage chunk or an unknown decoder estimate can still prevent
a bounded read. See [HDF5 data](hdf5-data.md) for supported encodings and source
identity checks.

## Shape and squeezing apply to the whole output

Streaming writes one array, rather than separate per-chunk arrays. The global
selection and squeeze setting determine its final shape before writing starts.
A short final chunk does not remove an axis merely because that chunk has
length one.

Add `--no-squeeze` to preserve singleton axes in the selected output:

```bash
arrayscope input.npy --recipe steps.json --export selected-3d.npy --stream \
  --range 0=2:8 --range 1=1:9 --range 2=1:2 --no-squeeze
```

This produces `(6, 8, 1)`. Equal start and stop values make an empty selection;
empty and scalar results retain their ordinary NumPy shape semantics. Empty
output has no payload chunks; an eligible scalar result uses one chunk.

## Read the plan correctly

Standard output contains one JSON summary; progress and errors go to standard
error. Dry-run reads metadata without evaluating source values or writing
output. It succeeds even when the valid plan requires cost consent.
Planning can still take time because it validates the source bounds of every
piece.

Streaming summaries retain `schema_version: 1` and identify
`execution_mode: "streaming"`. Their main size and cost fields mean:

| Field | Meaning in streaming mode |
| --- | --- |
| `output_shape`, `dtype` | Complete saved array shape and dtype. |
| `output_bytes` | Logical values in the entire output, excluding its file header. |
| `source_read_bytes` | Total planned logical source-region reads across chunks. |
| `chunk_shape`, `chunk_count` | Maximum piece shape in final output coordinates, and number of pieces. |
| `estimated_chunk_peak_bytes` | Estimated working memory for an evaluated chunk. |
| `estimated_peak_bytes` | Per-chunk working estimate in this execution mode. |
| `warnings`, `needs_confirmation` | Reasons to inspect and explicitly accept costly work. |
| `paths`, `dry_run` | Written destinations and whether this was planning only. |

Logical source reads are not measured disk traffic. HDF5 chunk decoding,
overlap, filesystem caches, and output writes can make physical I/O different.
An unknown working estimate causes refusal; it is never treated as zero.

Each chunk's known working estimate must fit the default 64 MiB target. A
single exact output slab that exceeds this target is refused; cost consent
cannot force a larger chunk. This is not a hard process-RSS limit: library
temporaries, allocator retention, runtime overhead, and operating-system caches
are not bounded by that number. Hard source-read
admission still applies to each planned piece. The output must also fit on the
destination filesystem, and staging needs space for a complete new file while
an existing destination remains present.

A large total output can still require consent even when individual chunks
fit the target. After inspecting the plan, use `--allow-expensive` to accept
supported cost warnings. It does not override hard source-read limits,
unknown working estimates, or an unsupported streaming plan.

## Which recipes can stream

Eligibility is conservative and is checked from metadata. Pieces run along
the first axis of the final output, after the global squeeze decision. The
planner needs those pieces to be independently evaluable with known working
costs and monotonically advancing, nonoverlapping bounds on one fixed input
axis, including reverse traversal. If `--no-squeeze` preserves a leading
singleton axis, one leading slab covers its complete trailing region, which
must fit the target. It refuses plans that would repeatedly evaluate a whole
source or opaque stage for each output chunk. It does not probe source values
to discover an otherwise unknown operation contract or silently fall back to
full-array evaluation.

An FFT can be eligible when a chunk contains its full transform axis and the
array is split along another independent axis. Splitting the FFT's own axis
would not compute the same transform; repeatedly transforming the whole input
for successive output fragments is also refused. Inspect the actual plan
rather than assuming all FFT recipes are supported or unsupported.
For example, an array ordered as `(batch, x, y)` can stream batches while
transforming the complete `x` and `y` axes, if each chunk fits. A transpose or
other operation whose region mapping revisits source bounds may be refused,
even when another theoretical partition could work.

Enabled `fftshift` and `split_axis` steps are currently refused by streaming,
including small cases. Their general coordinate mappings are outside this
initial bounded metadata contract. Ordinary batch export still supports them.

Named FFT operations use the viewer's established MRI convention, including
its shift order and orthonormal normalization; see
[recipe replay](recipe-replay.md#fft-conventions-match-the-viewer).

User-authored operations, command wrappers, optional packs, and third-party
plugin IDs remain outside the headless built-in recipe contract, including
unsupported steps marked disabled. `--allow-expensive` does not enable them.

## Keep the processing record separately

The streamed `.npy` contains scientific values only. It does not embed the
operation recipe, selection metadata, display settings, or source dataset
identity, and it does not create automatic sidecars. Keep your original recipe
and capture the JSON summary alongside the result. The summary describes the
input, dataset, selection, plan, and destinations; it is not a replacement for
the full recipe or the original input.

Keep input files unchanged while the job runs. HDF5 validates captured file
and dataset metadata before reads and publication; these checks are not
durable content hashes or snapshot isolation. NumPy input uses a read-only
mapping, which does not prevent or automatically detect another process
writing to that file. Output aliases to the input or authored recipe are
rejected, including symlink and hard-link aliases.

## Cancellation and failures

Before publication, Ctrl+C or a failed chunk removes staged output and
preserves an existing destination. Once publication has committed, a late
interrupt cannot undo the saved result and success is reported. There is no
partial output to resume after cancellation.

Exit codes follow [recipe replay](recipe-replay.md#cancellation-and-exit-codes):
`0` for success or dry-run, `2` for invalid options or an unsupported,
unpriceable, or over-target streaming plan, `3` for cost consent or source
read-budget refusal, `1` for I/O/runtime failure,
and `130` for interruption before commit. Missing `--export` or a streamed
`.npz` destination is a usage error; neither starts the viewer.
