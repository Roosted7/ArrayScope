# Inspect and process HDF5 arrays lazily

ArrayScope can browse an HDF5 file using metadata and open a supported dataset
without loading its entire array. The same dataset source powers viewing,
explicit Python loading, and headless recipe export.

## Open a dataset in the viewer

Open an `.h5` or `.hdf5` file from the launcher or terminal:

```bash
arrayscope measurement.h5
```

With several displayable arrays, the dataset browser shows their paths,
shapes, dtypes, logical array sizes, and access modes. Type part of a name or
path to filter, then press Enter or choose **Open**. Groups stay visible around
matching children. A single supported lazy array opens directly.

**Lazy** means the viewer reads requested regions through the normal array
planner. **Array size** describes the uncompressed values, not an immediate
RAM allocation or the file's compressed size. Compressed or chunked datasets
can require decoding complete chunks even for a small selection.

Some legacy compound fields contain an array inside each record. They remain
available through **Load all**, with a visible explanation of the full logical
array size and decoding buffers. The browser also appears when such a field
is the only choice, so opening it is an explicit selection. These fields do
not yet have lazy region support.

Open a known supported dataset directly, without a selection dialog:

```bash
arrayscope measurement.h5 --dataset scan/image
```

Dataset keys are exact, case-sensitive HDF5 paths. A leading slash is
accepted. `--dataset` selects one input file; it does not guess a dataset from
the file name or from the first array it finds. Selected lazy sources need the
file for later reads, so `--dataset` cannot be combined with `--consume`.

## Try a small reproducible file

```python
import h5py
import numpy as np

values = np.arange(12 * 10 * 5, dtype=np.float32).reshape(12, 10, 5)
with h5py.File("measurement.h5", "w") as output:
    output.create_dataset(
        "scan/image", data=values, chunks=(4, 5, 1), compression="gzip"
    )
    output.create_dataset("reference", data=values[:, :, 0])
```

Browse its catalog from Python without importing Qt or reading array values:

```python
from arrayscope.io.hdf5_catalog import catalog_hdf5

catalog = catalog_hdf5("measurement.h5")
for dataset in catalog.datasets:
    print(dataset.key, dataset.shape, dataset.dtype, dataset.lazy_supported)
```

The catalog contains immutable descriptors. It closes its HDF5 file before
returning; it does not leave a live `h5py.Dataset` attached to the browser.

## Read a bounded region from Python

```python
from arrayscope.io.file_interpreters import load_path

loaded = load_path("measurement.h5", dataset_path="scan/image")
source = loaded.data
try:
    region = source.read_region((slice(2, 6), slice(1, 4), 2))
    assert region.shape == (4, 3)
finally:
    source.close()
```

The returned source retains the dataset's original shape, dtype, and array
axis order. An integer index drops its axis; a slice retains it. Scalar and
empty arrays remain valid in the source API even when the viewer cannot
display them. Explicit loading does not combine a size-two axis merely
because its name contains `complex`.

Low-level `read_region` reads the region you request. Viewer and export
planners apply their source-read budgets above that adapter. Avoid requesting
the whole source unless you intend to allocate it. For a deliberately eager,
budget-checked load of an adapter-supported dataset, use
`load_path(..., dataset_path="scan/image", lazy=False)`.
The legacy `H5DatasetSelector.load_data(key)` remains an explicit eager API;
normal GUI selection uses the separate lazy loading path.

## Save a selected array from the viewer

Press Ctrl+S to choose the values to export before evaluation starts. Review
the shape, dtype, and logical bytes, then choose the destination. NumPy `.npy`
stores the values alone. Choose `.npz` to keep the operation recipe and
separate selection/source metadata with those values. Display colors and
window levels are not applied to the scientific data.

## Export an HDF5 region without a window

```bash
arrayscope measurement.h5 --dataset scan/image --export selected.npz \
  --range 0=2:6 --range 1=1:4 --range 2=2:3 --dry-run
```

The dry-run reads metadata, reports an honest JSON plan, and writes no output.
It reports planned reads, not measured disk traffic. Run the export by
removing `--dry-run`; add `--no-squeeze` to preserve the final singleton axis.
The default output shape here is `(4, 3)`; with `--no-squeeze` it is `(4, 3, 1)`.

Use `--recipe steps.json` to apply a built-in operation recipe first. Ranges
always select the derived output after those operations. A transform may need
larger input regions than the output selection, and compressed chunks may
increase the estimated read working footprint further. JSON `source_read_bytes`
counts logical source-region values; `estimated_peak_bytes` includes estimated
read working buffers. Neither measures physical disk traffic. Review any
warnings before using `--allow-expensive`. Consent does not remove hard
source-read limits, and estimated peak memory is not a hard RAM or process-RSS
cap.

NPZ exports carry the original operation recipe and a separate selection
manifest. Those operations describe processing of the original HDF5 dataset;
applying them to the already-derived exported values would process them again.
See [recipe replay](recipe-replay.md) for JSON output, cancellation, and exit
codes.

## Supported encodings and current boundaries

Lazy sources support ordinary numeric and boolean datasets, native complex
datasets, scalar numeric compound fields, and recognized real/imaginary
record encodings with scalar fields. The recognized pairs use `real`,
`realdata`, or `r`, and `imag`, `imagdata`, or `i`. Native complex values and
recognized records preserve both components; they do not discard the
imaginary part.

Compound subarray fields use the explicitly labeled GUI eager fallback.
Explicit lazy loading and batch export reject that fallback rather than
silently loading the whole field. Null datasets, nonnumeric data, virtual
datasets, external storage, and soft/external links are not supported lazy
sources. Unsupported entries remain visible with an explanation where the
catalog can describe them.

The browser keeps its historical convenience for a single real array whose
name contains `complex` and which has a size-two axis: the viewer can interpret
that axis as real/imaginary components. Explicit `--dataset`, Python source
loading, and batch export use raw dataset semantics and never apply that name
heuristic.

## Reload after the file changes

Sources capture file and dataset identities. Every read reopens the file
read-only, checks those identities, and closes its handle afterward. A source
that detects replacement or a metadata change refuses to serve new values;
it does not silently combine an old plan with a different dataset.

Use the viewer's reload control after a file change. Reload recaptures the
catalog and opens the same dataset key lazily, preserving compatible
operations and view state. If that key has disappeared, the usual dataset
selection flow lets you choose a replacement or cancel. Python callers should
open a new source after a change.

Identity checks are not cryptographic content hashes or snapshot isolation.
Keep inputs unchanged during an inspection or export; external writers can
still race a read. The source is read-only, and closing a selector does not
invalidate the independently opened selected source.

Eligible terminal exports can use [incremental NPY output](streaming-export.md)
with `--stream`. Chunk planning includes HDF5 decoded-chunk working costs and
keeps the same source identity and per-read budget checks.
