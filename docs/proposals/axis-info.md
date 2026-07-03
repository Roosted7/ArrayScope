# AxisInfo Proposal

## Status

Second increment (2026-07-03, rich axis metadata). `AxisInfo` now carries
optional physical metadata (`unit`, `spacing`, `origin`) alongside identity.
File loaders provide axis metadata where the format knows it: NIfTI zooms and
xyzt units become per-axis spacing/units (including DICOM directories converted
through dcm2niix, whose stacked series axis is labeled by its varying key),
single-file DICOM provides row/column/frame labels and pixel/slice spacing, and
Philips XML/REC labels its canonical axis layout. `load_path` returns the axis
metadata, the launch routes pass it to `ArrayScopeWindow`, and the dimension
strip shows custom labels with a metadata tooltip (unit, spacing, origin,
coordinate). Operations propagate physical metadata conservatively (see
Operation Semantics).

First increment: `arrayscope.core.axis_info.AxisInfo` exists and
`ArrayDocument` exposes `base_axes` and `current_axes` matching the current
operation stack.

Full session role matching, linked-window sync, editable labels, coordinate
arrays, physical-unit cursor readout, orientation, and xarray integration
remain future work.

## Problem

ArrayScope currently treats axes as integer positions. That is enough for basic
viewing, but shape-changing operations can shift axis positions:

```text
input:      (readout, phase, slice, coil)
RSS(axis=3) -> (readout, phase, slice)
Mean(axis=1) -> (readout, slice)
```

After reductions, later operations and saved recipes can still be numerically
valid while losing the scientific meaning of each axis. This will become a
problem for montage roles, linked-window sync, labels/units, session recipes,
and reconstruction-pipeline workflows.

## Proposed Model

ArrayScope has a Qt-free axis metadata model in `arrayscope.core.axis_info`:

```python
from dataclasses import dataclass, replace
from typing import Optional, Tuple


@dataclass(frozen=True)
class AxisInfo:
    id: str
    label: str
    size: int
    unit: Optional[str] = None
    coordinate: Optional[str] = None
    source_index: Optional[int] = None
    spacing: Optional[float] = None   # physical step per index, in `unit`
    origin: Optional[float] = None    # physical coordinate of index 0, in `unit`


AxisInfoTuple = Tuple[AxisInfo, ...]
```

`spacing`/`origin` describe an affine index-to-coordinate mapping
(`coordinate(i) = origin + i * spacing`). They stay `None` when the source does
not know them; nothing forces a data source into a medical-imaging model.

Initial arrays get stable generated IDs and conservative labels:

```text
axis-0, axis-1, axis-2 ...
Dim 0, Dim 1, Dim 2 ...
```

File loaders may later provide better labels, units, or coordinates when known.

## Operation Semantics

Every shape-changing operation should transform axis metadata alongside shape:

- Slice-preserving operations such as FFT, IFFT, reverse, conjugate, and
  fftshift keep axis IDs and labels.
- Crop keeps the axis ID and label but updates `size`; when spacing and origin
  are known, the origin advances by `start * spacing`.
- Reverse keeps the affine mapping exact: spacing negates and the origin moves
  to the previous last sample.
- FFT shift rotates samples, so the index-to-coordinate mapping is no longer
  affine: spacing and origin are cleared.
- Centered FFT/IFFT move the axis to a reciprocal domain: unit, spacing, and
  origin are cleared.
- Reduction operations such as mean, sum, min, max, and RSS remove the reduced
  axis.
- Combine real/imag keeps the combined axis ID, updates `size` to 1, annotates
  `coordinate="complex"`, and clears physical metadata.
- Split complex keeps the axis ID, updates `size` to 2, annotates
  `coordinate="real-imag"`, and clears physical metadata.

Shape prediction and axis metadata prediction are paired internally through
helper functions. A future operation API can expose:

```python
def output_shape(self, shape: Shape) -> Shape: ...
def output_axes(self, axes: AxisInfoTuple) -> AxisInfoTuple: ...
```

## ViewState Integration

`ViewState` should continue to store axis positions for fast indexing. Axis
identity should be attached to the current document/evaluator state, not copied
into every `ViewState` mutation.

UI controls can display:

```text
label [size]
```

while callbacks still pass integer axis positions. Sync/session features can
match by `AxisInfo.id` first, then fall back to compatible label/size only when
the user explicitly accepts ambiguity.

## Recipes and Sessions

Operation recipes should remain operation-only for now. Full session recipes
should store:

- source data identity;
- base axis metadata;
- operation stack;
- derived axis metadata or enough information to recompute it;
- `ViewState` roles by axis ID where possible.

When loading old recipes without axis metadata, ArrayScope should generate
positional `AxisInfo` values and mark the session as positional.

## Acceptance Criteria For First Implementation

- [x] `AxisInfo` and helpers live in `arrayscope.core` and import no Qt.
- [x] `ArrayDocument` can expose `current_axes` matching `current_shape`.
- [x] Existing operation tests gain axis metadata cases for crop, reduction,
  combine/split, and FFT-like preservation.
- [x] The UI can keep using integer axes while showing labels from `AxisInfo`.
- [x] No montage or sync feature relies only on post-operation integer axis
  positions.

## Acceptance Criteria For Second Increment (Rich Axis Metadata)

- [x] `AxisInfo` carries optional `spacing`/`origin` and stays Qt-free.
- [x] Operations propagate physical metadata conservatively: crop shifts the
  origin, reverse negates spacing exactly, FFT-like operations clear metadata
  they invalidate rather than keeping stale values.
- [x] Loaders that know axis metadata provide it (`LoadedPath.axes`): NIfTI
  spacing/units, DICOM file and directory, Philips REC labels. Loaders that do
  not stay conservative (`None`).
- [x] The dimension strip shows loader/custom labels and a metadata tooltip
  while callbacks still pass integer axis positions; positional default axes
  render exactly as before.
- [x] Axis metadata that cannot be aligned with the final data shape is
  discarded rather than misassigned.
