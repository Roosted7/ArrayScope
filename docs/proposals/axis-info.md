# AxisInfo Proposal

## Status

Proposal. Do not implement montage, linked-window sync, or full view recipes
against positional axes alone.

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

Add a Qt-free axis metadata model in `arrayscope.core.axis_info`:

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


AxisInfoTuple = Tuple[AxisInfo, ...]
```

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
- Crop keeps the axis ID and label but updates `size`.
- Reduction operations such as mean, sum, min, max, and RSS remove the reduced
  axis.
- Combine real/imag keeps the combined axis ID, updates `size` to 1, and may
  annotate `coordinate="complex"`.
- Split complex keeps the axis ID, updates `size` to 2, and may annotate
  `coordinate="real-imag"`.

Shape prediction and axis metadata prediction should be paired. A future
operation API can expose:

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

- `AxisInfo` and helpers live in `arrayscope.core` and import no Qt.
- `ArrayDocument` can expose `current_axes` matching `current_shape`.
- Existing operation tests gain axis metadata cases for crop, reduction,
  combine/split, and FFT-like preservation.
- The UI can keep using integer axes while showing labels from `AxisInfo`.
- No montage or sync feature relies only on post-operation integer axis
  positions.
