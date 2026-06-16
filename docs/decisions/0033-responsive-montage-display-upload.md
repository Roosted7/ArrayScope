# 0033 Responsive Montage Display Upload

## Status

Accepted.

## Context

Operation-backed montage can make tile evaluation fast through the stage cache, but the UI can still
freeze when committing a large updated montage canvas to pyqtgraph. `ImageItem.setImage()` and
histogram updates run on the Qt thread, and moving them to a worker would violate Qt ownership rules.

The expensive phase can include more than the visible image upload: a separate histogram image item,
histogram recomputation, RGB/complex re-windowing, and duplicate level updates can all contribute.
Repeated calls to pyqtgraph `HistogramLUTItem.setImageItem()` also connect image-change signals each
time, so careless rebinding can multiply histogram recomputation.

## Decision

Display upload remains on the Qt thread, but ArrayScope reduces and measures that work.

`ImageView2D` owns upload instrumentation for visible image upload, histogram plot upload, histogram
recompute, RGB re-windowing, level synchronization, and profile-bound refresh. Runtime diagnostics show
these phases separately from tile evaluation, stage cache, and canvas composition.

Histogram binding is idempotent and centralized. ArrayScope binds the histogram widget through one
helper, disconnects pyqtgraph's automatic image-change callback, and explicitly refreshes the
histogram plot once per committed state.

Programmatic presentation commits upload pixels with already-decided levels, then synchronize
histogram handles without causing a second image re-window or upload. Explicit user histogram edits
still update displayed pixels immediately.

Progressive montage commits are coalesced to the latest state when upload is slow. Final complete
state is always committed.

For large or previously slow montage displays, `ImageView2D` switches to an internal exact tile-layer
mode. The full canvas remains the committed value source for hover/status and semantics, but per-tile
`ImageItem`s paint visible loaded tiles. Updating one tile then uploads that tile rather than the full
canvas. This mode is internal and does not change the public viewer API.

## Consequences

Diagnostics can distinguish slow operation evaluation from slow Qt upload.

Large progressive montage is more responsive because intermediate canvas commits can be coalesced and
tile-layer mode avoids repeated full-canvas uploads.

The display implementation is more complex: ImageView2D now owns two montage paint modes, and tests
must guard that both use the same levels, histogram source, and hover/value semantics.
