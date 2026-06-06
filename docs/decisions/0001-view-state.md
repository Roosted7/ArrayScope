# 0001 — ViewState boundary

## Status

Accepted.

## Decision

`arrayscope/view_state.py` is the Qt-free, authoritative description of the
current view of the derived array.

It owns:

- shape and dimensionality;
- image axes;
- profile/line axis;
- slice indices for non-displayed axes;
- channel and scale modes;
- per-axis display flags such as flip and fftshift.

Widgets mirror `ViewState`; they do not define it. User actions update
`ViewState`, then `ArrayScopeWindow.render()` derives controls, images, profiles,
and labels from that state.

## Non-Goals

`ViewState` does not own data arrays, file loading, Qt widgets, histogram
widgets, RGB conversion, operation history, or linked-window synchronization.

## Invariants

- `ViewState` must not import Qt.
- Axis indices must be valid for `shape`.
- Slice indices must be in bounds.
- `image_axes`, when present, must contain two distinct valid axes.
- `line_axis`, when present, must be valid.
- Per-axis flags must have length `ndim`.
- Shape changes must go through `ViewState.for_shape()` so surviving state is
  preserved and invalid state is repaired.
