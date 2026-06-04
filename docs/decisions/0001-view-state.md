# 0001 — ViewState boundary

## Status

Planned / initial implementation.

This document defines the intended role of `ViewState`. Update it only when the boundary changes in a meaningful way.

## Problem

The current viewer stores much of its view state implicitly inside Qt widgets and the main window. That is workable for a small viewer, but it makes future features harder:

* extracting a pure slice/display engine;
* syncing multiple windows;
* exporting the same view that is shown on screen;
* testing slicing and display behavior without Qt;
* avoiding duplicated display logic in export/video code.

We need an explicit state object that says what the user is currently looking at.

## Decision

Add a Qt-free `arrayscope/view_state.py`.

`ViewState` represents the current interpretation of an n-dimensional array for viewing. It should be a small dataclass, supported by small enums where useful.

It should describe:

* the input array dimensionality and shape;
* which axes are shown as the 2D image axes;
* which axis, if any, is used for line plotting;
* the selected index for non-displayed dimensions;
* the selected complex/channel display mode;
* the selected display scale mode;
* per-axis view flags such as flip and fftshift.

## Non-goals

`ViewState` does not own:

* the actual ndarray data;
* file loading;
* Qt widgets;
* pyqtgraph items;
* histogram computation;
* RGB conversion;
* dimension operations such as FFT, mean, crop, RSS;
* derived-data history;
* linked-window synchronization.

Those belong elsewhere.

## Intended direction

After `ViewState` exists, a later `slice_engine.py` should take:

```python
data, view_state
```

and return display-ready 2D image, RGB image, line profile, or histogram input.

The main window should eventually become mostly responsible for:

* updating `ViewState` from user actions;
* passing `data + ViewState` to the display engine;
* showing the returned result.

## Invariants

* `ViewState` must not import Qt.
* Axis indices must be valid for `shape`.
* Slice indices must be in bounds.
* `image_axes`, when present, must contain two distinct valid axes.
* `line_axis`, when present, must be a valid axis.
* Per-axis flags must have length `ndim`.
* State updates should return a valid state or raise a clear exception.

## Initial migration rule

The first implementation should preserve behavior. It is acceptable if the main window still contains most slicing/display logic after this step. The purpose of this step is to create an explicit state boundary, not to complete the architecture.
