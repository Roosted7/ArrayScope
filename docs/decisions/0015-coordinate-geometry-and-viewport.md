# 0015 — Display geometry and explicit viewport policy

ArrayScope uses one internal 2D image coordinate convention:

- NumPy display images are row-major `(height, width)`.
- Display `x` is column and display `y` is row.
- `ViewState.image_axes` is `(y_axis, x_axis)`.
- ROI geometry is stored in display/image coordinates `(x, y)`.
- Montage tile shapes are `(height, width)`.

`arrayscope.display.geometry.DisplayGeometry` is the only mapper from display
points to array indices/profile states. Pixel hover, live profile markers,
montage tile lookup, and marker clamping use the same geometry object that was
created with the committed image. Montage assembly returns the
`MontageGeometry` it used, so hit testing and rendered tile layout cannot drift.

Axis flips are view-only. They invert the PyQtGraph ViewBox direction and do
not change the array index encoded by image-item coordinates.

Rendering commits image data and display geometry together. A worker result is
discarded if the document key or `ViewState` changed after the snapshot was
taken. This keeps stale background work from replacing newer user intent.

Viewport changes are explicit. Normal render calls preserve the current
ViewBox range. The view fits/resets only on the first image, display-shape
changes, or direct user actions such as Fit and 1:1. `ImageView2D.setImage`
accepts `ViewportPolicy` instead of relying on implicit auto-range.

ROI statistics remain display-space in this phase. They sample the displayed
scalar image or histogram source. Montage histogram sources contain `NaN` in
tile gaps, so ROI statistics naturally ignore inter-tile spacing. Full nD ROI
back-projection is out of scope for Phase 4a.

Input arrays are treated as immutable between renders unless the owner calls
`notify_data_changed()`. `ArrayDocument.revision` participates in evaluator
cache keys, so explicit data-change notifications invalidate image, profile,
scalar, and export-frame caches even when object identity and shape are
unchanged.

`ARRAYSCOPE_STRICT_UI=1` enables development/test strictness for GUI failures:
programming exceptions are logged with traceback and re-raised instead of being
silently swallowed.

