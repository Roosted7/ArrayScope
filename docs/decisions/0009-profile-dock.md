# 0009 — Profile dock

Profiles are shown in a dockable `Profile` panel rather than a second central
tab. Live image profiles should be visible while the image remains visible, so
the profile plot lives in a `QDockWidget` registered in the bottom dock area but
hidden by default. Hidden docks do not consume layout space or trigger profile
line queries.

The dock owns the existing line plot controller and adds lightweight profile
controls:

- profile axis selection, backed by visible per-dimension `P` role buttons and
  the existing single `line_plot_dimension`;
- profile Y range mode: `Match image window` or `Auto`.

`Match image window` is the default because live profiles should usually compare
against the same numeric window as the image. The profile still uses
`slice_engine.make_line(...)` through `OperationEvaluator`, so channel, scale,
slice state, and operation-stack behavior stay aligned with other profile and
line display paths.

The old hidden line-plot tab is removed from normal use. For 1D data, the
Profile dock is shown and focused while the central image tab is disabled.

When `Live profile` is enabled and the Profile dock is hidden, the dock is shown
as a floating dock window. It remains a normal `QDockWidget` with close, move,
and float features, so users can drag the dock title bar, close it, or dock it
back into the main window. Closing the Profile dock disables live profile mode to
avoid hidden background processing.

Dimension roles are presented as compact per-dimension `Y`, `X`, and `P`
buttons. The model keeps profile axes as a tuple so multiple profile axes can be
added later, but this first version plots one active profile axis at a time.

The live image marker has draggable vertical and horizontal lines plus a
draggable center handle. Dragging a line changes one coordinate; dragging the
center handle changes both. Marker movement is clamped to valid image
coordinates before profile extraction.
