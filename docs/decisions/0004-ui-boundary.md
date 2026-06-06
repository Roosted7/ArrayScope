# 0004 — UI boundary

ArrayScope keeps the main window as the coordinator, but standalone concerns
should live outside `arrayscope.window.main`.

Current boundaries:
- `arrayscope.app.launch` owns QApplication creation, multiprocessing launch, and IPython Qt event-loop handling.
- `arrayscope.ui.dialogs` owns standalone dialogs such as save-range selection.
- `arrayscope.ui.widgets` owns reusable widgets such as `RangeSlider`.
- `arrayscope.display.colormaps` owns custom colormap construction.
- `arrayscope.display.line_plot` owns the line plot widget, crosshair, hover display, plot style, and plot-specific zoom behavior.
- `arrayscope.export.video` owns file encoding and progress UI, but uses `arrayscope.display.slice_engine` for export frame display data.
- `arrayscope.window` owns main-window orchestration through small mixins for rendering, state sync, operation actions, file reload, and domain indicators.

The old flat module layout was replaced with focused subpackages:
`app`, `core`, `operations`, `display`, `profiles`, `io`, `ui`, `export`,
and `window`. Root-level implementation imports are intentionally unsupported.
The public user API is the callable package module:

```python
import arrayscope as asc
asc(data)
```

Export frame preparation should use:

```python
data + ViewState -> slice_engine.make_export_frame(...)
```

This keeps screen display and export display behavior aligned. Export-specific
code may still normalize to `uint8`, apply the current view flips as pixels, resize
for pixel-ratio options, and write files.
