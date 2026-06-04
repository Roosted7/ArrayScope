# 0004 — UI boundary

ArrayScope keeps the main window as the coordinator, but standalone concerns
should live outside `arrayscope.py`.

Current boundaries:
- `launch.py` owns QApplication creation, multiprocessing launch, and IPython Qt event-loop handling.
- `dialogs.py` owns standalone dialogs such as save-range selection.
- `widgets.py` owns reusable widgets such as `RangeSlider`.
- `colormaps.py` owns custom colormap construction.
- `line_plot.py` owns the line plot widget, crosshair, hover display, plot style, and plot-specific zoom behavior.
- `video_export.py` owns file encoding and progress UI, but uses `slice_engine` for export frame display data.

Export frame preparation should use:

```python
data + ViewState -> slice_engine.make_export_frame(...)
```

This keeps screen display and export display behavior aligned. Export-specific
code may still normalize to `uint8`, apply the current view flips as pixels, resize
for pixel-ratio options, and write files.
