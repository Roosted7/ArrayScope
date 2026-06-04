# 0007 — Window/level mode

ArrayScope image display has two window/level modes:

- relative windowing: slice changes preserve the selected low/high positions as
  fractions of the previous histogram data range, then apply those fractions to
  the new histogram data range;
- absolute windowing: slice changes preserve the exact numeric low/high levels
  already set on the histogram.

The viewer opens in relative mode, matching the previous auto-window behavior on
first display while allowing user-adjusted relative levels to persist across
slices. The window-mode controls live in the existing Display controls as
`Relative` / `Absolute` radio buttons.
The main window still uses `slice_engine.make_image(...)` for display conversion;
window-mode logic only decides whether `ImageView2D.setImage(...)` receives
`autoLevels=True` or explicit levels.

For complex/RGB display, both modes use histogram-source levels from
`ImageView2D.getLevels()` and histogram-source bounds from
`ImageView2D.getHistogramDataBounds()`. Those levels belong to
`DisplayImage.histogram_data`, usually magnitude, not the RGB image item's
0-255 range.

Channel and scale changes force auto-window in either windowing mode.
Those changes alter the numeric meaning of the displayed values, so preserving a
previous fixed range would often be misleading. Slice changes preserve levels in
relative or absolute form depending on the active mode.
