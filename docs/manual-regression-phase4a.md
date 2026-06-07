# Phase 4a Manual Regression Checklist

Run these checks before release-level changes to display coordinates, montage,
ROI, profile interaction, or viewport behavior.

## Viewport

- Open a 2D array, pan/zoom, change channel and scale, and confirm the viewport
  stays fixed.
- Step a non-image slice in a 3D array and confirm the viewport stays fixed.
- Use Fit and 1:1 and confirm those actions intentionally change the viewport.
- Show/hide Profile, Operations, and Inspection docks and confirm the image
  viewport does not jump.

## Coordinates

- Hover a normal 2D image and confirm HUD pixel coordinates match array values.
- Use an image-axis range such as `0:2:100` and confirm hover/profile positions
  report actual ranged axis indices.
- Select reversed image axes and confirm row/column mapping still follows
  `(y_axis, x_axis)`.

## Montage

- Enter `:` or a stepped range on a non-image axis to create a montage.
- Hover a second-row/second-column tile and confirm HUD pixel value and live
  profile use the same montage index.
- Move the live profile marker onto a montage gap and confirm it clamps to or
  clears consistently.
- Check the last incomplete montage row: empty tile slots should not map to
  array indices.

## ROI

- Create rectangle and polyline ROIs on a normal image and confirm statistics
  match visible scalar values.
- Create a montage ROI spanning tile plus gap and confirm gap pixels do not
  contribute finite values.
- In complex RGB display, confirm ROI statistics use the magnitude histogram
  source rather than RGB color channels.

## Cache And Data Mutation

- Render an image, mutate the original NumPy array in place, call
  `win.notify_data_changed()`, and confirm displayed values refresh.
- Confirm cache diagnostics do not report stale image/profile/scalar reuse after
  the data-change notification.

## Strict UI Mode

- Run a focused UI test with `ARRAYSCOPE_STRICT_UI=1` and confirm callback
  exceptions fail the test instead of being swallowed.

