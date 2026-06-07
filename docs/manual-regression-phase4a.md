# Phase 4a Manual Regression Checklist

Run these checks before release-level changes to display coordinates, montage,
ROI, profile interaction, or viewport behavior.

## Phase 4b Resolution Notes

Automated coverage now guards the main regressions found here:

- Fit and 1:1 use `ViewportController`; 1:1 computes a ViewBox range from viewport pixels.
- Managed dock menu actions no longer use `toggleViewAction()`, and direct managed dock show/hide
  calls are forbidden by an architecture test.
- Closing a window clears queued local thread-pool work and stale async pixel/profile/render results
  are ignored.
- Montage hover context is produced by `DisplayGeometry`, so tiled axes are not duplicated.
- “Live profile from this axis” sets exactly one profile axis.
- Tiled dimension X/Y buttons are disabled and guarded defensively; empty range text clears to a
  midpoint scalar slice.
- File reload preserves compatible operation stacks and prompts before clearing incompatible stacks.
- ROI colors, table rows, histogram curves, selection, and delete paths are synchronized through
  `RoiStore`/`RoiTableModel`.

Manual checks to keep after Phase 4b: actual floating-dock dragging on each platform, high-DPI 1:1
visual fidelity, long-running real-data cancellation behavior, large montage performance, and the
file reload confirmation/save-recipe UX.

## Viewport
- Open a 2D array, pan/zoom, change channel and scale, and confirm the viewport
  stays fixed.
  - This seems okay, but suboptimal. The viewport should by default be used as much as possible.
  - When the displayed geometry changes, this behaviour should re-scale to optimally fit the new geometry
  - Unless the user has explicitly changed the viewport (zoomed / panned), in which case the viewport should be preserved as much as possible, even if the displayed geometry changes.
- Use Fit and 1:1 and confirm those actions intentionally change the viewport.
  - These viewing options seem broken! The image does not change at all...

- Show/hide Profile, Operations, and Inspection docks and confirm the image
  viewport does not jump.
  - Opening the profile dock does not resize the main window, so the 2D view shrinks in height
  - Opening the operations dock does not resize the main window, so the 2D view shrinks in width
    - Oddly enough, unlike with the profile dock, grabbing the edge or corner of the window instantly does resize the main window to the correct size
    - So it _almost_ works correctly! Perhaps the updated size needs beter propagation?
- Open the Inspection dock and confirm it defaults to a left-docked panel next
  to the 2D view, not a floating overlay.
  - The inspection dock opens to the left - but has exactly the same issues as the operations dock
- Close a non-floating dock and confirm the 2D view keeps its size when the main
  window can reasonably shrink or move to preserve it.
  - Closing the profile dock - does correctly resize the main window!
  - Closing the operations dock, does _not_ correctly shrink the width. Even after grabbing the edge or corner.
  - (Same for inspection dock, it has identical problems)
- Float each dock manually and confirm it has a usable title bar, can be moved,
  and exposes a bottom-right resize grip.
  - Floating the profile dock correctly shrinks the width of the main window!
    - While it has a title bar and working resize grip, it cannot be moved by dragging the title bar.
    - When the floated dock is closed, and later re-opened: it fails. It is drawn like a docked panel - but drawn over the 2D view. The window does not resize.
    - This also seems the happen when the re-dock button is pressed - although sometimes this button does nothing at all. (It never seems to work correctly, failing in 2 different ways)
  - The operations dock also cannot be moved by dragging the title bar. It also correctly shrinks the main window when floated
    - The redocking does seems to work correctly for the operations dock, but it does not resize the main window. It does seem to do something related to it, because as soon as the main window corner/edge is grabbed, the main window resizes to the correct size.
  - The profile dock has exactly the same issues as the operations dock
- Close the main window while docks have been shown/floated; the application
  should fully exit when no other ArrayScope windows remain.
  - The floating windows now correctly close! But sometimes the main process does not exit, as it seems to perform any remaining queued operations before exiting.
  - All actions should be canceled immediately - except file saving operations, if needed, which should be allowed to finish before exit. (Perhaps drawing a small "Saving..." window, with a cancel button)

## Montage

- Enter `:` or a stepped range on a non-image axis to create a montage.
- Hover a second-row/second-column tile and confirm HUD pixel value and live
  profile use the same montage index.
    - Seem correct, but the both the text in the bar above the 2D view and hover "tooltip" on the main 2D view incorrectly show dimensions twice when tiling (eg; "d2 = 50 d2 = 102 ) - the second value seems correct
- Move the live profile marker onto a montage gap and confirm it clamps to or
  clears consistently.
  - Seems okay!
  - However when opening the profile dock, by using the "Live profile from this axis" action in the dimensions menu, when the profile dock is closed: opens the profile docks with also a another profile axis
  - When the profile dock auto-opens due to a live profile being added/enabled, this should be the only profile shown.
- Check the last incomplete montage row: empty tile slots should not map to
  array indices.
  - Correct! No values are shown and the live-profile cross-hair does not enter this region
  - Perhaps... It would be nice to shown a thin border around the montage tiles, so that it is visually clear to the user where the tiles are, and why the live-profile cross-hair does not outside of them.

## ROI

- Create rectangle and polyline ROIs on a normal image and confirm statistics
  match visible scalar values.
  - Seems correct! Although it is hard to match the values to the ROIs.
  - Perhaps the info textbox should unique to each ROI (so the name can be removed), and be placed next to the ROI itself
  - The user can move the per-ROI info boxes around, so that the default position is not fixed
  - When the user moves the ROI or text info, a line should be drawn between the ROI and its info box, so that it is clear which info box belongs to which ROI
  - When the 2D view is zoomed or panned, the ROI info boxes should stay in the same position relative to the ROI itself - perhaps also scaling?
- In complex RGB display, confirm ROI statistics use the magnitude histogram
  source rather than RGB color channels.
  - Works!
  - But this entire panel is not very helpful nor easy to use.
  - The list with checkbox adds very little value
  - Adding ROIs in the panel is clumsy, and should perhaps be removed in favor of the right click (or a single button)
  - The table with the statistics is not very clear, and it is not obvious that the plot below is a histogram.
  - The colors in the histogram are meaningless
  - The items in the table/list above should match color to the histogram
  - The ROIs themselves in the 2D view should also match the colors in the histogram and table
  - The ROIs should be deletable by right clicking the table row (or perhaps add a small delete button in the row?)
  - The ROIs should also be deletable by right clicking the ROI itself in the 2D view (or perhaps add a small X that appears when hovering the ROI in the 2D view?)
  - When the user clicks on a row in the table, the corresponding ROI in the 2D view should be highlighted (perhaps by flashing or changing color)

## Cache And Data Mutation

- Render an image, mutate the original NumPy array in place, call
  `win.notify_data_changed()`, and confirm displayed values refresh.
- Confirm cache diagnostics do not report stale image/profile/scalar reuse after
  the data-change notification.
  - Not tested Numpy input.
  - File reload button seems to work, but also deletes all operations. That cannot happen! If something like that must happen, a confirmation dialog should be shown, and the user should be given the option to save the current recipe (operations + view state) before reloading the file.
  - The file reload button does not look like a button. It merely is a icon, with no change when clicked (or hovered)

## Strict UI Mode

- Run a focused UI test with `ARRAYSCOPE_STRICT_UI=1` and confirm callback
  exceptions fail the test instead of being swallowed.
  - When pressing a X or Y dimension button, on a tiled dimension - the error below is thrown. To prevent this, tiled dimensions should have the X and Y buttons grayed out. And a tooltip should be shown on hover, explaining that these buttons are not available for tiled dimensions.
  - When a (tiled) dimension slice string is fully deleted, it should default to a scalar (mid-dimension), instead of reverting the old string. This allows easy canceling of a montage

```
Traceback (most recent call last):
  File "/home/thomas/projects/ArrayScope/arrayscope/ui/dimension_controls.py", line 190, in set_dimension_role
    self._set_view_state(self.view_state.with_image_axis(role, axis))
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/thomas/projects/ArrayScope/arrayscope/core/view_state.py", line 147, in with_image_axis
    return self.with_image_axes(primary_axis, secondary_axis)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/thomas/projects/ArrayScope/arrayscope/core/view_state.py", line 130, in with_image_axes
    return replace(self, image_axes=(axis0, axis1))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/thomas/miniconda3/envs/arrayscope/lib/python3.12/dataclasses.py", line 1588, in replace
    return obj.__class__(**changes)
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "<string>", line 18, in __init__
  File "/home/thomas/projects/ArrayScope/arrayscope/core/view_state.py", line 85, in __post_init__
    self.validate()
  File "/home/thomas/projects/ArrayScope/arrayscope/core/view_state.py", line 319, in validate
    raise ValueError("montage axis cannot also be an image axis")
ValueError: montage axis cannot also be an image axis
Traceback (most recent call last):
  File "/home/thomas/projects/ArrayScope/arrayscope/ui/dimension_controls.py", line 190, in set_dimension_role
    self._set_view_state(self.view_state.with_image_axis(role, axis))
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/thomas/projects/ArrayScope/arrayscope/core/view_state.py", line 147, in with_image_axis
    return self.with_image_axes(primary_axis, secondary_axis)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/thomas/projects/ArrayScope/arrayscope/core/view_state.py", line 130, in with_image_axes
    return replace(self, image_axes=(axis0, axis1))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/thomas/miniconda3/envs/arrayscope/lib/python3.12/dataclasses.py", line 1588, in replace
    return obj.__class__(**changes)
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "<string>", line 18, in __init__
  File "/home/thomas/projects/ArrayScope/arrayscope/core/view_state.py", line 85, in __post_init__
    self.validate()
  File "/home/thomas/projects/ArrayScope/arrayscope/core/view_state.py", line 319, in validate
    raise ValueError("montage axis cannot also be an image axis")
ValueError: montage axis cannot also be an image axis
```
