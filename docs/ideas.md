# Ideas

Agents may append concise ideas here when discovered during implementation.

Rules:

* Do not promote ideas to `docs/roadmap.md` unless explicitly requested.
* Keep entries short and actionable.
* Prefer one-line context plus why it matters.
* Put speculative or risky work under “Maybe later” or “Maybe never”.

## UI / UX ideas

* For 1D arrays, consider making the profile plot the central widget instead of a dock so the first
  viewport has no sparse empty central area.
* Canvas-first default layout: for simple 2D arrays, show only image, histogram, compact dimension controls, and minimal display controls.
* Hide the Operations dock by default when there are no operations; show it automatically after the first operation.
* Hide the Profile dock by default unless live profile is enabled, data is 1D, or a profile axis is selected.
* Replace the remaining “Image View” tab chrome with a direct central canvas unless multiple central view modes are actually active.
* Add layout presets: Minimal, Inspect, Pipeline, Compare.
* Add a Reset Layout action plus named saved layouts.
* Add a command palette / locator, e.g. `Ctrl+K`, for commands such as “FFT dim 2”, “RSS dim 3”, “save recipe”, “show profile”.
* Add an operation palette/search: searchable list of operations with short descriptions and compatible dimensions.
* Add on-canvas HUD overlays for pixel value, cursor index, current window/level, zoom, and active operation stack summary.
* Consider adding optional profile/montage badges to dimension chips later if the per-dimension menu is not discoverable enough.
* Add hover tooltips for dimension roles and operation rows.
* Add a “compact controls” mode for quick plotting and a “full controls” mode for inspection.
* Add keyboard-first navigation: arrow keys/scroll for active slice dimension, shortcuts for active dimension selection, `F` fit, `1` 1:1 pixels, `A` auto window.
* Add transient toast/status messages instead of printing warnings to stdout.
* Add an overview/minimap dock for large 2D images.
* Add visually richer operation rows: drag handle, enable checkbox, operation name, output shape, cache status, delete button/context menu.
* Add operation row color/status bar: cached/ready/stale/error, but only when the status reflects real state.
* Add optional dark/light/native theme choice after the interaction model is stable.
* Add visual warnings before expensive full materialization or export.

## Dimension and viewing ideas

* Add first-class axis metadata: original axis id, label, size, units, spacing, and source axis after reductions.
* Support user-renamable dimensions, e.g. `x`, `y`, `z`, `coil`, `time`, `echo`, `channel`.
* Add a montage role (`M`) for creating 2D grid/collage views over one or two dimensions.
* Add multiple profile axes with either overlaid curves or stacked mini-plots.
* Add profile modes for complex data: magnitude, phase, real/imag pair, and magnitude with phase color strip.
* Add line-profile export to CSV/NPY.
* Add ROI tools: point, line, rectangle, ellipse, freehand later.
* Add ROI statistics: mean, max, min, std, RSS, histogram.
* Add linked crosshair between image and profile views.
* Add optional image marker snapping to integer pixels, center of pixel, or nearest local maximum.
* Add multi-array compare modes: difference, ratio, phase difference, overlay, linked cursor.
* Make ROI compare layers public, operation-aware, and session-backed instead of the current internal compatible-2D histogram scaffold.

## Operation / pipeline ideas

* Add enable/disable operation without deleting it.
* Add inline edit for operation parameters, starting with crop ranges.
* Add duplicate operation.
* Add operation groups or named sections.
* Add pipeline comments/labels.
* Add recipe save/load including ViewState, not just operation stack.
* Add full session save/load: operation stack, axes, slices, windowing, channel, scale, layout, profile state.
* Add derived-array export: save materialized result after operations as `.npy` / `.npz` first, with recipe sidecar.
* Add output-size estimation before materialization/export.
* Add undo/redo stack for view and operation changes.
* Add operation compatibility preview before applying or reordering.
* Add operation search by dimension compatibility, e.g. show only operations valid for selected dimension.
* Add batch/apply-to-all-open-windows recipe workflow later.

## Performance ideas

* Replace full derived-array materialization during image/profile updates with slab-based evaluation.
* Slice as early as possible, then apply operations only to the minimal data needed for the displayed image/profile.
* Support lazy/view-like ops for crop, reverse, conjugate, and simple slicing.
* Add bounded display-frame cache keyed by document, view state, channel, scale, windowing, and colormap.
* Add optional nearby-slice prefetch after the evaluator is slab-based.
* Add cancellation/ignore-stale logic for background workers.
* Add cache memory budget, e.g. max frames or max MB.
* Add performance HUD/debug panel showing cache hits/misses, materialized shape, and evaluation time.
* Add a user-facing cache budget setting with presets for laptop, workstation, and memory-constrained sessions.
* Add explicit export progress for derived-array `.npy/.npz` saves after materialization is complete.
* Add benchmark fixtures for representative MRI stacks so cache and slab changes can be compared over time.
* Add optional memory-mapped array support for large `.npy` files.
* Consider zarr/dask later, but do not introduce them before the internal lazy evaluator is clean.

## IO / MRI ideas

* Add BART `.cfl/.hdr` write/export.
* Add NIfTI export for 3D/4D derived arrays when affine metadata is available.
* Preserve metadata from source formats: affine, voxel size, orientation, echo/time/coil labels.
* Add better DICOM series grouping metadata display.
* Add ISMRMRD support later if useful.
* Add “coil dimension” helper operations: RSS, SoS, simple combine, phase reference, coil montage.
* Add k-space/MRI presets: centered FFT over selected axes, show log/symlog magnitude, crop center k-space.

## Testing / quality ideas

* Keep pure tests independent of Qt and pyqtgraph imports.
* Split GUI workflows from pure helpers when adding new features.
* Add smoke tests for minimal viewer launch, operation dock, profile dock, and dimension controls.
* Add screenshot/artifact tests for UI regressions, but keep them few and meaningful.
* Add a clean-environment import test: `python -c "import arrayscope"`.
* Add tests for recipe compatibility across shape-changing operations.
* Add tests for axis identity once axis metadata exists.
* Add benchmark-style tests for display refresh on representative 3D/4D arrays.

## Technical debt

* Replace print-based warnings with logging or user-facing status messages.
* Preserve axis identity metadata across shape-changing operation stacks.
* Add first-class scalar display support for operation stacks that reduce all dimensions.
* Clarify FFT naming: current centered FFT/IFFT follow viewer convention but may surprise users expecting NumPy direction.

## Maybe later

* Plugin system.
* Full node-graph workflow editor.
* GPU-backed display/evaluation.
* Dask/zarr execution backend.
* Full napari replacement behavior.
* Full MATLAB ArrayShow parity.
* QML rewrite.
