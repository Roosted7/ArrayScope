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

* Expose axis metadata in the UI: user-renamable labels, units, spacing, coordinates, and file-derived defaults.
* Support user-renamable dimensions, e.g. `x`, `y`, `z`, `coil`, `time`, `echo`, `channel`.
* Extend montage to two non-image dimensions with explicit row/column montage axes.
* Add line-profile export presets for CSV/NPY with axis metadata columns.
* Add ROI tools for point and ellipse selections.
* Add per-ROI draggable info callouts with connector lines that stay anchored correctly while panning,
  zooming, and resizing.
* Add linked crosshair between image and profile views.
* Add optional image marker snapping to integer pixels, center of pixel, or nearest local maximum.
* Add multi-array compare modes: difference, ratio, phase difference, overlay, linked cursor.
* Make ROI compare layers public, operation-aware, and session-backed instead of the current internal compatible-2D histogram scaffold.
* Add full nD ROI back-projection from display-space ROI geometry for operation-aware source-array measurement.

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

* Support lazy/view-like ops for crop, reverse, conjugate, and simple slicing.
* Add optional per-cache manual budget overrides on top of the current memory profile presets if users need finer control.
* Add richer diagnostics graphs/timelines for cache growth, scheduler activity, and prefetch usefulness.
* Add explicit export progress for derived-array `.npy/.npz` saves after materialization is complete.
* Add benchmark fixtures for representative MRI stacks so cache and slab changes can be compared over time.
* Keep future stage-cache entries tied to planner regions, operation prefixes, and document revisions;
  reusable cached data should not carry viewport placement, tile grid origin, or canvas-local coordinates.
* Consider a dedicated viewport-tile planner that owns montage canvas rect, tile coverage, and scheduling
  decisions so the window render path can stay thin as Phase 4g stage caching evolves.
* Add chunked/cancellable FFT and reduction execution so expensive transforms can be interrupted rather
  than only estimated and warned about.
* True cancellation inside one FFT call remains unsolved; current chunking cancels only between
  independent output chunks and major evaluation steps.
* Add optional memory-mapped array support for large `.npy` files.
* Measure prefetch usefulness on representative 3D/4D datasets before adding predictive cache heuristics
  beyond nearby-slice/profile prefetch.
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
* Add screenshot/artifact tests for UI regressions, but keep them few and meaningful.
* Add a clean-environment import test: `python -c "import arrayscope"`.
* Add tests for recipe compatibility across shape-changing operations.
* Add benchmark-style tests for display refresh on representative 3D/4D arrays.
* Add CI workflow updates for strict UI and screenshot artifact tests if the repository gains GitHub Actions.

## Technical debt

* Replace print-based warnings with logging or user-facing status messages.
* Add first-class scalar display support for operation stacks that reduce all dimensions.
* Clarify FFT naming: current centered FFT/IFFT follow viewer convention but may surprise users expecting NumPy direction.
* Add public data-mutation ergonomics beyond `notify_data_changed()`, such as context managers or observable data sources.
* Consider a debug overlay showing the current `DisplayGeometry` mapping under the cursor when strict UI mode is enabled.
* Consider a deliberate migration from current `start:step:stop` range text to Python `start:stop:step`
  syntax with a compatibility warning or explicit preference.
* Consider detached managed panels backed by `QDialog`/tool windows if platform-specific `QDockWidget`
  floating behavior remains problematic after the Phase 4c lifecycle cleanup.
* Prototype a detached `QDialog`/tool-window panel model for medium-term Wayland reliability instead
  of repairing `QDockWidget` lifecycle events with event filters.
* Add a priority scheduler for visible rendering, profile, ROI, hover, and prefetch work before
  reintroducing operation-backed predictive prefetch.
* Reintroduce physical FOV/aspect controls only after axis spacing/unit metadata is available.

## Maybe later

* Plugin system.
* Full node-graph workflow editor.
* GPU-backed display/evaluation.
* Dask/zarr execution backend.
* Full napari replacement behavior.
* Full MATLAB ArrayShow parity.
* QML rewrite.
