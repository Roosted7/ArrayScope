# Roadmap

This roadmap is intentionally practical. It should track accepted work, not every idea.

## Phase 0 — Foundation cleanup

* [x] Rename/fork into ArrayScope.
* [x] Add `ViewState`.
* [x] Add pure `slice_engine`.
* [x] Add pure dimension operations.
* [x] Add operation pipeline and recipes.
* [x] Add operation dock.
* [x] Add profile dock and live image-driven profiles.
* [x] Split code into focused package areas: app, core, display, operations, profiles, ui, window, export, io.
* [x] Fix pure/GUI test separation so pure tests do not require `pyqtgraph`.
* [x] Clean broad copy-pasted imports from mixins.
* [x] Split the remaining large UI builder into smaller components.
* [x] Decide and document theme strategy: built-in palette only.
* [x] Add pure-helper import checks.
* [x] Group tests by package area.
* [x] Add AxisInfo design proposal before montage/sync.

## Phase 1 — Small, fast, pleasant viewer

Goal: make `asc(data)` feel lightweight enough for quick plotting.

* [x] Canvas-first default layout.
* [x] Hide empty/unused docks by default.
* [x] Remove unnecessary central tab chrome unless multiple central views are active.
* [x] Compact dimension chips with clear `Y`/`X` roles and profile actions.
* [x] Clear, minimal display controls for channel, scale, aspect, and window mode.
* [x] Good default keyboard shortcuts for fit, 1:1, auto window, slice stepping, and profile toggle.
* [x] User-facing status/toast messages instead of stdout prints.
* [x] Layout persistence and reset layout.
* [x] Native/light/dark theme behavior that is reliable and readable.

## Phase 2 — Operation workflow v1

Goal: make the operation stack powerful but understandable.

* [x] Add operation registry.
* [x] Add recipe save/load for operations.
* [x] Add undo/clear/materialize.
* [x] Add operation delete/reorder.
* [x] Make operation rows visually richer and easier to manipulate.
* [x] Add enable/disable operation.
* [x] Add operation parameter editing for crop.
* [x] Add shape/dtype/size estimate for current derived output.
* [x] Add derived-array export to `.npy` / `.npz`.
* [x] Save recipe sidecar with derived-array export.
* [x] Add full view recipe: operations + `ViewState` + display settings.

## Phase 3 — Performance and large-array behavior

Goal: keep the UI responsive on real MRI/reconstruction arrays.

* [x] Add evaluation timing and cache diagnostics.
* [x] Add slab-based evaluator for image/profile views.
* [x] Avoid full materialization for display when operations permit slice-first evaluation.
* [x] Add bounded image/profile cache.
* [x] Add optional nearby-slice prefetch after slab evaluation works.
* [x] Add cancellation/ignore-stale behavior for background evaluation.
* [x] Add memory-budget controls or guardrails.
* [x] Warn before expensive full materialization/export.

## Phase 4 — Profiles, montage, ROI

Goal: cover the most useful ArrayShow-like inspection workflows.

* [x] Single profile axis.
* [x] Live image-hover profile.
* [ ] Multiple profile axes.
* [ ] Better complex profile modes: magnitude, phase, real/imag, phase color strip. Magnitude/phase/real/imag modes are implemented; phase color strip remains.
* [x] Profile export.
* [ ] Montage role (`M`) for 2D stacked/collage views.
* [x] One-axis plot mode as a natural fallback when only one display dimension is selected.
* [ ] ROI line/rectangle tools.
* [ ] ROI statistics.
* [ ] Histogram comparisons for ROI / multiple arrays.

## Phase 5 — Multi-window and sessions

Goal: make repeated inspection work reproducible and efficient.

* [ ] Session save/load: data reference, operation stack, view state, display settings, layout.
* [ ] Multi-window sync proposal.
* [ ] Opt-in sync groups for slice indices.
* [ ] Opt-in sync for window levels/channel/scale.
* [ ] Opt-in sync for operation stack.
* [ ] Opt-in sync for cursor/profile marker.
* [ ] Soft failure when sync target shape is incompatible.
* [ ] Copy/paste view recipe between windows.

## Phase 6 — Scientific/MRI-specific quality of life

* [ ] Axis labels and units.
* [ ] Source metadata display panel.
* [ ] Coil/RSS helpers.
* [ ] K-space presets.
* [ ] BART export.
* [ ] NIfTI export with affine metadata where available.
* [ ] Improved DICOM series metadata and grouping diagnostics.
* [ ] Optional Gyrotools `PRecon` integration for reading Philips raw or image data.
* [ ] Optional `ismrmrd` integration for reading ISMRMRD datasets.
* [ ] Optional Siemens `gt-twixtools` integration for reading Siemens raw data.

## Pre-release work

* [ ] Update README, pyproject URLs/authors, documented extras, and screenshots.
* [ ] Add examples for Julia and MATLAB usage

## Won't for now

* Full MATLAB ArrayShow parity.
* Full napari replacement.
* Plugin system.
* QML rewrite.
* GPU compute backend.
* General node-graph pipeline editor.
