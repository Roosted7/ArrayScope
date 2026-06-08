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
* [x] Multiple profile axes.
* [x] Better complex profile modes: magnitude, phase, real/imag, phase color strip.
* [x] Profile export.
* [x] Montage role (`M`) for 2D stacked/collage views.
* [x] One-axis plot mode as a natural fallback when only one display dimension is selected.
* [x] ROI line/rectangle tools.
* [x] ROI statistics.
* [x] Histogram comparisons for ROI / multiple arrays.

## Phase 4a - Improve coordinate usage and GUI

Goal: Interaction hardening, Coordinate-space contract, and GUI regression suite

* [x] AxisInfo / axis identity metadata
* [x] DisplayGeometry / coordinate mapping contract
* [x] Viewport preservation policy
* [x] Strict GUI error/logging mode
* [x] Manual regression checklist
* [x] pytest-qt interaction tests
* [x] Hypothesis/property tests for mapping
* [x] Cache revision/invalidation policy
* [x] ROI coordinate-space policy

## Phase 4b: hardening and debt burn-down

Goal: fix bugs, harden interactions, and burn down technical debt before phase 5

### P0 — correctness

* [x] Fix lazy slab crop/reverse/reduction differential failures.
* [x] Add materialized-vs-lazy differential tests.
* [x] Fix file reload semantics so operations are preserved when compatible.
* [x] Fix stale async pixel/profile/render commit guards.

### P1 — interaction/state

* [x] Replace toggleViewAction() with managed dock actions.
* [x] Route all dock visibility through WindowLayoutManager.
* [x] Add an architecture test forbidding direct managed dock show/hide.
* [x] Implement ViewportController.
* [x] Implement true Fit and true 1:1.
* [x] Disable tiled-dimension X/Y buttons and defensively guard state transitions.
* [x] Make empty montage/range text clear to scalar midpoint.
* [x] Fix live-profile axis selection to set exactly one axis.

### P2 — montage/ROI polish

* [x] Move hover/context labels into DisplayGeometry.
* [x] Fix duplicate montage context text.
* [x] Add tile borders and “showing N of M tiles” warning.
* [x] Convert ROI UI to RoiStore + QAbstractTableModel.
* [x] Add ROI color/selection/delete synchronization.

### P3 — performance and resilience

* [x] Fix image prefetch key bug.
* [x] Add in-flight prefetch dedupe and queue limits.
* [x] Add local thread-pool cancellation/clear-on-close.
* [x] Add cache diagnostics for hit/miss/prefetch usefulness.
* [x] Consider smarter predictive cache only after measuring.

### Testing infrastructure

Run in CI:

* [x] pure tests
* [x] Hypothesis differential tests
* [x] Qt interaction tests with pytest-qt
* [x] strict UI mode tests
* [x] screenshot smoke tests
* [x] architecture guard tests

At least one CI job should run: `ARRAYSCOPE_STRICT_UI=1 pytest tests/ui`

## Phase 4c: stabilization before Phase 5

Goal: fix remaining bugs and harden interactions before adding multi-window and session features.

### Phase 4c-1 — recovery stability

* [x] Fix display-axis range extraction bug.
* [x] Compare full async request keys, not partial document keys.
* [x] Fix channel auto/manual behavior.
* [x] Fix montage/visible-render status stuck on Computing.
* [x] Simplify dock manager by removing event-filter/snapshot direct-close repair machinery.
* [x] Add explicit user-hidden intent so Operations does not auto-reopen after user close.
* [x] Stop forced redocking when showing floating Inspection/Profile docks.
* [x] Remove FOV from UI for now.
* [x] Make Fit and 1:1 viewport-only commands.
* [x] Replace hover scalar evaluation with direct committed-display reads.
* [x] Move freehand/polyline ROI drawing to one-shot canvas interaction.

### Phase 4c-2 — performance discipline

* [x] Debounce ROI stats.
* [x] Move heavy ROI histograms/stats off UI thread.
* [x] Throttle live profile and hover requests.
* [x] Deduplicate prefetch requests.
* [x] Disable operation-backed image prefetch until a priority scheduler exists.
* [x] Avoid hidden profile/inspection render-tail work.

### Phase 4c-3 — real-path tests

* [x] Add deterministic tests for every known manual bug.
* [x] Expand Hypothesis tests to arbitrary image axes and axis ranges.
* [x] Add pytest-qt action-path tests.
* [x] Add strict UI mode in CI.
* [x] Add architecture guards for dock and high-frequency status behavior.
* [x] Add real operation dock user-close regression test.
* [x] Add real hover path tests that fail on scalar scheduling or “updating”.
* [x] Add viewport toolbar tests that fail if Fit/1:1 render data.

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

* [ ] Axis labels and units, from file metadata or from `asc` keyword arguments (or in UI manually).
* [ ] Source metadata display panel.
* [ ] Coil/RSS helpers.
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
