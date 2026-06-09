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

### 1 — recovery stability

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

### 2 — performance discipline

* [x] Debounce ROI stats.
* [x] Move heavy ROI histograms/stats off UI thread.
* [x] Throttle live profile and hover requests.
* [x] Deduplicate prefetch requests.
* [x] Disable operation-backed image prefetch until a priority scheduler exists.
* [x] Avoid hidden profile/inspection render-tail work.

### 3 — real-path tests

* [x] Add deterministic tests for every known manual bug.
* [x] Expand Hypothesis tests to arbitrary image axes and axis ranges.
* [x] Add pytest-qt action-path tests.
* [x] Add strict UI mode in CI.
* [x] Add architecture guards for dock and high-frequency status behavior.
* [x] Add real operation dock user-close regression test.
* [x] Add real hover path tests that fail on scalar scheduling or “updating”.
* [x] Add viewport toolbar tests that fail if Fit/1:1 render data.

## Phase 4d - Improve realworld usage

Goal: Make ArrayScope responsive, bounded, and predictable under real interactive use.

### P0 — correctness and memory safety

* [x] Fix slice_engine display-axis preservation
* [x] Add render memory estimates
* [x] Prevent giant montage allocations

### P1 — latest-only evaluation scheduler

* [x] Implement visible-render replacement semantics
* [x] Split evaluation categories
* [x] Add `EvalPriority`

### P2 — Tiled montage renderer

* [x] `MontagePlan` for separation
* [x] Evaluate only visible tiles
* [~] Cache tile results (tile byte accounting fixed in Phase 4e P0; predictive/byte-based selection deferred)
* [x] Sample histogram/levels

### P3 — Managed panels instead of floating QDockWidgets

* [x] Introduce `PanelManager` for authoritative state
* [x] Use QDockWidget only for docked panels, for detached panels; use `QDialog` / tool window
* [x] Ensure Wayland move functionality, through `startSystemMove()`
* [~] Preserve canvas size on every panel transition (setGeometry-based best effort remains; transaction rewrite deferred to Phase 4e P1)

### P4 — Viewport model and toolbar UX

* [x] Fit becomes a checkable locked mode (but 1:1 remains momentary)
* [x] All button should look and behave like buttons (hover, pressed, checked)

### P5 — ROI/profile responsiveness

* [x] One interaction-mode owner: `class InteractionMode(Enum)`
* [x] Debounce and background ROI stats (visible/selected ROI immediately, histograms when idle)
* [~] Hidden panels do not compute (only refresh when shown)

### P6 — Docs, metrics, and regression pipeline

* [x] Add performance budgets
* [~] Add benchmark/stress tests
* [x] Update manual regression docs (OS/window-manager behavior; Phase 4e P0 adds explicit panel lifecycle sequence)

## Phase 4e

### P0 — correctness and memory stop-the-bleeding

goal: No silent memory blowups, no broken panel ownership states, no misleading “done” docs.

* [x] Fix BoundedArrayCache byte accounting for RenderedTile.
* [x] Add tests for cached tile byte accounting and eviction.
* [x] Fix PanelManager DETACHED → HIDDEN and HIDDEN → DOCKED body ownership.
* [x] Remove or neuter StandardDockWidget.closeEvent override.
* [x] Add panel lifecycle tests for detach/hide/show/redock/reset.
* [x] Mark roadmap Phase 4d partial where appropriate.

### P1 — Wayland preserve-canvas transaction

Main goal: Panel transitions preserve the central viewer size as best as Wayland permits.

* [x] Replace setGeometry-based panel delta with post-layout central-widget correction.
* [x] Use resize(), not setGeometry(), for preserve-canvas behavior.
* [x] Add QTimer-based verification retries.
* [x] Do not move window position during preserve.
* [x] Add setting: preserve canvas on panel changes = best effort / off.
* [x] Add manual Wayland test doc.

### P2 — bounded montage renderer v1

Main goal: Montage never allocates based on total stack size during interaction.

* [x] Add byte-based visible tile selection.
* [x] Replace local mini-montage with viewport canvas + origin.
* [x] Make DisplayGeometry understand montage canvas origin.
* [x] Use sampled histogram data for montage levels instead of full hist collage where possible.
* [x] Add RSS stress test.
* [x] Remove abandoned multi-ImageItem path.

### P3 — operation cost model and FFT backend

Main goal: Slow operations become predictable, measurable, and configurable.

* [x] Add OperationCapabilities / OperationCost metadata.
* [x] Add peak-memory estimates for reductions, RSS, complex conversion, FFT.
* [x] Add scipy.fft backend with worker control.
* [x] Add optional pyFFTW backend, import-guarded at runtime and included in the conda dev/test env.
* [x] Add app settings for FFT workers and render memory budget.
* [x] Add benchmarks for raw slicing, FFT slicing, montage, ROI stats.

### P4 — scheduler v2

Main goal: Visible work always wins.

* [ ] Keep visible render max_workers=1 and latest-only.
* [ ] Add cost-aware decision: sync/cached, async, refuse, degraded preview.
* [ ] Add idle-only prefetch.
* [ ] Add cache hit-rate diagnostics.
* [ ] Add adaptive prefetch only after measurement.
* [ ] Add cooperative cancellation points for chunked operations.

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
