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

* [x] Keep visible render max_workers=1 and latest-only.
* [x] Add cost-aware decision: sync/cached, async, refuse, degraded preview.
* [x] Add idle-only prefetch.
* [x] Add cache hit-rate diagnostics.
* [~] Add adaptive prefetch only after measurement (diagnostics and conservative cost gates exist; predictive/adaptive expansion remains future).
* [x] Add cooperative cancellation points for chunked operations.

## Phase 4f — finish bounded interactive rendering

Main goal: Montage, panels, and visible rendering are stable, bounded, and predictable.

### P0 — fix current montage correctness

* [x] Remove loaded_rect intersection from make_montage_viewport_canvas().
* [x] Canvas rect must be stable and based on requested viewport.
* [x] Add unloaded/loading/skipped tile states, with skipped reserved for exceptional over-budget tiles.
* [x] Make hover/profile distinguish gap vs loading vs skipped vs loaded.
* [x] Ensure stale montage results do not clear current overlays.

Required tests:

- viewport canvas shape does not shrink when only one tile is loaded
- canvas origin stays stable while tiles load
- hover over unloaded tile reports loading, not NaN
- hover over gap reports empty
- live profile works on loaded tile and reports loading on unloaded tile
- stale montage tile result does not mutate current UI state

### P1 — progressive tile rendering

* [x] Add MontageRenderSession.
* [x] Commit cached tiles immediately.
* [x] Schedule missing visible tiles individually.
* [x] Copy each finished tile into the current canvas.
* [x] Throttle image updates to ~30 Hz.
* [x] Show per-tile loading overlays.

### P2 — memory policy unification

* [x] Replace static constants with MemoryPolicy.
* [x] Use psutil for total/available/RSS.
* [x] Tie visible, montage, tile, stage, and prefetch budgets together.
* [x] Add user profiles: conservative / balanced / aggressive / custom.
* [x] Remove duplicate hidden limits from operations panel.

### P3 — debug diagnostics window

* [x] Add simple floating QDialog.
* [x] Show memory, cache, scheduler, render-plan, montage, FFT stats.
* [x] Update on timer.
* [x] Open only from Developer menu or env flag.

### P4 — clean Wayland preserve-canvas code

* [x] Encapsulate current trick in CanvasPreserveTransaction.
* [x] Gate strong nudge path to Wayland or fallback mode.
* [x] Capture/restore actual size constraints.
* [x] Replace print() with logging/debug diagnostics.
* [x] Add manual Wayland regression doc.

## Phase 4g — operation planner and stage cache

Main goal: Expensive operation results are computed at the right granularity once, cached intelligently, and reused across slices, montage, profiles, and ROI.

### P0 — operation capabilities

Each operation declares and handles:

* [x] output shape
* [x] output dtype
* [x] blocking axes
* [x] chunkable axes
* [x] request expansion behavior
* [x] temp multiplier
* [x] cache-stage preference
* [x] fusion eligibility

### P1 — region planner

Render requests become:

* [x] requested final region
* [x] required input region
* [x] expanded intermediate regions
* [x] candidate stage-cache points
* [x] estimated peak memory

### P2 — stage cache

Add:

* [x] in-memory stage cache
* [x] byte budget
* [x] cache priority
* [x] document/operation-prefix/region keys
* [x] stage invalidation on operation edits

### P3 — transform-aware caching

For FFT/IFFT (and similar operations) over a sliced axis:

* [x] compute expanded full-axis result once
* [x] cache final expanded stage
* [x] serve future slices from cached stage

### P4 — operation simplifier

Add algebraic simplifications:

* [x] FFT followed by matching IFFT → dtype-preserving runtime identity/cast
* [x] Reverse twice → identity
* [x] Conjugate twice → identity
* [x] Crop composition
* [x] Adjacent elementwise operation fusion, scoped to current conjugate cancellation and dtype-cast coalescing

## Phase 4h — interaction latency and progressive-render performance

Goal: Keep UI interaction responsive while exact rendering catches up, and make cached/progressive montage updates cheap enough to feel immediate.

### P0 — stabilize and instrument hot paths

* [x] Fix optional `pyfftw` tests so the base suite does not require optional backends.
* [x] Add/reset fixtures for FFT runtime options and scheduler globals that can leak across broad test runs.
* [x] Investigate and fix the chunked cancellation broad-run flake.
* [x] Fix `EvaluationController.clear_group()` so queued prefetch bookkeeping cannot be orphaned by `QThreadPool.clear()`.
* [x] Route profile prefetch away from exact live-profile work so exact profile updates and prefetch cannot corrupt each other’s scheduler state.
* [x] Avoid duplicate slab planning in image/line/scalar/export snapshot evaluation.
* [x] Add render timing diagnostics for synchronous render orchestration, planning, queue wait, evaluation, display commit, levels/histogram, operation dock refresh, inspection refresh, montage canvas work, and overlay work.
* [x] Add montage timing diagnostics for tile cache hits/misses, stage cache hits/misses, last tile eval, canvas compose/patch, `ImageItem.setImage`, and overlay update.
* [x] Show Phase 4h timing diagnostics in Developer -> Diagnostics.
* [x] Add baseline tests for timing diagnostic presence and scheduler bookkeeping.

### P1 — render coalescer and fast interactive slice path

* [x] Add a render request coalescer owned by the window/render coordinator.
* [x] Add an interactive slice path that updates `ViewState` and slice controls immediately, then schedules rendering through the coalescer.
* [x] Ensure rapid scroll/slice bursts render only the latest state.
* [x] Clear/cancel stale visible work when a newer interactive render supersedes it.
* [x] Defer operation dock, profile, ROI, and inspection refreshes during interactive bursts unless their state is immediately visible and cheap.
* [x] Preserve correctness for normal exact render, degraded preview, chunked render, montage, profile, ROI, export frame, and cache-hit paths.
* [x] Add interaction tests proving slice text updates immediately while rendering is coalesced.
* [x] Add latency-oriented tests or benchmark assertions for rapid slice changes.

### P2 — progressive montage and worker/cache policy

* [x] Split full display commit from progressive display commit.
* [x] Ensure progressive montage commits do not refresh side panels, operation dock, histogram/levels, or stale/evaluation overlays unnecessarily.
* [x] Make `MontageRenderSession` own mutable canvas buffers, histogram buffers, tile states, and dirty rects.
* [x] Patch completed tiles into the existing montage canvas instead of rebuilding the canvas from all loaded tiles.
* [x] Throttle montage screen flushes so many tile completions produce one UI update per frame interval.
* [x] Replace per-commit montage overlay item recreation with one persistent/custom overlay item.
* [x] Add a dedicated montage tile evaluation controller, initially max 2 workers.
* [x] Keep visible image rendering latest-only on its own max-1 lane.
* [x] Keep prefetch idle-only and separate from exact visible/profile/ROI lanes.
* [x] Verify tile cache keys are independent of layout-only montage choices such as column count.
* [x] Improve StageCache retention scoring using estimated bytes, recompute cost, hit count, visible reuse, and prefetch-only penalties.
* [x] Add StageCache key tests proving viewport position, montage columns, and progress/loading state do not affect stage identity.
* [x] Add benchmark thresholds for hot cached tile display and cold cheap tile latency.

## Phase 4i — stage-first rendering and hot-path cleanup

Goal: finish the remaining interaction-latency work by making expensive reusable stages explicit, preventing stale commits, and making progressive montage display updates cheap.

### P0 — stale-work correctness and pure imports

* [x] Add render-generation guards advanced by every visible-output render request or state mutation.
* [~] Compare async callback generation and current evaluator keys before image, preview, montage, profile, ROI, or pixel commits.
* [x] Make `EvaluationController.clear_group()` invalidate group generations even when no replacement job is submitted.
* [x] Pass cancellation tokens into montage tile evaluation and stage-cache slab execution.
* [x] Remove the eager `ArrayScopeWindow` import from `arrayscope.window.__init__` and lazy-load it through `__getattr__`.
* [x] Fix broad-run chunked cancellation flakes.
* [~] Add stale-commit, group-invalidation, montage-cancellation, and pure-import regression tests.

### P1 — stage-first rendering and singleflight

* [x] Add `StageMaterializationManager` owned by `OperationEvaluator`.
* [x] Add in-flight singleflight for expanded stage keys.
* [~] Route duplicate stage requests to the in-flight job instead of recomputing.
* [x] Add a dedicated stage materialization lane with controlled FFT worker settings.
* [x] Surface stage candidate bytes, budget, decision, and recompute consequence in diagnostics.
* [x] Detect common cacheable expanded stages during montage session planning.
* [x] Materialize missing fitting stages before scheduling cold tile renders.
* [x] Attach loading tiles to in-flight stage work.
* [x] Render cached-stage tiles through rendered tile cache and patch the montage session canvas.
* [x] Keep visible image, montage tile, stage materialization, and prefetch work on separate lanes.
* [~] Add concurrent cold-stage, cancellation/error cleanup, revision invalidation, refusal-diagnostics, and FFT-over-montage-axis tests.

### P2 — true progressive image update path

* [x] Add an `ImageView2D` fast pixel-update API for same-shape progressive commits.
* [x] Split full display commits from progressive pixel commits at the window/render boundary.
* [x] Freeze levels during progressive montage except at explicit recompute boundaries.
* [x] Skip histogram, side-panel, operation dock, ROI, and profile refreshes during tile patch commits.
* [x] Patch display-ready dirty tile regions for complex/RGB progressive montage.
* [x] Add tests proving progressive tile patches avoid level scans, histogram refreshes, and unrelated dock refreshes.

### P3 — stage-aware predictive cache and compute policy

* [x] Add a compute policy coordinating Qt worker lanes and FFT worker counts.
* [x] Enforce conservative limits for tile workers multiplied by FFT workers.
* [x] Add idle stage pre-materialization for valuable expanded stages that fit.
* [x] Add near-viewport rendered tile prefetch only when the required stage is cached or in-flight.
* [x] Add directional next-slice pre-render only when cost is cheap or the needed stage already exists.
* [x] Forbid prefetch paths that compute the same expensive FFT separately per tile.
* [x] Add diagnostics and tests for predictive work decisions.
* [x] Add global resource-governor feedback for lane workers, UI fan-in budgets, prefetch admission,
  and realtime diagnostics.

### P4 — regression, benchmarks, and documentation

* [ ] Add latency benchmarks for hot rendered-tile display, hot stage/cold tile display, cold shared-stage montage warmup, and rapid slice bursts.
* [x] Add manual regression coverage for FFT montage, fast slice scrolling, cache-hit stale-result prevention, and progressive levels behavior.
* [x] Update architecture docs for stage materialization ownership, compute policy, and progressive image APIs.
* [ ] Keep Phase 4i items open until broad tests pass and manual interaction confirms the lag/jitter path is gone.

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

## Phase 4l — rendering backend experiment and foundation closure

Goal: answer whether VisPy should replace the PyQtGraph pixel display path while preserving the stable ArrayScope state, coordinate, ROI, and scheduling model.

* [x] Add thread-safe bounded cache access for demand-rendered tile/ROI regions.
* [x] Add status-silent evaluator accessors for offscreen demand rendering so ROI/profile requests do not pollute visible-render diagnostics.
* [x] Add an experimental VisPy image rendering backend selected from the Performance menu.
* [x] Keep PyQtGraph as the stable default until the VisPy backend is manually validated.
* [x] Document the hybrid VisPy strategy and trade-offs.
* [ ] Manually test VisPy normal image rendering, Fit, 1:1, hover, profile marker, ROI overlays, and histogram level dragging.
* [x] Benchmark PyQtGraph vs VisPy hot-cache level changes on large scalar and complex montages.
* [x] Prototype a VisPy shader path for RGB/complex intensity windowing from separate color and scalar textures.
* [x] Replace the per-tile VisPy montage prototype with a typed, batched atlas-backed tiled renderer.
* [ ] Prototype full VisPy shader mapping from complex scalar data to magnitude/phase/RGBA if the intensity-windowing experiment is promising.
* [ ] Decide whether VisPy becomes the only renderer before Phase 5 feature work.

## Phase 4m — unified frame scheduler and rendering surface

Goal: guarantee frame progress and bounded event-loop work while keeping one semantic pipeline across
normal images, montages, PyQtGraph, and VisPy.

### P0 — trace-driven freeze fixes

* [x] Commit fully cached montage pixels before complete semantic level sampling.
* [x] Bound actual tiled upserts per GUI callback and feed real item counts into latency feedback.
* [x] Replace front-drained list work queues with constant-time FIFO queues.
* [x] Remove the unsupported VisPy montage canvas fallback.
* [x] Add phase-level montage timings and a Qt-free JSONL trace summarizer.

### P1 — progress-preserving visible scheduler

* [ ] Add explicit presented, active, and latest frame targets.
* [ ] Replace cancel-on-every-interaction with active-plus-latest scheduling and cost-aware cancellation.
* [ ] Measure request-to-first-frame, exact completion, frame age, and discarded work milliseconds.
* [ ] Allow ROI/profile/histogram work to issue immediately at lower priority and refine incrementally.

### P2 — one planner, multiple storage strategies

* [ ] Unify normal-image and montage planning behind one semantic frame planner.
* [ ] Generalize tiled presentations to large single images without montage geometry.
* [ ] Select raster versus tiled/virtual storage by dimensions, bytes, update pattern, device limits,
  and backend capability.
* [ ] Make clean viewport/presentation commits O(changed regions), not O(all resident payloads).

### P3 — backend composition and interaction ownership

* [ ] Create one shared `ImageViewShell` containing a PyQtGraph or VisPy surface.
* [ ] Remove `VisPyImageView2D(ImageView2D)` after semantic and manual parity tests pass.
* [ ] Move pointer capture, drag lifecycle, hover, and cursor intent fully into the shared interaction
  controller.
* [ ] Run one backend conformance suite plus real Qt/Wayland/OpenGL interaction tests.

### P4 — adaptive residency and prediction

* [ ] Record CPU preparation, upload bytes/time, queue delay, frame age, and cancellation cost by
  backend and payload kind.
* [ ] Adapt worker, item, byte, and interval budgets independently.
* [ ] Add value-scored directional slice, viewport-ring, hovered, and selected-object prediction.
* [ ] Add device-budgeted multi-page and compatible multi-resolution residency.
