# Roadmap

This roadmap is ordered by risk reduction, not by feature excitement. Historical phase checklists are
archived at [`archive/roadmaps/phase-roadmap-through-v28.md`](archive/roadmaps/phase-roadmap-through-v28.md).

A roadmap item is complete only when its exit gate is met. “Code exists” is not completion.

## Now — stop semantic drift in the v30 rendering control plane

### N5. Backend-aware presentation convergence

**Status:** Done!

**Goal:** every global presentation command reaches a deterministic, observable latest target without
pretending that PyQtGraph and VisPy have the same physical cost.

Work:

- Preserve one level target/revision/source contract across both backends.
- Keep PyQtGraph CPU/item redraws in the existing prioritized, item/byte/time-bounded tile queue.
- Keep compatible VisPy changes as shader/uniform commits with zero source-pixel upload.
- Distinguish retained visibility from accepted upserts and current-revision acknowledgement.
- Keep automatic, restored, and explicit-user level sources separate from the current physical target.
- Define completion as current active-tile convergence, not “all tiles are visible.”
- Add transition matrices for rapid target supersession, auto-window, revert, partial montage,
  viewport changes during convergence, and tiles entering/leaving the active set.

Exit gate:

- one-tile-per-callback PyQtGraph tests eventually update every active tile exactly to the latest
  target;
- a deferred but still-visible upsert remains pending;
- older auto/restore work cannot overwrite a newer command;
- VisPy level-only tests report uniform work and zero texture upload;
- benchmark records include revision, stale count, pending count, settled flag, and backend-specific
  physical work;
- real PyQtGraph and VisPy manual traces agree with the automated semantic result.

See [ADR 0040](decisions/0040-backend-aware-presentation-convergence.md).

### N4. Histogram and level refinement discipline

**Status:** Done!

**Goal:** preserve responsive level control while maintaining semantic correctness.

Work:

- Measure adaptive histogram sampling/binning and manual edit paths on large/complex data.
- Keep automatic level bounds available immediately from semantic summaries while detailed plot
  refinement runs latest-only off-thread when it exceeds budget.
- Keep level coverage separate from detailed plot completion.
- Split histogram numerical/refinement ownership from PyQtGraph widget binding and manual editor UI.
- Replace or isolate private `HistogramLUTItem` rebinding (`imageItem` weakrefs and
  `_setImageLookupTable`) behind one compatibility adapter with version tests.
- Verify user-lock, auto-window, restore, partial montage, close/teardown, and backend parity
  transitions.

Exit gate:

- level drag/manual edit previews remain responsive on reference datasets;
- no deleted-widget callback or stale refinement result can apply;
- repeated zoom/range requests leave at most one latest histogram refinement per view;
- automatic levels never wait for detailed binning and never use incomplete semantic coverage for
  displayed pixels;
- PyQtGraph and VisPy conformance tests agree on level/value semantics;
- supported PyQtGraph versions pass explicit histogram-binding compatibility tests.

### N6. Extract the rendering control-plane state machines

**Status:** Done!

**Goal:** make local rendering fixes local again.

`MontageRenderSession` and `montage_renderer.py` currently combine materialization, stage waits,
payload admission, level convergence, viewport/residency hints, acknowledgement, timers, and
committed-frame publication. That concentration is now a correctness risk, not merely a file-size
preference.

Work:

- Extract a Qt-free `PresentationGenerationTracker` for target/revision/active coverage/pending/
  acknowledgement.
- Extract a `TileAdmissionQueue` that owns priority, aging, item/byte/deadline caps, and no semantic
  meaning.
- Define `LevelConvergenceStrategy` implementations for progressive PyQtGraph and uniform VisPy.
- Extract stage-wait/result fan-in batching from the session.
- Replace timer-implied ordering with explicit target/revision guards and resubmission reasons.
- Keep compatibility shims behavior-free and remove dead duplicate acknowledgement concepts.
- Remove legacy aliases after each extraction so tests, diagnostics, and profilers use the canonical
  owner directly instead of preserving both the old and new APIs.
- Add state-machine/property tests before moving callers.

Exit gate:

- `MontageRenderSession` no longer owns backend-specific level convergence or queue policy;
- each extracted model is Qt-free and property-tested for supersession, partial acknowledgement,
  active-set changes, cancellation, and bounded progress;
- no timer establishes semantic order without a target/revision guard;
- PyQtGraph and VisPy run the same semantic generation suite through separate strategy fixtures;
- representative callback work stays within configured limits or explicitly reschedules.

Completion notes:

- `PresentationGenerationTracker`, `TileAdmissionQueue`, `LevelConvergenceStrategy`, and
  `StageFanInState` are Qt-free and covered by focused state-machine/property tests.
- `MontageRenderSession` delegates level convergence, tile admission, and stage fan-in to those models
  while retaining montage identity, materialized tile state, payload cache, and canvas patch state.
- Legacy session aliases for extracted generation and fan-in state were removed; call sites now use
  `level_generation` and `stage_fan_in` directly.
- Montage timers carry explicit session/revision work tokens for commit, result fan-in, stage wait,
  and priority-retarget callbacks.
- PyQtGraph and VisPy retain backend-specific physical mechanics while sharing the semantic generation
  assertions.

### N7. Make native-only LOD an explicit, tested production policy

**Status:** Done!

**Goal:** eliminate misleading “selected but not applied” behavior while preparing a safe
multi-resolution implementation.

Work:

- Continue reporting desired factor, applied factor, policy, and reason separately.
- Test selection/hysteresis independently of materialization.
- Record source-texel-per-pixel values per axis and identify anisotropic/extreme-aspect cases.
- Prevent any synchronous pyramid construction from re-entering presentation commits.
- Design cache keys and storage classes for level/tile-shape/format/gutter compatibility.
- Define retained native/adjacent-level transition behavior and exact semantic-value sources.

Exit gate for this item (not for enabling LOD):

- diagnostics never imply factor >1 was presented when applied factor is 1;
- native-only policy is covered in runtime, benchmark, and session tests;
- the old synchronous path has no callable production entry;
- an implementation plan and benchmark matrix satisfy ADR 0041 before multi-resolution coding starts.

See [ADR 0041](decisions/0041-lod-selection-materialization-and-residency.md).

Completion notes:

- `arrayscope.display.lod` now exposes Qt-free LOD demand and native-only policy decisions with
  desired/applied factors, per-axis source texels, policy, and reason.
- `MontageRenderSession` stores one canonical `lod_policy_decision`; payloads remain native
  `LodInfo(level=0, factor=1, gutter=0)` until async compatible residency exists.
- The synchronous CPU pyramid/gutter construction entrypoints were removed from production code and
  guarded by architecture tests.
- Runtime diagnostics, profile JSONL, benchmark records, session tests, and LOD model tests cover the
  native-only contract.
- The future multi-resolution implementation and benchmark matrix are recorded in
  [the LOD proposal](proposals/lod-multires-implementation-plan.md).

## Next — converge the architecture after the control plane is stable

### X1. Unified frame planner and tiled image surface

**Status:** Done!

**Goal:** normal images and montages become one semantic presentation pipeline.

Work:

- Introduce explicit `FrameTarget`/quality and a unified region/tile model.
- Move normal and montage planning behind one `FramePlanner`.
- Represent single images, large planes, and montages as tile regions in the same semantic pipeline.
- Optimize one-tile and small-tile cases inside the tiled engine.
- Make one-tile montage and normal image share level/value/cache/scheduling tests.
- Generalize `DisplayTiledPresentation` so montage geometry is optional.
- Reuse the N6 generation/admission state machines rather than building a second scheduler.

Exit gate:

- no semantic branch depends on “normal versus montage”;
- a huge single plane can use internal tiling;
- conformance tests pass across one-tile, small-tile, large-tile, and montage cases on both backends;
- existing public interactions remain available throughout migration.

Completion notes:

- `FramePlanner` plans normal images, internally tiled large single planes, and montages as one
  semantic `FramePlan`/`FrameRegion` model with cached active/planned/near region IDs.
- `DisplayTiledPresentation` and both backend adapters accept montage-optional tiled geometry. Real
  PyQtGraph item tests and VisPy atlas/quad tests cover non-montage tiled single-plane commits.
- Large normal frames can commit through the same typed tiled surface as montage tiles. The committed
  frame owns tiled value semantics, so hover/value/ROI reads come from payloads rather than a
  placeholder canvas.
- Montage viewport retargets recompute the frame plan with the new active/near set, and scene
  conversion uses the current tile delta so stale frame-plan activity cannot leak into committed
  semantics.
- Montage resize/layout reflow now follows ADR 0042: manual resize preserves screen zoom in the
  viewport controller, same-source layout changes translate by source-local focus without another
  zoom change, and ROI selections remap through canonical source-local geometry.
- Shared tile-layout helpers keep backend placement logic out of semantic code and avoid scanning the
  complete montage population for active-payload quad generation.
- Rendering benchmarks now include a real `normal_large_tiled_initial` commit on both backends and
  assert tiled-surface work counters instead of only proving that a plan was produced.

### X2. Deadline work graph and visible admission

**Status:** Done!

**Goal:** replace debounce/timer-shaped render ordering with explicit frame-value work admission.

X1 unified the semantic frame surface. X2 implements ADR 0039's `WorkGraph`/deadline-admission half
as its own gate rather than hiding it inside backend composition or hardware benchmarking.

Work:

- Introduce a Qt-free `WorkGraph` for visible planning/cache lookup, materialization, display
  preparation, backend commit, histogram refinement, profile/ROI work, stage materialization, and
  speculative residency.
- Admit work by frame target, quality tier, supersession key, deadline, estimated cost, and expected
  value instead of by quiet-period timers alone.
- Keep camera-only retargeting and presentation-only edits from restarting source materialization.
- Make GUI result fan-in itself budgeted by item, byte, and elapsed time, with explicit resubmission
  reasons.
- Preserve active-plus-latest visible semantics while allowing already-running reusable work to finish
  when it is cheaper than cancellation/restart.
- Publish deterministic counters for queued, admitted, dropped, superseded, completed, and rescheduled
  work by lane.
- Keep the existing N6 `TileAdmissionQueue`, `PresentationGenerationTracker`, and `StageFanInState`
  as component models rather than replacing them with another scheduler.

Exit gate:

- continuous pan/zoom/level interaction cannot starve exact visible work indefinitely;
- worker-result bursts cannot produce an unbounded GUI callback;
- hidden panels and speculative residency admit no work without available budget/value;
- stale work is dropped before it can mutate the visible presentation set;
- request-to-first-frame, event-loop gap, and work-counter benchmarks cover normal, internally tiled
  large-plane, and montage paths;
- tests prove camera-only, presentation-only, semantic, and document-revision changes have distinct
  cancellation/materialization behavior.

Completion notes:

- `arrayscope.core.work_graph` now provides Qt-free `WorkGraph`, `WorkItem`, lane, admission, and
  counter models for visible planning/materialization, display preparation, backend commit, GUI
  fan-in, side analysis, stage materialization, and speculative residency.
- `EvaluationController` visible submissions now carry admitted work items when a window graph is
  present. Active-plus-latest behavior is preserved, queued obsolete work collapses by supersession
  key/value through a keyed queue index rather than a whole-queue scan, and stale reusable completions
  are counted without presenting stale pixels.
- Queued work re-admission does not advance supersession state, and budget-blocked queued work is
  counted by blocked state rather than by how often `admit_ready()` is polled.
- Normal rendering, montage planning, stage/tile materialization, bounded result fan-in, backend
  commits, profile/ROI work, stage warmup, and prefetch now publish lane-specific work metadata.
- Local idle, memory, dedupe, and in-flight caps run before optional work enters the graph, so rejected
  prefetch/warmup does not leave phantom active work. Retained stage warmup is optional work; exact
  visible stage materialization remains visible work.
- Runtime diagnostics and JSONL include a `Work Graph` section with queued, admitted, dropped,
  superseded, completed, failed, rescheduled, reusable-finished, deadline-missed, and budget-blocked
  counters by lane. Rendering benchmark records expose graph-derived backend-commit work counters.
- Focused model, controller, runtime-diagnostic, render-scheduler, frame-planner, pan/level, and
  presentation-only interaction tests cover the X2 contracts. Real hardware evidence is still part of
  X5, not this gate.

### X3. Backend composition

**Status:** Done!

**Goal:** replace backend inheritance with a shared shell and thin image surfaces.

Work:

- Define `ImageViewShell` ownership of controls, histogram, HUD, viewport, and semantic signals.
- Define `ImageSurface` protocol for tiled payload commit, camera, overlay state, pointer conversion,
  diagnostics, and teardown.
- Move remaining PyQtGraph/VisPy mechanics to their backend packages.
- Keep the histogram adapter and presentation strategies injected rather than inherited.
- Retire compatibility shims only after internal imports and tests use canonical paths.

Exit gate:

- `VisPyImageView2D` no longer subclasses the full PyQtGraph view;
- only one active scene/event system owns image interaction per backend;
- backend replacement/context loss has explicit lifecycle tests;
- feature-parity tests target the surface contract rather than widget class internals.

Progress notes:

- `ImageViewShell` is the canonical display shell name and owns the semantic widget state used by the
  window. Built-in views expose an `ImageSurface` contract directly through `surface`.
- `DisplayCommitter` commits raster and tiled presentations to `ImageSurface` instead of the retired
  built-in method-adapter scaffold.
- `VisPyImageView2D` now inherits `ImageViewShell`, not the PyQtGraph concrete `ImageView2D` class.
- The `ImageSurface` contract covers raster/tiled presentation, camera application, overlay
  coordinate mapping, diagnostics, context-loss reset, teardown, interaction-state visual sync, and
  declared shared interaction-event ownership.
- The current VisPy surface keeps the VisPy canvas mouse-transparent to the stacked Qt event layer
  while the shared Qt pointer driver owns ROI/profile capture and drag lifecycle for both built-in
  surfaces. Background pan/zoom uses backend-native VisPy event handling with shared
  `display.view_navigation` range math, so plain navigation avoids PyQtGraph scene drag.
- Surface lifecycle and feature-parity tests now target the contract and cover backend reset/teardown
  behavior for the built-in surfaces.

### X4. Shared pointer capture and drag lifecycle

**Status:** Done!

**Goal:** one semantic interaction controller governs both backends.

Work:

- Move pointer capture, press/move/release, handle drag, cancellation, and cursor policy out of backend
  event handlers.
- Keep backend hit primitives/coordinate conversion mechanical.
- Define deterministic priority among ROI handles, bodies, profiles, pixel hover, and camera gestures.
- Test drag interruption by mode change, frame replacement, window deactivation, and widget close.

Exit gate:

- both backends execute the same interaction state-machine tests;
- no duplicate semantic ROI/profile drag logic remains;
- pointer loss cannot leave a stuck active tool or cursor.

Progress notes:

- `DisplayInteractionController` now owns hover, pointer capture, drag update, cancellation, and cursor
  intent for the built-in PyQtGraph and VisPy surfaces.
- PyQtGraph ROI/profile graphics items are passive overlay views; VisPy mirrors shared interaction
  state for hover/capture emphasis, does not register duplicate PyQtGraph ROI/profile scene items,
  and uses native background viewport navigation for non-overlay pan/zoom.
- ROI hit candidates are indexed by display-space cells before exact semantic hit testing, so ordinary
  pointer motion does not scan every ROI or allocate full ROI snapshots.
- Hit testing uses unclamped display coordinates while active drags clamp committed image coordinates,
  so off-frame margin clicks cannot start edge-overlay drags.
- The shared Qt pointer driver cancels active capture on tool changes, frame replacement, target
  removal, window/application deactivation, button loss, and widget teardown.
- PyQtGraph and VisPy run the same viewport-event ROI/profile drag and off-frame hit-test checks.

### X5. Hardware evidence and residency policy

**Goal:** base GPU and multi-resolution decisions on real device behavior.

Work:

- Record queried texture/format limits and proven allocation outcomes.
- Separate estimated GPU residency from CPU caches and track eviction/reupload.
- Build Linux X11/Wayland, Windows, and macOS reference traces on integrated and discrete GPUs.
- Decide whether/where VisPy becomes default from measured latency, stability, memory, and parity—not
  theoretical throughput.
- Implement asynchronous/source-provided LOD materialization only after N6/N7 gates.
- Use separate compatible LOD pages/arrays or virtual textures; retain adjacent levels during
  transitions when budget allows.

Exit gate:

- published benchmark matrix includes request-to-frame, event-loop, RSS, residency, upload counters,
  and LOD transition traces;
- no fixed assumed max texture size drives policy;
- context loss and allocation failure recover without semantic corruption;
- repeated zoom threshold crossings do not rebuild/re-upload the full active set;
- exact inspection values remain independent of display LOD;
- backend-default and LOD-enable decisions have documented evidence.

## Later — product capabilities that fit the mission

These are candidates after the foundation gates, not parallel commitments.

### Linked windows and inspection groups

Adopt ArrayShow’s useful synchronized-window idea through explicit group objects and typed messages,
never a global workspace registry. Support selected dimensions, levels, cursor, ROI, or operation
recipe links independently. Prevent feedback loops with origin/revision IDs.

### Focused compare mode

Provide side-by-side or overlay comparison with shared coordinates/levels and a small set of difference
views. Keep registration/segmentation pipelines outside the core product unless they become narrow
inspection adapters.

### Rich axis metadata

Surface axis names, units, coordinates, spacing, and orientation without making every data source
conform to a medical-imaging model. Continue the `AxisInfo` proposal incrementally.

### Out-of-core and lazy sources

Add a source protocol for memory-mapped/chunked arrays and explicit region reads. Keep request planning,
cancellation, and memory budgets above the source adapter so “lazy” does not mean unbounded transport
or decoding.

### Invocation adapters

Improve Jupyter and editor launch routes only when they call one stable semantic API. Avoid duplicating
a frontend/state machine per host.

## Explicitly not now

- General plugin marketplace/layer ecosystem.
- Broad segmentation, registration, qMRI, or vector-field workbench.
- Remote multi-user server/collaboration architecture.
- Destructive workspace-style operations.
- Re-enabling the old synchronous LOD pyramid path.
- Another large renderer rewrite without incremental conformance tests and traces.
