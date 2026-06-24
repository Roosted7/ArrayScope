# Roadmap

This roadmap is ordered by risk reduction, not by feature excitement. Historical phase checklists are
archived at [`archive/roadmaps/phase-roadmap-through-v28.md`](archive/roadmaps/phase-roadmap-through-v28.md).

A roadmap item is complete only when its exit gate is met. “Code exists” is not completion.

## Now — stop semantic drift in the v30 rendering control plane

### N5. Backend-aware presentation convergence

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
- Add state-machine/property tests before moving callers.

Exit gate:

- `MontageRenderSession` no longer owns backend-specific level convergence or queue policy;
- each extracted model is Qt-free and property-tested for supersession, partial acknowledgement,
  active-set changes, cancellation, and bounded progress;
- no timer establishes semantic order without a target/revision guard;
- PyQtGraph and VisPy run the same semantic generation suite through separate strategy fixtures;
- representative callback work stays within configured limits or explicitly reschedules.

### N7. Make native-only LOD an explicit, tested production policy

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

## Next — converge the architecture after the control plane is stable

### X1. Unified frame planner and tiled image surface

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

### X2. Backend composition

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

### X3. Shared pointer capture and drag lifecycle

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

### X4. Hardware evidence and residency policy

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
