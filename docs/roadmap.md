# Roadmap

This roadmap is ordered by risk reduction, not by feature excitement. A roadmap
item is complete only when its exit gate is met. "Code exists" is not
completion.

Completed gates N4–N7 (v30 control-plane stabilization) and X1–X4 (frame
planner, work graph, backend composition, shared pointer capture) are archived
with their exit criteria and completion notes in
[`archive/roadmaps/completed-gates-n4-x4.md`](archive/roadmaps/completed-gates-n4-x4.md).
The v32 composition change (render orchestration off the window) is recorded in
[ADR 0045](decisions/0045-render-orchestrator-composition.md) and the
[v32 audit](reviews/v32-composition-audit.md).

## Now — finish ownership after the v32 composition change

### Y1. One generation contract and one admission path

**Status:** Done (2026-07-02). `window/render_contract.py` owns the staleness
vocabulary (render generation, session currency, per-kind work tokens);
orchestrator predicates delegate to it and architecture guards forbid local
reimplementations and context-free `singleShot` callbacks. The orchestrator
exposes `work_graph`, fixing silently dropped admission records after the v32
extraction; montage tile prefetch, level-evidence batches, and
viewport/priority retargets now record admissions. The redundant
prefetch-dispatch and frame-viewport-update revision counters were deleted
(coalescing is structural: one queued flag / one restarted timer).

**Goal:** a single staleness/ordering vocabulary inside `RenderOrchestrator`
so that no fix ever has to choose between parallel token schemes.

Work:

- Inventory the remaining revision counters, session keys, and staleness
  predicates in the orchestrator (the v32 audit counted 8/3/5 before the
  extraction; they now live in one namespace).
- Define one `RenderGeneration`-anchored contract: `(document_key,
  semantic_key, render_generation)` with derived session/level/histogram
  revisions, owned by the orchestrator.
- Replace per-site staleness checks with the shared predicate; delete the
  duplicates.
- Route the remaining ad-hoc work admission (montage tile scheduling and
  result fan-in paths that bypass the graph) through `WorkGraph` lanes.
- Keep timers as pure rescheduling; every deferred callback carries the
  generation guard and a receiver context (the ADR 0045 pattern).

Exit gate:

- one module defines staleness; grep finds no local reimplementation;
- every admission decision is observable in `WorkGraph` counters;
- the montage timer/token tests pass unchanged or are strengthened;
- no deferred callback can outlive the window or apply across generations.

### Y2. Backend de-duplication against the surface contract

**Status:** Done (2026-07-02). Measured reality differed from the audit
estimate: `ImageView2D` is an empty subclass (the shell *is* the PyQtGraph
implementation) and the two `tiles.py` files share zero functions — their
divergence is physical (CPU items vs. GPU atlas), with semantic tile
bookkeeping already owned above them by `montage_session`, `frame_planner`,
and `montage_viewport`. What was unified: the shared semantic drivers now
live once in `ImageViewShell` behind small backend hooks
(`_apply_preview_levels_to_display`, `_after_viewport_camera_change`,
`_after_profile_marker_sync`, `_viewport_content_shape`), eleven VisPy
overrides were deleted, the tile-layer stats contract moved to
`display/model/tile_stats.py` (deleting `GpuMontageLayerStats`, its
conversion layer, and the vispy→pyqtgraph import), and
`tests/display/test_imagesurface_contract.py` runs the surface contract on
both backends. Two real forks were found and fixed by those tests: VisPy's
close path had lost `_cancel_interaction`, and VisPy hid tiled presentation
into a private `"idle"` mode instead of the shared `"none"`.

**Goal:** one implementation of everything that is not texture/atlas vs.
QGraphicsItem mechanics.

Work:

- Hoist the ~40 methods implemented in both `ImageView2D` and
  `VisPyImageView2D` into `ImageViewShell` (audit: ~1,200 duplicated lines,
  ~465 directly hoistable).
- Extract the shared Qt-free tile bookkeeping from
  `display/backends/pyqtgraph/tiles.py` and `display/backends/vispy/tiles.py`
  into a common model; keep upload/visual mechanics per backend.
- Share the identical histogram preview handling.
- Feature-parity tests target the `ImageSurface` contract, not widget classes.

Exit gate:

- a behavior fix in shared shell logic cannot be applied to one backend only;
- the two `tiles.py` files contain only physical mechanics;
- display-tree tests pass on both backends with no per-backend semantic forks.

### Y3. Declarative UI sync, tools on production composition, one cache core

**Status:** Done (2026-07-02).

- `ui/state_binding.py` (`ViewStateBinder`) owns ViewState→widget mirroring:
  each control registers one binding where it is created, applies run with
  signals blocked and only on value change, and the sync entry points in
  `state_sync.py` are thin delegates (guarded by an architecture test).
  Widget-side drift recovery goes through `_reset_controls_to_view_state()`.
- The profiling tools already drive the production `ArrayScopeWindow`
  composition (`profile_montage_workflow.py` and `profile_scroll_input.py`
  construct the real window and call `render()`); the audit's
  re-implementation claim was resolved before this gate, so the remaining
  2,000 lines are scenario/measurement code, which is what "thin scripts over
  production wiring" means here.
- `core/bounded_cache.py` (`BoundedCache`) is the one eviction/priority
  implementation: byte/entry budgets, LRU order, and a pluggable
  `retention_key`. `BoundedArrayCache` (plain LRU), `StageCache`
  (priority-weighted retention score), and `RetainedTiledPayloadStore`
  (entry-bounded payload reuse) all build on it, with focused tests in
  `tests/core/test_bounded_cache.py` and a guard forbidding hand-rolled
  eviction loops.
- Idle stage warmup stays removed; if it returns it must be admitted through
  the `WorkGraph` speculative-residency lane (unchanged policy).

**Goal:** remove the remaining drift machines.

Work:

- Replace the 17 manual `_sync_*` fan-outs with one binder that observes
  `ViewState` revisions and updates registered widgets; controls emit intent
  only.
- Make `tools/profile_montage_workflow.py` and the scroll profiler drive the
  real `ArrayScopeWindow` composition instead of re-implementing it; profiling
  scenarios become thin scripts over production wiring.
- Unify the three cache eviction/priority implementations (stage cache, slab
  cache, display/payload caches) behind one bounded-cache core in `core/`.
- If idle stage warmup is wanted again, admit it through the `WorkGraph`
  speculative-residency lane; the removed bespoke scheduler is not the model.

Exit gate:

- adding a control requires registering it once, not editing sync methods;
- profiler output is produced by production composition;
- one eviction/priority implementation with focused tests.

## Now — evidence-first performance gates

### X5. Hardware evidence and residency policy

**Status:** Active after Y1–Y3; refined by
[ADR 0046](decisions/0046-evidence-first-performance-strategy.md). This is the evidence and residency
gate for tiled surfaces and physical strategy selection, not a general performance bucket. VisPy under
Xvfb/software GL is intermittently unstable; headless GL runs are not evidence for or against the VisPy
backend — only real-hardware traces count here.

**X5a done for Linux (2026-07-03):** the first real-hardware pass ran the presented-frame
micro/stress matrix, the production-window montage workflow, and 60 Hz scroll interaction on a live
Wayland session across Intel iGPU / NVIDIA dGPU and Wayland / XWayland
([reference traces](reviews/x5a-hardware-telemetry-linux-wayland.md)). It fixed the broken GPU
device-limit query (every record silently reported a 4096 fallback), two O(n²) VisPy commit costs
(per-commit histogram concatenation, resident-key recomputation; 272-tile stress submit
1180 ms → ~130 ms), and the PyQtGraph level-refinement starvation on large montages (a 272-tile
level drag never converged; now ~4.3 s). [ADR 0047](decisions/0047-auto-image-backend-selection.md)
adds the resulting `auto` backend choice: VisPy on Linux with hardware GL, PyQtGraph everywhere
else. **X5b done for montage tiled scenes (2026-07-05)** via
[ADR 0051](decisions/0051-single-owner-tile-lifecycle.md): presentation state is a machine whose
only path to `presented` is a backend-acknowledged commit, and acknowledgement is identity-aware
(backend slot identities vs. emitted payload identities, causally bound reports). Windows/macOS
traces and X5c–X5e remain open.

**Goal:** base GPU, backend-default, singleton/direct fast-path, viewport-residency, and
multi-resolution decisions on real device behavior.

Ordered gates:

1. **X5a — Telemetry baseline.** Record queried texture/format limits, proven allocation outcomes,
   upload timings, accepted/rejected tile counts, event-loop gaps, RSS, and context-loss/fallback
   behavior.
2. **X5b — Acknowledged residency (done for montage, 2026-07-05).** Treat committed tiled-scene
   residency as backend-acknowledged state only; requested upserts are not resident until accepted
   by the backend. Delivered by ADR 0051 P1+P2 for montage tiled scenes: identity-aware,
   causally-bound acknowledgement is the machine invariant, with conformance coverage for partial
   acceptance, declines, parking, stale reports, and session replacement. Normal-image tiled
   scenes inherit this when X5c routes them through the same machine. Field verification
   passed and machine-derived dispatch landed (2026-07-05): every event edge re-derives all
   pumps from the records, declined admissions arm capacity waiters, and the stall watchdog
   is an assertion. Remaining P2 work (legacy sets as views + stage fan-in events,
   delta-commit walk cost) is tracked in ADR 0051's phases.
3. **X5c — Viewport-scoped tiled scenes.** Change viewport retarget scheduling from montage-mode
   checks to tiled-scene/storage checks before enabling visible-only active regions for internally
   tiled normal images.
4. **X5d — Region-first materialization and physical strategy policy.** Introduce region-first display
   materialization so huge single-plane tiling can read and prepare visible regions without requiring
   a full display image first. Add a measured physical strategy policy below `ImageSurface`: small or
   one-region frames may use a singleton/direct surface when measured faster, while large planes and
   montages use resident or virtual tiled storage. This must not restore the old separate normal-image
   semantic path.
5. **X5e — Backend and LOD decisions.** Benchmark huge normal-plane first frame, pan into cold tiles,
   pan across warm/resident tiles, level-only changes, backend reset/context loss, and allocation
   fallback on both PyQtGraph and VisPy paths. Build Linux X11/Wayland, Windows, and macOS reference
   traces on integrated and discrete GPUs. Decide whether/where VisPy becomes default from measured
   latency, stability, memory, and parity — not theoretical throughput. Montage/tiled scenes
   already run resident asynchronous LOD (ADR 0050); this gate governs internally tiled normal
   images and source-provided pyramids, which land only after the acknowledged-residency,
   viewport-retarget, region-first materialization, and compatible-residency contracts are proven.

Policy constraints:

- Separate estimated GPU residency from CPU caches and track eviction/reupload.
- Keep exact inspection values independent of display LOD.
- Use separate compatible LOD pages/arrays or virtual textures; retain adjacent levels during
  transitions when budget allows.
- Warm/speculative residency must be queue-based, bounded before admission, and superseded by newer
  visible work. Do not copy the remaining payload map on every timer tick.
- Do not make every small image pay atlas or quad overhead just because its semantic presentation is
  tiled; semantic unification and physical storage are different decisions.

Exit gate:

- published benchmark matrix includes request-to-first-visible-tile, request-to-settled-frame,
  event-loop gap, RSS, residency, upload counters, accepted/rejected tile counts, and LOD transition
  traces;
- no fixed assumed max texture size drives policy;
- committed scene residency always reflects backend acknowledgement, including partial acceptance and
  context-loss recovery;
- internally tiled normal images can pan/zoom through viewport-scoped active and near regions without
  blanking, full-frame rematerialization, or montage-specific assumptions;
- region-first materialization works for eager array-backed sources and has a clear extension point for
  memory-mapped/chunked sources;
- singleton/direct and tiled physical strategies are chosen by measured capability and latency, while
  sharing the same semantic frame/value/interaction contracts;
- context loss and allocation failure recover without semantic corruption;
- repeated zoom threshold crossings do not rebuild/re-upload the full active set;
- exact inspection values remain independent of display LOD;
- backend-default and LOD-enable decisions have documented evidence.

See [ADR 0044](decisions/0044-viewport-scoped-tiled-residency.md) and
[ADR 0046](decisions/0046-evidence-first-performance-strategy.md).

## Later — product capabilities that fit the mission

These are candidates after the foundation gates, not parallel commitments.

### Linked windows and inspection groups

**Status:** First iteration shipped (2026-07-03), [ADR 0048](decisions/0048-linked-window-sync.md).
Per-facet sync toggles exist for window/level (display toolbar), dimension indexing (dimension
strip), operation recipes (operations dock), and ROIs (inspection dock). Sync works across
separately started processes on Linux, macOS, and Windows through per-user Qt local sockets with
broker-relay topology and re-election (`arrayscope/sync/`). Typed JSON envelopes carry
origin/revision ids for feedback-loop suppression; dimension indices clamp per axis on shape
mismatch; incompatible operation recipes are skipped with a status toast. Remaining from the
original idea: cursor links, viewport links, and named groups beyond the default one (the
envelope already carries a group field).

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

Progress (2026-07-03): `AxisInfo` carries optional unit/spacing/origin with conservative operation
propagation, loaders that know axis metadata provide it (NIfTI, DICOM, Philips REC), and the
dimension strip surfaces labels plus a metadata tooltip. See
[the AxisInfo proposal](proposals/axis-info.md) for what remains (coordinate arrays, orientation,
physical cursor readout, session role matching).

### Out-of-core and lazy sources

**Status:** First slice done (2026-07-03), [ADR 0049](decisions/0049-out-of-core-lazy-sources.md).
`core/array_source.py` defines the source protocol (`ArraySource`, `LazySourceArray`);
`operations/source_read.read_base_region` is the single budgeted, cancellable base-data read seam
under slab/stage evaluation; `io/lazy_sources.py` provides memory-mapped `.npy`/`.cfl` adapters and
`load_path(lazy="auto")` opens large supported files lazily. Remaining: a chunked (Zarr/HDF5-like)
adapter behind the same protocol, lazy dataset selectors, chunk-aligned planning hints, and UI
surfacing of budget refusals.

Add a source protocol for memory-mapped/chunked arrays and explicit region reads. Keep request planning,
cancellation, and memory budgets above the source adapter so “lazy” does not mean unbounded transport
or decoding.

### Invocation adapters

Improve Jupyter and editor launch routes only when they call one stable semantic API. Avoid duplicating
a frontend/state machine per host.

Julia and MATLAB launch wrappers exist under `wrappers/` (2026-07-03) and follow this rule: both are
thin adapters that write a raw `.npy` handoff and invoke the CLI (`--mmap --consume`), re-implementing
no viewer behavior. The handoff contract is documented in [`invocation.md`](invocation.md) and pinned
by `tests/io/test_language_handoff.py`.

## Explicitly not now

- General plugin marketplace/layer ecosystem.
- Broad segmentation, registration, qMRI, or vector-field workbench.
- Remote multi-user server/collaboration architecture.
- Destructive workspace-style operations.
- Re-enabling the old synchronous LOD pyramid path.
- Re-introducing the removed refuse/degraded/chunked normal-path render decisions or the bespoke
  idle stage-warmup scheduler (superseded by tile budgets and `WorkGraph` lanes).
- Another large renderer rewrite without incremental conformance tests and traces.
