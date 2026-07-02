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

**Status:** Ready after Y1 (independent of it in code, ordered by risk).

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

**Status:** Ready.

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

## Next — hardware evidence

### X5. Hardware evidence and residency policy

**Status:** Gated behind Y1–Y3. This is the evidence and residency gate for tiled surfaces, not a
general performance bucket. Note: VisPy under Xvfb/software GL is intermittently unstable; headless
GL runs are not evidence for or against the VisPy backend — only real-hardware traces count here.

**Goal:** base GPU, backend-default, viewport-residency, and multi-resolution decisions on real device
behavior.

Work:

- Record queried texture/format limits and proven allocation outcomes.
- Separate estimated GPU residency from CPU caches and track eviction/reupload.
- Treat committed tiled-scene residency as backend-acknowledged state only; requested upserts are not
  resident until accepted by the backend.
- Add conformance coverage for partially accepted, deferred, rejected, evicted, and context-lost tiled
  commits so `DisplayScene.resident_region_ids` follows acknowledged payloads.
- Change viewport retarget scheduling from montage-mode based to tiled-scene based before enabling
  visible-only active regions for internally tiled normal images.
- Introduce region-first display materialization so huge single-plane tiling can read and prepare visible
  regions without requiring a full display image first.
- Benchmark huge normal-plane first frame, pan into cold tiles, pan across warm/resident tiles, level-only
  changes, backend reset/context loss, and allocation fallback on both PyQtGraph and VisPy paths.
- Build Linux X11/Wayland, Windows, and macOS reference traces on integrated and discrete GPUs.
- Decide whether/where VisPy becomes default from measured latency, stability, memory, and parity—not
  theoretical throughput.
- Implement asynchronous/source-provided LOD materialization only after the acknowledged-residency,
  viewport-retarget, and region-first materialization contracts are proven.
- Use separate compatible LOD pages/arrays or virtual textures; retain adjacent levels during
  transitions when budget allows.

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
- context loss and allocation failure recover without semantic corruption;
- repeated zoom threshold crossings do not rebuild/re-upload the full active set;
- exact inspection values remain independent of display LOD;
- backend-default and LOD-enable decisions have documented evidence.

See [ADR 0044](decisions/0044-viewport-scoped-tiled-residency.md).

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
- Re-introducing the removed refuse/degraded/chunked normal-path render decisions or the bespoke
  idle stage-warmup scheduler (superseded by tile budgets and `WorkGraph` lanes).
- Another large renderer rewrite without incremental conformance tests and traces.
