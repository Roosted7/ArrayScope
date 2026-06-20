# 0039 — Unify image presentation and schedule work by frame value

## Status

Proposed; accepted as the target direction for the post-v27 rendering work.

## Context

ArrayScope now has useful backend-neutral presentation models, committed semantic value sources,
backend capabilities, PyQtGraph and VisPy adapters, a staged evaluator, tiled deltas, cache policy,
and a resource governor. Those pieces should be retained.

The remaining control flow is still split by normal image versus montage and by legacy widget
behavior. High-frequency requests are handled primarily with debounce, latest-only cancellation, and
quiet-period refreshes. That can reduce duplicate work, but it also creates three failure modes:

1. continuous interaction can cancel every exact single-image render before completion;
2. one progressive callback can still contain unbounded per-item work unless every consumer enforces a
   real item/byte/time budget;
3. low-priority analysis and prediction are delayed globally rather than admitted according to
   measured spare resources and expected latency value.

The VisPy widget also remains a subclass of the complete PyQtGraph widget. It preserves feature parity
but keeps two scene/event systems active and makes backend lifecycle, pointer ownership, camera state,
and overlays difficult to reason about.

## Decision

ArrayScope will use one semantic image-presentation pipeline and one deadline-aware work scheduler.
Raster and tiled/virtual storage remain interchangeable physical strategies selected by a planner.
They are not separate semantic render pipelines.

The target flow is:

```text
ViewIntent
  -> FramePlanner
  -> FramePlan / WorkGraph
  -> DeadlineScheduler
  -> PresentationCommit
  -> ImageSurface
       -> raster strategy
       -> tiled / virtual strategy
```

### Semantic frame identity

The controller owns three explicit identities:

```python
@dataclass(frozen=True)
class FrameTarget:
    semantic_key: object
    viewport_key: object
    presentation_key: object
    quality: "QualityTier"
    deadline_ns: int

@dataclass
class FrameProgress:
    presented: FrameTarget | None
    active: FrameTarget | None
    queued_latest: FrameTarget | None
```

- `semantic_key` identifies source data and operation state.
- `viewport_key` identifies slice, range, camera, and visible regions.
- `presentation_key` identifies levels, LUT, component, and display uniforms.
- quality is explicit: retained, coarse, exact-visible, exact-complete.

Presentation-only changes must not invalidate materialized pixels. Camera-only changes must not
restart evaluation. A new target replaces the queued target, not automatically the active work.

### Work graph and lanes

A frame plan expands into independently cancellable work units:

```python
@dataclass(frozen=True)
class WorkItem:
    key: object
    lane: "WorkLane"
    deadline_ns: int
    estimated_cpu_ms: float
    estimated_gpu_bytes: int
    dependency_keys: tuple[object, ...]
    supersession_key: object
    quality_gain: float
```

Required lanes are:

- visible planning and cache lookup;
- visible materialization;
- display preparation;
- GPU/Qt commit;
- histogram/level refinement;
- ROI/profile/pixel analysis;
- stage materialization;
- speculative prefetch/residency.

Cancellation is by supersession key and value, not by a global render generation alone. Visible
exact work supersedes an older queued target, but an already-running item may finish when its
remaining cost is lower than restart cost or when it produces reusable cache data.

### Admission score

After hard visible deadlines, optional work is admitted using an explicit value score:

```text
score = probability_of_use * expected_latency_saved * quality_gain
        / max(estimated_cost, epsilon)
```

Predictions use recent direction and velocity, selected/hovered entities, and stage availability.
Speculative work is suspended immediately when event-loop delay, memory pressure, or visible backlog
rises. It must never be allowed merely because a debounce timer expired.

### GUI-thread performance contract

Every path that mutates Qt or OpenGL state must obey all of these rules:

- no loop over an unbounded user/data-sized collection in one callback;
- interactive callbacks target at most 4 ms;
- idle presentation callbacks target at most 8 ms;
- 16 ms is a hard warning threshold, not a normal batch target;
- batches are bounded by items, bytes, and elapsed time;
- callbacks publish partial progress and reschedule remaining work;
- semantic histogram refinement never gates first pixel presentation;
- the last valid frame remains visible until a replacement is usable.

### One semantic surface, multiple storage strategies

A normal image is a one-region semantic scene. A montage is a multi-region semantic scene. Either may
use raster or tiled storage.

The storage planner chooses among:

- one texture / one `ImageItem` for a small, stable plane;
- partitioned tiles for a very large single plane;
- tiled montage storage for many semantic regions;
- explicit multi-resolution pages when source levels exist or can be built off-thread.

The choice considers dimensions, texture limits, estimated bytes, update frequency, viewport size,
backend capabilities, and available residency. A one-tile montage and a normal image must share level,
value, scheduling, cache, and interaction semantics even when their physical allocation differs.

`DisplayTiledPresentation` must therefore cease requiring montage geometry. Internal tile geometry for
a large single plane should be represented independently of montage-axis semantics.

### Backend composition

The destination widget shape is:

```text
ImageViewShell
  - histogram and display controls
  - HUD/status/ROI information
  - shared interaction controller
  - viewport controller
  - ImageSurface
      - PyQtGraphSurface
      - VisPySurface
```

`VisPyImageView2D(ImageView2D)` is a migration scaffold and will be removed. The shell owns semantic
signals and pointer policy. A surface owns concrete scene objects, textures, buffers, and drawing.
Neither backend may own the meaning of a level source, ROI hit target, or committed value.

## Feedback control

The current latency controller is retained but observations become typed and backend/path-specific.
At minimum, it must learn:

- callback milliseconds and item count;
- CPU preparation milliseconds per item;
- upload milliseconds per byte;
- queue delay and presented-frame age;
- cancellation milliseconds and reusable work produced;
- backend, payload kind, dtype/channel mode, and interaction state.

The controller independently adjusts:

- result callback count;
- presentation upsert count;
- upload byte budget;
- commit interval;
- worker count;
- speculative admission.

Cold start uses conservative batches and multiplicative growth. Recovery is gradual; overload backoff
is immediate. A complete callback must never be reported as one item when it processed many tiles.

## ROI, profile, histogram, and hover

Side analyses are not globally deferred until rendering is “finished.” They issue promptly at a lower
priority and can publish bounded partial results:

```text
interaction update
  -> cheap committed-frame read immediately
  -> coarse ROI/profile estimate if available
  -> exact background refinement
  -> final result guarded by semantic target identity
```

Their admission depends on visible deadline slack and lane-specific cost. Hidden panels do no work.
Selected or hovered entities receive higher prediction probability than unrelated panels.

## GPU memory and LOD

GPU residency and CPU semantic caches are separate budgets. GPU allocation is based on queried device
limits, proven allocation results, configured policy, and current pressure—not a fixed assumed texture
size.

Native-resolution tiles remain the production baseline. Multi-resolution rendering uses explicit
compatible storage classes: separate pages per LOD/tile shape, texture arrays grouped by dimensions,
or a virtual texture/page table. Arbitrarily sized LOD images must not share fixed atlas slots whose
sampling geometry assumes one tile shape.

## Migration plan

1. Enforce the GUI callback budget everywhere and add request-to-first-frame/event-loop benchmarks.
2. Introduce `FrameTarget`, presented/active/latest state, and an active-plus-latest visible queue.
3. Move normal image and montage planning behind one `FramePlanner` and storage-strategy decision.
4. Generalize tiled geometry so a huge single plane can use internal tiling.
5. Move pointer capture, drag lifecycle, hover, and cursor intent fully into the shared interaction
   controller.
6. Replace backend inheritance with `ImageViewShell` plus `ImageSurface` composition.
7. Add device-budgeted multi-page/multi-resolution residency after the scheduler and metrics are
   stable.

Each step must preserve a runnable backend and land with semantic conformance tests. Compatibility
shims may remain during migration but cannot acquire new behavior.

## Consequences

Positive:

- one set of semantics for 1, 2, or 500 regions;
- no cancellation starvation during sustained input;
- predictable GUI-thread latency independent of source size;
- useful idle compute without competition with visible work;
- backend comparisons based on actual frame delivery;
- large single images can gain tiled residency without becoming fake montages;
- PyQtGraph remains a reliable fallback while VisPy can exploit shaders and persistent GPU storage.

Costs and risks:

- the scheduler and widget shell are substantial changes;
- some legacy tests currently assert cancellation behavior that must be replaced with frame-progress
  invariants;
- retaining active work requires cost/progress estimates and careful stale-result guards;
- backend composition must be tested on real Qt/Wayland/OpenGL systems, not only pure models.

## Rejected alternatives

- **One monolithic tiled implementation for every image.** Small images may be faster and simpler as a
  single texture; semantics should be unified, not forced allocation.
- **Keep debounce/latest-only as the primary scheduler.** It can reduce duplicate work but cannot
  guarantee progress under continuous input.
- **Duplicate normal and montage schedulers per backend.** That would multiply the current drift and
  make performance behavior backend-dependent.
- **Remove PyQtGraph now.** VisPy still depends on the hybrid shell and lacks sufficient production
  validation.
- **Abandon VisPy now.** Shader mapping and persistent GPU residency remain valuable once orchestration
  and surface ownership are clean.
