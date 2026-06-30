# Architecture

ArrayScope separates **what the user means** from **how pixels are computed and drawn**.

The main rule is:

> Qt collects intent and presents results; Qt widgets are not the authoritative source of array-view state.

This overview defines ownership and invariants. Implementation details are split into four focused documents:

- [State and operations](architecture/state-and-operations.md)
- [Rendering](architecture/rendering.md)
- [Scheduling and memory](architecture/scheduling-and-memory.md)
- [Interaction and UI](architecture/interaction-and-ui.md)

## System map

```text
User / file / API
    |
    v
ArrayScopeWindow + focused UI controllers
    |
    +--> ViewState / ArrayDocument --------------------------+
    |                                                        |
    +--> operation planning, cache, evaluation               |
    |                                                        v
    +--> render coordination --> semantic presentation --> display commit
                                      |                         |
                                      v                         v
                               committed frame            backend adapter
                                      |                         |
                                      +--> hover/ROI/profile    +--> PyQtGraph
                                                                +--> VisPy
```

## Ownership

The authoritative models are deliberately split. `ArrayDocument` owns source
revision and operation history, `ViewState` owns the requested derived view,
operation planning owns evaluation regions, display models own presentation
meaning, and backend adapters own concrete textures/items/visuals only.

### Semantic state

- `arrayscope.core.view_state.ViewState` owns image axes, slice/range selections, line/montage roles, channel, scale, flips, FFT shifts, and display-relevant per-axis state.
- `arrayscope.operations.pipeline.ArrayDocument` owns the immutable source reference, explicit revision, and ordered operation steps.
- `arrayscope.core.axis_info.AxisInfo` and array metadata carry names/units/coordinates where available.
- `arrayscope.core.roi_store` and profile/ROI models own inspection entities independently of graphics items.

### Evaluation

- Operation classes declare shape/dtype behavior, blocking/chunkable axes, region expansion, memory/cost characteristics, and optimization eligibility.
- `operations.optimizer`, `regions`, and `planner` turn enabled steps and a request into a Qt-free execution plan.
- `OperationEvaluator` owns a unified display cache, profile/scalar cache, and reusable stage cache/materialization. Workers evaluate immutable snapshots and return results.
- `operations.slabs`, `chunked`, and `chunked_stage` execute bounded requests without moving operation semantics into UI callbacks.

### Presentation

- `display.planning` decides the semantic presentation shape.
- `display.model` holds backend-independent frame, tile, level, and commit state.
- `display.commit` applies a presentation through an adapter contract.
- `display.geometry` maps committed world/canvas/tile coordinates to array indices and profile states.
- `display.layers` owns image-view graphics-item insertion and z-order.
- Concrete PyQtGraph/VisPy modules own upload, texture, atlas, shader, visual, and scene mechanics only.
- `window.montage_viewport` owns Qt-free montage viewport reflow and source-local ROI remapping; renderers
  apply those decisions but do not redefine auto/manual semantics.

### Orchestration

- `window.render` and `window.frame_renderer` currently coordinate the visible image paths.
- `window.montage_session`, payload cache, viewport planning, tile provider, and extracted control-plane models separate parts of montage lifecycle from the main window.
- `window.evaluation_controller`, render generation, coalescing, prefetch, and stage warmup coordinate work around Qt.
- `core.memory_policy`, compute policy, latency feedback, telemetry, and resource governor decide limits/admission inputs.

### UI and interaction

- `ui.dimension_strip` and controls translate dimension intent into state changes.
- `display.interaction` and `overlay_hit_test` own backend-independent hover/target/cursor semantics where migrated.
- `display.view_navigation` owns backend-independent background pan/zoom range math; backends may
  provide native event paths that apply it to the canonical viewport range.
- `display.viewport` owns fit/preserve/reset/1:1 policy and recoverability constraints.
- Managed panels are coordinated by `window.layout_controller`; widgets do not become state authorities because they happen to be visible.
- Viewport continuity is a transaction shared by file-session restore, QSettings restore, and data
  reload. Layout consumes a saved `ViewportSession` and preserves the graphics viewport/canvas size;
  any outer-window resize is only a mechanical consequence. Viewport range is applied only after the
  first committed scene/plan; panel visibility is restored as intent for the managed-panel
  controller. See
  [ADR 0043](decisions/0043-file-session-restore-boundaries.md).

## Authoritative identities

Correct progressive behavior depends on not collapsing distinct identities. ArrayScope distinguishes changes that older viewers often collapse into one cache key.

### Document identity

Source object/revision plus operation steps. A caller that mutates an array in place must notify the document owner or replace the data so caches do not treat changed bytes as the same document.

### Semantic target identity

Which derived data and selection are requested: operation revision, axes, slice/ranges, montage indices, channel/component, and related semantic choices.

### Viewport identity

Which spatial region/camera is currently useful. Pan and zoom normally retarget visible/residency work; they do not change document semantics.

### Presentation identity

Levels, LUT, scale mapping, component, brightness/contrast, and other uniforms. Presentation-only edits must not invalidate materialized source pixels.

### Physical residency identity

The concrete tile/texture representation: source revision, semantic region, LOD/storage class, dtype/texture format, and backend context. It excludes ordinary window/level and LUT state.

## Lifecycle states

These terms are not interchangeable:

1. **Requested** — a work item exists or is queued.
2. **Materialized** — CPU/source payload is available.
3. **Resident** — a backend has usable texture/item storage.
4. **Presented** — a commit was acknowledged for the current revision.

Placeholders and dirty state clear only after presentation acknowledgement. Worker completion or queueing an upload is insufficient.

### Presentation generations

A global presentation command has one semantic target/revision, but convergence is capability-specific.
PyQtGraph may require prioritized, budgeted CPU/item redraws for every active tile; VisPy can normally
apply compatible level changes through shader uniforms. Retained visibility is not acknowledgement of
the current target. A backend advances only the work it actually accepted, and completion means all
currently active coverage is acknowledged at the latest revision. See
[ADR 0040](decisions/0040-backend-aware-presentation-convergence.md).
`PresentationGenerationTracker` owns target/revision/active-coverage state; `LevelConvergenceStrategy`
adapts PyQtGraph progressive redraw and VisPy uniform acknowledgement to that shared contract.

Persistence intent (for example an explicit user lock) is separate from the latest physical presentation
target. A newer concrete target supersedes older automatic work without recreating the materialization
session.

## Render flows

### Normal image (current)

```text
state/document change
  -> render coalescing and generation
  -> cache lookup + render cost decision
  -> synchronous / chunked / background evaluation
  -> DisplayImage + semantic geometry/levels
  -> FramePlanner chooses typed tiled regions
  -> presentation planning and commit
  -> committed frame
```

The last valid frame remains visible during slow or refused work. Small images become one tiled region;
large single planes and montages become multiple tiled regions through the same backend surface.
Evaluation scheduling still has separate normal/montage orchestration, but committed presentation
semantics are always `DisplayTiledPresentation` commits through `present_tiled`; obsolete direct
tile-layer widget APIs are intentionally absent.

### Montage (current)

```text
state/viewport change
  -> montage plan + visible/near tile priorities
  -> session retarget
  -> cache/stage lookup and tile materialization
  -> semantic level coverage
  -> ready presentation delta
  -> adapter upload/rebind/visibility work
  -> acknowledgement and committed frame
```

Pan/zoom retarget the session rather than recreating document work. Native-resolution persistent tiles are the production baseline. The LOD selector may report a desired factor, but the applied factor remains one until asynchronous materialization and compatible residency satisfy [ADR 0041](decisions/0041-lod-selection-materialization-and-residency.md).

Montage resize and layout reflow are viewport retargets, not semantic scope changes. Manual resize
preserves screen zoom through `ViewportController`; same-source column reflow may translate the range
by source-local focus but must not add another zoom change. Fit and genuinely near-auto views
recompute fitted ranges. ROI selections are remapped by `source_index` plus tile-local coordinate only
when the source set is unchanged. See
[ADR 0042](decisions/0042-montage-viewport-reflow-and-roi-ownership.md).

### Remaining target flow

[ADR 0039](decisions/0039-unified-image-surface-and-deadline-scheduler.md) defines the intended convergence:

```text
ViewIntent
  -> FramePlanner
  -> FramePlan / WorkGraph
  -> DeadlineScheduler
  -> PresentationCommit
  -> ImageSurface (unified tiled regions with backend-specific commit mechanics)
```

A small plane, huge plane, one-tile montage, and many-tile montage should share semantic planning.
One-tile and small-tile cases are optimized inside the tiled engine; backend surfaces may commit them
through different physical mechanics without changing their meaning.
The `FramePlanner`, typed tiled surface, explicit `WorkGraph` admission, and
`ImageViewShell`/`ImageSurface` boundary are implemented. X4 now owns the remaining shared pointer
capture and drag-lifecycle migration.

## Non-negotiable invariants

- Widgets do not own `ViewState` or operation semantics.
- The visible operation stack is never silently rewritten by runtime optimization.
- Worker callbacks never commit stale semantic or presentation revisions.
- Camera-only changes do not restart array evaluation.
- Levels/LUT edits do not evict unchanged source texture data; PyQtGraph may still require bounded CPU/item redraws while VisPy uses uniforms.
- First pixels do not wait for a detailed histogram plot; they do require a valid semantic level source for the pixels shown.
- GUI callbacks have item, byte, and elapsed-time limits; an item cap alone is not a time budget.
- Cold upload/preparation is measured separately from warm rebind/visibility work.
- Warm/speculative residency is separately budgeted and yields to visible residency.
- Hidden side panels do not continuously compute.
- The committed frame, not a compatibility placeholder, answers hover/value queries.
- Backend branches are based on declared capabilities, not concrete class-name tests.
- Clearing a backend requires an explicit reason such as context loss, replacement, document revision, or incompatible physical representation.
- VisPy visible tile admission is coherent: placeholders clear only after texture data, geometry,
  visibility, and draw invalidation are consistent.
- Desired and applied LOD are separate states; no commit performs synchronous pyramid construction.

## Placement guide

| Change | Primary owner |
|---|---|
| Axis/slice/range/channel state | `core.view_state`, `core.slice_selection` |
| New operation semantics | `operations.pipeline` + declarations/tests |
| Request expansion/planning | `operations.regions`, `planner`, `slabs` |
| Cache/stage behavior | `operations.cache`, `stage_cache`, evaluator |
| Frame/presentation meaning | `display.model`, `planning`, `commit` |
| Coordinate conversion | `display.geometry` |
| Montage viewport reflow and source-local ROI remapping | `window.montage_viewport` |
| Texture/atlas/visual mechanics | `display.backends.*` |
| Hover/drag/cursor policy | `display.interaction`, hit testing |
| Background pan/zoom range math | `display.view_navigation` |
| Widget controls/layout | `ui.*`, focused `window.*` controller |
| Admission/budget policy | `core.memory_policy`, compute/resource policy |
| Historical rationale | `docs/decisions/` |

Avoid adding major behavior directly to `window.main`, `window.render`, or a backend widget simply because a callback is nearby.

## Known architectural debt

- `VisPyImageView2D` now inherits the shared `ImageViewShell` rather than the PyQtGraph concrete
  `ImageView2D`. The VisPy canvas is mouse-transparent to the stacked Qt event layer; the shared
  pointer interaction controller owns ROI/profile hit priority, capture, drag lifecycle,
  cancellation, and cursor intent, while VisPy background pan/zoom uses backend-native events with
  shared `display.view_navigation` range math.
- Normal and montage evaluation scheduling remain separate, even though frame planning and committed
  tiled presentation semantics are now unified.
- `MontageRenderSession` is still large, but level convergence, tile admission, and stage fan-in now delegate to Qt-free control-plane models.
- `frame_renderer.py` remains large, but montage resize/reflow and ROI layout semantics now delegate
  to Qt-free viewport helpers per ADR 0042.
- Large renderer/backend modules still combine orchestration and mechanics.
- Histogram binding still reaches into private PyQtGraph state.
- Several timers serve as implicit sequencing. They must become bounded scheduler resubmission/admission signals, not semantic order.

These are roadmap items, not invitations to perform a single broad rewrite. Each migration step must leave at least one runnable backend and retain semantic conformance tests.
