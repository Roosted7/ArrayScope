# ArrayScope v27 rendering architecture and performance review

## Executive verdict

Do not throw away the semantic display, operation-planning, caching, or tiled-residency work. Those
parts are the strongest foundation in the project.

Do retire two current assumptions:

1. that “normal image” and “montage” need separate end-to-end render orchestration; and
2. that debounce plus latest-only cancellation is an adequate interactive scheduler.

The desired project has exactly two production pixel surfaces—PyQtGraph and VisPy—but only one
semantic presentation pipeline, one interaction model, one frame scheduler, one cache identity model,
and one performance contract. Raster versus tiled is a physical storage decision beneath that common
pipeline. It is not a separate viewer mode.

The v27 traces show two distinct classes of defects:

- a backend-independent synchronous cache-hit path blocked the UI for roughly 9.1 seconds in both
  backends;
- the PyQtGraph progressive path performed up to 173 item updates and 166 ms of CPU windowing in one
  GUI callback.

The review branch fixes those immediate defects and removes the broken VisPy canvas option. It does
not attempt the riskier scheduler and widget-shell rewrite without a real Qt/OpenGL environment. That
larger work is specified below in enough detail to implement deliberately.

## Repair addendum after the 14:16/14:18 captures

The follow-up traces and manual testing showed that the original seven review commits fixed the long
event-loop freezes but introduced a different class of regressions: VisPy histogram/window flicker,
slow one-by-one tile pop-in, destructive montage retargeting during scroll/pan, placeholders that
disappeared before content was actually visible, and occasional queued full refreshes.

The repair series keeps the freeze fixes and changes the invariants:

- semantic level identity is independent of the currently requested coverage; coverage is tracked by
  the level tracker and may only improve within one semantic generation;
- a tile may be presented before the detailed histogram curve is rebuilt, but never before the
  automatic level source includes that tile's semantic summary;
- materialized CPU/GPU-ready data is not the same thing as user-visible presentation; placeholders
  are removed only after the backend acknowledges presentation;
- resident rebinds, existing PyQtGraph items, geometry moves, and visibility changes are cheap work
  and must not be throttled by the same mechanism as cold texture uploads or new item creation;
- scrolling and panning preserve retained source identities and treat entering/leaving tiles as a
  small delta rather than as a full clear-and-reload;
- stale deltas are rejected and backend clears require explicit reasons such as context loss, backend
  replacement, or a true semantic revision.

This means the review branch should be read as two layers: the first seven commits removed the worst
freezes, and the repair commits restore the presentation invariants needed for smooth tiled use.

## What the recent work got right

### Backend-neutral presentation semantics

`DisplayRasterPresentation`, `DisplayTiledPresentation`, `DisplayGeometry`, `DisplayScene`,
`CommittedDisplayFrame`, and the value-source types establish the correct ownership rule: rendering
libraries draw a decided presentation; they do not decide what the data, levels, geometry, or hover
values mean.

The backend adapter boundary is also the right migration mechanism. It lets orchestration commit
semantic intent while legacy widget methods remain temporarily available. Rewriting both widgets and
all render behavior in one flag-day change would be much riskier.

### Semantic values are separated from display storage

`CanvasValueSource` and `TiledValueSource` prevent hover, ROI, profile, and export behavior from
reading a shader-ready RG plane, a CPU-windowed RGB tile, an atlas slot, a reduced LOD texture, or a
compatibility placeholder as though it were source data. This is essential for both correctness and
backend parity.

### Tiled delta and residency concepts are sound

Stable content identity, upserts/removals, active/planned/near sets, persistent source-keyed atlas
slots, and separation of draw visibility from residency are all worth retaining. They are the right
shape for GPU reuse and for a PyQtGraph fallback that updates only dirty items.

### Operation planning and reusable stages address the right compute problem

The operation region planner and stage cache distinguish reusable expensive transforms from cheap
per-view slicing. This is more valuable than merely adding tile workers. A broadcast FFT should be
materialized once at the correct granularity and then feed many tiles or slices; it should not be
recomputed speculatively once per predicted output.

### Resource feedback is the correct idea

A measured governor is better than fixed global worker counts and fixed debounce values. The traces
showed an observation bug—whole callbacks were counted as one item—but not that feedback control is a
bad direction. With typed measurements and hard GUI budgets it should become the central admission
policy.

### The package refactor follows ownership rather than renderer names

Moving display models, planning, commits, backend mechanics, and interaction semantics toward
`arrayscope.display` is correct. Creating matching copies of planning/ROI/window-level logic under
`pyqtgraph/` and `vispy/` would make the tree look neat while guaranteeing behavioral drift.

## Critical findings

## P0 — synchronous render orchestration can monopolize the event loop

Both captures recorded a cache-hit 272-tile montage with approximately 9.1 seconds in
`last_render_sync_ms`. Evaluation, cache lookup, payload construction, upload, and final commit were
all far smaller. The common synchronous path was doing semantic level/histogram accumulation before
showing already available pixels.

This violates the most important rendering rule: metadata refinement must never gate a usable frame.

### Implemented correction

The branch now:

- separates semantic montage identity from the requested or visible coverage;
- incorporates the level summaries for all tiles that are about to become active before committing
  those tiles to the backend;
- allows the detailed histogram curve to lag by a bounded timer slice, while the rendering level
  source may not lag presented content;
- retains the last valid level source and prevents lower-coverage summaries from replacing better
  coverage within the same semantic generation;
- excludes LUT/presentation state from the semantic level key;
- samples finite values in one pass.

Expected outcome: a fully cached restart has first-pixel latency proportional to planning, level
summary merging for the tiles being shown, and one bounded commit. New visible tiles no longer appear
with window levels computed from unrelated early data.

## P0 — progressive display work was not actually bounded

The old “batch” controlled how often a montage commit ran. A single commit could still carry every
pending tile. In the PyQtGraph trace, one callback CPU-windowed 173 tiles for 166 ms and spent 294 ms
in the full commit.

### Implemented correction

The first attempt sliced `TilePresentationDelta.upserts`, but follow-up testing showed that this
throttled resident/cheap work and could serialize an already-cached montage over many callbacks. The
repair changes the unit of control: backends now report cold work separately from resident rebinds,
existing-item shows, geometry moves, and visibility updates.

Cold texture uploads, new PyQtGraph items, and CPU windowing are the work that must obey deadlines.
Resident VisPy atlas entries and already-created PyQtGraph items should be rebound, moved, or shown in
the same commit whenever possible. Feedback is recorded on typed cold-work observations rather than on
one generic callback number.

Expected outcome: backend throughput remains high without allowing a cold burst to monopolize the GUI
thread, and an already-resident 272-tile montage is not artificially paced as 272 expensive uploads.

## P0 — continuous single-image interaction can cancel all progress

The current normal path does the following:

```text
request_render(interactive=True)
  -> advance render generation
  -> clear visible/profile/ROI/pixel/prefetch groups
  -> 16 ms coalesced render
       -> clear groups again
       -> start_latest(visible-image)
            -> cancel previous visible-image
```

The visible lane has one worker. When a slice takes longer than the request cadence, every job can be
cancelled before completion. The UI therefore keeps the previous frame until scrolling slows. The
scheduler counters in the traces support this: the PyQtGraph visible lane had 31 stale and 20
cancelled jobs against 31 completed jobs.

This is a fundamental scheduler problem. Changing VisPy texture upload or PyQtGraph debounce values
will not solve it.

### Required replacement

Use one active target plus one replaceable latest target:

```python
class VisibleFrameQueue:
    presented: FrameTarget | None
    active: FrameTarget | None
    latest: FrameTarget | None

    def request(self, target: FrameTarget) -> None:
        if self.active is None:
            self._start(target)
        else:
            self.latest = target       # replace queued target only

    def active_finished(self, result) -> None:
        self._cache_reusable_result(result)
        if result.target == self.latest:
            self._present(result)
        next_target, self.latest = self.latest, None
        if next_target is not None:
            self._start(next_target)
```

The real implementation should add cost/progress-aware cancellation: cancel active work only when the
new request invalidates reuse and the estimated remaining work is materially larger than restarting.
A completed stale result may still populate a cache even when it cannot be presented.

Expected outcome: continuous scrolling always makes bounded progress and periodically presents a
recent frame instead of waiting for input to stop.

## P1 — VisPy is still a hybrid widget, not an independent backend

`VisPyImageView2D` remains a 2,032-line subclass of the 1,821-line `ImageView2D`. PyQtGraph still owns
parts of interaction, camera/range state, histogram behavior, and overlay semantics while a VisPy
canvas owns pixels. This was a reasonable prototype, but it is not a maintainable final backend.

Consequences include:

- two scene/event systems;
- camera synchronization and duplicate redraw triggers;
- difficult Qt stacking and Wayland behavior;
- backend-specific imports crossing the boundary;
- unclear pointer capture and cursor ownership;
- a VisPy class that inherits fallback behavior it should never expose.

### Implemented correction

The broken VisPy montage canvas fallback is removed from capabilities and disabled in the UI. VisPy
has one tiled montage path. PyQtGraph retains its tiled path and optional raster canvas behavior.

### Required structural change

Create a shared `ImageViewShell` that contains, rather than is inherited by, an `ImageSurface`:

```python
class ImageSurface(Protocol):
    def present_raster(self, presentation, mode): ...
    def present_tiled(self, presentation): ...
    def apply_viewport(self, viewport): ...
    def apply_overlay_state(self, overlays): ...
    def diagnostics(self): ...
```

The shell owns histogram widgets, HUD, ROI information, semantic signals, shared pointer controller,
and viewport controller. PyQtGraph and VisPy surfaces own only concrete graphics objects and resource
lifecycle.

Expected outcome: backend parity is tested at one semantic contract; no renderer inherits another
renderer; camera and pointer events have one owner.

## P1 — single image versus montage is the wrong physical boundary

`DisplayScene` already states the correct model: a normal image is one semantic region and a montage is
multiple semantic regions. However, `DisplayCommitter._validate_presentation()` currently rejects a
tiled presentation without montage geometry. That prevents a very large single plane from using the
same virtual/tiled storage strategy.

The future planner should decide storage independently:

```python
def choose_storage(scene, backend, device, history):
    if scene.estimated_bytes <= device.safe_single_texture_bytes \
       and scene.update_rate <= history.raster_update_threshold:
        return RasterStorage()
    return TiledStorage(
        tile_shape=device.preferred_tile_shape,
        residency_budget=device.safe_residency_bytes,
    )
```

Do not literally put every small image in an atlas. A small stable plane is often best as one texture
or one `ImageItem`. A huge plane can be internally partitioned while preserving single-image
semantics. A one-tile montage and a normal image should share scheduling, levels, cache identity, and
interaction behavior even if their allocation differs.

Expected outcome: one optimal path at the semantic/scheduling level, with storage selected for the
actual workload rather than for the UI mode name.

## P1 — immutable tiled state still performs whole-map work per commit

The delta protocol is good, but `build_tile_presentation()` still snapshots and compares complete
payload maps, and `TilePresentationState.apply_delta()` copies the complete mapping. At 272 tiles the
captured payload-build cost was about 5–7 ms. It is not the cause of the multi-second freeze, but it
will become a frame-budget problem at larger residency counts or higher commit rates.

Do not optimize this by making old committed frames silently share a mutable dictionary. Instead,
separate:

- a session-owned mutable payload registry;
- a revisioned immutable semantic snapshot/key;
- a compact delta for backend mutation;
- a value source that resolves through the committed registry revision.

A persistent-map implementation or generation-owned registry can avoid O(n) copies while preserving
frame validity.

Expected outcome: clean viewport or level commits are O(changed tiles), not O(all resident tiles).

## P1 — feedback needs typed cost, not one generic callback number

The controller currently learns elapsed time and item count. The item-count fix makes it useful, but
maximum performance across PyQtGraph and VisPy needs more dimensions:

```python
@dataclass(frozen=True)
class UiCostObservation:
    channel: str
    backend: str
    payload_kind: str
    item_count: int
    byte_count: int
    cpu_prepare_ms: float
    upload_ms: float
    callback_ms: float
    queue_delay_ms: float
    interactive: bool
```

Use separate estimates for:

- CPU preparation per tile;
- upload time per byte;
- fixed callback overhead;
- draw/vertex submission overhead;
- queue delay and frame age;
- wasted/cancelled milliseconds.

Then choose item count and byte count independently. VisPy can ramp rapidly when upserts are cheap;
PyQtGraph can stay at a smaller batch when RGB windowing dominates. Cold start should be conservative,
followed by bounded multiplicative growth. Overload backoff should be immediate; recovery gradual.

Expected outcome: the same governor extracts available throughput without allowing backend-specific
work to violate the event-loop budget.

## P1 — debounce and quiet-period gating are overused

Debounce is useful for coalescing repeated intent. It is not a scheduler. Several side paths wait for
interaction or rendering to become globally quiet, which makes inexpensive or cached work feel late
and prevents partial ROI/profile feedback.

The desired rule is:

```text
issue immediately -> assign priority/deadline -> admit by measured slack -> refine incrementally
```

For ROI/profile/histogram work:

1. read the committed frame immediately for hover and cheap local values;
2. publish a bounded coarse/partial result when possible;
3. schedule exact work on its own lane;
4. pause only when visible deadline slack disappears;
5. guard the final commit by semantic target identity.

Hidden panels still do no work. Selected and hovered entities get higher priority than background
panel refreshes.

Expected outcome: tools feel live during rendering without stealing time from visible frame delivery.

## P1 — prediction should be value-based and stage-aware

Current prefetch is mostly idle-only, often disabled, and begins after an exact render. That cannot
hide latency during sustained motion. Prediction should use:

- recent slice direction and velocity;
- next/previous slice probability;
- the next viewport tile ring;
- selected or hovered ROI/profile targets;
- reusable stage availability;
- measured evaluation and upload cost.

A practical score is:

```text
probability of use * expected latency saved / estimated resource cost
```

Admit only while event-loop delay, visible backlog, memory pressure, and GPU residency pressure remain
below thresholds. Never speculate an expensive FFT separately per tile when one reusable stage is not
already cached or in flight.

Expected outcome: spare CPU/GPU/memory becomes latency reduction rather than random cache fill, and
speculation disappears immediately when the user-visible path needs resources.

## P2 — GPU memory and LOD need explicit compatible storage classes

Native-resolution residency is the correct current production default. The earlier CPU LOD prototype
mixed differently sized images into fixed atlas slots whose geometry assumed one tile shape. That can
produce padded sampling, distortion, rebuilds, and repeated uploads around zoom thresholds.

The production LOD path should use one of:

- separate pages by LOD and tile shape;
- texture arrays grouped by compatible dimensions;
- a virtual texture/page table;
- source-provided pyramids when available.

GPU budget must be derived from runtime limits and proven allocation behavior, then constrained by the
user profile. CPU semantic cache, CPU display-preparation cache, and GPU residency are separate
budgets with separate eviction costs.

Expected outcome: zoomed-out views reduce bandwidth and fill time without weakening hover/ROI/export
semantics or destabilizing residency.

## Code-structure assessment

The ownership direction is good, but several files still combine too many roles:

| File | Approximate lines | Main remaining responsibilities |
|---|---:|---|
| `display/vispy_imageview2d.py` | 2,032 | shell inheritance, camera bridge, pointer bridge, overlays, raster/tile commits, lifecycle |
| `window/montage_renderer.py` | 1,831 | planning, cache resolution, stages, scheduling, levels, commits, overlays, prefetch hooks |
| `display/imageview2d.py` | 1,821 | shared shell plus PyQtGraph surface mechanics |
| `display/backends/vispy/tiles.py` | 1,692 | atlas allocation, residency, payload batching, visuals, diagnostics |
| `display/backends/pyqtgraph/tiles.py` | 611 | item registry, CPU preparation, updates, diagnostics |

Splitting by arbitrary line count would not help. Split along ownership boundaries:

```text
display/
  model/               semantic frame, presentation, tiles, overlays
  planning/            frame/storage/quality plans
  interaction/         hit test, pointer capture, drag lifecycle, cursor intent
  backends/
    pyqtgraph/          raster, tiles, overlays, surface
    vispy/              raster, atlas, shaders, overlays, surface
  widget.py            shared Qt shell

rendering/
  target.py            presented/active/latest frame identity
  work_graph.py        dependencies and supersession keys
  scheduler.py         deadlines, admission, cancellation, progress
  feedback.py          typed cost models
```

Window code should translate user/application state into `ViewIntent` and consume committed frame
state. It should not own atlas details, per-item CPU windowing, or backend-name policy.

## Performance contract

Adopt these as tested invariants rather than aspirations:

- no data-sized loop in a GUI callback without item/time/byte slicing;
- interactive GUI work target: <= 4 ms;
- idle GUI work target: <= 8 ms;
- callback > 16 ms: regression/failure in performance tests;
- request to first retained/coarse frame measured separately from exact completion;
- last valid frame remains visible;
- pan/zoom with resident data performs zero materialization and zero texture upload;
- scrolling a resident tiled montage preserves retained source identities and does not clear the
  backend;
- placeholders persist until content has been acknowledged as presented;
- automatic levels for visible tiles include the tiles being presented, even when the detailed
  histogram plot is refined later;
- presentation-only level/LUT changes perform zero texture upload on shader-capable backends;
- cache-hit one-region and 500-region sessions have bounded first-commit latency;
- continuous input must produce frame progress and cannot require a quiet period;
- side analyses never gate pixel presentation;
- diagnostics report work discarded in milliseconds and bytes, not only counts;
- diagnostics distinguish cold work from resident rebinds, existing-item shows, and geometry moves.

## Recommended migration sequence

### Stage 0 — stop the observed freezes

Completed in this branch:

- decouple montage semantic identity from requested coverage;
- require level coverage for tiles before they become visible;
- separate materialized, loading, and presented tile states;
- replace universal upsert throttling with typed cold-work reporting;
- preserve retained tiled payloads across viewport retargeting;
- reject stale tiled deltas and make backend clears explicit;
- use constant-time work queues;
- remove VisPy canvas fallback;
- add phase timings, trace summarization, and tiled work diagnostics.

### Stage 1 — make regressions measurable

Add automated scenarios for both backends:

- 1, 2, 72, 272, and 500 cached tiles;
- cold tiles over cached stage;
- cold shared-stage montage;
- rapid single-image slice motion;
- pan/zoom with full residency;
- level/LUT/component changes;
- ROI/profile interaction while visible work is active;
- GPU budget pressure and eviction.

Measure event-loop gaps, request-to-first-frame, request-to-exact-frame, upserts, bytes, and cancelled
milliseconds. Do not accept setter-call duration as frame latency.

### Stage 2 — replace latest-only restart with frame progress

Introduce `FrameTarget` and active-plus-latest queue for normal images first. Preserve the old
presentation commit and evaluator APIs during this step. Then move montage onto the same target/work
model.

### Stage 3 — unify planning and storage choice

Create one `FramePlanner` that emits raster or tiled storage plans for either single or montage scenes.
Generalize tiled geometry beyond montage axes. Remove the split orchestration once conformance tests
cover both scene layouts.

### Stage 4 — finish interaction and widget composition

Move pointer capture and drag state into the shared interaction controller. Create `ImageViewShell` and
extract both surfaces. Delete `VisPyImageView2D(ImageView2D)` only after parity tests pass.

### Stage 5 — optimize residency and prediction

Add typed feedback, value-based prediction, multi-page/multi-resolution residency, and backend-specific
fast paths behind the shared contracts.

## Changes committed in this review branch

Starting from `f2a22c6`, the first review layer contains:

1. `1c0130f` — Defer cached montage level sampling
2. `c3cbdc6` — Remove VisPy montage canvas fallback
3. `1ce4225` — Bound progressive tiled display commits
4. `bc186a7` — Add actionable montage trace diagnostics
5. `fe5c88a` — Move montage level semantics into display model
6. `aa2115b` — Use constant-time montage work queues
7. `3efc7ee` — Document unified rendering and scheduling direction

The repair layer starts from `3efc7ee` and adds focused commits for the seven invariants above. Those
repair commits are intentionally additive so they can be applied on top of an existing review branch.

## Validation and limitations

The review environment does not contain PyQtGraph, Qt, VisPy, or an OpenGL display. The changes were
therefore constrained to pure models, architecture guards, static compilation, and tests that do not
construct the real widgets. Real validation is still required for:

- Qt event-loop pacing;
- PyQtGraph item updates and histogram behavior;
- VisPy context creation, texture allocation, and shader paths;
- Wayland stacking, resize, and cursor behavior;
- DPI scaling and pointer-coordinate parity;
- actual GPU memory limits and multi-page eviction;
- end-to-end first-presented-frame timing.

That limitation is exactly why the branch does not perform the scheduler or widget-shell rewrite.
Those changes should land against a real interactive benchmark harness, not by making unverified Qt
lifecycle assumptions.
