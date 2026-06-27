# v31 rendering and roadmap audit

Date: 2026-06-27

This audit reviews the rendering, scheduling, backend-composition, viewport, restore, and documentation
work currently sitting on top of `origin/main`. It is deliberately critical: the goal is not to defend the
recent direction, but to decide whether it is still the fastest path to a clean, future-ready renderer.

## Verdict

The broad route is correct. Do not throw away the whole code path. The recent work has moved ArrayScope
from renderer-specific behavior toward one semantic display pipeline with backend-specific physical
mechanics. That is the right foundation for mobile, integrated-GPU, workstation, and render-server
profiles.

However, the current route has three serious hazards that should drive the next roadmap gate:

1. **Large normal-image tiling is semantically unified but not yet physically complete.** A huge single
   plane can be represented as tiled frame regions, but the normal-image viewport retarget path and
   source-region materialization path are not complete enough to make visible-only residency the
   production default.
2. **Residency truth must come from backend acknowledgement, never from requested work.** A requested
   tile upsert is not resident until the backend reports that it accepted it. This audit repaired one
   instance of that bug.
3. **Backend and LOD defaults still need hardware evidence.** The architecture is ready for evidence,
   but policy should not be set from intuition. Device limits, allocation failures, upload counters,
   event-loop gaps, and context-loss behavior must decide default backend and multi-resolution policy.

In short: keep the architecture, but narrow the next phase. Do not start another broad renderer rewrite.
Do not enable aggressive viewport-only single-plane tiling until the retarget/materialization gap is
closed. Do not enable CPU-generated LOD in the production atlas.

## Review method

The review used the current git history, rendering documentation, ADRs, and the code shape in the
current tree. I focused on boundaries and hot-path correctness rather than rerunning existing tests.
The recent work clusters into these method families:

- **Control-plane extraction:** `PresentationGenerationTracker`, `TileAdmissionQueue`,
  `StageFanInState`, level-convergence strategies, and timer/revision guards reduce hidden ordering in
  `MontageRenderSession`.
- **Semantic unification:** `FramePlanner`, `FramePlan`, `FrameRegion`, `DisplayScene`,
  `DisplayTiledPresentation`, and `TiledValueSource` make normal images, internally tiled large planes,
  and montages use one semantic model.
- **Backend composition:** `ImageViewShell` plus `ImageSurface` replaces inheritance from the concrete
  PyQtGraph view. PyQtGraph and VisPy now share shell/UI/interaction state while owning separate
  mechanics.
- **Deadline/value scheduling:** `WorkGraph` makes visible, optional, side-analysis, and speculative
  work explicit and observable by lane.
- **Resource-governed fan-in:** the governor records UI observations and chooses item/byte/time budgets
  instead of letting worker completions flood the GUI thread.
- **Native-only LOD discipline:** the code now separates desired LOD from applied LOD and keeps the
  production display path native until compatible multi-resolution residency exists.
- **Viewport/restore repair:** recent commits address resize, fit behavior, restore camera locks, ROI
  remapping, and shared pointer capture.

These are good methods. They attack root causes instead of adding one more debounce timer or backend
branch. The weak point is that several pieces are now architecturally converged before the corresponding
physical data path is complete.

## How the pieces integrate

The intended hot path is now sensible:

```text
View/state change
    -> FrameTarget
    -> FramePlanner
    -> materialization / cache / display preparation
    -> DisplayPresentation
    -> DisplayCommitter
    -> ImageSurface
    -> backend-specific raster or tiled mechanics
    -> CommittedDisplayFrame / DisplayScene / value source
```

This separation matters. The renderer libraries should not define ArrayScope semantics. PyQtGraph and
VisPy should differ in allocation, upload, shader, camera, and overlay mechanics, not in the meaning of a
frame, region, ROI, level source, or inspection value.

The best integration decisions are:

- `DisplayScene` is backend-neutral and separates active, planned, near, and resident regions.
- `DisplayTiledPresentation` can represent montage and non-montage tiled surfaces.
- `DisplayCommitter` is the single gateway into `ImageSurface`, so backend acknowledgement can become a
  hard semantic boundary.
- `WorkGraph` and `ResourceGovernor` provide observable scheduling and fan-in pressure rather than
  implicit timer behavior.
- Shared pointer capture means ROI/profile behavior is not reimplemented once per backend.

The least complete integration point is large normal-image tiling. `FramePlanner` can split a single
plane into regions, but `ViewportBridge` still only schedules viewport retargets for montage view
states. Until normal-image retargeting and source-region reads are implemented, a large single plane
must either keep all planned regions active on first commit or risk blank panning.

## What the recent work does well

### It reduces semantic duplication

The project is much less likely to drift into separate PyQtGraph and VisPy products. Shared frame
planning, scenes, value sources, and interaction state are the right long-term shape.

### It gives the scheduler explicit vocabulary

`WorkGraph` is a major improvement over debounce-shaped rendering. It distinguishes visible work,
backend commit, GUI fan-in, histogram refinement, ROI/profile work, stage materialization, and
speculative residency. That makes performance debugging possible because counters can explain why work
was admitted, blocked, dropped, superseded, or completed.

### It treats backend differences honestly

The design no longer pretends that PyQtGraph item updates and VisPy uniform updates have the same cost.
That is important. A good cross-backend contract should require the same semantic outcome, not identical
physical work.

### It protects native inspection values

`TiledValueSource` and raster value sources keep hover/profile/ROI values independent from shader-ready
textures and future display LOD. This is non-negotiable for scientific inspection.

### It correctly keeps production LOD conservative

The old CPU LOD approach was structurally risky because variable-size reduced tiles do not belong in a
fixed-shape atlas page without a compatibility layer. Native-only production LOD is the right temporary
policy. The next LOD implementation should use compatible pages, arrays, or a virtual texture/page table,
not arbitrary mixed shapes inside one page.

### It removes the worst inheritance trap

VisPy no longer needs to be a subclass of the concrete PyQtGraph view. The shell/surface split is the
right destination: common UI and semantic state above thin backend mechanics.

## What it does not do well yet

### Large normal-image tiling still has a physical-data gap

The semantic model says a huge single image can be tiled. The current commit path can also send tiled
payloads to the backend. But the data source path still tends to operate on a materialized `display_image`
for the whole frame, and normal viewport changes do not yet retarget a tiled single-plane surface the way
montage viewport changes do.

This is the main reroute point. Do not spend the next phase polishing minor tile flags while first-frame
materialization can still be full-plane work. The next low-level design needs explicit region reads:

```python
class DisplayRegionSource(Protocol):
    def read_region(self, region: FrameRegion, *, quality: str, deadline_ns: int) -> DisplayTilePayload: ...
```

That source can initially wrap the existing eager array, but the caller contract must become region-first.
Only then can memory-mapped, chunked, remote, or server-side sources behave efficiently.

### Normal-image viewport retarget is missing

The bridge currently schedules montage viewport retargets only when `view_state.montage_axis is not
None`. That is correct for the old world, but not for internally tiled single planes. If the planner starts
marking only visible normal-image regions active, panning can reveal non-resident tiles without scheduling
new visible work.

The required shape is:

```python
if current_scene.storage is DisplayStorage.TILED:
    schedule_tiled_viewport_update(view_range=current_view_range)
```

That condition must be scene/storage based, not montage-mode based.

### Some compatibility surfaces remain larger than ideal

`montage_renderer.py` and `montage_session.py` are much better after extraction, but they remain a high
risk zone because they still coordinate Qt timers, payload caches, level convergence, viewport state,
overlay state, diagnostics, and commit publication. More extraction should be incremental and
state-machine driven; a big rewrite here would be risky.

### Hardware policy is still theoretical

The code now records many useful counters, but the roadmap must continue treating X5 as a gate. The
project should not decide “VisPy default everywhere,” “PyQtGraph fallback only,” or “enable LOD” until it
has traces from low-power integrated GPUs, discrete GPUs, macOS, Windows, Linux X11, and Linux Wayland.

### PyQtGraph histogram binding is still brittle

The histogram adapter isolates private API use, which is good, but private `HistogramLUTItem` behavior is
still a maintenance risk. Keep this behind compatibility tests and consider replacing the dependency for
critical level UI if PyQtGraph changes break the adapter.

## Bugs and low-hanging improvements fixed in this audit

### 1. Committed tiled scenes now use acknowledged residency

Before this audit, `commit_tile_layer()` built the `DisplayScene` from the requested tiled presentation
before backend acknowledgement was folded into the committed tile state. A backend could defer or reject
upserts, while `CommittedDisplayFrame.scene.resident_region_ids` could still claim the requested tiles
were resident.

The fix commits the tiled delta first, builds a committed presentation from the acknowledged state, then
builds the scene and value source from that state.

```python
self.commit_tiled_delta(presentation)
committed_state = self.last_tile_committed_state or presentation.base_tile_state
committed_presentation = replace(presentation, tile_state=committed_state)
scene = display_scene_for_presentation(committed_presentation)
return self._frame_for(committed_presentation, key, scene, tile_state=committed_state)
```

Expected outcome: diagnostics, hover/profile/ROI value availability, and scene residency now describe
what the backend actually accepted, not what the control plane hoped it would accept.

### 2. VisPy atlas cold allocation no longer scans every slot

`TextureAtlasPool._slot_for()` used `page.slot_owners.index(None)` for each cold tile. That is linear in
page capacity and becomes unnecessary allocation overhead during large visible commits.

The fix adds a free-slot stack to `TextureAtlasPage`:

```python
self._free_slots = list(range(self.capacity - 1, -1, -1))

def take_free_slot(self, owner: object) -> int | None:
    while self._free_slots:
        slot = int(self._free_slots.pop())
        if self.slot_owners[slot] is None:
            self.slot_owners[slot] = owner
            return slot
    return None
```

Expected outcome: cold atlas fill cost is bounded by the number of newly assigned tiles, not multiplied
by the number of slots already occupied in the page. Eviction behavior is unchanged.

### 3. Legacy PyQtGraph tiled delta calls no longer dereference missing payload maps

The legacy raster-backed PyQtGraph montage path accepts `tile_payloads=None`, but the active-set setup
was still testing membership in that missing map when a `tile_delta` was supplied. Direct legacy delta
callers could therefore raise `TypeError` before reaching the newer typed presentation path.

The fix makes the legacy active set come directly from `tile_delta.active_tiles` when present. The branch
should still be deleted once all callers use `DisplayTiledPresentation`, but it no longer has a fragile
`None` dereference.

## Bugs and risks not fixed here

### 1. First-frame large normal image can still be too eager

A tiled single-plane frame should not require full display-frame materialization before the first visible
tiles can appear. Today the architecture supports the target shape, but region-first source reads are not
the dominant contract yet.

### 2. Scene cache keys include object identity

`_frame_plan_regions_cached()` receives `id(frame_plan)` as a cache argument. That is safe enough for
short-lived cache invalidation, but it reduces reuse across equivalent plans. This is not urgent, because
the region signature is also present and scene construction is not the main hot path, but the cache should
not grow around plan object churn if frame plans become frequent during normal tiled panning.

### 3. Backend capability flags need to encode failure modes

`direct_montage_tile_payloads` is currently used as a capability gate for tiled storage. Hardware and
runtime failure modes are more nuanced: max texture size, supported internal formats, allocation success,
context loss, and shader support may change the viable strategy after startup. X5 should turn capability
flags into measured, invalidatable backend evidence.

### 4. Restore and viewport code are better, but still sensitive

The recent restore work is valuable, but restored camera locks, fit policy, ROI remapping, and pointer
capture now intersect with tiled surfaces. Add manual traces specifically for: restore huge single-plane
tiled view, pan before first warm residency, switch backend after restore, and drag ROI during restored
viewport release.

## What should happen now

Treat X5 as a focused evidence-and-residency gate, not a generic performance bucket.

Immediate work:

1. **Keep committed residency acknowledgement as a hard contract.** Add conformance tests around
   deferred/rejected backend upserts and assert that `DisplayScene.resident_region_ids` follows the
   acknowledged state.
2. **Add normal tiled viewport retarget before visible-only active regions.** The condition should be
   “current committed scene is tiled,” not “montage mode.”
3. **Introduce region-first materialization.** Start with an adapter over the eager array, then move lazy
   and out-of-core sources behind the same protocol.
4. **Benchmark huge single-plane first frame and pan.** Measure request-to-first-visible-tile,
   event-loop gap, upload bytes, accepted tiles, resident tiles, stale tiles, and RSS.
5. **Collect hardware traces before changing defaults.** Use integrated GPU, discrete GPU, low-power
   laptop, Linux X11/Wayland, Windows, and macOS. Include allocation failures and context-loss recovery.
6. **Delete or hard-guard legacy tiled presentation entrypoints.** Compatibility shims should import or
   adapt only; they should not keep behavior branches alive.

## What should happen later

After X5 evidence is available:

- Choose default backend policy by measured latency/stability, not by renderer preference.
- Implement async/source-provided LOD with compatible page classes or texture arrays.
- Add adjacent-level retention under budget so zoom threshold crossings do not cause full churn.
- Move remaining montage session responsibilities into smaller Qt-free models only where a test can own
  the state machine.
- Replace private histogram binding if compatibility tests show recurring fragility.
- Extend the source protocol to memory-mapped, chunked, remote, and server-backed arrays without changing
  rendering semantics.

## Roadmap and ADR updates made with this audit

- Added ADR 0044 to make viewport-scoped tiled residency and backend acknowledgement explicit.
- Updated the roadmap X5 gate with normal tiled viewport retargeting, region-first materialization,
  acknowledgement tests, and hardware evidence requirements.
- Added this review to the review index.
