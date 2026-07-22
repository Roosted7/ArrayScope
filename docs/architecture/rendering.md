# Rendering

Rendering is split into semantic presentation and concrete backend mechanics.

## Semantic inputs and outputs

An evaluation result supplies data plus semantic metadata such as texture kind, histogram/value source, component, and exact/degraded status. Presentation planning combines that with:

- committed geometry;
- levels and histogram domain/source;
- LUT and scale mapping;
- viewport intent;
- tiled region payloads;
- dirty/retained/presented tile state.

The resulting display presentation is backend-independent. A backend adapter translates it into concrete widget/texture calls and reports commit work/acknowledgement.

## Committed frame

A frame is the visible semantic truth. It owns:

- document/semantic/presentation revision;
- geometry used to draw the pixels;
- value source used for hover/inspection;
- level source and mapping;
- visible tile/payload identity;
- exact/degraded and dirty/acknowledged state.

Hover and ROI/profile mapping use this frame, not whatever `ViewState` happens to be queued next. That prevents a pointer value from being read from new state against old pixels.

## Geometry

`display.geometry` maps among:

- backend world/ViewBox coordinates;
- committed-display coordinates;
- montage tile-local coordinates;
- array indices and profile states.

Geometry is committed with the frame. Normal and montage paths must not invent separate coordinate conventions inside event handlers.

## Levels and histograms

Three concepts are separate:

1. semantic values/pixel source;
2. level source/coverage used for automatic windowing;
3. detailed histogram plot data shown to the user.

A progressive tile can be shown before a high-detail plot is complete, but automatic levels for that tile must be based on semantic coverage that includes it. User-locked levels are not overwritten by later refinement. Preview drags/manual edits may update pixels immediately while only final edits emit the semantic level-change signal.

Montage level evidence is ranked separately from coverage: rough preview,
rough target/full, then refined. Rough preview evidence may seed the first
VisPy shader levels and histogram plot, but it remains provisional. A reduced
target LOD is rough target evidence, not merely preview evidence, when it is
the requested final display target. Refined evidence is admitted after visible
presentation settles and may update the histogram/levels without a tile upload.
Lower-quality evidence never replaces higher-quality evidence for the same
semantic source. See
[ADR 0054](../decisions/0054-montage-level-evidence-phasing.md).

## Unified Tiled Surface

ArrayScope presents normal images, large planes, and montages through one semantic tiled image
surface. A small/stable image is one tile; a large single plane may use internal tiles; a montage uses
multiple semantic tile regions. Those are layouts inside one presentation model, not separate semantic
renderers.

A tiled presentation is a set of semantic regions and payloads. PyQtGraph uses persistent per-tile
image items; VisPy uses atlas/texture-backed visuals. Tile identity is based on materialized data and
compatible physical representation, not levels/LUT. Level/window/LUT changes are presentation
updates, preferably shader/uniform updates where the backend supports them, and do not imply new
source pixels.

Presentation command order is semantic data, not a backend preference. The
canonical flow is `montage_priority_focus` → `TilePriorityContext` /
`tile_priority_key` → ordered materialized rows → ordered
`TilePresentationDelta.upserts` → ordered backend work and acknowledgement.
Shared-transform results are layout-independent, so each bounded fanout batch
must project them through the current layout before admission. A backend may
sort geometry keys, page indices, or unordered membership snapshots for stable
mechanics; it may not sort or setify the ordered command collection. Backend
hover state and cache iteration order never choose semantic priority.

`FrameSession` does not persist a second tile-work queue. Required target debt
is read from `TileLifecycle` whenever `FramePipeline` plans; running work is a
task/evaluation claim; stage waiting is a stage-fan-in binding; deferred stage
planning retains only its immutable missing-tile input. Completion and
diagnostics derive from those owners. A priority queue may exist as a local,
ephemeral ordering utility for prefetch, but its membership is never semantic
or lifecycle state.

VisPy atlas residency is a data-keyed cache, not a mirror of the current viewport. `active_tiles`
controls which retained tile mappings are visible; source identity, texture kind, LOD, tile shape,
storage mode, budget eviction, reset/context loss, or teardown are the only valid reasons for texture
residency to become cold.

Presentation-generation and admission state are Qt-free. `PresentationGenerationTracker` owns the
latest level target, revision, active coverage, pending work, and acknowledgement state.
`TileAdmissionQueue` owns priority/aging/item/byte/deadline admission without knowing array semantics.
`LevelConvergenceStrategy` keeps PyQtGraph progressive tile redraws and VisPy uniform updates behind
one semantic convergence contract. The kernel sits above these component models and owns real
lane-level execution/counters for visible planning, materialization, display preparation, GUI fan-in,
backend commit, side analysis, stage materialization, and speculative residency. Work visibility is
carried by target quality as well as lane: exact visible stage materialization is visible work, while
retained stage warmup is optional and subject to available-budget admission.

A montage is one reason to have semantic regions, but not the only one. Internal tiling of one huge
plane is implemented without inventing a montage axis; `FramePlan` region bounds and data slices are
the source of truth for backend tile placement.

Montage resize and column reflow use shared viewport policy. `ViewportController` preserves manual
screen zoom across widget resize; then the Qt-free `montage_viewport` policy handles montage-specific
layout retargeting. The renderer supplies the current Fit/near-auto facts and applies the returned
`MontageViewportReflow`. Manual views do not refit because the viewport grew, shrank, changed aspect,
or temporarily intersects zero tiles. Near-auto re-entry requires all four current edges to be near the
next fitted auto range. When the same source indices are laid out in different columns, manual reflow
translates by `source_index` and tile-local focus without applying another zoom change. When the tiled
dimension scrolls to a different source set, the range remains in world coordinates and samples the
new content. ROI geometry follows the same source-local rule through canonical
`RoiSelection` remapping. See
[ADR 0042](../decisions/0042-montage-viewport-reflow-and-roi-ownership.md).

### Multi-resolution

Resident multi-resolution LOD is the default for VisPy tiled scenes
([ADR 0050](../decisions/0050-async-multi-resolution-tile-residency.md)): asynchronous worker-side
pyramid materialization keyed by semantic tile identity; atlas pages classed by
`(level, texture shape, format, gutter)` so a reduced tile never enters a native-shaped slot; a
presentation floor that presents any resident level instead of a placeholder; and a retained,
pinned preview level. Tile lifecycle state (what is evaluated, resident, presented) is owned by
the Qt-free `TileLifecycle` machine
([ADR 0051](../decisions/0051-single-owner-tile-lifecycle.md)); backends acknowledge commits with
slot identities and never own semantic bookkeeping. Exact inspection stays native-resolution;
PyQtGraph adoption and ops-input LOD remain roadmap work.

## Backend contract

Shared code asks for capabilities such as:

- typed tiled presentation via `DisplayTiledPresentation`/`TilePresentationDelta`;
- persistent residency;
- shader windowing for scalar/complex data;
- native pointer/viewport interaction;
- whether an X/Y axis-order swap is applied as a display transform
  (`display_axis_transpose`, see below);
- diagnostics and acknowledgement.

It must not branch on `isinstance(...VisPy...)` to decide semantic meaning.

### Canonical orientation and display-only axis swap

A backend that declares `display_axis_transpose` renders tiles that are
materialized, cached, uploaded, and identified **once in canonical
(sorted-image-axes) orientation**; an X/Y axis-order swap (transpose) is then a
pure display transform — the same cost as an axis flip — instead of
re-materializing tiles ([ADR 0058](../decisions/0058-canonical-tile-orientation-and-display-transpose.md)).
The per-window evaluator's `canonical_orientation` flag (set per frame from the
capability) gates canonical extraction and sorts `image_axes`/`keep_axes` in
cache keys and semantic identities, so a transposed view reuses the unswapped
view's payloads and GPU residency. wgpu applies the swap with a swapped UV walk
in the vertex shader; PyQtGraph feeds the `ImageItem` a transposed view of the
canonical buffer. Value readout indexes the canonical array with swapped
coordinates. LOD **factor** selection and montage layout stay display-oriented,
but page **source** rectangles are canonical. A non-capable backend (VisPy)
keeps the legacy re-render-on-swap path.

Concrete backend code may own:

- image items, visuals, buffers, textures, atlas pages;
- upload/window preparation specific to the library;
- shader sources/uniforms;
- camera synchronization and scene redraw mechanics;
- backend-native background viewport navigation mechanics that apply shared range math;
- resource/context-loss handling.

It may not own:

- what constitutes a frame target;
- which ROI/profile target wins;
- whether levels are user-locked;
- cache/document identity;
- the meaning of a viewport request;
- ROI/profile hit priority or drag lifecycle;
- montage reflow or ROI source-local remapping semantics.

## Current backends

### PyQtGraph

The default path is mature and provides the complete feature baseline. Its tiled implementation avoids rebuilding a composed montage image, but large item counts and per-item updates can become GUI/scene-graph bottlenecks. Warm item visibility/geometry changes should not be reported as cold CPU windowing/upload.
It accepts typed tiled presentations for internally tiled single planes as well as montages. The
old direct tile-layer widget API has been removed; tile-layer commits enter through
`present_tiled` with committed tile state and revisioned tile deltas.
PyQtGraph now declares persistent CPU/item tile residency separately from VisPy shader residency.
Inactive tiles are retained as prepared `ImageItem` state under a bounded inactive pool and can be
rebound by source/content identity without recreating the item or rebuilding display arrays. Viewport,
camera, and active-set changes may hide, move, or rebind residents, but they must not clear them.
Only explicit reset/teardown, incompatible source/content identity, or residency-budget eviction can
destroy PyQtGraph tile residency.

### VisPy

VisPy supports shader mapping and persistent tiled residency with atlas-backed drawing. It can avoid
repeated CPU windowing and reduce many-item overhead. `VisPySurface` now reaches presentation commits
through the shared `ImageSurface` contract rather than inheriting the PyQtGraph concrete surface.
VisPy does not expose a direct tile-layer presentation API; all tiled updates use the typed
`DisplayTiledPresentation` path so histogram identity, payload revisions, residency acknowledgement,
and committed value semantics stay on one control plane.
Normal-image commits hide/deactivate the tiled presentation but do not reset compatible VisPy atlas
residency or retained acknowledged tile payloads. Explicit surface reset, context loss, teardown, and
incompatible physical representation changes are the reset boundaries that destroy residency.
The VisPy canvas remains mouse-transparent for the stacked Qt event layer. ROI/profile hover and drag
use the shared pointer interaction controller, while background pan/zoom uses
`display.view_navigation_driver` plus `display.view_navigation` range math to update the canonical
`ViewBox` range and camera without PyQtGraph scene drag.
VisPy is the preferred backend for sustained large tiled rendering, pending small-view latency and
platform validation. Its active visible commit should be a coherent GPU presentation transaction:
admitted payloads are acknowledged only after texture data, atlas/page geometry, visibility, and draw
invalidation are consistent.
The layer-wide shader-mapping key is only a desired-state cache; each atlas
page visual owns physical uniform state. Every touched/active page synchronizes
that state, even when the layer-wide key is unchanged. A levels-only action
updates levels only and must never replay cached component, display, LUT, or
scale state.
Its atlas/quad path uses frame-plan tile geometry for internally tiled single planes and montage
geometry for montage presentations.

Widget close stops warm-tile work, cancels queued histogram refresh, and closes the VisPy canvas.

## Progressive presentation contract (Thomas, 2026-07-17/18 — binding)

Preview work exists because it is CHEAPER than target-quality work: a
preview pass over the whole required scope completes sooner than the full
render — **but only if the expensive work is not running at the same time,
competing for the same workers**. The barrier is therefore about WHERE
COMPUTE GOES, never about what may be shown:

1. **Two strictly ordered compute phases.**
   *Phase 1 (coverage):* preview/floor evaluation for every required tile,
   rough level statistics, and the single rough histogram publication.
   Canonical materialized pages retain bounded, source-area-weighted
   summaries. Incumbent backends aggregate those summaries across the ADR 0056
   non-overlapping coverage frontier in the existing worker task. The wgpu
   backend instead dispatches the same frontier over the committed plane's
   physically resident pages; a coverage-lane worker waits the submission's
   completion token, resolves the bounded readback, and installs it through
   the same `MontageLevelTracker`/publication machinery. No GUI callback scans
   pixels or aggregates bins. Physical page acknowledgement remains immediate;
   the scheduling-policy owner keeps phase 1 open on an explicit evidence
   barrier until the matching fenced evidence is installed through the existing
   level tracker and its rough publication commits. Display-ready RGB is exempt;
   windowable RGB samples its resident scalar signal alongside scalar and
   complex planes. Stronger semantic evidence already attached to a preview is
   never replaced.
   A parent remains the sole contributor until its complete finer cover is
   available. Presentation construction binds every ordered upsert and
   retained active payload to the accepted level generation before physical
   draw; the gateway rejects a stale crossing.
   *Phase 2 (refinement):* target-quality evaluation, refined level
   sampling, refined histogram updates, and every refinement-adjacent job.
   **Nothing from phase 2 is submitted, scheduled, or evaluated until
   phase 1 has completed for the whole required scope.** Running the two
   in parallel silently destroys preview's entire reason to exist
   (2026-07-18 field report: parallel exact work made scroll shuffles a
   slideshow).
2. **Presentation never withholds better data.** If a better payload is
   ready, show it immediately — even if that upgrades tiles "out of
   order", even if the whole frame jumps to final quality at once. An
   instant all-at-once upgrade from ready resident data is the ideal
   outcome, not a defect. There is NO quality-ordering gate at commit;
   pacing comes only from the ordinary bounded-batch commit caps.
   (A commit-side refinement-withholding gate was built 2026-07-17 and
   rejected 2026-07-18 — see the graveyard.)
3. **Every unit of work obeys the priority system** — inside phase 1 and
   inside phase 2 alike, ordering is the canonical tile priority
   (viewport distance from the CURRENT camera), re-targeted on every view
   change (user gesture or auto-fit). Inside phase 1, already-covered
   tiles never outrank missing tiles. The final presentation boundary
   reasserts this order and carries its immutable rank snapshot with the
   delta; later camera callbacks cannot rewrite the evidence for an already
   emitted transaction.
4. **Quality upgrades never blank.** Refinement replaces pixels atomically
   per tile; eviction returns to coarse pixels, never black (ADR 0056).
5. **`ProgressiveSchedulingPolicy` is the one phase owner.** Each frame
   session gives it the required lifecycle scope (slot plus semantic source)
   and reads its immutable `SchedulingVerdict`; no ladder, admission wave,
   kernel lane, level/histogram producer, atomic transaction, or commit batch
   may derive coverage state independently. The machine opens `COVERAGE` for
   a progressive required-scope generation and closes it only when
   `TileLifecycle.first_pixels_presented(required_tiles)` is true. That close
   edge changes the verdict to `REFINE` and owns the single refinement
   replan. Shader-windowing scopes always open `COVERAGE`; CPU-LUT scopes do
   so when resident LOD makes them progressive. A stuck-open phase silently
   starves refinement forever (PyQtGraph
   zoom regression, 2026-07-18: LOD never upgraded until a reindex).
6. **CPU-LUT windowing (PyQtGraph):** levels bake into pixels at commit, so
   level phasing is bounded, never per-batch (the auto-levels crawl stays
   forbidden). A cold scope at or below one refined evidence batch
   (`MONTAGE_LEVEL_STATS_FIRST_CPU_BATCH` sources) keeps single-pass
   semantics: final levels are decided before the first tile renders.
   A larger cold scope must not hold every evaluated floor hostage to the
   full evidence sweep (montage-entry blackout, 2026-07-18): its first
   pixels window with the refined first batch as a provisional source
   (rank `MONTAGE_VISIBLE_SUBSET`, published to the histogram/levels
   widgets with those pixels), and the full-population sweep then delivers
   exactly one refined re-window through the settled-metadata refresh.
   The visible-dependency evidence producers run at the same INTERACTIVE
   priority as the tiles they gate — evidence queued behind the fill it
   unblocks is a self-inflicted wait. Rough→refined *preview* level
   phasing remains shader-windowing (VisPy) behavior only.

Work classification follows the presentation dependency, not a historical
function or lane name. In particular, PyQtGraph semantic level evidence and
histogram aggregation required to make the first CPU-LUT pixels valid are
`COVERAGE` work; the equivalent post-first-pixel updates are `REFINEMENT`.
Both forms ask the policy owner. Presentation remains acknowledge-driven and
bounded on both backends; it never advances the machine from evaluation,
queue, or commit-emission counts.

## Presentation performance contract

- Keep the last valid frame until a replacement is usable.
- Reject stale commits by revision/key.
- Timer callbacks carry explicit session/revision work tokens; timers reschedule bounded work but do
  not establish semantic order.
- Do not clear because an identity is merely unknown.
- Apply backpressure before visible admission; once admitted, visible payloads commit coherently or
  the previous placeholder/retained frame remains in force.
- Bound cold preparation/upload by items, bytes, and elapsed time.
- A VisPy transaction that performs zero texture uploads, zero upload bytes,
  and zero vertex uploads is a resident mapping rebind, not a cold batch. It
  may bypass the item cap so presentation does not withhold ready pixels;
  commits that upload any pixels remain capped.
- Do not count a batch of many tiles as one feedback item.
- Separate submission time, preparation time, upload bytes/time, queue delay, and first-frame/presented age.
- Publish work-graph counters by lane so dropped, superseded, reusable, and budget-blocked work are
  observable separately from backend physical work.
- Run local prefetch/warmup gates before graph admission; rejected optional work must not appear as
  active admitted work.
- Changes to levels/LUT/scale should update uniforms or prepared display state without re-materializing unchanged source pixels.

## Migration direction

`ImageViewShell` is the shared widget contract for controls, histogram, HUD, viewport intent,
interaction state, and display timing. PyQtGraph and VisPy expose concrete `ImageSurface`
implementations with declared capabilities; `DisplayCommitter` commits semantic tiled presentations
directly to that surface contract. The contract also covers camera application, overlay
coordinate mapping, diagnostics, context-loss reset, teardown, interaction-state visual sync, and
declared shared interaction-event ownership.

Since 2026-07-17 the shell is the single owner of the tiled-commit flow
(prepare → backend apply → commit report), ROI/interaction emphasis state
(`_roi_visual_style`), and tiled-layer queries; concrete surfaces implement the
declared backend hooks (`_apply_backend_tiled_presentation`,
`_after_tiled_commit`, `_tiled_presentation_layer`, and the ROI/profile visual
hooks) with scene/texture mechanics only. `ImageView2D` carries the PyQtGraph
tile-layer mechanics; `VisPySurface` no longer constructs a dormant PyQtGraph
tile layer.
