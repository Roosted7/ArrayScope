# Architecture

Main rule: Qt collects intent and displays results; Qt widgets are not the
source of array-view state.

## Ownership

- `arrayscope.core.ViewState`: authoritative current view of the derived array: shape, image
  axes, profile axis, montage axis, slice indices, channel, scale, and per-axis flags.
- `arrayscope.display.slice_engine`: converts `data + ViewState` into display-ready images and
  lines.
- `arrayscope.display.montage`: plans montage tile geometry, identifies visible tiles, composes the
  bounded viewport canvas used by interactive montage display, and keeps the small `make_montage()`
  helper for pure utility/test use.
- `arrayscope.display.geometry`: the pure display-coordinate contract for normal image and montage
  views. It maps stable ViewBox/world points to canvas-local points, tile-local points, array
  indices, and profile states using the geometry committed with the current image.
- `arrayscope.display.layers`: the only owner for adding image-view graphics items to the ViewBox.
  It applies the shared z-order policy for image/tile pixels, ROI graphics, profile markers, montage
  loading overlays, and HUD graphics.
- `arrayscope.display.viewport`: explicit viewport update policy and `ViewportController` for
  preserving, fitting, resetting, and true 1:1 2D ViewBox ranges.
- `arrayscope.operations.pipeline`: immutable NumPy operations plus shape, dtype, and capability
  declarations. Each operation owns its blocking axes, chunkable axes, request expansion behavior,
  temporary multiplier, stage-cache preference, and fusion eligibility.
- `arrayscope.operations.optimizer`: Qt-free internal runtime operation simplifier. It produces an
  optimized execution plan for enabled operations without mutating `ArrayDocument.steps`, recipes, row
  IDs, undo/redo history, or operation-dock display. It preserves derived shape and dtype, including
  dtype adapters for optimized FFT/IFFT pairs.
- `arrayscope.operations.capabilities`: Qt-free operation capability vocabulary consumed by cost and
  planner code.
- `arrayscope.operations.cost`: Qt-free operation kind, output dtype, output shape, and conservative
  memory-cost estimates for operation stacks. These estimates are derived from operation-declared
  capabilities and feed warnings, diagnostics, visible render decisions, and cost-aware prefetch gates.
- `arrayscope.operations.regions` / `arrayscope.operations.planner`: Qt-free runtime region and
  stage-planning infrastructure. Display/profile/scalar/export slab evaluation uses optimized
  `RegionPlan` transitions; the same plans expose final/required regions, optimization summaries, and
  candidate stage-cache points.
- `arrayscope.operations.stage_cache`: Qt-free in-memory cache for expanded operation-stage arrays.
  It is owned by `OperationEvaluator`, keyed by document identity/revision, operation prefix, region,
  dtype, and shape, and budgeted by `MemoryPolicy.stage_cache_budget_bytes`. Planner candidates carry
  a retained/skipped flag: slab execution stores retained useful candidates by default and falls back
  to an earlier fitting candidate when the preferred retained stage is oversized.
- `arrayscope.operations.stage_materialization`: Qt-free singleflight coordinator for explicit
  reusable stage materialization. It is owned by `OperationEvaluator`, deduplicates in-flight
  expanded stage requests by `StageKey`, records stage decision diagnostics, and refuses oversized
  candidates without forcing them into `StageCache`.
- `arrayscope.operations.chunked_stage`: Qt-free reusable-stage materialization helper that chooses
  allowed chunk axes independently from view image axes, then chunks only over operation non-blocking
  axes. Blocking axes such as FFT axes remain complete in every chunk. A reusable stage that already
  fits the stage-cache budget is materialized unchunked by default because chunking reduces peak
  memory but can be slower when the same full stage must still be stored.
- `arrayscope.core.memory_policy`: Qt-free runtime memory policy. It samples system total,
  available memory, and process RSS through psutil with a deterministic fallback, then derives visible
  render, montage canvas/tile, image cache, montage tile cache, profile cache, future stage-cache, and
  prefetch budgets from the selected profile plus the per-render hard cap.
- `arrayscope.core.memory_budget`: byte-estimation and formatting helpers only. Runtime budgets are
  owned by `MemoryPolicy`, not static constants.
- `arrayscope.core.runtime_diagnostics`: Qt-free diagnostics snapshots and plain-text formatting for
  memory policy, caches, schedulers, render decisions, montage state, FFT, and operation state.
- `arrayscope.core.compute_policy`: Qt-free lane worker policy. It derives per-lane Qt worker counts
  and FFT worker counts from runtime settings so visible/stage work can use capped multi-worker FFTs
  while montage tile and prefetch lanes avoid native-worker oversubscription.
- `arrayscope.core.resource_telemetry` and `arrayscope.core.resource_governor`: Qt-free adaptive
  scheduling inputs and decisions. Resource telemetry samples system/process CPU and memory without
  blocking the UI. The resource governor combines `ComputePolicy`, `MemoryPolicy`, resource
  telemetry, scheduler busy state, and UI latency feedback to choose effective lane worker counts,
  callback fan-in budgets, commit intervals, and prefetch admission. It is bounded and damped:
  worker changes step gradually, UI pressure backs off immediately, and recovery requires healthy
  feedback.
- `arrayscope.operations.fft_backend`: FFT backend abstraction and worker-count runtime settings.
  `auto` resolves to SciPy when available, with NumPy fallback and an optional pyFFTW backend when
  explicitly selected and importable. The centered FFT/IFFT naming follows ArrayScope's MRI/k-space
  convention: centered FFT uses an inverse FFT internally, and centered IFFT uses a forward FFT.
- `arrayscope.operations.slabs`: builds planner-backed slab requests and executes `RegionPlan`
  transitions for image, profile, scalar hover, and export-frame requests. It can look up and store
  StageCache entries at planner candidate boundaries. Registered operation region expansion belongs to
  operation-owned region methods, not ad hoc slab branches.
- `arrayscope.operations.cache`: bounded LRU caches and cache diagnostics for evaluated display
  results. Image views/export frames, montage tile payloads, and profile/scalar results use separate
  budgets supplied by `MemoryPolicy`.
- `arrayscope.operations.pipeline.OperationStep`: ordered operation rows with enable/disable
  state. `ArrayDocument.operations` is the active enabled operation sequence; `steps` is the
  pipeline UI/document sequence.
- `arrayscope.operations.pipeline.ArrayDocument.revision`: explicit input-data revision token used
  in evaluator cache keys. Base arrays are treated as immutable until the owner calls
  `notify_data_changed()` or replaces the data.
- `arrayscope.operations.coordinator`: owns the operation document, evaluator, stack edits,
  and materialization.
- `arrayscope.operations.evaluator.OperationEvaluator`: UI-thread owner of display/profile caches,
  stage cache/materialization, evaluation status, and diagnostics. Background workers use immutable
  document snapshots and pure evaluation helpers; they must not mutate the live evaluator directly.
- `arrayscope.profiles.model` / `arrayscope.profiles.coordinator`: profile display policy and line
  result orchestration. Display-point to profile-state mapping belongs to `arrayscope.display.geometry`.
- `arrayscope.core.roi`: Qt-free ROI geometry, line/polyline/freehand sampling, masks, and finite-value
  statistics for inspection workflows.
- `arrayscope.core.histograms`: Qt-free histogram and shared-range comparison helpers for ROI values.
- `arrayscope.core.compare`: minimal compatible-layer model used by ROI histogram comparisons.
- `arrayscope.core.window_levels`: decides image window/level reuse or auto-level behavior.
- `arrayscope.display.ImageView2D`, `arrayscope.ui`, and `arrayscope.ui.docks`: Qt display and controls only.
  Progressive montage uses explicit raster or tiled presentation APIs that preserve caller-decided
  levels, histogram range, transform, and viewport instead of asking a widget to infer semantic state
  from partial pixels. Complex/RGB windowability is explicit display metadata; RGB array rank alone
  does not mean pixels are already windowed. PyQtGraph paints typed tiled payloads with per-tile
  `ImageItem`s; VisPy paints the same payload contract with a batched atlas visual. The committed
  frame's `FrameValueSource`, never a compatibility placeholder, owns hover/value semantics.
- `arrayscope.display.backend_contract`: backend capability declarations used by presentation planning.
  Shared code asks whether a backend supports direct tiled payloads, persistent residency, shader
  windowing, or native pointer interaction; it does not branch on backend class names. The intended
  long-term shape is a shared widget/interaction shell composed with a thin pixel-rendering backend,
  as recorded in ADR 0038.
- `arrayscope.display.histogram_controller` and `arrayscope.display.image_upload`: focused display
  helpers for histogram preview/final level interaction, ImageItem upload preparation, and shared
  RGB/complex windowing. Histogram level drags update display pixels as a throttled preview, while
  `userLevelsChanged` remains the semantic final user edit signal emitted on drag finish.
- `arrayscope.display.shader_mapping`: pure NumPy shader-equivalent display mapping model. VisPy
  raster and tiled paths can upload raw scalar `float32` or raw complex `complex64`/`RG32F` texture
  planes, then apply component extraction, phase LUT color, linear/log/symlog scale, and
  window/level mapping in shader code. PyQtGraph remains CPU display-prepared and uses the same pure
  mapping functions as the oracle/fallback path. Histograms and levels are computed from semantic
  CPU samples of the same scalar field the shader windows, not from rendered RGB pixels.
- `arrayscope.display.lod`: CPU-side montage tile LOD helpers. LOD selection is based on current
  view range and viewport size; finite-aware 2x2 reduction builds uploaded texture planes, and
  guttering duplicates edge texels for linear-filtered tile seams. LOD texture payloads never own
  hover, ROI, profile, or export semantics; exact semantic data and semantic histogram data stay with
  `DisplayTilePayload`.
- `arrayscope.display.backends.pyqtgraph.tiles`: Qt display helper owned by `ImageView2D` for exact
  per-tile montage painting. It keeps per-item source, histogram, local-rect, level, and RGB-windowing
  state so known-clean tile-layer flushes skip pixel uploads entirely, dirty flushes update only the
  affected tile items, all-cached newly composed sessions can reuse unchanged rendered tile sources,
  and RGB/complex level changes reuse cached float32 tile bases.
- `arrayscope.display.backends.vispy.tiles`: VisPy display helper owned by `VisPyImageView2D` for
  first-class tiled montage painting. It applies revisioned `TilePresentationDelta` updates to
  persistent tiled state, assigns stable source-keyed atlas slots across multiple pages, and maps the
  current active tile numbers onto those resident source slots for drawing. Tile numbers are geometry,
  not GPU content identity: shifted index windows must reuse already-resident semantic sources without
  re-uploading pixels. Inactive tiles remain GPU-resident until byte-budgeted LRU pressure requires
  eviction, and viewport-near resident sources are protected ahead of farther inactive sources. It
  queries runtime texture limits, allocates only the scalar, complex RG, or color planes required by
  the presentation, and uses tile-sized staging arrays rather than full CPU shadow atlases. One
  batched visual draws each active page; level/scale/LUT changes update uniforms, clean commits skip
  texture uploads, and dirty commits upload only changed atlas regions. Cached tiled sessions can seed
  materialized payload wrappers from the previous committed tiled frame so GPU-resident LOD textures
  do not rebuild CPU pyramids just to confirm that no upload is needed.
- `arrayscope.display.overlay_hit_test`: Qt-free ROI/profile handle and outline hit testing shared by
  backend visual adapters. Interaction semantics must move here or into a future shared interaction
  controller rather than being reimplemented by each graphics library.
- `arrayscope.display.overlays`, `arrayscope.display.roi_items`, and
  `arrayscope.display.profile_marker`: focused Qt display helpers for montage/loading overlays, ROI
  graphics item conversion, ROI info panels, and profile marker bounds. `ImageView2D` keeps the
  widget-facing API and delegates concrete helper ownership to these modules.
- `arrayscope.display.planning`: Qt-free display presentation decisions. It normalizes
  window/level bounds, keeps display levels separate from histogram/data ranges, accepts accumulated
  semantic montage tile coverage as provisional level sources, and is the only place that chooses
  committed display levels for normal, degraded, initial montage, progressive montage, and explicit
  Auto Window commits. Relative mode preserves level fractions across normal, montage, cached, and
  uncached semantic source changes; absolute mode preserves numeric levels while histogram metadata
  may improve.
- `arrayscope.display.model.commit`: Qt-free immutable request, payload, presentation-decision, and
  commit-plan models used at the boundary between render orchestration and display mutation. Render
  orchestration provides a `PresentationInput`; presentation policy returns a `PresentationDecision`;
  Qt code only receives the decided `DisplayPresentation`. `DisplayRasterPresentation` owns raster
  pixels and `DisplayTiledPresentation` owns typed `DisplayTilePayload` mappings; they are distinct
  first-class variants, so core code never represents a tiled montage as a fake giant raster.
- `arrayscope.display.commit`: the single gateway from a decided presentation to
  `ImageView2D`. Window render code must not call image pixel or histogram setters directly.
  `DisplayCommitter` validates display shape, local histogram-source shape, optional sampled
  histogram plot sources, finite increasing levels/histogram ranges, and forwards montage dirty-tile
  metadata only to the tile-layer presentation path before mutating Qt state. Committed frames store a
  `FrameValueSource`: `CanvasValueSource` reads normal/viewport-canvas pixels, while
  `TiledValueSource` reads typed tile payloads for hover/status and demand ROI/profile regions.
- `arrayscope.window.diagnostics_snapshot`: window-owned runtime diagnostics snapshot construction.
  Menu code opens the diagnostics dialog only; it does not assemble scheduler, cache, render, montage,
  operation, or upload timing snapshots.
- `arrayscope.window.montage_backend`: explicit montage display backend policy. The Performance menu
  stores Auto, Tile layer, or Canvas fallback. Auto always sends large RGB/complex and previously slow
  upload paths to tiled painting; a backend that declares a tiled-montage preference also receives
  large scalar montages through that path. The chosen backend, reason, fallback, and any warning are
  recorded in diagnostics.
- `arrayscope.window.stage_warmup`: idle-only reusable stage warmup. It uses the stage-cache budget,
  schedules on the stage lane with a stage `EvaluationContext`, attaches to existing singleflight
  requests, and records the latest warmup decision for diagnostics.
- `arrayscope.window.montage_prefetch`: stage-aware rendered montage tile prefetch. It predicts
  nearby tiles after visible montage commits, schedules only on the prefetch lane, requires expensive
  operation stages to be cached or in-flight, and refuses paths that would recompute the same FFT once
  per predicted tile.
- `arrayscope.display.model.frame`: committed display-frame keys and value source ownership for
  hover/status. Tiled frames must not report values from placeholder arrays; they resolve values and
  tile-local regions through their `TiledValueSource`.
- `arrayscope.window.montage_levels`: semantic montage histogram coverage tracking keyed by montage
  scope, independent of viewport origin, visible canvas shape, and the current tiled index window.
  It stores deterministic sampled per-source-index tile stats so shifted tiled ranges reuse overlap
  while excluding removed indices.
- `arrayscope.window.montage_renderer`, `arrayscope.window.normal_renderer`, and
  `arrayscope.window.viewport_bridge`: focused ownership modules for montage orchestration, normal
  image orchestration, and ViewBox range events. `RenderMixin` keeps high-level interaction/status
  glue and delegates actual visible-image paths to these modules.
- `arrayscope.ui.dimension_strip`, `arrayscope.ui.display_toolbar`, `arrayscope.ui.command_palette`,
  `arrayscope.ui.diagnostics`, `arrayscope.ui.docks.inspection`, and `arrayscope.ui.hud`: compact
  viewer controls, operation discovery, developer diagnostics, ROI inspection controls, and on-canvas pixel
  feedback. They emit user intent and do not own view state.
- `arrayscope.core.view_recipe`: serializes operations, `ViewState`, and display settings for
  full-view restore. It is pure and does not contain dock geometry.
- `arrayscope.window.main.ArrayScopeWindow`: wires Qt signals to state changes, then either calls
  `render()` directly for immediate/full render workflows or requests an interactive render through
  the render coordinator. It owns the committed display frame used for pixel hover/status values:
  displayed data, optional histogram source, display geometry, window levels, document key, request
  key, render generation, and optional montage level key.
- `arrayscope.window.render_coordinator.RenderCoordinator`: owns high-frequency render request
  coalescing, latest-state flushing, interaction quiet detection, cancellation of stale
  render-dependent work, and deferred side-panel refresh after interactive bursts.
- `arrayscope.window.render_generation.RenderGeneration`: per-window visible-output generation guard.
  Render requests and visible-output state changes advance the generation; async callbacks commit only
  when their captured generation and current evaluator/session key still match current viewer state.
- `arrayscope.window.evaluation_controller`: owns categorized background display/profile/ROI/prefetch
  dispatch, latest-only replacement groups, local thread pools, queue clearing, cancellation tokens,
  and stale-result ignoring.
- `arrayscope.window.panels.PanelManager`: owns managed panel state and panel body reparenting. Docked
  panels use `QDockWidget`; detached panels use `QDialog`/tool windows with a `startSystemMove()` move
  handle. Hidden and detached panels are removed from the `QMainWindow` dock layout so they cannot
  leave stale minimum-size constraints behind. Hidden panels keep their body in the hidden dock;
  detached panels keep their body in the dialog. `panel.dialog` is `None` unless the panel location is
  `DETACHED`, and `StandardDockWidget` has no custom close lifecycle override.
- `arrayscope.window.layout_controller.WindowLayoutManager`: owns first-run layout restore, reset
  layout, progressive panel visibility, managed panel menu actions, dock default sizes, shutdown dock
  closing, and panel transition geometry. Panel show/hide/detach/redock preserves the central viewer
  size with a post-layout transaction: it records the central widget size, applies the panel change,
  then corrects the top-level size with `resize()` over a short `QTimer` retry loop. The final retry
  temporarily fixes the top-level `QWidget` and `QWindow` minimum and maximum size to make the Wayland
  size request harder for the compositor to ignore, then restores the original constraints. When Qt
  reports success without a remaining central-widget delta, the transaction still briefly advertises
  the current top-level size as fixed and repeats QWidget/QWindow resize/update requests as Wayland
  commit pokes. It does not call
  `setGeometry()` or intentionally move the window position; users can turn this best-effort behavior
  off in the View menu. Managed panels still avoid native `QDockWidget` floating state.
- `arrayscope.app.launch`: QApplication creation, multiprocessing launch, and IPython Qt event-loop handling.
- `arrayscope.io`: file loading, dataset selectors, and save workflows.
- `arrayscope.export`: video/frame export workers and UI workflow.

The public package surface is intentionally small: users should prefer
`import arrayscope as asc` followed by `asc(data)`. Internal code should import
concrete submodules rather than relying on package-root re-exports.

## Render Flow

User actions update `ViewState` or the operation coordinator. `render()` is the immediate render
execution primitive for initial render, operation/data changes, channel/scale changes, tests, and
other non-high-frequency workflows. High-frequency slice interaction updates `ViewState` and visible
slice controls immediately, then calls `request_render(..., interactive=True)`. The
`RenderCoordinator` coalesces those requests at a short frame cadence, renders only the latest state,
cancels stale visible/profile/ROI/pixel/prefetch work, and refreshes side panels once the interaction
burst is quiet.

Progressive montage rendering has a narrower commit path than full image rendering. Presentation
policy chooses a raster canvas or a direct tiled presentation. Raster sessions patch a bounded
session-owned canvas. Tiled sessions retain typed payload/source wrappers and progressively commit the
currently loaded mapping; renderer compatibility placeholders are created only inside the legacy
widget shell. A successful commit records a `CanvasValueSource` or `TiledValueSource` as the semantic
hover/ROI/profile source. Progressive commits update the selected pixel representation, display
geometry, axis flips, viewport preservation, committed frame, and loading/skipped overlays; side
panels and expensive dock/profile/ROI refreshes run only on full commits or after the interaction
burst is quiet. Montage tile evaluation uses a dedicated `montage` scheduler lane sized by
`ComputePolicy` (auto uses roughly half the CPU, capped, with one FFT worker per tile). Shared expanded
operation stages use a
separate max-1 `stage` lane and are materialized before most cold dependent tiles are rendered; one
lead visible tile may render directly while a cold stage warms so the montage does not appear frozen
behind a single reusable-stage job. If an attached stage is no longer cached or in-flight, waiting
tiles fall back to direct tile evaluation instead of remaining in a loading state. Visible
exact image rendering remains latest-only on the max-1 `visible` lane and prefetch remains idle-only
on the separate `prefetch` lane. `ComputePolicy` supplies these lane widths and the FFT worker count
attached to each `EvaluationContext`: visible and stage lanes can use the capped runtime FFT worker
count (auto resolves to up to half the machine, capped at eight workers), while montage tile and
prefetch lanes use one FFT worker by default.

UI-thread fan-in is governed by the resource governor, backed by `LatencyFeedbackController`, not
fixed duration tiers. Subsystems record measured costs under named channels and ask the governor for
a work budget, batch limit, or commit interval. Montage uses `montage_tile_result` to decide how many
completed tile results to patch in one UI tick and `montage_commit` to decide how quickly to flush
canvas/display updates. Histogram preview, ROI refresh, live profile updates, pixel hover, and
diagnostics callbacks use the same feedback model where they have explicit debounce or callback
limits. This keeps worker throughput high while adapting UI-thread fan-in to recent cost, profile
tuning, memory pressure, CPU headroom, and interactive vs idle state.

Display upload is a UI-thread phase separate from render/evaluation. `ImageView2D` measures visible
image upload, histogram plot upload, histogram recompute, RGB re-windowing, level synchronization, and
profile-bound updates so runtime diagnostics can distinguish slow evaluation from slow Qt painting.
Histogram image-item binding is idempotent: pyqtgraph connects image-change signals in
`setImageItem()`, so ArrayScope routes all histogram item switches through one helper and explicitly
refreshes the histogram plot once per committed state. User histogram drags have a preview/final
split: preview updates are throttled to the visible display path, while the semantic level state is
emitted once on drag finish. PyQtGraph may re-window its visible RGB tile items from bounded retained
float bases; VisPy changes shader uniforms without uploading clean tile pixels.

Progressive tiled commits are coalesced when upload is slow and carry a revisioned
`TilePresentationDelta`: upserts, removals, active tiles, planned visible tiles, viewport-near tiles,
and level/histogram/viewport revisions. The committed `TilePresentationState` owns semantic payloads
for hover, ROI, and profile reads; renderers apply only the delta needed for upload and draw state.
The VisPy atlas keeps stable source-keyed slot ownership and separates active draw visibility from
retained GPU residency. It reserves the complete non-skipped visible plan when budget allows, falls
back to active capacity when the plan is larger than the derived budget, and evicts far inactive
sources before viewport-near inactive sources. Diagnostics report visible, resident, capacity, pages,
active pages, budget, device texture limit, near/warm residency, updated/skipped, texture and vertex
submissions, uploaded bytes, storage rebuilds/evictions, estimated GPU bytes, CPU shadow bytes, and
capacity warnings. Clean commits, pan/zoom-only updates, and shifted active tile windows whose sources
are already resident should report zero texture uploads.

Predictive compute is stage-aware and governor-admitted. Stage warmup runs only while visible work is
idle and only when a retained cacheable candidate fits the stage-cache budget. Rendered tile and
next-slice prefetch run only while idle; expensive FFT-backed prefetch is allowed only when the
required reusable stage is already cached or in-flight. Otherwise the prefetch path records a skip
decision instead of computing the same expanded transform separately for each predicted output. During
cold montage redraws, diagnostics distinguish cache hits, direct lead tiles, stage-backed tiles, and
tiles waiting for a shared stage so the operation log does not imply per-tile repetition of the
retained FFT/IFFT stage.

`render()` then:

1. migrates `ViewState` to the current derived shape;
2. syncs controls from `ViewState`;
3. requests image/profile data through the evaluator;
4. applies view-only axis flips;
5. updates docks, labels, HUD text, compact controls, and cache status.

Display images are row-major `(height, width)`. Display `x` is image column,
display `y` is image row, and `ViewState.image_axes` is `(y_axis, x_axis)`.
ROI geometry uses the same display coordinates. Montage tile shapes are also
`(height, width)`. Axis flips are view-only ViewBox inversions and do not change
the array index represented by image-item coordinates.

Every committed 2D image has a matching `DisplayGeometry`. Pixel hover, live
profiles, profile marker clamping, montage tile lookup, and future linked
cursor behavior must use that geometry instead of reconstructing coordinates
from widget state. Display point mapping uses pixel cells: `[x, x+1)` maps to
column `x`, and `[y, y+1)` maps to row `y`. Hover/status context text also
comes from `DisplayGeometry` so montage axes are labelled once.
Pixel hover reads the committed displayed scalar image or histogram source
directly from the window-owned committed display frame. For montage, the
committed frame is indexed by display canvas coordinates while tile-local
coordinates remain status/context text. A committed frame is usable only when
its document key, request key, render generation, geometry, and display shape
still match current visible state. Pixel hover does not schedule scalar
evaluation or show an intermediate “updating” value during normal mouse
movement.

Montage window/level state is tracked separately from both the rendered pixel canvas and the
committed value source. `MontageLevelTracker` maintains per-source-index sampled histogram stats with
finite bounds, expected indices for the current tiled range, and a coverage rank: none, visible
subset, complete, or sampled-full. It returns ranked `LevelSource` objects and sampled histogram plot
data to the presentation model. Viewport culling may reduce rendered pixel work only. It must not make
hover semantics stale and must not replace broader semantic window/level bounds with a narrower
visible subset. Shifted tiled ranges reuse retained per-source stats for overlapping indices and
exclude indices that left the range. Partial tiles may display immediately; implicit relative
windowing maps existing level fractions onto improving semantic histogram ranges, while absolute mode
keeps numeric levels. Explicit Auto Window uses the best semantic bounds currently available.
Degenerate bounds are normalized before display, so zero-width windows are never committed.
User-edited levels are stored as relative intent or absolute locks according to the active window mode,
not rediscovered from the histogram widget during rendering.

The committed display frame has two validity uses. Hover/status reads use strict current-frame
validity: document key, request key, render generation, geometry, and display shape must still match
the visible state. Window/level history uses a relaxed validity check so render generation advances
and in-flight uncached renders do not erase the previous committed levels before replacement pixels
arrive.

Do not read widget values to reconstruct `ViewState`. Widget state is an output
of render, except transient UI-only state such as dock visibility and histogram
interaction. The fast slice path is still a projection of `ViewState`: it updates only the active
axis controls immediately after mutating `ViewState`, before the coalesced render catches up.

Managed panel visibility uses supported ArrayScope paths only: the managed title-bar Hide button,
View menu actions, or `WindowLayoutManager` programmatic methods. Native `QDockWidget.closeEvent`
semantics are not an app-level lifecycle path for managed panels.
Preserve-canvas panel transactions are owned by
`arrayscope.window.canvas_preserve.CanvasPreserveController`; `WindowLayoutManager` delegates panel
size transitions to it. `PanelResizeBehavior` supports `off`, `best_effort`, and `strong_wayland`.
Best effort requests top-level size correction with `resize()` and accepts bounded compositor error.
Strong Wayland is explicit, Wayland-gated, and may temporarily apply captured QWidget/QWindow min/max
constraints plus commit pokes/nudges when ordinary correction does not settle. Detach transitions skip
the strong path so detached tool windows can map cleanly. Preserve-canvas diagnostics are exposed in
Developer -> Diagnostics; no stdout preserve diagnostics are emitted by default.

The window tracks derived-array metadata from `ArrayDocument.current_shape` and dtype estimates.
It must not materialize the derived array as normal state. Full derived evaluation is reserved for
explicit materialize/save/export actions. Display evaluation uses slab-first requests through
`OperationEvaluator`; slow Qt-visible requests run through `EvaluationController` so the previous
image/profile remains visible and stale worker results cannot overwrite newer user intent. Workers
capture immutable `ArrayDocument`/`ViewState` snapshots and return results to the UI thread; only the
UI thread commits cache/status updates to the live evaluator. Visible image, profile, pixel,
montage, and prefetch callbacks compare full evaluator request keys rather than only document keys,
so stale work for the same document but a different `ViewState` cannot replace newer user intent.

Viewport changes are explicit. `ViewportController` tracks untouched, user,
locked-fit, and one-to-one modes. Normal renders preserve the current ViewBox range.
The first image fits, display-shape changes fit only while untouched or while locked Fit is enabled,
Fit disables pan and zoom, and explicit 1:1 computes a range where one image pixel maps to one
viewport pixel.

File reload is distinct from data mutation and replacement. In-place mutation
uses `mark_base_data_changed()` / `notify_data_changed()` and preserves
operations while incrementing the revision. File reload uses
`reload_base_data(..., preserve_steps=True)` and preserves compatible operation
stacks. Explicit replacement/materialization uses
`replace_base_and_clear_steps()`.

Background evaluation uses categorized local per-window `QThreadPool` instances. Visible rendering,
profile updates, stage materialization, montage tiles, ROI inspection, and prefetch have separate
controllers. Visible/profile/ROI/stage pools use one worker and `start_latest()` replacement groups so
newer requests clear queued stale work. Clearing a replacement group advances that group's generation
even if no replacement job is submitted. Closing a window clears queued work, increments generations,
stops polling, and ignores late results.
Cancellation tokens are checked before/after major evaluation steps and between chunks. Visible image
rendering uses a cost-aware decision before work is submitted: use cache, run exact async, run exact in
cooperative chunks, show a marked degraded preview, or refuse while keeping the previous image visible.
Degraded previews are not stored in the exact image cache. Chunking is only across independent
output/display axes; a single FFT axis is not split, so one SciPy/pyFFTW FFT call is still not
cancellable mid-call. Prefetch requests are keyed, deduped, bounded, off by default, idle-only, skipped
while visible work is busy, skipped for montage, and allowed for operation-backed views only when cost
estimates are below conservative thresholds. Cache diagnostics include hit rate, prefetch outcomes,
render refusal/degraded/chunk counters, and scheduler pending/running/stale/cancelled counters.
Phase 4h timing diagnostics are internal developer diagnostics only. They sample synchronous render
orchestration, planning, worker queue wait, evaluation, display commit, image setting, levels/histogram
work, operation-dock refresh, inspection refresh, montage tile evaluation, tile and stage cache lookup,
montage canvas composition/patch/commit, and montage overlay updates; they do not define public API or
user-facing behavior.
Render coalescer diagnostics are likewise internal and report pending/interacting request state plus
requested/flushed/coalesced/deferred-refresh counters.
The Developer -> Diagnostics dialog is a plain `QDialog`, not a managed dock, so it does not
participate in panel layout or canvas-preservation transactions. It shows color-coded filling bars
for memory/cache budget usage and compact text sections for deeper state. The Operations panel does
not show cache summaries; cache detail lives in Diagnostics.

App settings include theme, nearby-slice prefetch, panel resize behavior, FFT backend, FFT worker count,
memory profile, and render memory budget. The render memory budget is a per-render hard cap for
visible image and interactive montage tile/canvas guardrails. Cache and prefetch budgets adapt from
the selected memory profile and sampled system memory. StageCache is in-memory only and uses a
score-based retention policy that favors reusable visible, high-cost, frequently hit stages while
penalizing large and prefetch-only entries; disk/memmap cache is not implemented. Operation
simplification is a runtime/internal execution optimization, not a recipe rewrite or user-facing stack
transformation.

Exact profile work and profile prefetch must not share cleanup-sensitive scheduler bookkeeping.
Live/visible profile requests use the profile evaluation controller; profile prefetch uses the
prefetch controller so `start_latest()` replacement for exact profile work cannot clear unrelated
queued prefetch keys.

Channel mode tracks automatic versus user-selected intent. Invalid channels are
coerced when dtype changes, for example complex-only channels fall back to real
when the derived output becomes real. When output becomes complex, an untouched
default real channel switches to complex display, while a user-selected real
channel remains real.

In development and tests, `ARRAYSCOPE_STRICT_UI=1` makes GUI programming
exceptions log their traceback and re-raise instead of being silently swallowed.

## Interdependency Map

When changing `ViewState`, check:

- `arrayscope.display.slice_engine`
- `arrayscope.profiles`
- `arrayscope.operations.evaluator`
- `arrayscope.operations.coordinator`
- `arrayscope.window.render.RenderMixin.render()`
- `arrayscope.export`
- Qt smoke/artifact tests

When changing `slice_engine`, check:

- image display
- profile extraction
- video export
- window-level behavior

When changing `operation_pipeline`, check:

- operation registry and recipes
- operation dock
- evaluator cache keys
- `ViewState.for_shape()`
- `arrayscope.operations.slabs`

## Placement Guide

- Axis validation: `arrayscope.core.axis_utils`.
- Pure array transforms: `arrayscope.operations.dim_ops` or `arrayscope.operations.pipeline`.
- Display conversion: `arrayscope.display.slice_engine`.
- Colormap creation: `arrayscope.display.colormaps`.
- Window/level decisions: `arrayscope.core.window_levels`.
- User-action orchestration: `arrayscope.window` mixins or a focused coordinator.
- Qt UI controls: `arrayscope.ui` and `arrayscope.ui.docks`.
- Full-view serialization: `arrayscope.core.view_recipe`.

## Progressive Disclosure

The quick-glance layout keeps the central image and histogram first. The Operations dock is hidden
while the operation step list is empty, and it appears automatically after the first operation. That
automatic reveal does not mark the dock as user-pinned; clearing the stack hides it again unless the
user explicitly showed the dock from the View menu. The Profile dock is hidden unless the data is 1D,
live profile is enabled, or the user explicitly shows it.

The Inspection dock is optional and hidden by default. Basic ROI creation and live-profile toggling are
available from the image context menu, so ROI use does not require opening a dock. The Inspection dock
manages analysis and immediate line/rectangle ROI creation; freehand and polyline drawing are one-shot
canvas interactions owned by `ImageView2D`. ImageView2D owns ROI graphics items, emits complete ROI
geometry, and displays a movable semi-transparent ROI info overlay.
The window debounces ROI statistics and histogram updates, then sends settled results back to the dock
and overlay. Normal image snapshots can compute directly from the committed scalar image or histogram
source. Montage ROI work uses demand tile-region requests derived from world-coordinate ROI geometry:
loaded visible canvas data is reused when it covers the region, otherwise rendered-tile, region, and
stage caches are checked before exact tile evaluation is scheduled on the ROI lane. The visible montage
session, viewport canvas, and main-view loading overlays are not mutated by offscreen ROI demand work.
ROI rows remain visible while stale stats are replaced atomically when the current request key matches.
Extra comparison layers are internal scaffolding for same-ROI histogram comparison and are not full
session/sync support.

The Profile dock can plot multiple active profile axes. The window evaluates each profile state through
the existing line evaluator/cache and sends all results to one plot. Complex profile mode is dock-local:
when global channel mode is `complex`, line evaluation preserves complex samples so the Profile dock can
choose magnitude, phase, real, imaginary, or magnitude plus phase strip.

Montage is a display mode driven by range text in a non-image dimension slice field, stored on
`ViewState.montage_axis`. The current image axes remain the tile Y/X axes. Range text on image axes is
stored as per-axis display ranges, allowing image-axis subsetting such as `0:2:100`. Montage uses
`MontagePlan` to derive full grid geometry and visible tiles. The interactive canvas rectangle is based
on the requested viewport clipped to full montage bounds; it is never shrunk to the loaded tiles.
Visible tiles are evaluated through the same image snapshot path as normal views, cached individually,
and composed into one bounded `MontageViewportCanvas` with `origin_x`/`origin_y` in full montage
coordinates. The canvas carries per-tile states (`loaded`, `loading`, `skipped`, `unloaded`) so hover
and live profile can distinguish real data from loading placeholders, hard budget-skipped tiles, and
inter-tile gaps. ViewBox coordinates are full montage world coordinates: the bounded canvas `ImageItem`
is positioned at `canvas.origin_x/origin_y`, and the exact tile-layer mode positions tile `ImageItem`s
at their full montage tile origins. `DisplayGeometry` maps world points to canvas-local points,
tile-local points, and array indices; hover/value lookup requires loaded committed pixels, while
ROI/profile demand mapping can resolve valid offscreen or unloaded tiles. Missing visible tiles are
scheduled sequentially through the one-worker visible controller,
and each completed tile can update the current canvas with visual commits throttled to roughly 30 Hz.
Visible tiles are not skipped merely because there are many of them; progressive rendering evaluates
them one at a time. A tile is skipped only when an individual tile would exceed the visible render
budget, and that path shows a detailed warning with the estimate, budget, tile shape, and recovery
options. Montage tile cache entries store only layout-independent image payloads; the current
`MontagePlan` supplies placement when composing the canvas, so cached pixels cannot carry stale grid
coordinates. Stale tile callbacks are ignored without mutating overlays, geometry, or the current
canvas. Histogram/ROI sources are canvas-sized and contain `NaN` for gaps and non-loaded regions.
Full giant montage allocation is blocked by render memory estimates and bounded viewport canvas
allocation.

Operation creation is intentionally available from three places:

- dimension chip operation buttons / context menus use the clicked axis;
- the Operations dock Add/Search controls use the last valid operation axis, then the first
  non-display non-singleton axis, then profile, X, Y, then the first valid axis;
- `Ctrl+K` uses the same defaulting, with a focused dimension chip taking precedence.
