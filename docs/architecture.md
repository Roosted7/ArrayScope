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
  views. It maps display points to array indices and profile states using the geometry committed with
  the current image.
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
- `arrayscope.core.memory_policy`: Qt-free runtime memory policy. It samples system total,
  available memory, and process RSS through psutil with a deterministic fallback, then derives visible
  render, montage canvas/tile, image cache, montage tile cache, profile cache, future stage-cache, and
  prefetch budgets from the selected profile plus the per-render hard cap.
- `arrayscope.core.memory_budget`: byte-estimation and formatting helpers only. Runtime budgets are
  owned by `MemoryPolicy`, not static constants.
- `arrayscope.core.runtime_diagnostics`: Qt-free diagnostics snapshots and plain-text formatting for
  memory policy, caches, schedulers, render decisions, montage state, FFT, and operation state.
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
  evaluation status, and diagnostics. Background workers use immutable document snapshots and pure
  evaluation helpers; they must not mutate the live evaluator directly.
- `arrayscope.profiles.model` / `arrayscope.profiles.coordinator`: profile display policy and line
  result orchestration. Display-point to profile-state mapping belongs to `arrayscope.display.geometry`.
- `arrayscope.core.roi`: Qt-free ROI geometry, line/polyline/freehand sampling, masks, and finite-value
  statistics for inspection workflows.
- `arrayscope.core.histograms`: Qt-free histogram and shared-range comparison helpers for ROI values.
- `arrayscope.core.compare`: minimal compatible-layer model used by ROI histogram comparisons.
- `arrayscope.core.window_levels`: decides image window/level reuse or auto-level behavior.
- `arrayscope.display.ImageView2D`, `arrayscope.ui`, and `arrayscope.ui.docks`: Qt display and controls only.
- `arrayscope.ui.dimension_strip`, `arrayscope.ui.display_toolbar`, `arrayscope.ui.command_palette`,
  `arrayscope.ui.diagnostics`, `arrayscope.ui.docks.inspection`, and `arrayscope.ui.hud`: compact
  viewer controls, operation discovery, developer diagnostics, ROI inspection controls, and on-canvas pixel
  feedback. They emit user intent and do not own view state.
- `arrayscope.core.view_recipe`: serializes operations, `ViewState`, and display settings for
  full-view restore. It is pure and does not contain dock geometry.
- `arrayscope.window.main.ArrayScopeWindow`: wires Qt signals to state changes, then either calls
  `render()` directly for immediate/full render workflows or requests an interactive render through
  the render coordinator.
- `arrayscope.window.render_coordinator.RenderCoordinator`: owns high-frequency render request
  coalescing, latest-state flushing, interaction quiet detection, cancellation of stale
  render-dependent work, and deferred side-panel refresh after interactive bursts.
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

Progressive montage rendering has a narrower commit path than full image rendering. The initial
montage viewport composes a bounded canvas, then completed tiles patch that session-owned canvas in
place and flush to screen at a frame cadence. Progressive commits update pixels, display geometry,
axis flips, viewport preservation, and montage loading/skipped overlays; side panels and expensive
dock/profile/ROI refreshes run only on full commits or after the interaction burst is quiet. Montage
tile evaluation uses a dedicated `montage` scheduler lane with two workers, while visible exact image
rendering remains latest-only on the max-1 `visible` lane and prefetch remains idle-only on the
separate `prefetch` lane.

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
directly. It does not schedule scalar evaluation or show an intermediate
“updating” value during normal mouse movement.

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
profile updates, ROI inspection, and prefetch have separate controllers. Visible/profile/ROI pools use
one worker and `start_latest()` replacement groups so newer requests clear queued stale work. Closing a
window clears queued work, increments generations, stops polling, and ignores late results.
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
The window debounces ROI statistics and histogram updates from the current displayed scalar image or
histogram source, then sends settled results back to the dock and overlay. Small snapshots compute
directly; larger ROI/image combinations run through the ROI evaluation controller and commit only if
their ROI/image request key is still current. Extra comparison layers are internal scaffolding for
same-ROI histogram comparison and are not full session/sync support.
ROI sampling is display-space in Phase 4a. Montage histogram sources contain
`NaN` in gaps, so ROI statistics ignore inter-tile spacing. Full nD ROI
back-projection is intentionally deferred.

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
inter-tile gaps. Interactive montage display keeps a single `ImageItem` plus lightweight tile-state
overlays; `DisplayGeometry` maps canvas-local display points through the canvas origin into the full
montage grid before resolving source tile indices, and only loaded tiles produce array/profile
mappings. Missing visible tiles are scheduled sequentially through the one-worker visible controller,
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
