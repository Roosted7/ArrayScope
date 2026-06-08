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
- `arrayscope.operations.pipeline`: immutable NumPy operations plus shape prediction.
- `arrayscope.operations.slabs`: plans and evaluates the smallest exact base-data slab needed
  for image, profile, scalar hover, and export-frame requests.
- `arrayscope.operations.cache`: bounded LRU caches and cache diagnostics for evaluated display
  results.
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
  `arrayscope.ui.docks.inspection`, and `arrayscope.ui.hud`: compact viewer controls, operation discovery,
  ROI inspection controls, and on-canvas pixel
  feedback. They emit user intent and do not own view state.
- `arrayscope.core.view_recipe`: serializes operations, `ViewState`, and display settings for
  full-view restore. It is pure and does not contain dock geometry.
- `arrayscope.window.main.ArrayScopeWindow`: wires Qt signals to state changes, then calls `render()`.
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

User actions update `ViewState` or the operation coordinator. `render()` then:

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
interaction.

Managed panel visibility uses supported ArrayScope paths only: the managed title-bar Hide button,
View menu actions, or `WindowLayoutManager` programmatic methods. Native `QDockWidget.closeEvent`
semantics are not an app-level lifecycle path for managed panels.
The preserve-canvas transaction is best effort because the window manager or compositor may constrain
top-level sizes; ArrayScope requests size correction with `resize()`, escalates once with temporary
QWidget/QWindow min/max size constraints, sends repeated commit pokes when Qt already reports the
target layout, and accepts the remaining error after bounded retries. Detach transitions intentionally
skip the strong fixed-size/nudge escalation so the new detached tool window can map cleanly. Temporary
stdout diagnostics with the `[ArrayScope preserve-canvas]` prefix remain while Wayland behavior is being
debugged.

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
window clears queued work, increments generations, stops polling, and ignores late results. Prefetch
requests are keyed, deduped, bounded, off by default, skipped for operation-backed and montage views,
and counted in cache diagnostics.

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
`MontagePlan` to derive full grid geometry and visible tiles. Visible tiles are evaluated through the
same image snapshot path as normal views, cached individually, and composed into one bounded
`MontageViewportCanvas` with `origin_x`/`origin_y` in full montage coordinates. Interactive montage
display keeps a single `ImageItem`; `DisplayGeometry` maps canvas-local display points through the
canvas origin into the full montage grid before resolving source tile indices. Histogram/ROI sources
are canvas-sized and contain `NaN` for gaps and unloaded regions. Full giant montage allocation is
blocked by render memory estimates and byte-budgeted visible tile selection.

Operation creation is intentionally available from three places:

- dimension chip operation buttons / context menus use the clicked axis;
- the Operations dock Add/Search controls use the last valid operation axis, then the first
  non-display non-singleton axis, then profile, X, Y, then the first valid axis;
- `Ctrl+K` uses the same defaulting, with a focused dimension chip taking precedence.
