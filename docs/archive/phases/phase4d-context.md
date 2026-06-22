# Phase 4d - Responsive, bounded, predictable viewer foundation

## Current state (post Phase 4c)

Hover was mostly fixed.
Dock management was simplified, but still depends on native QDockWidget floating/redocking too much.
Fit/1:1 were made functional, but not modeled as proper viewport modes.
Prefetch was partly restrained, but there is still no real evaluation scheduler.
Montage remains fundamentally all-tiles/all-at-once and can explode memory.
Some docs/roadmap items are marked done even though implementation is partial.

## Goals

Make ArrayScope responsive, bounded, and predictable under real interactive use.

That means:

- No accidental 10 GB allocations.
- No stale render pile-ups.
- No native dock state mysteries.
- No UI control that lies about mode.
- No hidden panel doing expensive work.
- No manual test catching basic invariants.

### Issue analysis

####  The biggest thing we got wrong

We improved individual symptoms without adding the missing central system: a bounded evaluation scheduler.

Right now, when ones scroll through a sliced dimension with operations applied, the viewer can start expensive evaluations for old states. Generation checks can prevent stale results from committing, but they do not stop stale NumPy work from consuming CPU and memory.

That distinction matters.
Dropping stale result after it finishes: protects correctness.
Stopping stale work from running: protects responsiveness and memory.

We mostly have the first. We do not yet have the second.

Qt’s QThreadPool defaults to a thread count based on the machine’s ideal thread count, and setMaxThreadCount() is the intended way to control concurrency. QThreadPool.clear() can remove queued runnables that have not started, but it cannot magically kill already-running NumPy operations.

So for a viewer, the visible-render path should not behave like a normal background batch queue. It should behave like:

- Only the newest visible image matters.
- Cancel/drop queued older visible work.
- Let at most one visible image evaluation run at a time.
- Never let prefetch outrank visible rendering.
- Never let montage allocate beyond a memory budget.

That is Phase 4d’s core.

#### Why scrolling is slow now

1. Visible image renders are not “latest-only” enough

EvaluationController.start() increments a generation, but it does not aggressively clear queued visible image jobs every time a newer slice request arrives. If the user scrolls through 30 slices quickly, old jobs can still run or sit queued.

For normal 2D slicing without operations, that may be acceptable. With operations, it is fatal.

You need a visible-render controller with this semantic:

```
def start_latest_visible(self, fn, *, on_done, on_error):
    self._generation += 1
    self._pool.clear()          # drops queued, not-yet-started old work
    self._pending.clear()
    self._handlers.clear()

    token = CancellationToken()
    self._active_token.cancel()
    self._active_token = token

    self._pool.start(RenderRunnable(fn, token, generation=self._generation))
```

And the pool should be constrained: `self._pool.setMaxThreadCount(1)`

This will not stop a currently-running NumPy call mid-flight, but it prevents a pile-up.

2. Montage is still all-tiles/all-at-once

The current montage render path still does roughly this:

```
for each montage tile up to 256:
    evaluate full tile image
    store tile image in list
    store histogram image in list

allocate one huge montage image
copy all tiles into it
allocate histogram montage
copy all histogram tiles into it
display final giant image
```

That is exactly how a 0.5 GB dataset can lead to 10+ GB resident memory.

Example memory multipliers:

- base data
- operation temporary arrays
- per-tile rendered arrays
- per-tile histogram source arrays
- final montage array
- final histogram montage array
- complex display temporaries
- finite/histogram masks
- stale montage worker still running while a newer one starts

Even if each single step looks reasonable, the combination is not bounded. Montage needs to stop being “make one big image.” It needs to become a tile renderer.

### Hidden bug: one-pixel image ranges can collapse to 1D

This is a concrete correctness bug in slice_engine.py.

make_image() and make_image_from_slab() still use broad np.squeeze() logic. If a display/image axis range has length 1, the resulting image can collapse from 2D to 1D.

Example:

```
data = np.arange(2 * 3).reshape(2, 3)

state = (
    ViewState.from_shape((2, 3))
    .with_image_axes(0, 1)
    .with_axis_range(0, indices=(0,), text="0")
)

image = make_image(data, state).data
```

Current behavior can become:

```
actual shape:   (3,)
expected shape: (1, 3)
```

That is not a harmless edge case. A one-row image is still an image. Hover, ROI, montage, histogram, and PyQtGraph display code should not suddenly receive a line.

Fix: Stop using blanket squeeze() in display extraction.

The display extraction rule should be:

- Non-display axes are indexed with integers and removed.
- Display axes are always preserved, even when length 1.
- Image output is always 2D: (height, width), or 3D only for RGB/RGBA.
- Line output is always 1D.

The helper should look conceptually like this:

```
def _slice_to_display_axes(data, state, display_axes):
    index = []
    present_axes = []

    for axis in range(state.ndim):
        if axis in display_axes:
            index.append(slice(None))
            present_axes.append(axis)
        else:
            index.append(int(state.slice_indices[axis]))

    result = np.asarray(data[tuple(index)])
    return result, tuple(present_axes)


def _apply_display_axis_ranges(data, state, present_axes):
    result = data

    for result_axis, original_axis in enumerate(present_axes):
        indices = state.axis_range_indices[original_axis]
        if indices is not None:
            result = np.take(result, indices, axis=result_axis)

    return result
```

Then explicitly assert: `assert image_data.ndim == 2`

Add regression tests immediately:

```
def test_make_image_preserves_one_row_axis_range():
    data = np.arange(2 * 3).reshape(2, 3)
    state = (
        ViewState.from_shape(data.shape)
        .with_image_axes(0, 1)
        .with_axis_range(0, indices=(0,), text="0")
    )

    image = make_image(data, state)

    assert image.data.shape == (1, 3)
    np.testing.assert_array_equal(image.data, data[[0], :])
```

Also add the same for make_image_from_slab().

### The dock situation: stop relying on floating QDockWidget as the primary design

On Wayland, applications cannot freely move top-level windows by setting positions the old X11 way; the compositor owns that. Qt’s QWindow::startSystemMove() is the official way to ask the platform to start a native move operation.

Qt’s QDockWidget does support closable, movable, and floatable dock widgets, but that model does not map cleanly to all Wayland compositors. KDAB’s KDDockWidgets documentation explicitly discusses this Wayland mismatch and describes a two-titlebar approach because one titlebar cannot cleanly serve both as a dock-drag area and as a native top-level move area on Wayland.

So the answer is not “add more visibilityChanged hacks.” That made earlier versions worse.

Best design: managed panels, not native floating docks. Like:

```
Docked panel:
  QDockWidget inside QMainWindow.

Detached panel:
  QDialog / tool window containing the same panel body widget.

Hidden panel:
  Body is owned but not visible.
```

In other words, QDockWidget is used for docking. QDialog is used for floating.

That avoids the broken middle ground where a floating QDockWidget is half dock item, half top-level window, and your layout manager tries to infer what happened after the fact.

#### Proposed model

```
class PanelLocation(Enum):
    HIDDEN = "hidden"
    DOCKED = "docked"
    DETACHED = "detached"


@dataclass
class ManagedPanel:
    name: str
    dock: QtWidgets.QDockWidget
    body: QtWidgets.QWidget
    dialog: QtWidgets.QDialog | None
    location: PanelLocation
    last_dock_area: QtCore.Qt.DockWidgetArea
    user_visible: bool | None = None
```

Operations:

```
show_docked(panel)
hide_panel(panel)
detach_panel(panel)
redock_panel(panel)
toggle_panel(panel)
reset_panel_layout()
```

Every operation goes through PanelManager.

No native close/float/redock should be treated as authoritative. The panel manager is authoritative.

#### Layout behavior

When showing a docked right-side panel:

- increase main-window width by dock width, so canvas size stays constant

When hiding/detaching a right-side panel:

- decrease main-window width by dock width, so canvas size stays constant

When showing a bottom profile dock:

- increase main-window height by dock height

When hiding/detaching bottom profile:

- decrease main-window height by dock height

When detaching:

- hide/remove the dock from QMainWindow
- move the panel body into QDialog
- shrink the main window as if the dock closed

When redocking:

- move the body back into QDockWidget
- show dock
- grow main window as if dock opened
- close detached dialog

This directly fixes your current issue:

- opening grows
- closing/detaching shrinks
- redocking grows
- detached windows move natively
- Wayland move handle

For detached QDialog, use a custom title/move handle that calls:

```
window = self.window().windowHandle()
if window is not None:
    window.startSystemMove()
```

This is cleaner than trying to make floating QDockWidget move.

### Fit and 1:1: the user model should be explicit

Fit should be a toggle.
1:1 should be a momentary command.

Right now the UI uses toolbar actions that look like static icons/labels. In Qt, toolbar actions commonly become QToolButtons; QToolButton can be checkable and has auto-raise behavior inside a QToolBar. QAction also supports checkable actions, which is exactly what Fit should be.

#### Correct behavior

Fit enabled:

- Fit action checked.
- All content always visible.
- Aspect ratio may stretch.
- Pan disabled.
- Zoom disabled.
- Resize refits.
- New image/slice refits.
- 1:1 action: unchecks Fit first.

Implementation:

```
fit_action.setCheckable(True)
fit_action.toggled.connect(self.set_fit_locked)
def set_fit_locked(self, enabled: bool):
    self.viewport_controller.set_fit_locked(enabled)

    self.img_view.view.setAspectLocked(not enabled)
    self.img_view.view.setMouseEnabled(x=not enabled, y=not enabled)

    if enabled:
        self.img_view.view.autoRange(padding=0)
```

You may also need to block wheel zoom explicitly. setMouseEnabled(False, False) disables mouse-drag interaction, but wheel behavior may need to be ignored in a ViewBox subclass depending on PyQtGraph behavior.

Fit disabled:

- Fit action unchecked.
- Square pixels restored.
- Pan/zoom enabled.
- Current center/range preserved as much as possible.

1:1 pressed: 

- Momentary action.
- Fit unchecked.
- Square pixels enabled.
- Pan/zoom enabled.
- View range set so one image pixel maps to one screen pixel.

Do not trigger a render! Fit and 1:1 are viewport operations, not data operations.

#### Make the buttons look like buttons

Use actual QToolButtons or retrieve the toolbar-created widgets:

```
button = toolbar.widgetForAction(fit_action)
button.setAutoRaise(False)
```

Then style:

```
QToolButton {
    padding: 4px 7px;
    border: 1px solid palette(mid);
    border-radius: 4px;
}

QToolButton:hover {
    border-color: palette(highlight);
}

QToolButton:pressed {
    background: palette(midlight);
}

QToolButton:checked {
    background: palette(highlight);
    color: palette(highlighted-text);
}
```

Qt stylesheets support button states such as pressed and checked; for custom backgrounds to show reliably, buttons often need explicit border styling.

### Montage must be redesigned

The current montage model is:

- input: many tiles
- output: one giant NumPy image

That is simple but fundamentally not scalable.
For a responsive viewer, montage should become:

- input: tile layout + view range
- output: only visible tiles, cached individually

New concept: 

```
MontagePlan
@dataclass(frozen=True)
class MontageTile:
    montage_index: int
    source_index: int
    row: int
    col: int
    x0: int
    y0: int
    width: int
    height: int
    view_state: ViewState


@dataclass(frozen=True)
class MontagePlan:
    axis: int
    tile_shape: tuple[int, int]
    grid_shape: tuple[int, int]
    gap: int
    tiles: tuple[MontageTile, ...]
```

Then: `visible_tiles = plan.tiles_intersecting(view_box.viewRange())`

Only these get evaluated.

#### Fix: multiple ImageItems, one per tile

Pros:

- no huge montage allocation
- tile cache is natural
- only visible tiles update
- memory can be bounded

Extra attention needed for:

- more PyQtGraph items
- need tile lifecycle management
- histogram/levels need separate sampled source

### Add memory budgets

Before allocating, estimate memory:

```
def estimate_montage_bytes(tile_shape, tile_count, dtype, *, histogram=True, rgb=False):
    tile_bytes = np.prod(tile_shape) * np.dtype(dtype).itemsize
    image_bytes = tile_bytes * tile_count

    if rgb:
        image_bytes *= 4

    if histogram:
        image_bytes += np.prod(tile_shape) * tile_count * np.dtype(np.float32).itemsize

    return int(image_bytes)
```

Then enforce:

visible render budget: 512 MB
montage hard budget: 1 GB by default
prefetch budget: 256 MB

When over budget: Do not allocate.
Show a clear message:
"Montage would allocate 4.8 GB. Showing visible tiles only / reduce tile count / reduce range."

No viewer should silently allocate 10+ GB for a 0.5 GB array!

### Cache tiles, not whole montages

Cache key:

TileKey(
    document_revision,
    operation_steps,
    view_state_without_montage_axis,
    montage_axis,
    source_index,
    channel,
    levels_mode,
    colormap_key,
)

Cache value:

RenderedTile(
    image: np.ndarray,
    histogram_source: np.ndarray | None,
    source_shape: tuple[int, ...],
)

The current montage path does not benefit enough from the normal image cache because it directly evaluates each tile into local lists and then discards the tile-level structure. That is wasted work.

### Evaluation scheduling

We need a proper scheduler and execution model.

Recommended API:

```
class EvalPriority(IntEnum):
    VISIBLE_IMAGE = 0
    LIVE_PROFILE = 10
    SELECTED_ROI = 20
    HOVER_EXACT = 30
    PREFETCH = 40
@dataclass(frozen=True)

class EvalRequest:
    key: object
    priority: EvalPriority
    generation: int
    replace_group: str
    memory_budget_bytes: int | None = None
```

Replace groups:

- visible-image: only newest matters
- live-profile: only newest marker position matters
- selected-roi: newest selected ROI stats matter
- prefetch: many may matter, but low priority and bounded

Perhaps it might be better to use separate controllers first:

VisibleRenderController:
  QThreadPool maxThreadCount = 1
  start_latest()
  clear queued old work

ProfileEvalController:
  QThreadPool maxThreadCount = 1
  debounce marker motion

RoiEvalController:
  QThreadPool maxThreadCount = 1
  debounce stats/histograms

PrefetchController:
  QThreadPool maxThreadCount = 1
  disabled for operations/montage until measured

Or unify them (now or later if better)

Immediate rule for slice scrolling

new slice event:
  cancel/drop queued visible jobs
  mark current running token canceled
  start latest visible job
  do not prefetch
  do not recompute hidden profile/ROI panels

That will make scrolling feel much better even before deeper optimization.

### Hidden memory issue: histogram/levels can allocate large temporary arrays

Several display paths use patterns like: `finite_data = data[np.isfinite(data)]`

That creates a boolean mask and then a copied finite array. On large images or giant montage images, this can be a major memory spike.

Prefer:

```
np.nanmin(data)
np.nanmax(data)
```

or sample: `sample = data[::stride_y, ::stride_x]`

For auto-levels in an interactive viewer, approximate is usually better than exact if exact blocks or allocates.

Policy:

- Small image: exact levels.
- Large image: sampled levels.
- Montage: sampled visible tiles only.
- ROI: exact only for selected ROI after debounce.

### Prefetch setting is currently not wired strongly enough

We have an app setting for prefetch, but the render path appears to call prefetch based mostly on internal conditions.
Make the setting authoritative:

```
if not self.app_settings.prefetch_nearby_slices:
    return

if self.document.steps:
    return

if self.view_state.montage_axis is not None:
    return

if estimated_slice_bytes > self.prefetch_budget_bytes:
    return
```

Default should be off until the scheduler is real and implemented correctly.
Predictive caching is worthwhile later, but not while visible rendering is slow.

### ROI/live plot responsiveness

Multiple ROIs + live plots + operations will get slow unless you split interaction into tiers.

During mouse drag:

- Update overlay only.
- Do not recompute all ROI stats.
- Do not recompute histograms.
- Do not resize table columns.
- Do not evaluate operation-backed scalar/profile per mouse event.

After drag ends:

- Debounced selected ROI stats.
- Maybe selected ROI histogram.
- Other ROI stats lazy/idle.

While panel hidden:

- Do not update ROI table/histogram.
- Mark stale and refresh when shown.
- Table model

Avoid full model resets and column resizing during live updates. Prefer stable row IDs and dataChanged() for changed rows.
This is less exciting than new features, but it is exactly what makes a scientific viewer feel professional.

### The docs and roadmap need to become sharper

Right now roadmap.md and the Phase 4 docs are a bit too optimistic.
Some items are effectively marked complete even though the implementation is partial.

Examples that should not be treated as complete yet:

- priority scheduler
- hidden panel work avoidance
- ROI async/debounce completeness
- viewport Fit toggle
- Wayland floating-panel solution
- montage memory bounding
- predictive caching discipline

Also add decision records:

docs/decisions/0016-evaluation-scheduler-and-memory-budget.md
docs/decisions/0017-managed-panels-and-wayland.md
docs/decisions/0018-viewport-fit-toggle.md
docs/decisions/0019-tiled-montage-renderer.md

Each decision record should include:

- Problem
- Decision
- Consequences
- Rejected alternatives
- Tests required
- Manual checks required

That will keep this from becoming another vague “we fixed responsiveness” phase.