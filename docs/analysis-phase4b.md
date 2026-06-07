


For extra context, to the phase 4b items, here are my thoughts and ideas issues and how to fix them - around which I tried to craft phase 4b (add items to it, if you're missing something that needs to be covered):

1. P0: lazy slab evaluation is wrong for some operation combinations

This is the most serious bug I found because it can silently show wrong data.

A deterministic differential check found failures for combinations like:
```
Crop(axis=1, start=1, stop=4)
ReverseAxis(axis=1)
```
The lazy slab result does not match full materialization. Example failure shape:
```
actual image shape:   (4, 2)
expected image shape: (4, 3)
```
The root cause is in arrayscope/operations/slabs.py. The current slice algebra handles crop and reverse too simplistically. In particular, _reverse_spec() effectively turns many requests into: `slice(None, None, -1)`  and _crop_spec() does not correctly compose that with crop bounds when the same axis is involved.

This is exactly the kind of bug that can create “weird but no error” behavior: the viewer renders valid arrays, just not the correct arrays.

Fix: Do not handwave slice composition. Implement index composition explicitly.

Conceptually:
```
def output_indices_for_item(item, size):
    # Convert scalar/slice/requested output coordinates into explicit output indices.
    return np.arange(size)[item]

def crop_source_indices(item, start, stop):
    output = output_indices_for_item(item, stop - start)
    return start + output

def reverse_source_indices(item, size):
    output = output_indices_for_item(item, size)
    return size - 1 - output
```
Then convert back to a simple slice only when the resulting sequence is arithmetic. If it is not safely representable as a basic slice, either use carefully isolated advanced indexing or fall back to applying that operation to the slab. Prefer correctness over laziness.

Add this immediately:
```
def test_slab_matches_materialized_after_crop_reverse_same_axis():
    data = np.arange(4 * 5 * 6).reshape(4, 5, 6)
    doc = ArrayDocument(
        data,
        steps=(
            Crop(axis=1, start=1, stop=4),
            ReverseAxis(axis=1),
        ),
    )

    state = ViewState.for_shape(doc.output_shape).with_image_axes((0, 1))
    lazy = evaluate_image_snapshot(doc, state)
    full = make_image(doc.materialize(), state)

    np.testing.assert_array_equal(lazy.data, full.data)
```
Then generalize it with Hypothesis: generate small shapes, random operation sequences, and random view states, and require: `lazy slab image/profile/scalar == materialized image/profile/scalar`
This should become one of your most important test families. Hypothesis is well suited here because it generates and shrinks edge cases rather than only testing examples humans thought of. Its stateful testing can also generate action sequences, not just values.

2. Dock opening/closing is bypassing your new layout manager

The manual tests notes say:
```
Opening Profile dock does not resize main window
Opening Operations dock shrinks 2D view
Grabbing edge/corner immediately resizes to correct size
Floating docks reopen weirdly
```
The likely root cause is very specific. In arrayscope/ui/menus.py, you use Qt’s built-in dock toggle actions:
```
operation_action = self.operation_dock.toggleViewAction()
operation_action.triggered.connect(
    lambda visible: self._set_operation_dock_visible_from_user(visible)
)
```
But QDockWidget.toggleViewAction() already creates a checkable action that shows/closes the dock. Qt owns that visibility behavior. Then your slot runs after the dock may already be visible. Your layout manager sees:
```
if dock.isVisible() == bool(visible):
    return
```
and exits without doing the preserve-canvas resize work.
So the menu action itself is probably bypassing the logic you just wrote.

Qt documents toggleViewAction() as the dock’s own show/close action, and visibilityChanged is the signal to observe dock visibility changes.

Fix: Do not use toggleViewAction() for docks whose visibility needs custom layout behavior.

Create your own checkable actions:
```
def _make_managed_dock_action(self, text, dock, setter):
    action = QtGui.QAction(text, self)
    action.setCheckable(True)
    action.setChecked(dock.isVisible())

    def on_triggered(checked):
        setter(bool(checked))

    def sync_checked(visible):
        blocker = QtCore.QSignalBlocker(action)
        action.setChecked(bool(visible))

    action.triggered.connect(on_triggered)
    dock.visibilityChanged.connect(sync_checked)
    return action
```
Use QSignalBlocker while syncing action state so you do not trigger recursive visibility changes; Qt’s QSignalBlocker is specifically designed for exception-safe temporary signal blocking.

Then route all dock changes through one manager API:
```
self.layout_manager.set_dock_visible_preserving_canvas(
    self.profile_dock,
    True,
    reason="user-menu-profile",
)
```
Also remove these direct calls outside the layout manager:
```
profile_dock.show()
operation_dock.show()
inspection_dock.show()
dock.setVisible(...)
dock.close()
```
I found direct dock show paths in render/profile/operation/state-sync code. They should become manager calls.

Add a guard test; Add a simple AST or grep-style architecture test:
```
No direct `.show()`, `.hide()`, `.setVisible()`, or `.close()` on managed docks
outside layout_controller.py and dock implementation files.
```
This sounds strict, but it will save us! These are exactly the bypasses that create layout ghosts.

3. Fit and 1:1 are not really implemented as distinct viewport modes

The notes say: `Fit and 1:1 seem broken; image does not change.`
That is consistent with the code.

ImageView2D.oneToOne() currently does:
```
self.setDisplayMode('square_pixels')
self.view.autoRange()
```
That is not 1:1. It means “square pixels and fit-ish range.” True 1:1 means one image pixel maps to one screen pixel, centered around a chosen point.

Also, _on_aspect_toolbar_changed() calls render(reason="aspect"); the render path often sees the same image shape and preserves the old viewport. So pressing Fit/FOV/1:1 can become visually no-op.

PyQtGraph’s ViewBox.autoRange() sets the range so children are visible; setAspectLocked() controls aspect ratio locking. They are useful, but they are not a complete viewport policy by themselves.

Fix: Introduce a small ViewportController.

It should own:
```
class ViewportMode(Enum):
    AUTO_UNTOUCHED = "auto_untouched"
    USER = "user"
    FIT = "fit"
    ONE_TO_ONE = "one_to_one"
```
Policy:
```
Fresh image:
  fit image.

New slice, same display shape:
  preserve range.

New display shape and user never panned/zoomed:
  fit image.

New display shape and user did pan/zoom:
  preserve center/scale as much as possible.

Explicit Fit:
  enter FIT mode and auto-range.

Resize while in FIT:
  re-fit.

Explicit 1:1:
  enter ONE_TO_ONE mode and compute view range from viewport pixel size.

User pan/zoom:
  enter USER mode.
```
Pseudo-code for 1:1:
```
def set_one_to_one(view_box, image_shape, viewport_size_px):
    height, width = image_shape

    visible_w = viewport_size_px.width()
    visible_h = viewport_size_px.height()

    cx, cy = current_or_image_center()
    x0 = cx - visible_w / 2
    x1 = cx + visible_w / 2
    y0 = cy - visible_h / 2
    y1 = cy + visible_h / 2

    view_box.setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0)
```
You will need to handle high-DPI/device-pixel-ratio carefully, but the key point is: 1:1 is a range calculation, not autoRange().

4. Montage status text duplicates dimensions

Manual note: `status bar/HUD duplicate tiled dimension: d2=50 d2=102`

This is a straightforward bug. In render.py, getPixel() does:
```
context = self._slice_context_text()
...
context = f"{context} d{self.display_geometry.montage.axis}={index[...]}"
```
But _slice_context_text() uses view_state.non_display_axes(), and the montage axis is still considered a non-display axis. So you get: old scalar slice for d2 + actual montage tile index for d2

Fix: DisplayGeometry should build hover context labels.

Pixel hover, profile hover, ROI info, and HUD should all ask the same object: `geometry.context_for_display_point(x, y)`

That prevents the next duplicate/inconsistent label bug.

5. Live profile axis selection is using a toggle when it should set exactly one axis

Manual note: `Live profile from axis while profile dock closed opens profile dock with another profile axis.`

Root cause: `_enable_live_profile_for_axis(dim)` calls: `self.set_dimension_role("p", dim)`

But "p" means “toggle/add this profile axis.” For the command “live profile from this axis,” you do not want toggle semantics. You want:
```
profile_axes = (dim,)
line_axis = dim
live_profile = enabled
```

Fix: Add an explicit method:
```
def set_profile_axes_exactly(self, axes):
    axes = tuple(axes)
    state = self.view_state.with_line_axis(axes[0])
    self.profile_axes = axes
    self._set_view_state(state)
```
Then:
```
def _enable_live_profile_for_axis(self, dim):
    self.set_profile_axes_exactly((dim,))
    self._set_live_profile_enabled(True)
    self.layout_manager.set_dock_visible_preserving_canvas(
        self.profile_dock,
        True,
        reason="live-profile",
    )
```
Do not open the dock with profile_dock.show().

6. Tiled dimension X/Y buttons should be disabled, but also guarded defensively

Manual note:
```
clicking X/Y dimension button on tiled dimension raises:
ValueError: montage axis cannot also be an image axis
```
The compact dimension chip enables X/Y buttons for the montage axis. The lower-level ViewState correctly rejects the invalid state, but the UI should not offer it.

Fix in DimensionChip.update_state
```
can_use_as_image = (
    not is_singleton
    and not is_m
    and view_state.image_axes is not None
)

self.x_button.setEnabled(can_use_as_image)
self.y_button.setEnabled(can_use_as_image)
```
Tooltip: `Tiled dimension cannot also be image X/Y. Clear the range first.  

Also add a defensive guard in set_dimension_role():
```
if role in {"x", "y"} and self.view_state.montage_axis == axis:
    show_status_message(...)
    return
```
This way strict UI mode stays useful instead of becoming a user-facing traceback generator.

7. Empty tiled range should clear montage and become scalar midpoint

Manual note: `fully deleting tiled dimension slice string should default to scalar mid-dim`

Currently empty text raises parse failure and the UI restores the old text. Add this at the start of _on_slice_text_changed():
```
if text.strip() == "":
    mid = max(0, self.data.shape[axis] // 2)

    state = self.view_state.with_slice(axis, mid)
    state = state.with_axis_range(axis, None)

    if state.montage_axis == axis:
        state = state.with_montage_axis(None)

    self._set_view_state(state)
    self.render(reason="slice-empty-midpoint")
    return
```
This is also a good example of a general rule: `UI edits should reduce to explicit state transitions, not parse-error/revert behavior unless the input is genuinely invalid.`

8. File reload currently deletes operations

Manual note: `file reload works but deletes operations; unacceptable without confirmation/save recipe.`
This is real.

ArrayDocument.with_data_changed(base_data) currently means too many things:
```
same object changed in-place
new file reloaded
materialized current pipeline into new base
replace dataset entirely
```
Those are not equivalent.

Fix the API.
Split it:
```
mark_base_data_changed()
# Same object, revision++, preserve operations.

reload_base_data(data, *, preserve_steps=True)
# New base data from same file/reload. Preserve operations if compatible.

replace_base_and_clear_steps(data)
# Explicit reset/materialize/new dataset.
```
Reload behavior should be:
```
Same shape and operations still valid:
  preserve operations and increment revision.

Different shape but operations validate:
  preserve operations, update output shape.

Different shape and operations fail:
  prompt:
    - Reload and clear operations
    - Cancel reload
    - Save recipe first
```
Materialize behavior should still clear operations, because the operations have been baked into the new base.

Also save recipes/session files with atomic writes. Qt has QSaveFile, which writes to a temporary file and commits atomically to avoid partial output.

9. Close/cancel is not strong enough

Manual note: `Closing main with floated windows mostly closes, but sometimes process doesn’t exit until queued operations finish.`

EvaluationController.cancel_pending() increments a generation and clears pending metadata, but it does not cancel queued QRunnables or stop already-running work.

Qt’s QThreadPool.clear() removes queued runnables that have not started, while waitForDone() waits for threads to finish. The global thread pool is shared process-wide.

Fix: Use a local thread pool per main window/controller instead of only QThreadPool.globalInstance().

On close:
```
self._closing = True
self.evaluation_controller.cancel_pending()
self.evaluation_controller.clear_queued()
self.evaluation_controller.stop_polling()
```
Inside the controller:
```
def clear_queued(self):
    self._pool.clear()
    self._pending.clear()
    self._handlers.clear()
    self._prefetch_handlers.clear()
    self._generation += 1
```
For running tasks, use cooperative cancellation:
```
@dataclass
class CancellationToken:
    cancelled: bool = False
```
Long-running loops check:
```
if token.cancelled:
    return Cancelled
```
You cannot safely kill arbitrary NumPy calls mid-operation, but you can:
```
clear queued work
ignore stale results
avoid starting more work
cancel cooperative loops
make close deterministic once running work returns
```
Saving/exporting should use a separate path because you probably do not want to silently cancel file writes.

10. Image prefetch is probably not storing anything

Found a concrete bug in _prefetch_nearby_slices(). It computes a cache key for the current state once: `document_key = image_key(current_view_state)`
Then it schedules nearby states: `prefetch_state = ...`

But the done callback stores using the current-state key while _store_prefetch_image_if_current() compares against the prefetch-state key. Those keys differ, so the result is discarded.

Fix

Compute the key per prefetch state:
```
prefetch_key = self.operation_evaluator.image_key(
    prefetch_state,
    colormap_lut=colormap_lut,
    document=document,
)[1]

self.evaluation_controller.start_prefetch(
    fn,
    on_done=lambda result, state=prefetch_state, key=prefetch_key:
        self._store_prefetch_image_if_current(state, result, key),
)
```
Add a unit test that calls the real _prefetch_nearby_slices() path with a fake evaluator/controller, not only the store method.

After this, add prefetch discipline:
```
dedupe in-flight keys
limit queued prefetches
prioritize visible render > pixel/profile > nearby slice prefetch
drop prefetches on state generation change
```
Predictive caching should come later. First make basic prefetch correct and bounded.

# Testing strategy

what should catch this before manual testing?

The test method is moving in the right direction, but it is not yet layered correctly.

Manual testing is currently catching too much. Manual tests should catch subjective UX and OS/window-manager quirks, not basic state invariants.

Add four automated test layers
1. Differential pure tests

These should run without Qt. Core rule: `lazy result == materialized result`

For:
```
image render
line profile
scalar pixel lookup
ROI sampling
histogram source data
```
Across:
```
crop
reverse
mean
RSS
FFT/fftshift where applicable
axis ranges
montage axes
complex display modes
small/singleton/odd shapes
```
This would have caught the crop+reverse slab bug.

2. DisplayGeometry property tests

We already have DisplayGeometry. Lean into it.

Test:
```
display point → array index
array index → context label
display point in montage gap → None
display point in incomplete final montage row → correct/None
image-axis range maps display index to original index
float boundaries are deterministic
```
Important: add float tests. Right now geometry uses round() internally in places, while pixel hover floors before calling geometry. Decide one convention.

Recommendation:
```
Display pixel cell [x, x+1) maps to column x.
Use floor for mouse/view coordinates.
Use center positions only when drawing markers.
```
Then enforce it everywhere.

3. Qt interaction tests using real user paths

Your current Qt tests are useful but too direct. They call methods like: `_set_profile_dock_visible_from_user(...)`
That misses the menu-action bug.

Use pytest-qt to trigger the actual actions and signals. pytest-qt provides qtbot for Qt widgets/signals and simulated interaction; it can also surface exceptions from Qt virtual methods/slots during tests.

Test the real paths:
```
View menu → Profile Dock
View menu → Operations Dock
View menu → Inspection Dock
toolbar Fit
toolbar 1:1
compact chip X/Y/M/P buttons
slice text edit clear
live profile from axis menu
float dock → close → reopen → redock
main close during queued operation
```

4. Architecture guard tests

These are cheap and powerful. Add tests that fail when someone bypasses ownership rules:
```
No toggleViewAction() for managed docks.
No direct managed_dock.show() outside layout manager.
No direct profile_dock.show() in render/operation/state-sync code.
No render commit without generation check.
No broad except Exception without logging traceback in strict mode.
```
This is not overengineering. Your current bug class is architectural bypasses.

Specific manual tests to keep

Manual testing should continue, but with a narrower purpose.

Keep manual tests for:
```
actual dock dragging/floating on Windows/macOS/Linux
high-DPI 1:1 behavior
visual ROI readability
long-running operation cancellation with real data
large montage performance
file reload/recipe UX
```
Use index-coded arrays: `data[z, y, x] = 10000*z + 100*y + x`

Then hover/profile/ROI mismatches become obvious.

For montage, use:
```
first tile
second tile
gap
last complete tile
last incomplete-row tile
empty final row area
```
For viewport, test:
```
fresh open
slice change without pan
slice change after pan
dock open after pan
dock close after zoom
resize in Fit
resize in 1:1
Architecture: what to refactor now
```

# Refactoring

Do not do a broad rewrite. Do a targeted ownership refactor.

1. Make UI actions produce intents, not direct widget mutation

Right now some paths do:
```
profile_dock.show()
self.view_state = ...
self.render(...)
```

Instead, aim for: `UI event → Intent → State reducer → Render/layout transaction`

For example:
```
@dataclass(frozen=True)
class UiIntent:
    kind: str
    axis: int | None = None
    value: object | None = None

@dataclass(frozen=True)
class StateUpdate:
    view_state: ViewState
    render_reason: str
    viewport_intent: ViewportIntent = ViewportIntent.PRESERVE
    dock_intent: DockIntent | None = None
```
You do not need a giant Redux framework. Just make state transitions explicit and testable.

2. Centralize dock ownership

The layout manager should be the only owner of:
```
show/hide
float/redock
preserve canvas
sync menu checked state
restore default dock area
close behavior
```

Qt’s QMainWindow.resizeDocks() adjusts dock sizes within the main window, but it does not resize the main window itself. Qt’s saveState()/restoreState() stores main-window dock/toolbar layout state, and object names matter for restoring state reliably.

So our custom “preserve canvas by resizing main window” logic is legitimate. It just needs to be the only path.

3. Add a real ViewportController

Viewport behavior is currently spread across:
```
ImageView2D.setImage()
ImageView2D.setDisplayMode()
ImageView2D.fitToView()
ImageView2D.oneToOne()
render._viewport_policy_for_display_shape()
layout restore snapshots
dock resize restore
```

Make one object decide:

```
when to preserve
when to fit
when to preserve center
when user has modified view
when aspect mode implies range recalculation
```

This is the highest-value refactor after slab correctness.

4. Let DisplayGeometry own display labels too

It already maps display points to array indices. Extend it to produce:
```
hover context
montage tile label
ROI source mapping
profile marker state
```

Then remove duplicated logic from render.py.

5. Use Qt Model/View for ROI and operations UI

Our ROI UX complaints are a sign that widget-local state is not enough anymore:
```
ROI list
ROI table
ROI overlay
ROI histogram
ROI color
ROI selection
ROI deletion
per-ROI info boxes
```

These should share one model:

```
class RoiTableModel(QtCore.QAbstractTableModel):
    ...
```

Qt’s model/view architecture is built around QAbstractItemModel, where views/delegates present data that can live in a separate data structure.

Recommended structure:
```
RoiStore
  owns ROI objects, colors, names, visibility, selection

RoiTableModel
  exposes RoiStore to table/list widgets

Image ROI overlays
  are views of RoiStore

Histogram dock
  reads selected ROI ids from RoiStore

Then you get deletion, selection, color sync, and highlighting without duct tape.
```

6. Keep PyQtGraph; do not jump to napari/Dask/Zarr yet

Our choice of PyQtGraph still makes sense. We now explicitly use row-major ImageItem(axisOrder="row-major"), which matches NumPy (height, width) and avoids the old PyQtGraph col-major default trap. PyQtGraph documents both axis orders and notes that row-major is typically preferable for performance.

Do not replace the viewer with napari unless the goal changes dramatically. Napari is excellent, but adopting it would change our product into a layer-centric image viewer rather than a compact array-inspection tool.

Dask/Zarr are worth designing for later, not adopting now. Dask Array targets blocked/larger-than-memory array computation, and Zarr targets chunked compressed N-dimensional storage. They are relevant future backends, but adding them before the state/rendering contract is hardened would multiply complexity.

7. CI screenshot path looks stale

The workflow references: `python test/screenshot_test.py --style Fusion`
This is the old manual method. Replace this with our new set of automated (screenshot generating) tests.

8. Slice syntax is confusing

The UI examples use: `0:2:100` meaning: `start:step:stop`

But Python users expect: `start:stop:step`.
Use Python slicing syntax: `0:100:2`, by default. If it is easy, add a toggle in the menu (under view?) to opt into MATLAB-style slicing.

9. Montage silently truncates at 256 tiles

The montage render path truncates tile indices: `indices = indices[:256]`

That is fine as a safety limit, but it should not be silent.

Show: `Showing first 256 of 1024 tiles`

and maybe offer:
```
increase limit
change range
export montage
Pixel hover can update stale values
```

10. Pixel value requests are async.
The callback checks document/view-ish keys, but it does not fully guard against the cursor having moved to another pixel before the scalar result returns.

Add a pixel-hover request generation:
```
self._pixel_request_id += 1
request_id = self._pixel_request_id
...
if request_id != self._pixel_request_id:
    return
```
Do the same for profile marker async updates if needed.

11. Floating dock drag issue

The floating docks might be caused by our custom title bar? And the lack of:
```
QDockWidget.DockWidgetClosable
QDockWidget.DockWidgetMovable
QDockWidget.DockWidgetFloatable
```
and stable objectName.

All managed docks should have these. Perhaps even a test simulating dragging to move a floating dock? To ensure that works OK!
