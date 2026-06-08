# Bugs (from symptoms to ideas on fixes)

1. The dock bugs are still architectural, not random

We have these symptoms:

```
floated docks will not move normally
docks misbehave after repeated open/close
docked panels sometimes will not close unless undocked first
```

The most suspicious part is StandardDockWidget:

```
class StandardDockWidget(QtWidgets.QDockWidget):
    def close(self):
        snapshot, layout_manager = self._preserve_snapshot()
        result = super().close()
        self._restore_snapshot(snapshot, layout_manager)
        return result

    def setVisible(self, visible):
        snapshot, layout_manager = (None, None)
        if not bool(visible) and self.isVisible():
            snapshot, layout_manager = self._preserve_snapshot()
        super().setVisible(bool(visible))
        self._restore_snapshot(snapshot, layout_manager)

    def closeEvent(self, event):
        snapshot, layout_manager = self._preserve_snapshot()
        super().closeEvent(event)
        self._restore_snapshot(snapshot, layout_manager)
```

This is too invasive. QDockWidget already has a complicated internal state machine for floating, docking, closing, visibility, and top-level window behavior. Qt documents docks as movable/floatable/closable panels and warns that dock visibility can differ from plain QWidget.isVisible() in some cases. It also notes platform-specific floating-dock drag sensitivity, especially around native window handles and title bars.

So I think: The dock manager is better than before, but StandardDockWidget is still interfering with Qt’s own dock lifecycle.

Fix: Remove the close(), setVisible(), and closeEvent() overrides from StandardDockWidget.

Make StandardDockWidget boring:

```
class StandardDockWidget(QtWidgets.QDockWidget):
    pass
```

Keep feature flags:

```
dock.setFeatures(
    QtWidgets.QDockWidget.DockWidgetClosable
    | QtWidgets.QDockWidget.DockWidgetMovable
    | QtWidgets.QDockWidget.DockWidgetFloatable
)
```

Then move all preserve-canvas logic into WindowLayoutManager, and only around explicit managed actions.

The layout manager should be the only owner of:

```
show/hide
float/redock
preserve canvas
default dock sizes
menu checked-state sync
progressive dock behavior
layout reset
restore layout
```

Right now even inside layout_controller.py, there are still scattered direct calls like:

```
profile_dock.show()
profile_dock.hide()
operation_dock.show()
operation_dock.hide()
inspection_dock.setFloating(False)
```

Since they are inside the layout manager, they do not violate the architecture guard, but they still make behavior harder to reason about. Internally, use one raw helper:

```
def _apply_dock_visibility_raw(self, dock, visible):
    blocker = QtCore.QSignalBlocker(dock)
    dock.setVisible(bool(visible))
```

And one public route:

```
def set_managed_dock_visible(self, dock, visible, *, reason, preserve_canvas=True):
    ...
```

Also fix the menu action logic

In make_managed_dock_action(), avoid this pattern: `setter(not dock.isVisible())`

Use the action’s checked state and your own desired-state model:

```
def on_triggered(checked):
    self.set_managed_dock_visible(dock, bool(checked), reason="menu")
```

Do not use dock.isVisible() as the source of truth. For dock widgets, Qt’s visibilityChanged signal can behave differently from simple widget visibility.

Use a more robust detached-panel model! Perhaps we should stop fighting QDockWidget?

Let's use this split:

```
Docked mode:
  QDockWidget inside QMainWindow

Detached mode:
  QDialog / tool window containing the same panel widget
```

This gives us normal OS window behavior. For profile, operations, and inspection panels, this may be cleaner than relying on floating QDockWidget forever.

Qt’s QMainWindow.saveState() / restoreState() are still the right tools for storing dock/toolbar layout, but object names and stable ownership matter. resizeDocks() can size docks inside the main window, but it does not resize the main window itself, so your custom “preserve canvas” logic remains valid if centralized.

2. FOV mode should probably be removed for now

FOV currently feels broken because the UI mixes two different ideas:

```
aspect mode:
  square pixels / physical aspect / stretch

zoom mode:
  fit / 1:1 / user zoom
```

Right now the toolbar labels appear to map like this:

```
"1:1" → square_pixels
"FOV" → square_fov
"Fit" → fit
```

That is not quite right.

setAspectLocked() only controls aspect ratio. It does not mean “fit image” and it does not mean “1 image pixel equals 1 screen pixel.” autoRange() fits children into view; viewPixelSize() tells you how much data-space one screen pixel covers. These are separate concepts in PyQtGraph.

Fix: Remove FOV from the visible UI for now.

Keep two clear controls:

```
Fit
1:1
```

Expose them as a simple "Fit" button with a stretch icon, and a "1:1" button with a pixel icon.

Later, we can reintroduce FOV as Physical aspect only after you have axis metadata:

AxisInfo(
    name="x",
    spacing=0.5,
    unit="µm",
)

Then physical aspect can mean something real:

```
x spacing != y spacing
display physical field of view with correct proportions
```

Without axis spacing, “FOV” is ambiguous and mostly just another aspect-ratio lock.

Fix toolbar behavior

Toolbar icon “Fit” should not merely render with preserve policy. It should issue a viewport intent: `ViewportIntent.FIT`
Toolbar icon “1:1” should issue: `ViewportIntent.ONE_TO_ONE`
Do not map “1:1” to square_pixels. Square pixels means equal x/y scale. 1:1 means one image pixel maps to one screen pixel. This disable user zoom.

3. Channel mode needs explicit auto/manual state

We need to implement this desired behavior:

```
If data becomes complex through operations, auto-switch to complex
unless the user manually chose a channel.
```

If data becomes real again, invalid complex-only channels must switch away.
Current behavior does not track whether the user chose the channel manually. That makes it impossible to distinguish these cases:

```
User intentionally selected Real while viewing complex data.
Program defaulted to Real because the source data was real.
```

Those require different behavior.

Add this state `self._channel_user_selected = False`

When the user clicks/selects a channel:

```
def _on_channel_clicked(self, channel):
    self._channel_user_selected = True
    self._set_view_state(self.view_state.with_channel(channel))
    self.render(reason="channel-user")
```

When the program changes channel automatically, do not set that flag.

Then centralize dtype/channel coercion:

```
def _coerce_channel_for_current_dtype(self):
    is_complex = self._current_is_complex()
    channel = self.view_state.channel

    complex_only = {
        ChannelMode.COMPLEX,
        ChannelMode.IMAG,
        ChannelMode.ANGLE,
    }

    if not is_complex and channel in complex_only:
        self._set_view_state(self.view_state.with_channel(ChannelMode.REAL))
        return

    if is_complex and not self._channel_user_selected and channel == ChannelMode.REAL:
        self._set_view_state(self.view_state.with_channel(ChannelMode.COMPLEX))
```

Call this after every document/output-type transition:

```
open file
reload file
clear operations
add operation
remove operation
materialize operations
recipe load
operation parameter change
data revision change
```

Also add tests for this exact matrix:

```
real data → add imaginary op → untouched channel becomes complex
real data → user selects real → add imaginary op → remains real
complex data → clear ops to real → complex/imag/angle coerces to real
complex data → user selected abs → clear ops to real → abs can remain valid
reload real file after complex op → no stale complex channel
```

This one will pay off fast.

4. “Computing” / “updating” is too eager and too global

This is a UX bug, but it has an architectural cause.

In hover handling, the code sets: `value = updating...` before it knows whether the scalar is already cached or directly available. That makes the app feel slower than it is.

The current behavior is probably:

```
mouse moves
label says updating
cache returns immediately
label updates again
```

That still feels clunky.

Fix hover status: use this order:

```
cached = evaluator.cached_scalar(view_state, index)
if cached is not None:
    show_pixel_value(cached)
    return

if no_operations_are_active:
    value = base_data[index]
    show_pixel_value(value)
    return

start_delayed_busy_timer(100_ms)
request_async_scalar(...)
```

Only show updating... if the async request has not completed after a short delay.

Also, do not write hover updates into the global status bar. The status bar should not update on every mouse move. Keep hover info in the HUD/top label only.

Fix: global cache status

I also found a likely montage status bug. The montage render path sets something like: `last_status = Computing: Evaluating montage view`

But after montage completes, it applies the display image without necessarily setting a corresponding “ready” status. That can leave the operation/status UI stuck saying “Computing.”

Also, prefetching appears to reuse the same broad status channel as visible rendering. That makes status text misleading.

Split status into separate concepts:

```
@dataclass
class EvaluationStatus:
    visible_image: TaskStatus
    scalar_hover: TaskStatus
    live_profile: TaskStatus
    roi_stats: TaskStatus
    prefetch: TaskStatus
```

Then the top bar should mostly show: `visible image status`

Not scalar hover status, not prefetch status, not stale montage status.

5. Hidden correctness bug: display-axis ranges can break

slice_engine.py currently preserves non-display axes as singleton dimensions, then applies display-axis ranges using a result_axis counter. That assumes the sliced array contains only display axes. It does not.

A minimal failing case:

```
import numpy as np
from arrayscope.core.view_state import ViewState
from arrayscope.display.slice_engine import make_line, make_image

data = np.arange(2 * 3).reshape(2, 3)

state = (
    ViewState.from_shape(data.shape)
    .with_image_axes(0, 1)
    .with_line_axis(1)
    .with_axis_range(1, (0, 2))
)

make_line(data, state)
```

This can raise: `IndexError: index 2 is out of bounds for axis 0 with size 1`

Another failing case:

```
data = np.arange(2 * 3 * 4).reshape(2, 3, 4)

state = (
    ViewState.from_shape(data.shape)
    .with_image_axes(1, 2)
    .with_axis_range(1, (0, 2))
)

make_image(data, state)
```

This is exactly the kind of bug that only appears once users choose non-leading image axes or ranged axes.

Fix: Do not preserve non-display axes as singleton dimensions in display extraction. Slice them out with integer indexing.

Use a helper like this:

```
def _slice_to_display_axes(data, state, display_axes):
    display_axes = tuple(display_axes)
    index = []
    present_axes = []

    for axis in range(state.ndim):
        if axis in display_axes:
            index.append(slice(None))
            present_axes.append(axis)
        else:
            index.append(int(state.slice_indices[axis]))

    return data[tuple(index)], tuple(present_axes)


def _apply_axis_ranges(data, state, present_axes):
    result = np.asarray(data)

    for result_axis, original_axis in enumerate(present_axes):
        indices = state.axis_range_indices[original_axis]
        if indices is not None:
            result = np.take(result, tuple(indices), axis=result_axis)

    return result


def _reorder_axes(data, present_axes, desired_axes):
    if tuple(present_axes) == tuple(desired_axes):
        return data

    permutation = tuple(present_axes.index(axis) for axis in desired_axes)
    return np.transpose(data, permutation)

Then make_image() becomes conceptually:

image_data, present_axes = _slice_to_display_axes(data, state, state.image_axes)
image_data = _apply_axis_ranges(image_data, state, present_axes)
image_data = _reorder_axes(image_data, present_axes, state.image_axes)
```


Add tests immediately:

```
def test_make_image_applies_axis_range_when_image_axes_are_not_front():
    data = np.arange(2 * 3 * 4).reshape(2, 3, 4)

    state = (
        ViewState.from_shape(data.shape)
        .with_image_axes(1, 2)
        .with_axis_range(1, (0, 2))
    )

    image = make_image(data, state)

    np.testing.assert_array_equal(image.data, data[0, [0, 2], :])


def test_make_line_applies_axis_range_when_line_axis_is_not_front():
    data = np.arange(2 * 3).reshape(2, 3)

    state = (
        ViewState.from_shape(data.shape)
        .with_image_axes(0, 1)
        .with_line_axis(1)
        .with_axis_range(1, (0, 2))
    )

    line = make_line(data, state)

    np.testing.assert_array_equal(line.data, data[0, [0, 2]])
```

This is a priority fix.

6. Async stale-result checks should compare full request keys

I saw places where callbacks compare only the document part of a key, not the full request key.

Pattern to avoid:

```
document_key = evaluator.scalar_key(view_state, index, document=document)[1]

...

if document_key != evaluator.scalar_key(view_state, index)[1]:
    return
```

That only says “same document.” It does not fully say “same scalar request.”

Use the full key:

```
request_key = evaluator.scalar_key(view_state, index, document=document)

...

if request_key != evaluator.scalar_key(view_state, index):
    return
```

Do the same for live profile callbacks.

You already have request IDs/generations in some paths, which is good. But for vague bugs, be paranoid:

Every async result must prove:
- same document revision
- same view state
- same request parameters
- same UI generation

Otherwise it should be dropped.

7. ROI is improved, but it is becoming a performance hotspot

The ROI model direction is good. Multiple ROIs, ROI histogram, line/freehand/rectangle stats, and overlays are the right features.

But with multiple ROIs, live plots, montage, and operation-backed data, this can easily become sluggish.

Likely hotspots:

```
ROI stats recomputed synchronously
histograms recomputed too often
ROI table reset on every refresh
columns resized repeatedly
selection restored after model reset
live profile updates during mouse movement
hover scalar requests on every mouse move
prefetch competing with visible rendering
Fix pattern
```

Use event tiers:

```
During drag:
  update overlay only
  optionally show approximate bounding box

After drag finishes:
  compute exact ROI stats

After render completes:
  invalidate ROI stats
  recompute selected/visible ROIs only

Idle:
  recompute all ROI histograms if needed

Add debounce timers:

roi_stats_timer.setSingleShot(True)
roi_stats_timer.start(100)
```

Move heavier ROI stats/histograms off the UI thread. The UI should never block because ten ROI histograms are recomputing.

Also avoid full table resets where possible. Instead of:

```
beginResetModel()
...
endResetModel()
resizeColumnsToContents()
```

prefer stable row IDs and emit:

```
dataChanged(...)
rowsInserted(...)
rowsRemoved(...)
```

Resize columns sparingly, not after every live update.

8. Testing is better, but not yet catching the important failures early enough

The testing direction is now good: pure tests, architecture guards, viewport tests, slab tests, and Hypothesis scaffolding.

But the missed bugs show the gaps:

```
display-axis ranges not tested with arbitrary image axes
channel auto/manual state not tested as a state machine
dock actions not tested through actual Qt user paths
hover/status not tested for cached/direct paths
viewport toolbar does not test visible effect
ROI performance behavior not tested
```

Add three kinds of tests!

8A. More deterministic regression tests

Add exact tests for every bug above:

```
non-leading image axes + axis ranges
line axis not at front + axis range
Fit toolbar changes view range
1:1 toolbar changes viewPixelSize
FOV removed or disabled
real → complex op auto-switch
manual channel selection prevents auto-switch
clear/reload coerces invalid complex channel
hover cached scalar never shows updating
montage render clears Computing status
dock action open/close repeated 10 times
```

8B. Hypothesis property tests

Hypothesis is especially useful here because it generates edge cases humans forget: singleton axes, non-leading axes, reversed axes, odd ranges, empty-ish ranges, last montage tile, and repeated state transitions. Its stateful testing can generate action sequences and check invariants after each step.

Property tests I would add:

```
make_image result equals direct NumPy extraction for arbitrary image_axes/ranges
make_line result equals direct NumPy extraction for arbitrary line_axis/ranges
DisplayGeometry point → index matches make_image pixel value
montage point → source index matches tile extraction
operations lazy result equals materialized result for generated operation sequences
channel state machine preserves user intent
```

8C. Real Qt interaction tests

Use pytest-qt for actual user paths: menu actions, toolbar buttons, dock close buttons, floating, signal waiting, and event-loop settling.
qtbot provides wait helpers, mouse/key interaction, screenshots, and signal assertions.

Important tests:

```
View menu → Profile dock open/close/open/close
close dock using dock close button
float dock, close, reopen, redock
clear layout, restore layout
toolbar Fit causes fit behavior
toolbar 1:1 causes 1:1 behavior
hover over cached image does not show updating
operation completes and top bar leaves Computing
```

Also run GUI tests with: `ARRAYSCOPE_STRICT_UI=1`

Broad except Exception paths should fail tests in strict mode.
