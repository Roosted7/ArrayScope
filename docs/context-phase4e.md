# Phase 4e (and completing Phase 4d)

Be creative and thorough! Re-use existing libraries and features where sensible (as opposed to re-inventing the wheel).
Internal backwards code compatibility is not important. At all! So prefer clean rewrites, that are optimal going forward.

If you find bugs, or potential (performance, usability) issues - fix them please. If they are very complex, or not suitable for this scope; ensure to keep track of them in the .md. Same for nice future ideas!

## Serious issue 1: cached montage tiles are counted as 1 byte

This is the most concrete memory bug I found.

BoundedArrayCache._nbytes() handles objects with .data, plain ndarray, and scalars:

```
def _nbytes(value):
    if hasattr(value, "data") and isinstance(getattr(value, "data"), np.ndarray):
        ...
    if isinstance(value, np.ndarray):
        ...
    if np.isscalar(value):
        ...
    return 1
```

But cached montage tiles are RenderedTile objects:

```
@dataclass(frozen=True)
class RenderedTile:
    tile: MontageTile
    image: np.ndarray
    histogram_data: np.ndarray | None
    ...
```

They do not have .data. So every cached rendered tile is counted as 1 byte.

I confirmed this directly:

bytes_used 1
expected   80000

That means your tile cache can blow far past its byte budget while reporting that it is tiny. This is a direct explanation for memory spikes.

### The fix

Add a cache-size protocol:

```
def _nbytes(value):
    nbytes_method = getattr(value, "nbytes", None)
    if callable(nbytes_method):
        return int(nbytes_method())

    if hasattr(value, "image") and isinstance(getattr(value, "image"), np.ndarray):
        total = int(value.image.nbytes)
        histogram_data = getattr(value, "histogram_data", None)
        if isinstance(histogram_data, np.ndarray):
            total += int(histogram_data.nbytes)
        return total

    if hasattr(value, "data") and isinstance(getattr(value, "data"), np.ndarray):
        total = int(value.data.nbytes)
        histogram_data = getattr(value, "histogram_data", None)
        if isinstance(histogram_data, np.ndarray):
            total += int(histogram_data.nbytes)
        return total

    if isinstance(value, np.ndarray):
        return int(value.nbytes)

    if np.isscalar(value):
        return int(np.asarray(value).nbytes)

    return 1
```

Actually... It is better to abstract it like this:

```
@dataclass(frozen=True)
class RenderedTile:
    ...

    def nbytes(self) -> int:
        total = int(self.image.nbytes)
        if isinstance(self.histogram_data, np.ndarray):
            total += int(self.histogram_data.nbytes)
        return total
```

### Testing

Add this:

```
def test_bounded_cache_counts_rendered_tile_bytes():
    tile = MontageTile(...)
    rendered = RenderedTile(
        tile=tile,
        image=np.zeros((100, 100), dtype=np.float32),
        histogram_data=np.zeros((100, 100), dtype=np.float32),
        eval_ms=0.0,
        slab_shape=(),
        slab_nbytes=None,
    )

    cache = BoundedArrayCache(max_bytes=1024, max_entries=96)
    cache.put("tile", rendered)

    assert cache.bytes_used == 80_000
    assert len(cache._items) == 0  # evicted because over budget
```

This is P0!

## Serious issue 2: the montage renderer is only partially tiled

We now have MontagePlan, MontageTile, visible-tile selection, and tile caching. Good.

But the final commit path still does this:

```
rendered = make_montage(
    [tile.image for tile in rendered_tiles],
    histogram_images=[...],
    columns=min(plan.columns, len(rendered_tiles)),
    indices=tuple(tile.tile.source_index for tile in rendered_tiles),
)
```

So you are still assembling visible tiles into one local collage. That is better than assembling the full montage, but it is not a true tiled renderer yet.

### The problem

Subtle but important:

- MontagePlan describes global tile positions.
- make_montage(rendered_visible_tiles) creates a new local mini-montage.
- DisplayGeometry then receives local mini-montage geometry.

If visible tiles are not the first tiles in the full montage, the committed image no longer has the same coordinate space as the full montage plan. This can cause panning, hover mapping, profile mapping, and viewport preservation to behave strangely.

It also still allocates:

- visible tile arrays
- visible histogram tile arrays
- final local montage array
- final local histogram montage array
- possibly complex/RGB temporaries

A fixed 64-tile cap is not enough. Sixty-four 4096×4096 float tiles is still enormous.

### Fix (direction)

For Phase 4d, choose one of these two approaches.

#### Option A — viewport-sized composite canvas

Try this one first!
This is probably the best next step, since many ImageItems caused segfaults.

Instead of making a mini-montage from visible tiles, create a canvas bounded by viewport/display budget:

- canvas shape ≈ visible viewport in data pixels, plus margin
- tile images are copied into their correct clipped positions
- DisplayGeometry knows canvas_origin_x/canvas_origin_y
- hover maps canvas point + origin back into full montage coordinates

New object:

```
@dataclass(frozen=True)
class MontageViewportCanvas:
    data: np.ndarray
    histogram_data: np.ndarray | None
    origin_x: int
    origin_y: int
    full_plan: MontagePlan
```

Then geometry maps like:

```
global_x = canvas_x + origin_x
global_y = canvas_y + origin_y
tile = full_plan.tile_at(global_x, global_y)
```

This keeps one ImageItem, avoids segfaults from many items, and preserves global montage coordinates.

#### Option B — keep local mini-montage but make it explicit

This is simpler but less correct. Only use when option A is not possible.

You would treat montage as a “page” of visible tiles, not a pan-able full coordinate space. Then disable continuous panning through montage and add paging controls.

I do not recommend this for your goal. You want a real viewer.

Add byte-based tile limits

Replace:

```
if len(visible_tiles) > 64:
    visible_tiles = visible_tiles[:64]
```

with something like:

```
selected = []
used = 0

for tile in visible_tiles:
    estimated = estimate_display_image_bytes(
        tile_shape,
        output_dtype,
        rgb=is_rgb,
        histogram=True,
    )
    if used + estimated > VISIBLE_RENDER_BUDGET_BYTES:
        break
    selected.append(tile)
    used += estimated
```

Count bytes, not tiles.

## Serious issue 3: PanelManager has body ownership holes

The PanelManager direction is correct, but there are state/reparenting bugs.

This is suspicious:

```
def hide_panel(self, name, *, reason, preserve_canvas=True):
    panel = self._panels_by_name[str(name)]
    if panel.dialog is not None:
        panel.dialog.hide()
    else:
        self.window.removeDockWidget(panel.dock)
    panel.dock.setVisible(False)
    panel.location = PanelLocation.HIDDEN
```

If a detached panel is hidden, the dialog still exists and still owns the body widget. But the panel location becomes HIDDEN.

Later show_docked() does not reliably take the body back from the dialog before docking:

```
if QtWidgets.QDockWidget.widget(panel.dock) is not panel.body:
    panel.dock.setWidget(panel.body)
```

That can leave the same body conceptually owned by a hidden dialog and a dock, or leave the dock with a placeholder. This matches your symptoms:

- empty dock space
- panel appears briefly then closes
- redocking closes instead
- View menu sometimes fails to open
- actions work once then break after interaction
- Fix panel lifecycle strictly

Make these transitions total and explicit:

DOCKED → HIDDEN:
  remove dock from main window
  keep body inside dock or detach body to manager-owned hidden parent
  no dialog

DOCKED → DETACHED:
  remove body from dock
  remove dock from main window
  create dialog
  put body in dialog

DETACHED → HIDDEN:
  take body from dialog
  close/delete dialog
  store body in manager-owned hidden container or back in hidden dock
  no dialog remains

DETACHED → DOCKED:
  take body from dialog
  close/delete dialog
  put body into dock
  add dock to main window

Add a helper:

```
def _destroy_dialog_and_take_body(panel):
    if panel.dialog is None:
        return panel.body

    body = panel.dialog.take_body()
    panel.dialog._closing_for_redock = True
    panel.dialog.close()
    panel.dialog.deleteLater()
    panel.dialog = None

    if body is not None:
        panel.body = body

    return panel.body
```

Then hide_panel() for detached panels should do this:

```
def hide_panel(self, name, *, reason, preserve_canvas=True):
    panel = self._panels_by_name[str(name)]

    if panel.location == PanelLocation.DETACHED:
        body = self._destroy_dialog_and_take_body(panel)
        if body is not None:
            body.setParent(None)
            panel.dock.setWidget(body)
        panel.dock.setVisible(False)
        self.window.removeDockWidget(panel.dock)
        panel.location = PanelLocation.HIDDEN
        self._sync_view_action(panel)
        return

    if panel.location == PanelLocation.DOCKED:
        panel.dock.setVisible(False)
        self.window.removeDockWidget(panel.dock)
        panel.location = PanelLocation.HIDDEN
        self._sync_view_action(panel)
        return
```

Also fix _hide_detached_from_dialog():

```
def _hide_detached_from_dialog(self, name):
    panel = self._panels_by_name[str(name)]
    body = panel.dialog.take_body() if panel.dialog is not None else panel.body
    panel.dialog = None
    panel.body = body
    if body is not None:
        body.setParent(None)
        panel.dock.setWidget(body)
    panel.location = PanelLocation.HIDDEN
    self._sync_view_action(panel)
```

Right now this method only changes state and action check state. That is not enough.

## Serious issue 4: StandardDockWidget still intercepts close events

You had simplified it before, but v15 still has:

```
class StandardDockWidget(QtWidgets.QDockWidget):
    def closeEvent(self, event):
        ...
        layout_manager.set_*_visible_from_user(False)
        event.ignore()
        return
```

Since you now have a PanelManager and custom dock title buttons, this override is probably unnecessary and can still create lifecycle weirdness.

If native dock close buttons are no longer used, remove the override entirely.

If you keep it as a fallback, it should call only one authoritative method:

```
def closeEvent(self, event):
    parent = self.parent()
    layout_manager = getattr(parent, "layout_manager", None)
    if layout_manager is not None:
        layout_manager.hide_panel_for_dock(self, reason="native-close")
        event.ignore()
        return
    super().closeEvent(event)
```

But my recommendation, that you need to try first, is simpler:

```
class StandardDockWidget(QtWidgets.QDockWidget):
    pass
```

The panel title bar already has a Hide button. Avoid multiple close semantics.

## Wayland resizing: why it fails and how to improve it

The observation is important:

- QT_QPA_PLATFORM=xcb fixes it.
- Manual resize makes later programmatic resize work.
- After failed programmatic resize, grabbing an edge makes it jump correct.

That strongly suggests this is not just a bad width calculation. It is an ordering/compositor/layout synchronization problem.

On Wayland, the compositor owns top-level window positioning. Qt’s own documentation recommends startSystemMove() instead of position-setting APIs because it lets the window manager handle native behavior; the docs explicitly note that on Wayland setPosition is not supported for that use case. KDAB’s KDDockWidgets documentation describes the same Wayland mismatch for floating dock windows: one title bar cannot cleanly be both a dock-drag area and a native move area, which is why they use separate title-bar concepts.

You already did the right thing for moving detached panels: use a QDialog and call windowHandle().startSystemMove().

The remaining issue is programmatic main-window resize while Qt’s dock layout and the Wayland compositor are still settling.

### Current problem

layout_controller.py still does this:

```
target = QRect(win.geometry())
target.setWidth(...)
target.setHeight(...)
target = self._clamp_to_available_screen(target)

if target != win.geometry():
    win.setGeometry(target)
```

This changes size and position together. Under Wayland, position control is not reliable, and top-level geometry changes are asynchronous. Even if Qt accepts the call, the compositor may apply it later or differently.

### Better approach

Do not preserve the canvas by precomputing a new top-level rectangle. Preserve it by measuring the central widget after the panel transition and correcting the top-level size.

Use this model:

1. Record central widget target size.
2. Apply panel transition.
3. Let QMainWindow layout settle.
4. Measure central widget actual size.
5. Resize main window by the difference.
6. Verify again after the compositor/layout has responded.

Use resize(), not setGeometry(), because you only want to change size, not position.

Pseudo-code:

```
def run_panel_transition_preserving_canvas(self, transition):
    win = self.window
    central = win.centralWidget()

    if win.isMaximized() or win.isFullScreen() or central is None:
        transition()
        return

    target_canvas_size = central.size()

    transition()

    self._schedule_canvas_size_correction(target_canvas_size, attempts=3)


def _schedule_canvas_size_correction(self, target_canvas_size, attempts):
    QtCore.QTimer.singleShot(
        0,
        lambda: self._correct_canvas_size(target_canvas_size, attempts),
    )


def _correct_canvas_size(self, target_canvas_size, attempts):
    win = self.window
    central = win.centralWidget()

    if central is None or attempts <= 0:
        return

    layout = win.layout()
    if layout is not None:
        layout.invalidate()
        layout.activate()

    current = central.size()
    dx = int(target_canvas_size.width()) - int(current.width())
    dy = int(target_canvas_size.height()) - int(current.height())

    if dx or dy:
        new_width = max(win.minimumWidth(), win.width() + dx)
        new_height = max(win.minimumHeight(), win.height() + dy)
        win.resize(new_width, new_height)

        # Wayland/compositor/layout may need another configure/layout pass.
        QtCore.QTimer.singleShot(
            16,
            lambda: self._correct_canvas_size(target_canvas_size, attempts - 1),
        )
    else:
        self.refresh_view_geometry()
```

For panel open/close/detach/redock, call:

```
self.run_panel_transition_preserving_canvas(
    lambda: panel_manager.show_docked(...)
)
```

or:

```
self.run_panel_transition_preserving_canvas(
    lambda: panel_manager.detach_panel(...)
)
```

### Important details

Do not call setGeometry() for this. Use resize().

Do not move the top-left position during panel preserve. If the window would exceed the screen, accept that the canvas cannot be perfectly preserved and show a one-time message or let the compositor constrain it.

Do not use a single immediate resize. Use a small verification loop, because the problem is ordering.

Do not use QApplication.processEvents() as the normal app solution. It can mask reentrancy bugs. Use QTimer.singleShot() in app code; use qtbot.wait() in tests.

Use QMainWindow.resizeDocks() only to size the dock area itself after the dock is present. QMainWindow.saveState()/restoreState() are for dock/toolbar layout state relative to the main window, not for your “preserve central canvas pixel size” contract.

### If Wayland still refuses

Add a setting:

Panel resize behavior:

- Preserve canvas size, best effort
- Do not resize main window automatically
- Ask / manual

You can make best-effort preserve the default. But under Wayland, do not promise absolute behavior. The compositor gets a vote.

## Performance: why FFTs and sliced scrolling still feel slow

The scheduler got better, but it is still not enough.

You now have separate one-worker controllers:

- visible
- pixel
- profile
- ROI
- prefetch

That is good. But the expensive function still often runs as one large NumPy call:

- start worker
- call np.fft / reduction / complex transform / montage tile loop
- cannot interrupt until NumPy returns
- newer request waits or competes
- old request may be stale but still burns CPU/RAM

Qt’s QThreadPool lets you set max thread count and clear queued tasks; the default max thread count is based on QThread::idealThreadCount(), and setMaxThreadCount() changes the limit. But clearing queued work does not stop a running NumPy FFT.

So the remaining issue is not “more threads.” It is:

- work must be chunked, bounded, cancellable between chunks, and cost-aware

### Do not immediately “fully use all available threads”

That can make the viewer worse.

For a viewer, responsiveness beats throughput. The visible render should usually have:

- max_workers = 1
- latest-only semantics
- high priority
- small memory budget

Heavy prefetch should use spare capacity only after visible work is idle.

If FFT internally uses multiple workers and you also launch multiple Qt workers, you can oversubscribe the CPU badly:

- Qt worker 1 uses 8 FFT workers
- Qt worker 2 uses 8 FFT workers
- ROI worker uses NumPy threads
- BLAS also uses threads
- system becomes sluggish

Phase 4e should explicitly manage this.

### FFT optimization plan

SciPy is already a project dependency, so the easiest win is to stop using np.fft directly and add an FFT backend abstraction.

SciPy’s FFT module supports worker control; scipy.fft.set_workers(workers) sets the default number of FFT workers within a context, and individual FFT functions also expose a workers argument.

#### Step 1: add operations/fft_backend.py

```
class FFTBackend:
    name: str

    def centered_fft(self, data, axis, *, workers=1):
        raise NotImplementedError

    def centered_ifft(self, data, axis, *, workers=1):
        raise NotImplementedError
```

Default backend:

```
from scipy import fft

class ScipyFFTBackend(FFTBackend):
    name = "scipy"

    def centered_fft(self, data, axis, *, workers=1):
        shifted = fft.fftshift(data, axes=axis)
        transformed = fft.ifft(shifted, axis=axis, norm="ortho", workers=workers)
        return fft.ifftshift(transformed, axes=axis)

    def centered_ifft(self, data, axis, *, workers=1):
        shifted = fft.fftshift(data, axes=axis)
        transformed = fft.fft(shifted, axis=axis, norm="ortho", workers=workers)
        return fft.ifftshift(transformed, axes=axis)
```

Your current naming uses CenteredFFT → ifft and CenteredIFFT → fft, probably because of MRI/k-space convention. That may be intended, but it is surprising. Keep the convention only if the UI labels make it clear.

#### Step 2: add optional pyFFTW backend

pyFFTW provides interfaces intended as drop-in replacements for NumPy/SciPy FFT APIs. It can be faster for repeated same-shape transforms because FFTW planning/wisdom can be reused, but planning itself costs time. So use it as an optional backend:

FFT backend:
  auto
  scipy
  pyfftw
  numpy fallback

Do not make pyFFTW mandatory yet. Packaging can be annoying.

#### Step 3: expose FFT worker budget

Add app settings:

FFT workers:
  auto
  1
  2
  4
  all minus one

Default should probably be: `min(4, max(1, os.cpu_count() // 2))`

Not “all threads.” The UI needs breathing room.

#### Step 4: add FFT cost metadata

Every operation should expose cost/capability metadata:

```
@dataclass(frozen=True)
class OperationCost:
    kind: Literal["view", "elementwise", "reduction", "transform"]
    requires_full_axis: tuple[int, ...] = ()
    output_dtype: np.dtype | None = None
    temp_multiplier: float = 1.0
    can_chunk: bool = False
```

FFT should report:

```
kind = transform
requires_full_axis = (axis,)
temp_multiplier ≈ 3–6
can_chunk = only across non-transform axes
```

That lets the scheduler decide whether to:

- run now
- defer
- warn
- use fewer workers
- skip prefetch
- reduce montage tiles

### Lazy compute needs an operation capability model

The current slab evaluator is useful, but it is still mostly a clever recursive slicer. To make future operations sustainable, add an explicit operation classification:

View/index-remap operations:
  Crop
  ReverseAxis
  FFTShift sometimes
  axis range
  slicing

Elementwise operations:
  Real
  Imag
  Abs
  Angle
  Conjugate
  scaling/windowing

Axis reductions:
  Mean
  Sum
  Min/Max
  RSS

Transforms:
  FFT/IFFT
  convolution/filtering later

Each operation should declare:

```
class OperationCapabilities(Protocol):
    def output_shape(self, shape): ...
    def output_dtype(self, dtype): ...
    def remap_request(self, request): ...
    def can_apply_to_slab(self, request): ...
    def required_input_region(self, request): ...
    def estimate_peak_bytes(self, request, input_dtype): ...
```

Then the evaluator becomes:

- request visible output
- walk operations backward to determine required input region
- estimate memory/cost
- if safe: evaluate slab
- if too expensive: refuse/degrade
- if chunkable: evaluate chunks with cancellation checks

This is the foundation for predictive caching too.

### Predictive caching: not yet, but design for it now

Your ideas.md already has the right instincts:

lazy/view-like ops
cache memory budget
performance HUD
benchmark MRI stacks
memmapped .npy
measure prefetch usefulness before clever heuristics
zarr/dask later
priority scheduler before operation-backed predictive prefetch

That is good. Keep those.

But predictive caching should not come before basic bounded rendering. The order should be:

1. Visible render is latest-only and bounded.
2. Tile/image cache reports real bytes.
3. Montage memory is controlled.
4. Operation cost estimates exist.
5. Then prefetch.

A good prefetch policy:

Only prefetch after 100–250 ms idle.
Never prefetch while visible render is running.
Never prefetch montage until tile renderer is stable.
Never prefetch if cache is >70–80% full.
Never prefetch op-backed data unless estimated cost is below threshold.
Prefetch in the direction the user is scrolling.
Prefetch at most 1–2 slices initially.
Measure hit rate before expanding.

Your current _prefetch_nearby_slices() still schedules up to 8 neighbors. That is fine for cheap raw slicing, but too aggressive for heavy data. Default off is right.

## Other bugs or weak spots that need tests

### Hidden detached panel re-show

Test this exact sequence:

- open Inspection dock
- detach Inspection
- close detached dialog
- open Inspection from View menu
- redock
- hide from View menu
- open again

Assertions:

- PanelManager.location is correct at every step
- panel.dialog is None when hidden/docked
- dock.widget() is panel.body when docked/hidden
- body parent is not the old dialog
- no placeholder remains in dock

### Cache byte accounting for all cached result types

Add tests for:

- DisplayImage
- RenderedTile
- line/profile arrays
- scalar values
- RGB images
- histogram_data

### Montage geometry after visible subset

Create a plan with 20 tiles, visible range over tiles 10–12, then render/apply visible tiles.

Assert:

- hover over tile 10 maps to source index 10
- display geometry does not renumber tile 10 as tile 0 unless intentionally using viewport-canvas origin
- panning does not reset coordinate space

### RSS memory during montage

Use psutil, not only Python object estimates. NumPy allocations often do not show fully in tracemalloc.

Test shape:

data shape: maybe (128, 1024, 1024) float32
montage range: 128 tiles
budget: small, e.g. 256 MiB

Assert:

- render refuses or loads bounded subset
- RSS increase stays below budget + tolerance

### Rapid scroll with slow FFT

Monkeypatch FFT to sleep and allocate a temporary, then scroll through many slices.

Assert:

- visible controller queue does not grow
- only latest result commits
- old stale callbacks do not clear overlay
- RSS does not climb monotonically

### Fit toggle

Test:

- Fit checked
- wheel event ignored
- drag ignored
- resize refits
- new image refits
- 1:1 unchecks Fit
- Fit disabled restores square-pixel aspect
- Fit/1:1 do not call render()
