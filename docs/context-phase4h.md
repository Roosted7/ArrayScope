# Phase 4h — interaction latency and progressive-render performance

## Issues and problems

### There is no true render coalescer yet

Every scroll/slice event still enters render() and does a lot synchronously:

```
_sync_controls_from_view_state()
_update_channel_controls()
update_dimension_controls()
update_complex_indicators()
update_shift_indicators()
update_image_view()
update_line_plot()
_update_operation_dock()
_sync_progressive_docks()
```

That is too much for high-frequency interaction.

The slice number freezing is probably not because NumPy alone is slow. It is because the UI event handler is doing full render orchestration before returning control to Qt.

A viewer should behave like this:

user scrolls
→ slice number updates immediately
→ view_state updates immediately
→ render request is coalesced
→ only the latest state renders
→ old render work is cancelled/ignored

Right now it is closer to:

user scrolls
→ full render orchestration starts
→ controls/docks/status/profile may update
→ cost planning may run
→ worker scheduling happens
→ Qt event loop returns later

That is why it feels sticky.

### Montage commits are still too expensive

This is the largest performance issue I found.

Each montage tile completion calls: `_commit_montage_session_canvas()`

That then does:

```
canvas = make_montage_viewport_canvas(...)
_apply_display_image(DisplayImage(data=canvas.data, histogram_data=canvas.histogram_data), ...)
_update_montage_tile_overlays(canvas)
```

That means every progressive tile update still:

- rebuilds the viewport canvas
- copies already-rendered tiles again
- recomputes display bounds
- updates histogram machinery
- calls ImageItem.setImage
- updates operation dock
- clears stale/evaluation overlay
- refreshes inspection dock
- recreates overlay graphics items

So yes, even if “data → rendered tile” is cached, the path from “cached tile → visible screen” is not instant.

PyQtGraph’s ImageItem docs explicitly warn that performance depends heavily on image data, levels, LUTs, and defaults; they recommend row-major data and note that automatic level calculation samples image pixels for performance. Repeatedly pushing new large arrays through setImage() plus histogram/level updates will cost real time.

### Montage tiles are scheduled one at a time

In _schedule_next_montage_tile() we use: `self.visible_evaluation_controller.start_latest(...)`

with:

```
max_workers = 1
replace_group = "visible-montage"
```

That gives correctness and avoids runaway memory, but it guarantees large montages pop in tile-by-tile.

For cold cache, some pop-in is expected. But the current architecture makes even cached/progressive updates heavier than necessary.

### The scheduler has priorities, but not full priority behavior

EvalPriority exists now. Good.

But this is still not a real priority queue. You have multiple controllers and latest-only semantics, which is useful, but not enough for:

- visible image
- montage tile work
- stage materialization
- profile
- ROI
- prefetch

The app needs clearer lanes:

UI thread:
  instant state/control updates only

visible render lane:
  latest-only, max 1

tile render lane:
  visible tiles, 2-ish workers, cancellable by session

stage materialization lane:
  expensive reusable stage, max 1, high value

ROI/profile lane:
  debounced, max 1

prefetch lane:
  idle-only, low priority

Right now montage tile work uses the visible render lane, so long-running tile work can block the next visible request.

### Stage cache retention is too simple

The current stage cache is a real improvement, but retention policy is still basic:

- cache the latest reusable cacheable stage
- earlier cacheable stages are mostly treated as superseded

That is fine for the first implementation. But it leaves performance on the table for pipelines where:

- an earlier expensive transform is reused by several downstream requests
- the final retained stage is too large
- a smaller intermediate stage would be more valuable
- different views/profiles/ROIs reuse different stage prefixes

The next version should score cache candidates by:

- estimated bytes
- estimated recompute cost
- operation kind
- hit count
- last access
- whether it serves visible render
- whether it serves only prefetch

### The test suite still has optional-backend and cancellation rough edges

The pyfftw test should not fail the base suite. pyfftw is optional.

The chunked cancellation test failing only in a broad run means some global or module state leaks across tests. That needs fixing before you trust the scheduler/cancellation layer.

## Why the viewer still feels laggy

### Cause 1: input events are not decoupled from rendering

The number should update at UI speed. Rendering should follow.

Scrolling should not call the full render path directly. It should call something like:

```
def set_slice_interactive(axis, index):
    self.view_state = self.view_state.with_slice(axis, index)
    self.dimension_controls.update_slice_text_immediately(axis, index)
    self.request_render_interactive(reason="slice-scroll")
```

Then:

```
def request_render_interactive(self, reason):
    self._pending_render_reason = reason
    self._pending_render_state = self.view_state
    self._render_timer.start(0)      # or 8–16 ms
```

Only the final state in a burst renders. The UI controls update immediately.

This alone will make scrolling feel dramatically better.

### Cause 2: cached tile display is not cheap enough

A cache hit saves evaluation time, but it does not save:

- canvas rebuild
- array copy
- histogram/level recompute
- ImageItem.setImage conversion
- overlay recreation
- side-panel refresh

We need a fast progressive-commit path, separate from full render commit.

Current _apply_display_image() is too broad. It should not be used for every tile update.

Split it:

```
_apply_full_display_image()
  full render completion
  levels/histogram
  operation dock
  inspection dock
  stale/overlay final state

_apply_montage_canvas_progress()
  update canvas image only
  preserve viewport
  no operation dock refresh
  no inspection refresh
  no auto levels unless forced
  do not clear session overlay if session incomplete
```

### Cause 3: tile canvas is rebuilt rather than patched

The session should own mutable canvas buffers:

```
session.canvas_data
session.canvas_histogram
session.tile_states
session.dirty_rects
```

When one tile finishes:

- copy only that tile’s clipped region into session.canvas_data
- mark that tile loaded
- schedule a throttled UI flush

The UI flush should happen at maybe 30–60 Hz:
many tile completions → one screen update

Right now every commit conceptually reconstructs the current canvas from all loaded tiles. That scales poorly.

### Cause 4: histogram/levels are in the hot path

ImageView2D.setImage() calls updateImage(), which calls _updateImageLevels(), which calls finite_bounds(). For progressive montage, this happens repeatedly.

For progressive tile updates, levels should usually be frozen:

first commit:
  use previous levels or sampled visible estimate

during tile loading:
  keep levels fixed

final commit:
  optionally update sampled levels once

Do not recompute histogram bounds per tile.

### Cause 5: overlay graphics items are recreated every commit

setMontageTileOverlays() clears and recreates QGraphicsRectItem and TextItem objects each time.

For many tiles, this churn is expensive.

Better:

- one custom QGraphicsItem paints all tile-state overlays
- or persistent overlay items keyed by tile index

A single custom item is cleaner, so lets try that first!

### Cause 6: old work cannot truly be killed inside NumPy/SciPy calls

Cancellation tokens help only between calls/chunks. Once a NumPy/SciPy FFT is running, Python cannot safely kill it mid-call. SciPy FFT does support a workers argument, but the docs note that workers split independent 1D FFTs within the input, requiring enough non-transformed axes; small inputs may use fewer workers than requested.

So the fix is not “more threads everywhere.” The fix is:

- do not start obsolete work
- chunk where safe
- cache expanded stages
- use conservative FFT workers
- keep UI thread free

## Concrete bugs / suboptimal code paths I found

### EvaluationController.clear_group() can corrupt queued prefetch state

clear_group() calls: `self.pool.clear()`

but only cleans normal _requests. Prefetch runnables live in:

```
_prefetch_keys
_runnables
_handlers
```

If a prefetch runnable is queued and pool.clear() removes it, your Python bookkeeping may still think it exists. That can make diagnostics wrong and can cause has_running_or_pending() to remain true longer than reality.

This matters because profile prefetch uses the same controller as profile live evaluation.

#### Fix

Use separate controllers for profile exact work and profile prefetch.

### _apply_display_image() is too broad for progressive montage

It does:

```
self.img_view.setImage(...)
self.display_geometry = geometry
self._update_operation_dock()
self.apply_axis_flips()
self.img_view.setImageStale(False)
self.img_view.setEvaluationOverlay(False)
self._refresh_inspection_dock()
```

This is fine for final full render. It is wrong for every montage tile commit.

This likely explains your “slow update but no overlay” observation: a session-level slow overlay can be set, then a tile commit calls _apply_display_image() and clears it even though more tiles are still pending.

### _commit_montage_session_canvas() rebuilds too much

This path:

```
make_montage_viewport_canvas(...)
_apply_display_image(...)
_update_montage_tile_overlays(...)
```

is the main reason tiles still pop in slowly.

The session should not rebuild the canvas from scratch every time.

### evaluate_image_snapshot() plans twice

It does:

```
plan = plan_slab(document, request)
slab = evaluate_slab(document, request, ...)
```

and evaluate_slab() calls plan_slab() again.

This is not the biggest bottleneck today, but it is unnecessary synchronous work and will grow as planning becomes smarter.

#### Fix

```
plan = plan_slab(...)
slab = evaluate_slab_from_plan(document, request, plan, ...)
```

Same for line/scalar/export.

### StageCache.get_containing() is linear over all entries

For now this is acceptable. But once stage cache becomes important, scanning every cached stage and running region_contains() each time may become noticeable.

Add an index by:

- document_key
- operation_prefix
- dtype
- shape

Then only run region containment checks inside the matching bucket.

### image_key() includes montage fields

The general image key includes:

- montage_axis
- montage_columns
- montage_indices

For normal image caching that may be okay. For tile caching, verify that the tile key is independent of layout-only montage choices. A rendered tile for source index 42 should not become invalid just because the montage column count changed.

You probably avoid most of this by using tile.view_state.with_montage_axis(None), but I would add a test:

```
render tile index 10 with 4 columns
change montage columns to 6
same source tile cache key still hits
```

## Should “data → rendered tile” be instant?

Sometimes, yes. But not always.

A fully hot path should be almost instant:

- stage cache hit
- tile cache hit
- same channel/levels
- same display dtype/layout
- canvas patch only
- no histogram recompute
- no side-panel refresh

But currently a hot tile still travels through too much display machinery.

Cold tile rendering can still take time because it may require:

- region planning
- stage cache lookup
- operation execution
- FFT/reduction/complex conversion
- display conversion
- histogram source generation
- tile cache store
- canvas copy
- ImageItem update

The target should be:

hot tile:
  < 5–10 ms end-to-end, ideally below a frame

cold cheap tile:
  appears quickly, possibly within 1–2 frames

cold expensive tile:
  visible per-tile loading state, then appears when ready

Right now hot cached tiles can still be slowed by full canvas/application work.

### Split out the hot path

window/render.py is still doing too much. It is currently the place where state sync, rendering, montage, cache, scheduler, side panels, viewport, and diagnostics all meet. That file is now over 1,600 lines. It is the pressure point.

The next refactor should not be aesthetic. It should split the hot path:

RenderCoordinator
  coalesces render requests, owns latest state

MontageRenderer
  owns MontageRenderSession, tile scheduling, canvas patching

DisplayCommitter
  full commit vs progress commit

ViewportController
  fit/1:1/range logic

SidePanelRefreshController
  debounced profile/inspection/operations refresh

This will make performance easier because each path can say “I do not update side panels” or “I do not recompute levels.”

## Proposed subphases

### instrument before optimizing further

Add timings around the real hot path:

```
render() total synchronous time
control sync time
dimension controls update time
operation dock update time
planning time
worker queue wait time
evaluation time
stage cache lookup time
tile cache lookup time
canvas compose/patch time
ImageItem.setImage time
histogram/levels time
overlay update time
inspection refresh time
profile refresh time
```

Show these in the diagnostics window.

For montage sessions, show:

```
visible tiles
cached tiles
missing tiles
loading tiles
skipped tiles
tile cache hit rate
stage cache hit rate
last tile eval ms
last canvas compose ms
last setImage ms
last overlay ms
```

Without this, we will keep guessing.

### add render coalescing and a fast slice path

Implement:

```
def request_render(self, *, reason, interactive=False):
    self._pending_render_state = self.view_state
    self._pending_render_reason = reason
    self._render_request_timer.start(0 if not interactive else 16)
```

Then for scroll/slice changes:

- update slice index control immediately
- update ViewState immediately
- request render interactively
- return to Qt

Do not call full render() directly on every wheel/slider event.

During rapid scrolling:

- only latest state renders
- old queued visible work is cleared
- profile/ROI refresh is deferred
- operation dock refresh is deferred

This is probably the highest-impact responsiveness fix.

### split full display commit from progress commit

Create two paths:

```
apply_full_display_image(...)
apply_progress_display_image(...)
```

Full commit can do:

- levels
- histogram
- operation dock
- inspection refresh
- profile refresh if visible
- clear overlays

Progress commit should do:

- update image data
- preserve viewport
- update geometry
- maybe update lightweight tile overlay
- nothing else

Montage tile commits should use the progress path.

### make montage canvas mutable and patch-based

Change MontageRenderSession so it owns:

- canvas_data
- canvas_histogram
- tile_states
- dirty_rects

When a tile finishes:

- copy only that tile into the existing canvas buffer
- mark dirty
- schedule one UI flush

Do not call make_montage_viewport_canvas() on every tile.

Use it only when:

- new montage session
- viewport rect changes
- tile shape changes
- channel changes
- memory budget changes

### Phase 4h.5 — replace tile overlay churn

Replace:

```
clear all QGraphicsRectItem/TextItem
create all again
```

with custom QGraphicsItem that paints tile-state overlays. With a simple and cached loading/processing icon.

### give montage tiles their own worker lane

Do not schedule each tile through the visible image controller.

Use:

visible_evaluation_controller:
  full image / non-montage latest-only, max 1

tile_evaluation_controller:
  montage visible tiles, max 2 initially

stage_evaluation_controller:
  large reusable stage materialization, max 1

prefetch_evaluation_controller:
  idle-only, max 1

For montage:

- commit cached tiles immediately
- schedule missing visible tiles first
- schedule near-viewport margin tiles only after idle
- cancel whole session on new montage state

Start with max_workers=2 for tiles. More than that risks CPU/memory pressure.

### improve stage-cache policy

Move from “latest reusable stage wins” to scoring:

```
score =
    recompute_cost_weight
  + visible_reuse_weight
  + hit_count_weight
  - memory_cost_weight
  - prefetch_only_penalty
```

Keep final expanded stages high priority, but do not automatically discard all earlier expensive stages.

Also add a cache-key test for render-only state:

- stage cache key must not include viewport position
- stage cache key must not include montage columns
- stage cache key must not include loading/progress state

### Phase 4h.8 — optimize FFT carefully

Use SciPy FFT workers conservatively. SciPy exposes workers, and scipy.fft.set_workers() can set a default in a context.

Do not “use all cores” by default. A viewer needs spare CPU for the UI.

Recommended defaults:

visible exact render:
  FFT workers = min(4, max(1, os.cpu_count() // 2))

prefetch:
  FFT workers = 1

background materialization:
  user configurable

tile rendering:
  tile workers × FFT workers must not exceed sane CPU budget

Optional pyFFTW is still worth keeping, but only as optional. Fix the test so the base suite does not require it.

For future large stage cache, consider disk-backed stages using numpy.memmap; NumPy documents memmap as array-like access to small segments of large disk-backed arrays without reading the whole file into memory.

## Immediate fixes, we need ASAP

### Fix EvaluationController.clear_group()

Do not let pool.clear() orphan prefetch bookkeeping.

Fix: do not mix exact profile work and profile prefetch in one controller.

### Fix optional pyFFTW test

Use: `pytest.importorskip("pyfftw")` or mark it optional.

### Investigate the chunked cancellation flake

Add reset fixtures for global FFT/runtime options and any scheduler/runtime globals.

This test should not depend on order.

### Stop using _apply_display_image() for montage progress

This is the big performance bug.

### Add a render coalescer before doing more FFT work

If you optimize FFT first, the UI will still feel sticky because the event path is overloaded.