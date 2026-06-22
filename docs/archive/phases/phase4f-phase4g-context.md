# Context for phase 4f and phase 4g (and fully completing other phase 4's)

Implemented note: Phase 4f P2/P3 now provide a psutil-backed `MemoryPolicy`, split image/tile/profile cache budgets, policy-driven render and prefetch gates, and Developer -> Diagnostics with visual usage bars plus text sections. The policy exposes a stage-cache budget that Phase 4g now uses for the in-memory StageCache.

Implemented note: Phase 4g P0 added operation-declared capabilities plus pure region/planner
contracts. Cost estimates now consume the operation declarations, and slab plans expose final/required
regions and candidate stage-cache metadata for diagnostics. Runtime region execution and StageCache
allocation remain future Phase 4g steps.

Implemented note: Phase 4g P1 moved runtime slab evaluation onto `RegionPlan` transitions. Registered
operations now own their region mapping and regional application, and `operations.slabs` executes the
planner output instead of carrying operation-specific request-expansion branches. Developer Diagnostics
shows final/required regions, expanded axes, transitions, candidates, and peak estimates. StageCache
allocation remains future work.

Implemented note: Phase 4g P2/P3 added an in-memory, policy-budgeted StageCache owned by
`OperationEvaluator`. Planner candidates are used as cache lookup/store boundaries, so expanded
FFT/IFFT stages can be reused across slices, montage tiles, profiles, scalar reads, prefetch, and
evaluator-backed export. Disk-backed cache moved back to ideas. The later P4 optimizer handles safe
runtime operation simplification without rewriting user recipes.

Implemented note: Phase 4g P4 added a Qt-free internal runtime optimizer. It preserves the user-authored
operation stack while simplifying same-axis FFT/IFFT pairs with dtype preservation, reverse pairs,
conjugate pairs, adjacent same-axis crops, and adjacent dtype casts. StageCache now retains only useful
planner candidates by default, with fallback to an earlier fitting candidate when the preferred retained
stage is oversized. Disk-backed cache remains an idea; broader future work is chunked/cancellable long
operations and optional backends.

## P0: viewport montage currently shrinks/crops to loaded tiles

This likely explains several of our montage symptoms:

- tiles cropped midway
- missing tiles after update
- NaN hover where an image should eventually exist
- live profile cannot access expected tiles
- fragile behavior when the tiled dimension is scrolled

The core issue is in make_montage_viewport_canvas():

```
loaded_rect = plan.display_rect_for_tiles(loaded_tiles)

if view_range is None:
    rect = loaded_rect
else:
    rect = _rect_for_view_range(view_range, plan, viewport_shape)
    rect = _intersect_rect(rect, (0, 0, full_width, full_height))
    rect = _intersect_rect(rect, loaded_rect) or loaded_rect
```

That last line is the problem: `rect = _intersect_rect(rect, loaded_rect) or loaded_rect`

It means the displayed canvas is not “the requested viewport area.” It is “the part of the requested viewport area that happens to have loaded tiles.”

That makes the image coordinate system unstable. If only some tiles are loaded, the canvas can shrink around those loaded tiles. Then the displayed image no longer represents the viewport. That can break hover, live profile, panning, and tile edges.

### Fix

The canvas rectangle must be based on the requested viewport, not the loaded tiles.

Use this rule:

- Canvas rect = requested viewport rect, clipped to full montage bounds and memory budget.
- Loaded tiles are copied into that rect.
- Unloaded tiles remain visible as placeholders/loading areas.
- Gaps remain visibly empty.

Conceptually:

```
if view_range is None:
    rect = initial_viewport_rect(plan, viewport_shape)
else:
    rect = _rect_for_view_range(view_range, plan, viewport_shape)
    rect = _intersect_rect(rect, (0, 0, full_width, full_height))

# Do NOT intersect with loaded_rect.
```

Then initialize:

```
data = placeholder_image(canvas_shape)
histogram_data = np.full((height, width), np.nan, dtype=np.float32)
loaded_mask = np.zeros((height, width), dtype=bool)
tile_state_map = ...
```

For unloaded tile regions, show a clear placeholder/overlay instead of silently showing zero-valued pixels. Zero looks like real data. A missing tile should look missing.

## P0: montage is not actually tile-by-tile yet

update_montage_view() still evaluates all missing visible tiles inside one worker:

```
def evaluate():
    return tuple(
        (tile, evaluate_image_snapshot(document, tile.view_state, colormap_lut=colormap_lut))
        for tile in missing_tiles
    )
```

So the UI cannot progressively update tile by tile. It only receives results after all missing visible tiles finish.

This explains:

- slow montage update but no useful incomplete overlay
- tiles only appear at the end
- no per-tile loading indicator

### Fix

Add a MontageRenderSession.

```
@dataclass
class MontageRenderSession:
    generation: int
    key: object
    plan: MontagePlan
    canvas_rect: tuple[int, int, int, int]
    loaded_tiles: dict[int, RenderedTile]
    missing_tiles: set[int]
    loading_tiles: set[int]
    skipped_tiles: set[int]
```

Then:

1. Build stable viewport canvas rect.
2. Commit cached tiles immediately.
3. Mark missing tile regions as loading.
4. Schedule tile jobs individually through the one-worker visible controller.
5. When each tile finishes:
   - check generation
   - store tile in cache
   - copy it into the existing canvas
   - update the image, throttled to ~30 Hz
6. If the user scrolls/pans:
   - cancel/ignore old session
   - keep any already cached tiles

We do not need many ImageItems. Keep one image canvas, but update it progressively.

Add one extra overlay layer for tile states:

- loaded
- loading
- missing because over budget
- gap
- outside montage

Hover/profile behavior should distinguish those:

- gap                   → no value
- unloaded/loading tile → “tile loading…”
- budget-skipped tile   → “tile skipped by memory budget”
- loaded tile           → normal value/profile

Do not return plain NaN for “tile should exist but is not loaded yet.” NaN is a data value. Loading/missing is a state.

## P0: operation caching is still too shallow

Take this FFT/iFFT example as the key design insight.

When doing this:

- 3D dataset
- view axes 0 and 1
- slice axis 2
- operation stack: FFT(axis=2), IFFT(axis=2)
- scroll axis 2

Current behavior effectively says:

```
for each z slice:
  request image for z=k
  evaluator realizes FFT over axis 2 needs full axis 2
  computes full axis-2 transform for this request
  computes full axis-2 inverse transform for this request
  extracts z=k
```

Then for z=k+1, it does it again.

That is insane!

The current cache stores final display images keyed by exact ViewState. That means it can cache:

- image at z=0
- image at z=1
- image at z=2

But it does not cache:

- the transformed stage containing all z
- the final post-IFFT stage containing all z

For FFT-style operations, that is the wrong caching level.

### What we need

A stage cache, not just an image cache!

Think of the pipeline as stages:

stage 0: base data
stage 1: after FFT(axis=2)
stage 2: after IFFT(axis=2)
stage 3: display channel/render conversion

When a render request for z=12 forces the evaluator to compute all of axis 2, it should cache the larger result it had to compute anyway:

```
stage 1, region [:, :, :]
stage 2, region [:, :, :]
```

Then scrolling to z=13 should be:

```
cache hit: stage 2, region [:, :, :]
extract [:, :, 13]
render
```

No FFT. No IFFT. No recompute.

### Proposed data model

```
@dataclass(frozen=True)
class AxisRegion:
    kind: Literal["point", "slice", "indices", "all"]
    value: object


@dataclass(frozen=True)
class RegionSpec:
    axes: tuple[AxisRegion, ...]


@dataclass(frozen=True)
class StageKey:
    document_key: object
    operation_prefix_hash: str
    region: RegionSpec
    dtype: str
    shape: tuple[int, ...]
@dataclass
class StageValue:
    data: object
    region: RegionSpec
    nbytes: int
    stage_index: int
    priority: CachePriority
    last_access: float
    recompute_cost: float
```

Cache priorities:

highest:
  final expanded stage needed for current interaction

high:
  expensive transform output reused by downstream stages

medium:
  selected ROI/profile exact results

low:
  prefetch results

lowest:
  speculative intermediate stages

For the FFT/iFFT example, the final post-IFFT expanded result should be kept ahead of the intermediate FFT result. If memory pressure rises, evict the intermediate first.

## P1: operation capabilities should move into operations, not stay as type checks

arrayscope/operations/cost.py is useful, but right now it is still mostly central logic that recognizes operations by type/name.

That will not scale.

Each operation should declare its own capabilities:

```
@dataclass(frozen=True)
class OperationCapabilities:
    kind: Literal["view", "elementwise", "reduction", "transform"]
    blocking_axes: tuple[int, ...]
    chunkable_axes: tuple[int, ...]
    expands_request_axes: tuple[int, ...]
    cache_stage: bool
    temp_multiplier: float
    can_fuse: bool
```

Examples:

Crop:
  kind = view
  blocking_axes = ()
  cache_stage = false

Reverse:
  kind = view
  blocking_axes = ()
  cache_stage = false

Abs / Real / Imag:
  kind = elementwise
  chunkable_axes = all
  can_fuse = true
  cache_stage = usually false

Mean(axis=2):
  kind = reduction
  blocking_axes = (2,)
  cache_stage = maybe true

FFT(axis=2):
  kind = transform
  blocking_axes = (2,)
  expands_request_axes = (2,)
  cache_stage = true
  temp_multiplier = 4–6

Then the evaluator can plan intelligently:

requested final region
→ walk backwards through ops
→ discover required input region
→ discover expanded stage outputs
→ decide which stage to cache
→ estimate peak memory
→ choose exact / chunked / preview / refuse

This is the foundation for advanced operations. Without it, every expensive operation will become a special case.

## P1: the scheduler exists, but it is not yet the real scheduler

EvalPriority now exists:

```
class EvalPriority(IntEnum):
    VISIBLE_IMAGE = 0
    LIVE_PROFILE = 10
    SELECTED_ROI = 20
    HOVER_EXACT = 30
    PREFETCH = 40
```

Good. But the scheduler is not fully using the priority/cost model yet.

For example, start_prefetch() accepts:

```
memory_budget_bytes
idle_deadline_ms
```

but discards them. That is a smell. Enforce those arguments and remove unused code that adds confusion.

The visible render path is closer to “latest only,” but heavy NumPy/SciPy calls still cannot be interrupted mid-call. That means cancellation only works between chunks/stages. This is why the next evaluator design must chunk work around operation capabilities, not simply around image display axes.

For FFT specifically, do not blindly “use all available threads.” SciPy’s workers argument parallelizes independent FFTs, but it only helps when the non-transformed axes provide enough independent work; otherwise fewer jobs may be used. Oversubscribing Qt workers plus FFT workers can make the viewer feel worse.

Recommended policy:

visible render pool:
  max workers = 1
  latest-only
  highest priority

stage materialization pool:
  max workers = 1
  cancellable between chunks/stages

ROI/profile pool:
  max workers = 1
  debounced

prefetch pool:
  max workers = 1
  idle-only
  disabled under pressure

Then FFT workers are controlled separately:

FFT workers default:
  auto = conservative, e.g. min(4, max(1, os.cpu_count() // 2))

## Wayland resizing: your hack makes sense, but encapsulate it

Our current preserve-canvas logic is ugly, but I understand why it works.

It does roughly this:

```
record central widget size
apply panel transition
activate layout
resize window by central-widget delta
retry after short timers
on final failure/success, temporarily force min=max to desired size
poke QWindow.resize()
requestUpdate()
nudge width +1
then set it back
release constraints
```

That is not pretty, but it is consistent with our earlier observation:

- manual resize makes future size settle
- after failed programmatic resize, grabbing an edge makes it jump correct

The nudge/constraint trick is basically forcing the compositor/Qt backing window to acknowledge a size transaction that a plain resize() did not commit reliably under Wayland.

Qt’s docs support part of your direction: setGeometry() changes both position and size, and Qt recommends resize() when you want fixed size without forcing position. Qt also notes that not all systems support setting/querying top-level positions; Wayland is specifically called out for startSystemMove() being the way an app can influence position.

### Keep the working behavior, but clean the ownership

I would not spend a week finding the mathematically minimal Wayland trick. If this version works reliably on your compositor, keep it for now.

But make it less invasive:

1. Move it into a dedicated class:
   WaylandCanvasPreserver or CanvasPreserveTransaction.

2. Gate the strong constraint/nudge path:
   - use normal resize correction first
   - only use strong nudge on Wayland or after failed normal attempts

3. Capture and restore real constraints:
   current QWidget minimum/maximum
   current QWindow minimum/maximum
   not hard-coded 0 / QWIDGETSIZE_MAX

4. Remove production print() calls.
   Use logging or the debug window.

5. Add a setting:
   Panel resize behavior:
     off
     best effort
     strong Wayland

Implemented note: Phase 4f P4 moved this behavior into `CanvasPreserveController`, added explicit
Off / Best effort / Strong Wayland modes, removed the preserve-canvas stdout prints, and surfaced
preserve state in Developer -> Diagnostics. StageCache remains future Phase 4g work despite the memory
policy already exposing a placeholder budget.

Also, do not use direct window-manager tricks beyond Qt. Under Wayland, the compositor owns top-level window management; Qt’s startSystemMove() and startSystemResize() are the supported native interactive hooks.

## Memory limits: unify them and make them adaptive

Right now we have:

```
old operation-panel cache sizes/limits
render_memory_budget_mb setting
VISIBLE_RENDER_BUDGET_BYTES
MONTAGE_BUDGET_BYTES
PREFETCH_BUDGET_BYTES
hidden/static state inside render/evaluator code
```

That is too many concepts.

### Add one MemoryPolicy

Create:

```
@dataclass(frozen=True)
class MemoryPolicy:
    system_total_bytes: int
    system_available_bytes: int
    process_rss_bytes: int
    input_nbytes: int | None

    visible_render_budget_bytes: int
    montage_canvas_budget_bytes: int
    image_cache_budget_bytes: int
    tile_cache_budget_bytes: int
    stage_cache_budget_bytes: int
    prefetch_budget_bytes: int

    user_hard_cap_bytes: int | None
    profile: Literal["conservative", "balanced", "aggressive", "custom"]
```

Use psutil for system memory. psutil.virtual_memory() reports total/available memory, and the docs show available as the practical “can be used without swapping” metric across platforms.

Qt is good for screen/window geometry, not system memory. Use Qt for display geometry and psutil for RAM.

### Use adaptive budgets, but with hysteresis

Do not simply say: `budget = 50% of currently free memory`

That will fluctuate and make behavior unpredictable.

Use: `budget = function(system_total, system_available, input_nbytes, user profile)`

Then update slowly:

- never shrink an active render budget mid-render
- only shrink cache budgets between operations
- use hysteresis, e.g. update only if available memory changes by >20%

A reasonable first policy:

```
available = psutil.virtual_memory().available
total = psutil.virtual_memory().total
input_nbytes = document_base_nbytes(document)

visible_render_budget = clamp(
    min(available * 0.10, total * 0.04),
    lower=128 MiB,
    upper=user_hard_cap_or_2 GiB,
)

montage_canvas_budget = visible_render_budget

tile_cache_budget = clamp(
    min(available * 0.15, max(visible_render_budget, input_nbytes)),
    lower=256 MiB,
    upper=user_hard_cap_or_4 GiB,
)

stage_cache_budget = clamp(
    min(available * 0.30, max(2 * input_nbytes, 512 MiB)),
    lower=256 MiB,
    upper=user_hard_cap_or_8 GiB,
)

prefetch_budget = min(256 MiB, stage_cache_budget * 0.10)
```

The exact numbers can change, but the structure is right:

- visible render budget: small and safe
- tile cache: bounded, viewer-oriented
- stage cache: larger, because it avoids recomputing expensive transforms
- prefetch: small and disposable

## Add the simple floating debug window

Yes! We need this to get insight into what the viewer is doing.
Do not make it a managed dock. Do not involve it in canvas-preserving resize logic. Make it a plain developer QDialog.

Open it from: `Developer → Diagnostics`

Show:

```
Process RSS
System total / available RAM
Input data nbytes
Visible render budget
Montage canvas budget
Image cache bytes / entries / hit rate
Tile cache bytes / entries / hit rate
Stage cache bytes / entries / hit rate
Prefetch queue length
Visible/profile/ROI/prefetch scheduler diagnostics
Current render decision
Current render generation
Current montage plan: loaded/loading/missing/skipped tiles
Last render time
Last FFT backend/workers
Last operation plan
Last memory refusal reason
```

Update every 250–500 ms. Keep it ugly and useful. This will save you hours.

## FFT optimization: yes, but cache first

PyFFTW can help, but it should not be the first fix.

The first fix is to stop recomputing the same transform result on every scroll.

Order should be:

1. Stage cache / operation planner
2. SciPy FFT workers policy
3. Optional pyFFTW backend
4. Optional FFTW wisdom/plans
5. Optional disk-backed stage cache

PyFFTW can be useful as an optional backend, especially for repeated same-shape FFTs, but keep it optional.

The test should become: `pyfftw = pytest.importorskip("pyfftw")` or:

```
if "pyfftw" not in available_fft_backends():
    pytest.skip("pyFFTW not installed")
```

Do not require it in the base test suite.

Disk-backed stage caching is now explicitly future work in `docs/ideas.md`. The in-memory planner/cache
path should collect real usage data before adding memmap or Joblib-inspired persistence.

Dask and Zarr are still future backend ideas, not immediate fixes. Dask Array is built around blocked algorithms over chunks and can work larger-than-memory, while Zarr is a chunked N-dimensional storage format; both are relevant later, but neither replaces the need for your viewer-specific render planner, stage cache, and scheduler.

## Other bugs and suboptimal things

### Roadmap/docs are too optimistic

Your docs now say many Phase 4e items are done that are really partial.

Mark these as partial, not done:

- bounded montage rendering
- progressive tile rendering
- cooperative cancellation
- memory-budget enforcement
- hidden-panel compute avoidance
- scheduler v2
- stress tests

The code has scaffolding, but the user-visible behavior is not stable yet.

### start_prefetch() accepts arguments it ignores

This is misleading:

```
memory_budget_bytes=None
idle_deadline_ms=None
```

but the function discards them. Lets properly enforce them, and prevent such misleading situations in the future!

### Chunked cancellation has a test-order smell

The cancellation test passes alone but failed in the broader subset I ran. That usually means either:

- global runtime state leaking between tests
- module-level monkeypatch sensitivity
- test collection side effect

Do not ignore this. Test flakes are early warning signs in stateful GUI/evaluator code.

### Stale visible/montage results may still clear overlays

In update_montage_view.done():

```
if generation_key != current_key:
    self.img_view.setImageStale(False)
    self.img_view.setEvaluationOverlay(False)
    return
```

A stale result should usually do nothing. It should not clear an overlay that may belong to a newer render.

Same rule everywhere:

stale result:
  drop it
  increment stale counter
  do not mutate UI

### ChunkedImageResult appears unused

Wire it in! (Or remove it!) Dead scaffolding makes the evaluator harder to reason about.

### Memory stress test depends on qtbot

That is fine if it is a GUI test, but then mark it as such and make sure the dev/CI environment has pytest-qt. Otherwise it should skip cleanly.
