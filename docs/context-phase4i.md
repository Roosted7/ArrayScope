# Phase 4i — stage-first rendering and hot-path cleanup

This phase follows the Phase 4h latency work. Phase 4h added the render coalescer,
fast slice-control updates, progressive montage sessions, patch-based montage
canvas updates, a dedicated montage controller, expanded timing diagnostics, and
StageCache scoring. Phase 4i finishes the remaining latency work by making stage
materialization explicit, invalidating stale work by generation, and giving
progressive montage a display path that avoids full image, histogram, and panel
refresh work.

## Review findings

### Finding 1 — stale-work commits are still possible

Problem: async image callbacks compare request keys against captured old state
instead of the current viewer state. A failure path is:

1. An old async render for slice 0 is running.
2. The user scrolls to slice 1.
3. The newer state is a cache hit, degraded preview, or refusal, so no replacement
   visible-image worker is started.
4. The old worker finishes.
5. The callback's stale check compares against its captured `ViewState` and can
   commit the old image.

This explains the reported backward-then-forward jitter.

Selected fix: add an explicit render generation owned by the render coordinator.
Every visible-output state change advances the generation. Every async callback
captures the generation. Every callback compares both the captured generation and
the current evaluator key before committing.

`EvaluationController.clear_group()` also needs to invalidate the group even when
no replacement job is submitted. Clearing queued work must not leave callbacks or
prefetch bookkeeping able to appear current.

Montage tile jobs need cancellation tokens. A tile job cannot interrupt a
single NumPy/SciPy FFT call, but it can stop before/after stage transitions and
avoid scheduling follow-up work for stale sessions.

### Finding 2 — pure tests import GUI dependencies

Problem:
`tests/core/test_runtime_diagnostics.py` imports
`arrayscope.window.evaluation_controller`, which triggers
`arrayscope/window/__init__.py`. That initializer eagerly imports
`ArrayScopeWindow`, so pure diagnostics tests pull in `pyqtgraph`.

Selected fix: keep `arrayscope.window` package initialization lightweight. Remove
the eager GUI import and lazy-load `ArrayScopeWindow` only when the attribute is
requested.

Example:

```python
"""Main window package."""

from arrayscope.window.domain import Domain

__all__ = ["Domain", "ArrayScopeWindow"]


def __getattr__(name):
    if name == "ArrayScopeWindow":
        from arrayscope.window.main import ArrayScopeWindow

        return ArrayScopeWindow
    raise AttributeError(name)
```

### Finding 3 — FFT montage recomputes expensive stages

For a 3D dataset with image axes `(0, 1)`, montage/slice axis `2`, and
`CenteredFFT(axis=2)`, the planner correctly expands a tile request:

```text
final requested tile: [:, :, z]
required input:       [:, :, :]
stage candidate:      FFT result for [:, :, :]
```

Required behavior:

1. Compute the expanded FFT stage once.
2. Store the stage in StageCache and share the in-flight result.
3. Render each montage tile by slicing from the transformed stage.

Current failure modes:

- when the stage candidate is larger than `stage_cache_budget_bytes`, every tile can
  recompute the same FFT even when visible render and montage budgets look fine;
- when two cold tile workers start concurrently, both can miss StageCache and both
  can compute the same expanded stage;
- tile workers can directly trigger expensive stage computation instead of
  waiting on one explicit stage materialization job.

Selected fix: use stage-first montage planning. Montage sessions detect the common
expanded cacheable stage needed by visible tiles, schedule one materialization job
for that stage, attach duplicate requests to the in-flight job, and then render
tiles from the cached stage.

Singleflight example:

```text
tile 0 asks for FFT stage [:, :, :]
  cache miss -> reserve one stage materialization job

tile 1 asks for FFT stage [:, :, :]
  sees the stage in flight -> attach and wait

stage job finishes
  tile 0 and tile 1 render from the shared stage
```

Stage-first montage flow:

```text
update montage session
planner finds the common expanded FFT stage for visible tiles
StageCache lookup
if missing:
  schedule one stage materialization job
  show affected tiles as loading
when stage is ready:
  render visible tiles from cached stage
  cache rendered tiles
  patch the session canvas
```

### Finding 4 — StageCache refusal is hard to understand

When a reusable FFT stage is refused, diagnostics need to state the practical
consequence, not just expose separate counters.

Good diagnostic text should make this obvious:

```text
FFT stage candidate: 1.08 GiB
Stage cache budget: 1.00 GiB
Decision: refused
Consequence: each tile may recompute FFT
```

Reason: a user can have no visible memory pressure while the stage
cache budget alone rejects the one object needed for responsive montage.

### Finding 5 — StageCache hits still require tile rendering

Even after a hot StageCache hit, the renderer still has work:

- extract the tile slice;
- apply channel, scale, and complex display mode;
- preserve the histogram source policy;
- patch the montage canvas;
- upload pixels to `ImageItem`;
- update loading/skipped overlays.

The right cache hierarchy is:

```text
StageCache:
  expensive operation outputs, for example FFT result [:, :, :]

Rendered tile cache:
  tile image plus histogram data for a specific channel, scale, LUT, and view

Canvas/session cache:
  current viewport canvas patched with rendered tiles
```

Selected fix: coordinate all three cache layers explicitly. A hot rendered tile
cache hit must be immediate. A hot stage cache hit with a cold rendered tile must
avoid FFT and do only tile rendering. A cold FFT stage must show loading, compute
once, and be shared by all waiting tiles.

### Finding 6 — progressive commits use the full ImageView path

Phase 4h patched the montage canvas in place, but
`_apply_progressive_display_image()` still calls the broad `setImage()` path.
`ImageView2D.setImage()` updates levels and histogram state, then pushes the full
image through `ImageItem.setImage()`. For RGB/complex display it can rebuild
full-canvas floating-point RGB state on every progressive flush.

Progressive montage needs a narrower API:

```python
updateImageDataFast(
    image,
    *,
    histogramData=None,
    levels=None,
    rgb_already_windowed=False,
)
```

Contract: same shape, same levels, same histogram policy, same viewport, only
pixel data changed. This path must not recompute levels, refresh histogram
widgets, rebuild RGB base images, touch side panels, or clear session-level
loading overlays.

During progressive montage, levels are frozen. Recompute levels/histogram only
for the first full image, explicit Auto Window, final
montage completion, idle refresh, or channel/scale/window-mode changes. For
complex RGB montage, maintain a display-ready RGB canvas and patch only the
completed tile region before uploading the canvas.

### Finding 7 — predictive caching must be stage-aware

Use idle compute only after obsolete work and duplicated stage work are fixed.
Predictive work runs in this order:

1. Stage pre-materialization for valuable expanded stages that fit.
2. Rendered tile prefetch only after the required stage exists.
3. Directional next-slice pre-render only when cost is cheap or the stage is
   already cached.
4. Never prefetch by recomputing the same expensive FFT per tile.

Prefetch diagnostics explain why a prefetch ran, skipped, waited for a stage, or
was refused.

Forbidden prefetch pattern:

```text
prefetch tile 5 -> compute full FFT
prefetch tile 6 -> compute full FFT
prefetch tile 7 -> compute full FFT
```

Correct prefetch pattern:

```text
prefetch shared FFT stage once
render visible missing tiles from that stage
render near-viewport tiles from that stage
render likely next-direction tiles from that stage
```

### Finding 8 — CPU and FFT workers need one policy

Using more threads everywhere can make the UI worse. Montage can currently imply:

```text
2 tile workers * auto FFT workers + Qt/UI + other controllers
```

Selected fix: add a global compute policy that keeps native FFT workers and
Python workers under a conservative budget.

```python
@dataclass
class ComputePolicy:
    total_cpu_budget: int
    visible_workers: int = 1
    stage_workers: int = 1
    tile_workers: int = 2
    fft_workers_for_visible: int = 2
    fft_workers_for_stage: int = 4
    fft_workers_for_tile: int = 1
    fft_workers_for_prefetch: int = 1
```

Key rule: `tile_workers * fft_workers_for_tile` must stay within the
interactive compute budget. Expensive stage materialization should run as one job
with controlled FFT workers; tile rendering from a cached stage should stay cheap.

## Implementation plan

### P0 — stale-work correctness and pure imports

Goal: old workers cannot commit newer-invalid results, and pure tests do not
import GUI dependencies.

- Add a render-generation guard advanced on every render request or user state
  mutation that can affect visible output.
- Compare async callback generation and current evaluator request key before any
  image, preview, montage, profile, ROI, or pixel commit.
- Make `EvaluationController.clear_group()` invalidate group generation even when
  no replacement job is submitted.
- Pass cancellation tokens into montage tile evaluation.
- Add cancellation checks inside stage-cache slab execution before and after
  planner transitions.
- Remove the eager `ArrayScopeWindow` import from `arrayscope.window.__init__` and
  lazy-load it through `__getattr__`.
- Re-run and fix broad-run cancellation flakes instead of documenting them away.

Required tests:

- stale visible-image worker cannot commit after a newer cache-hit state;
- stale visible-image worker cannot commit after a newer refused/degraded state;
- clearing a group invalidates queued callbacks without a replacement worker;
- montage tile cancellation token is passed and observed;
- pure diagnostics tests do not import `pyqtgraph`.

### P1 — stage-first rendering and singleflight

Goal: montage over a transform axis renders visible tiles from one shared stage
instead of letting tile workers compute expensive stages independently.

- Add a `StageMaterializationManager` owned by `OperationEvaluator`, not by
  individual tile workers.
- Add in-flight singleflight keyed by document identity, operation prefix, region,
  dtype, shape, and relevant operation settings.
- Let duplicate stage requests attach to the in-flight job instead of recomputing.
- Use a dedicated stage lane with one worker and controlled FFT worker settings.
- Surface stage candidate bytes, budget, decision, and recompute consequence in
  diagnostics.
- Keep StageCache pure and in-memory; do not add disk or memmap cache in this
  phase.
- During montage session planning, detect common cacheable expanded stages needed
  by visible tiles.
- If a common stage is missing and fits, schedule stage materialization before
  scheduling cold tile renders.
- If the stage is in-flight, mark affected tiles loading and attach to the shared
  job.
- If the stage is refused, render conservatively and show diagnostics explaining
  that per-tile recomputation may occur.
- Render cached-stage tiles through the rendered tile cache, then patch the
  session canvas.
- Keep visible image rendering on a latest-only max-1 lane, tile rendering on the
  montage lane, stage materialization on the stage lane, and prefetch idle-only.

Required tests:

- concurrent cold requests for the same expanded FFT stage materialize it once;
- waiting requests receive the shared materialized result;
- refused stage diagnostics include candidate bytes, budget, and consequence;
- in-flight stage state is cleared on cancellation, error, and document revision
  changes;
- montage over an FFT axis calls the FFT operation once when the expanded stage
  fits;
- multiple visible tiles hit StageCache after warmup;
- layout-only choices such as montage columns do not affect stage identity;
- stale montage sessions do not continue scheduling tiles after their generation
  is invalidated.

### P2 — true progressive image update path

Goal: cached/progressive tile display avoids full display, histogram, side-panel,
and RGB-rebuild work.

- Add an `ImageView2D` fast pixel-update API for same-shape progressive commits.
- Split full display commits from progressive pixel commits at the window/render
  boundary.
- Freeze levels during progressive montage unless the user explicitly changes
  windowing or Auto Window is requested.
- Skip histogram updates during tile patch commits.
- Avoid full-canvas RGB floating-point rebuilds for complex/RGB progressive
  montage; patch display-ready tile regions instead.
- Recompute levels and histogram once on final commit, idle refresh, explicit Auto
  Window, or channel/scale/window-mode change.

Required tests:

- progressive tile patch does not call finite-bounds or histogram level scanning;
- progressive tile patch does not refresh operation dock, inspection dock, ROI, or
  profile panels;
- complex/RGB progressive montage patches only the dirty tile region before the
  screen flush;
- final montage completion performs exactly one allowed levels/histogram refresh
  when policy requires it.

### P3 — stage-aware predictive cache and compute policy

Goal: idle work improves the next interaction without stealing responsiveness or
duplicating expensive transforms.

- Add a compute policy that coordinates Qt worker lanes and FFT worker counts.
- Enforce conservative limits for tile workers multiplied by FFT workers.
- Add idle stage pre-materialization for valuable expanded stages that fit.
- Add near-viewport rendered tile prefetch only when the required stage is cached
  or in-flight through the stage manager.
- Add directional next-slice pre-render only when cost is cheap or the needed
  stage already exists.
- Forbid prefetch paths that would compute the same expensive FFT separately per
  tile.
- Add diagnostics for why prefetch ran, skipped, waited for a stage, or was
  refused.

Required tests:

- prefetch never launches duplicate expensive FFT stages for neighboring tiles;
- prefetch waits for or reuses stage singleflight;
- compute policy caps effective FFT worker usage for tile rendering;
- visible interaction cancels or pauses lower-priority predictive work.

### P4 — regression, benchmarks, and roadmap accuracy

Goal: phase completion is tied to user-visible responsiveness and broad-suite
stability, not just local implementation.

- Add latency benchmarks for hot rendered-tile display, hot stage/cold tile
  display, cold shared-stage montage warmup, and rapid slice bursts.
- Add manual regression coverage for FFT montage, fast slice scrolling, cache-hit
  stale-result prevention, and progressive levels behavior.
- Keep roadmap items unchecked or partial until broad tests pass and manual
  interaction confirms the lag/jitter path is gone.
- Update architecture docs if ownership changes around stage materialization,
  compute policy, or progressive image APIs.

Required checks:

- broad pure test suite;
- broad Qt test suite;
- strict UI mode tests;
- latency/benchmark assertions where deterministic enough;
- manual FFT-over-montage-axis inspection with diagnostics open.

## Accepted design decisions for this phase

- Use render generations plus current-key checks; do not rely only on request-key
  comparison against captured state.
- Make stage materialization explicit and singleflight; do not hide duplicated FFT
  work inside tile workers.
- Keep StageCache in memory for now; disk-backed or memmap caches remain future
  ideas.
- Add a narrow progressive pixel-update API; do not send every tile patch through
  the full display-image commit path.
- Freeze levels during progressive montage by default; recompute at deliberate
  boundaries only.
- Add a compute policy; do not solve lag by simply increasing worker counts.
