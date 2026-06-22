# Manual Regression Phase 4i

Use a 3D array with an FFT/IFFT operation over the sliced or montage axis.

## Hot Cached Montage

1. Enable montage with tile-layer display.
2. Scroll until the stage cache and montage tile cache are hot.
3. Stop on an already-rendered slice without changing levels.
4. Open diagnostics.

Expected:
- `Patched tiles last flush: 0`
- `Tile layer items: visible=N updated=0 skipped=N`
- `Tile layer RGB window tiles: 0`
- `Upload: visible=0 B ...`

## One Dirty Tile

1. With the same hot montage, force one tile to be recomputed.
2. Let the commit flush.

Expected:
- Exactly one tile item updates.
- RGB window tile count is `1` only for complex/RGB tile-layer display.
- The slice controls remain responsive while the tile update trails.

## Slice Controls During Slow Render

1. Rapidly scroll the active slice control.
2. Watch the spinbox/dimension strip while rendering catches up.

Expected:
- The displayed control value follows the latest requested slice immediately.
- Rendering may coalesce and trail, but stale renders do not reset the control.

## Stage Warmup

1. Add an FFT operation whose retained stage candidate fits `Stage cache budget`.
2. Leave visible work idle.
3. Open diagnostics.

Expected:
- Stage warmup reports `scheduled`, `hit`, or `in_flight`.
- It uses the stage-cache budget, not the small prefetch budget.
- It does not run while visible or montage work is busy.

## Tile And Slice Prefetch

1. Inspect diagnostics after a successful montage commit.
2. Compare behavior before and after the required stage is cached.

Expected:
- Missing expensive stage: rendered tile/slice prefetch is skipped.
- Cached or in-flight stage: nearby tile/slice prefetch may schedule on the prefetch lane.
- Prefetch never computes a separate expanded FFT per predicted tile.

## Resource Governor And Redraw Throughput

1. Trigger a cold FFT montage redraw.
2. Keep the diagnostics dialog open on Realtime, Feedback, and Montage.

Expected:
- Reusable stage reports `scheduled`, `hit`, or `in_flight`.
- `repeated per tile=no` for fitting reusable stages.
- Tile compute distinguishes cache hits, direct lead tiles, stage-backed tiles, and waiting tiles.
- UI remains responsive while completed tiles draw in bounded batches.
- Feedback shows UI pressure, CPU headroom, memory pressure, worker decisions, and channel
  batch/budget/interval values.
- The All diagnostics tab contains the same sections available as individual tabs.

## Chunked Stage Cancellation

1. Start a large reusable stage materialization.
2. Change slice/operation state before it completes.

Expected:
- Cancellation is honored between chunks.
- No partial stage is stored in the stage cache.
