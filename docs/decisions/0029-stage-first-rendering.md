# 0029: Stage-First Rendering

## Problem

Progressive montage could still submit multiple cold tile workers that each expanded and computed the
same expensive operation stage before `StageCache` had a chance to store the first result. Stale async
callbacks could also appear current when a newer visible state was a cache hit, degraded preview, or
refusal and therefore did not submit a replacement worker.

## Decision

Add a per-window render generation and require visible-output async callbacks to match both the
captured generation and the current evaluator/session key before committing.

Add a Qt-free `StageMaterializationManager` owned by `OperationEvaluator`. It deduplicates reusable
expanded stage requests by `StageKey`, records hit/in-flight/scheduled/refused/failed diagnostics, and
keeps stage materialization on a dedicated max-1 scheduler lane. Montage planning now detects retained
cacheable stages shared by visible cold tiles, schedules the stage first, and releases waiting tiles to
the tile lane only after the shared stage is ready.

Progressive montage tile flushes use `ImageView2D.updateImageDataFast()` for same-shape pixel updates.
This preserves levels, histogram range, transform, and viewport during progressive tile patches.

## Consequences

Cold FFT montage over the montage axis computes the expanded FFT stage once for visible tiles that
share it, then renders individual tiles from that stage. Oversized stages are refused according to the
stage-cache budget and diagnostics explain that tiles may recompute the expensive operation.

The initial montage canvas remains a full display commit. Progressive tile patches freeze levels until
an explicit full recompute boundary such as Auto Window, channel/scale/window-mode change, final/idle
refresh, or a new first canvas.

Amendment: montage progressive rendering separates the session render canvas from the committed value
source and from semantic window/level bounds. Hover/status values read only the committed display
frame for the current document, request, render generation, geometry, and display shape; montage
values index that frame with display canvas coordinates, not tile-local coordinates. Window/level
updates use coverage-ranked montage histogram stats. Viewport culling can limit rendered tile work,
but it cannot narrow semantic bounds after broader bounds are known, and zero-real-tile canvases do
not replace the displayed image with placeholders.

See also decision 0030 for the explicit presentation/commit boundary that keeps
viewport refreshes from altering semantic display scaling.

## Rejected Alternatives

- Let every tile worker race to populate `StageCache`: rejected because it permits duplicate expensive
  FFT work under normal cold montage interaction.
- Put stage singleflight in Qt scheduler code: rejected because stage identity and refusal decisions
  are pure operation/evaluator concerns.
- Force oversized reusable stages into cache: rejected because `MemoryPolicy.stage_cache_budget_bytes`
  is the explicit guardrail for this cache layer.

## Tests Required

Tests cover group-generation invalidation, stage singleflight scheduling/refusal/invalidation,
cancellation before stage cache put, staged slab evaluation equivalence, fast image pixel updates, and
FFT montage scheduling one shared stage before tile workers.
