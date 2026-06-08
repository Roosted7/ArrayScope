# 0019 - Bounded montage rendering

## Problem

The old montage path evaluated many tiles, stored tile arrays and histogram arrays, then allocated one
large final collage. Large montage ranges could multiply memory use far beyond the input array size.

## Decision

Montage rendering is planned with `MontagePlan` and `MontageTile`. The window computes visible tiles,
evaluates only that bounded tile set, stores each tile in the image cache with a montage-tile key, and
assembles only the loaded bounded set for display. Full giant montage allocation is blocked by the
montage memory budget.

The current Qt commit path uses one stable `ImageItem` assembled from loaded tiles because multiple
simultaneous PyQtGraph `ImageItem`s caused offscreen paint instability during implementation. The pure
plan and tile cache are in place so a future multi-item renderer can be reintroduced behind the same
planning/cache contract.

## Consequences

Large montage ranges no longer silently allocate a giant full collage. Visible tile work is reusable and
bounded. ROI and hover continue to work on the committed displayed montage source.

## Rejected alternatives

Keeping the 256-tile cap as the only guardrail was rejected because it still permits excessive
allocations for large tiles.

## Tests required

Pure tests cover plan geometry, visible tile intersection, source-index preservation, and memory
estimates. Qt tests cover montage status and ROI gap behavior.

## Manual checks required

Open a large stack, enable montage over a broad range, verify the memory warning appears when relevant,
and pan/zoom through the loaded montage without stale "Computing" status.
