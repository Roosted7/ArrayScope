# 0019 - Bounded montage rendering

## Problem

The old montage path evaluated many tiles, stored tile arrays and histogram arrays, then allocated one
large final collage. Large montage ranges could multiply memory use far beyond the input array size.

## Decision

Montage rendering is planned with `MontagePlan` and `MontageTile`. The window computes visible tiles,
selects them by estimated bytes, evaluates only that bounded tile set, stores each tile in the image
cache with a montage-tile key, and composes the loaded tiles into a single viewport-sized
`MontageViewportCanvas`. The canvas carries `origin_x`/`origin_y` in full montage coordinates and the
canvas `ImageItem` is positioned at that origin, so interactive display can stay bounded while
hover/profile/ROI mapping uses stable world coordinates and global source indices.
Full giant montage allocation is blocked by the montage memory budget and the interactive visible
render budget.

The default Qt commit path uses one stable canvas `ImageItem`. For large or previously slow uploads,
ArrayScope can switch to an internal exact tile-layer paint path. Tile-layer items are also positioned
in full montage coordinates and are owned by the same layer/z-order policy as ROI and profile marker
graphics. Inactive tile items are removed from the scene immediately rather than kept as hidden
graphics objects.
`make_montage()` remains only as a small pure helper for utility and tests, not for interactive
rendering.

## Consequences

Large montage ranges no longer silently allocate a giant full collage. Visible tile work is reusable and
bounded by bytes rather than tile count. Canvas and tile-layer painting both share full montage world
coordinates, so ROI and live profile markers do not jump when the bounded canvas origin changes.
Hover/value lookup reads committed loaded pixels; ROI/profile demand rendering can evaluate offscreen
tile regions without changing the current viewport canvas or main-view loading overlays.

## Rejected alternatives

Keeping the 256-tile cap as the only guardrail was rejected because it still permits excessive
allocations for large tiles.

## Tests required

Pure tests cover plan geometry, visible tile intersection, source-index preservation, viewport canvas
composition, world-aware display geometry, tile-region demand lookup, and memory estimates. Qt tests
cover montage status, ROI gap behavior, source-index hover/profile mapping for later visible tiles,
z-order policy, tile-layer world positioning, and hidden tile removal.

## Manual checks required

Open a large stack, enable montage over a broad range, verify the memory warning appears when relevant,
pan to later tiles and confirm hover/profile labels show their real source indices, draw ROI over
gaps/unloaded/offscreen regions and confirm demand stats update without visible overlays or ROI jumps,
and pan/zoom repeatedly without stale "Computing" status or visible RSS growth.
