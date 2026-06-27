# 0019 - Bounded montage rendering

## Problem

Montage can expose far more source pixels than a small interaction should materialize at once.
Large montage ranges can multiply memory use far beyond the input array size unless evaluation,
residency, and presentation are bounded by visible tile regions.

## Decision

Montage rendering is planned with `MontagePlan` and `MontageTile`. The window computes visible tiles,
selects them by estimated bytes, evaluates only that bounded tile set, stores each tile in the display
cache with a montage-tile key, and commits typed tile payloads through the same tiled presentation
model used by single-image display. Tile items are positioned in full montage coordinates and are
owned by the same layer/z-order policy as ROI and profile marker graphics. Inactive tile items are
removed from the scene immediately rather than kept as hidden graphics objects.

## Consequences

Large montage ranges no longer silently allocate a full image-sized presentation. Visible tile work is reusable and
bounded by bytes rather than tile count. Tile-grid painting uses full montage world
coordinates, so ROI and live profile markers do not jump when the bounded visible origin changes.
Hover/value lookup reads committed loaded pixels; ROI/profile demand rendering can evaluate offscreen
tile regions without changing current visible residency or main-view loading overlays.

## Rejected alternatives

Keeping the 256-tile cap as the only guardrail was rejected because it still permits excessive
allocations for large tiles.

## Tests required

Pure tests cover plan geometry, visible tile intersection, source-index preservation,
world-aware display geometry, tile-region demand lookup, and memory estimates. Qt tests
cover montage status, ROI gap behavior, source-index hover/profile mapping for later visible tiles,
z-order policy, tile-layer world positioning, and hidden tile removal.

## Manual checks required

Open a large stack, enable montage over a broad range, verify the memory warning appears when relevant,
pan to later tiles and confirm hover/profile labels show their real source indices, draw ROI over
gaps/unloaded/offscreen regions and confirm demand stats update without visible overlays or ROI jumps,
and pan/zoom repeatedly without stale "Computing" status or visible RSS growth.
