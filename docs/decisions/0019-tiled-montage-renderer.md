# 0019 - Bounded montage rendering

## Problem

The old montage path evaluated many tiles, stored tile arrays and histogram arrays, then allocated one
large final collage. Large montage ranges could multiply memory use far beyond the input array size.

## Decision

Montage rendering is planned with `MontagePlan` and `MontageTile`. The window computes visible tiles,
selects them by estimated bytes, evaluates only that bounded tile set, stores each tile in the image
cache with a montage-tile key, and composes the loaded tiles into a single viewport-sized
`MontageViewportCanvas`. The canvas carries `origin_x`/`origin_y` in full montage coordinates, so
interactive display can stay bounded while hover/profile/ROI mapping still uses global source indices.
Full giant montage allocation is blocked by the montage memory budget and the interactive visible
render budget.

The Qt commit path uses exactly one stable `ImageItem`. The earlier multi-`ImageItem` tiled display path
caused offscreen paint instability and has been removed rather than kept as dormant compatibility code.
`make_montage()` remains only as a small pure helper for utility and tests, not for interactive
rendering.

## Consequences

Large montage ranges no longer silently allocate a giant full collage. Visible tile work is reusable and
bounded by bytes rather than tile count. One `ImageItem` remains stable; hover and live profile labels
use full montage source coordinates, not local canvas order. ROI statistics are exact for loaded canvas
pixels and ignore gap/unloaded pixels because the histogram source marks them as `NaN`.

## Rejected alternatives

Keeping the 256-tile cap as the only guardrail was rejected because it still permits excessive
allocations for large tiles.

## Tests required

Pure tests cover plan geometry, visible tile intersection, source-index preservation, viewport canvas
composition, origin-aware display geometry, and memory estimates. Qt tests cover montage status, ROI gap
behavior, source-index hover/profile mapping for later visible tiles, and the absence of the old
interactive mini-montage call path.

## Manual checks required

Open a large stack, enable montage over a broad range, verify the memory warning appears when relevant,
pan to later tiles and confirm hover/profile labels show their real source indices, draw ROI over
gaps/unloaded regions and confirm `NaN` portions are ignored, and pan/zoom repeatedly without stale
"Computing" status or visible RSS growth.
