# 0032 Semantic Montage Histograms

## Status

Accepted.

## Context

Montage rendering is progressive and viewport bounded. At high zoom, only a subset of tiles may be
rendered, and some tiles may be intentionally deferred until the user pans or zooms out. Histogram and
window/level state still need to describe the semantic tiled index range, not just the current canvas
pixels.

The previous bounds tracker stored only a unioned range for a full montage key. That made shifted
tiled ranges hard to update efficiently, made overlap reuse imprecise, and could make relative levels
reset when uncached tiles arrived. Complex RGB display added another ambiguity: RGB array rank was
treated as display-ready color, even when the RGB colors still needed magnitude-based windowing.

## Decision

Montage histogram ownership lives in `arrayscope.window.montage_levels` as sampled per-source-index
semantic stats. The key excludes viewport origin and the current montage index list; the current
expected tiled indices are tracked separately. Building the current `LevelSource` selects stats for
the expected indices, reuses overlapping tiles, and excludes indices that left the tiled range.

Each tile stores finite bounds and a deterministic finite sample. Provisional stats are cheap, refined
stats can use a larger bounded sample, exact sampling is allowed for small tiles, and aggregate
histogram plot data is capped. Viewport culling never downgrades a broader semantic source.

Presentation separates local value data from histogram plot data. The local `histogram_data` remains
canvas-shaped for hover-compatible values and complex RGB intensity scaling. Optional
`histogram_plot_data` may be sampled semantic tile data for the histogram widget.

RGB windowability is explicit metadata on `DisplayImage`. RGB shape alone does not imply that pixels
are already windowed. Complex phase-color images set `rgb_already_windowed=False` so histogram level
changes remap magnitude intensity; display-ready RGB images set it to `True`.

User histogram edits preserve intent according to the active window mode. Relative edits preserve
fractions across improving histogram ranges, so levels may numerically refine as more tiles for the
same montage population are sampled. Absolute edits preserve numeric low/high values while histogram
metadata may improve. Explicit Auto Window clears user intent once and uses the best available
semantic source.

## Consequences

Progressive montage can show sensible levels from the first loaded tiles, improve relative levels as
more semantic stats arrive, preserve broader stats across zoom culling, and reuse overlap when the
tiled range shifts. A loading-only montage presentation may be committed before the first tile to
clear stale renderer contents; that placeholder must not become the semantic level baseline.
Cached tiles that are visible in the first commit of a montage session must all contribute
provisional semantic stats before being drawn, rather than deferring some stats behind the initial
display update.

The histogram widget may show semantic sampled data that is not the same shape as the displayed
canvas. Display commit validation therefore checks local value-source shape separately from optional
plot-source presence.

Persistent GPU tile residency is keyed by semantic tile identity plus texture-content identity. The
semantic base key is still used for overlap and warm-residency reuse, but a clean commit may skip
upload only when the resident slot already contains the same texture content.

Montage operation performance depends on two caches: per-tile display cache and shared stage cache.
If a new montage session attaches to an in-flight stage materialization, tile workers wait for the
shared stage instead of recomputing expensive operations per tile.
