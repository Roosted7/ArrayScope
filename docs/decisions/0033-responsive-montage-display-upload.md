# 0033 Responsive Montage Display Upload

## Status

Accepted.

## Context

Operation-backed montage can make tile evaluation fast through the stage cache, but the UI can still
freeze when committing a large updated montage canvas to pyqtgraph. `ImageItem.setImage()` and
histogram updates run on the Qt thread, and moving them to a worker would violate Qt ownership rules.

The expensive phase can include more than the visible image upload: a separate histogram image item,
histogram recomputation, RGB/complex re-windowing, and duplicate level updates can all contribute.
Repeated calls to pyqtgraph `HistogramLUTItem.setImageItem()` also connect image-change signals each
time, so careless rebinding can multiply histogram recomputation.

## Decision

Display upload remains on the Qt thread, but ArrayScope reduces and measures that work.

`ImageView2D` owns upload instrumentation for visible image upload, histogram plot upload, histogram
recompute, RGB re-windowing, level synchronization, and profile-bound refresh. Runtime diagnostics show
these phases separately from tile evaluation, stage cache, and canvas composition.

Histogram binding is idempotent and centralized. ArrayScope binds the histogram widget through one
helper, disconnects pyqtgraph's automatic image-change callback, and explicitly refreshes the
histogram plot once per committed state.

Programmatic presentation commits upload pixels with already-decided levels, then synchronize
histogram handles without causing a second image re-window or upload. Explicit user histogram edits
still update displayed pixels immediately.

Progressive montage commits are coalesced to the latest state when upload is slow. Final complete
state is always committed.

For large or previously slow montage displays, `ImageView2D` switches to an internal exact tile-layer
mode. The full canvas remains the committed value source for hover/status and semantics, but per-tile
`ImageItem`s paint visible loaded tiles. Updating one tile then uploads that tile rather than the full
canvas. This mode is internal and does not change the public viewer API.

Tile-layer presentation is stateful and dirty-aware. Presentation models carry optional dirty tile
numbers: `None` means the tile state is unknown and visible loaded items should refresh, `()` means a
known-clean flush, and non-empty tuples identify the loaded items whose pixels changed. `ImageView2D`
delegates the per-item state to `arrayscope.display.montage_tile_layer`, which tracks item source
identity, histogram identity, local canvas rect, levels, RGB-windowing policy, and cached display data.
Known-clean commits skip tile image uploads and, when histogram source/range/levels are unchanged,
skip histogram image upload as well.

Tile item source identity comes from the rendered tile cache payload, not from the transient montage
viewport canvas. Rebuilding a canvas from already-cached rendered tiles therefore does not by itself
force tile item uploads or RGB re-windowing; items refresh only when the tile source, visible crop,
levels, RGB policy, or dirty-tile set requires it.

Complex/RGB tile-layer windowing uses float32 working arrays. Each tile stores one RGB base and one
histogram source for the current item state; unchanged levels reuse the display cache, changed levels
recompute only visible RGB tiles from those cached bases, and dirty-tile commits recompute/upload only
the affected tile items. Diagnostics include tile-layer visible, updated, skipped, and RGB-windowed
item counts in addition to existing upload and RGB-window timings. Diagnostics also split
tile-layer-specific `ImageItem.setImage()` time and tile-layer RGB windowing time from the aggregate
visible-upload and RGB-window totals.

Each tile keeps one display-ready RGB cache variant for the current levels. ArrayScope intentionally
does not keep a second level variant by default because large visible montages can contain hundreds
of tiles; duplicating uint8 display tiles would trade CPU latency for another large memory pressure
source. Level changes remain real work, but unchanged-level hot-cache commits are expected to reuse
the single cached display variant and upload nothing.

Tile-layer float32 RGB source bases are bounded separately from the rendered-tile cache and montage
canvas. They are useful for immediate RGB/complex level changes, but retaining one base for every
visible tile in a large montage can duplicate hundreds of MiB. ArrayScope therefore prunes older
per-tile source bases while keeping the displayed `ImageItem` data intact; unchanged-level clean
commits still upload nothing, and later presentation commits can re-window pruned tiles from the
current committed canvas when levels change.

## Consequences

Diagnostics can distinguish slow operation evaluation from slow Qt upload.

Large progressive montage is more responsive because intermediate canvas commits can be coalesced and
tile-layer mode avoids repeated full-canvas uploads. Hot cached tile-layer flushes with unchanged
levels and no dirty tiles, including all-cached sessions that rebuild the viewport canvas, are expected
to report zero tile item updates and zero visible upload bytes.

The display implementation is more complex: ImageView2D now owns two montage paint modes, and tests
must guard that both use the same levels, histogram source, hover/value semantics, and dirty-only
update behavior.
