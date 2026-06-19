# 0037 — First-class VisPy tiled montage renderer

## Status

Accepted. Backend ownership is refined by [0038 — Compose rendering backends behind shared presentation semantics](0038-render-backend-composition.md).

## Context

The first VisPy montage path avoided full-canvas uploads but created one VisPy image visual per tile.
Profiling showed that this was slower than PyQtGraph for large montages: object churn, camera sync,
histogram/viewport bookkeeping, and per-visual management dominated the texture upload savings.

The old path also carried direct tile data as generic objects through a placeholder canvas commit.
That made hover/status, ROI/profile region reads, and display ownership depend on special cases.

## Decision

Make direct typed tile payloads the tiled presentation contract.

`DisplayTilePayload` carries tile number, source index, image, optional histogram/intensity data, and
a stable source identity. Committed frames own a `FrameValueSource`: canvas frames use
`CanvasValueSource`, and tiled frames use `TiledValueSource`. Hover/status and demand tile-region
reads go through the committed value source, never through placeholder pixels.

VisPy tiled montage rendering uses `arrayscope.display.vispy_tiled_renderer`:

- visible tile payloads use stable slots in mode-aware scalar and/or color texture atlases;
- inactive tiles remain resident until LRU pressure requires their slots;
- one batched visual draws the active tile quads;
- level-only changes update shader uniforms;
- clean commits skip texture and vertex uploads;
- dirty commits upload only changed atlas regions;
- PyQtGraph remains the interaction, histogram, ROI, profile, HUD, and context-menu owner.

PyQtGraph tile-layer fallback continues to exist, but it consumes the same typed payload contract.

## Consequences

Large complex/RGB montage initial commits avoid CPU RGB windowing and per-tile VisPy visual creation.
Clean VisPy tiled commits are now true no-op texture commits. The committed display value source is
explicit, which simplifies hover/status and offscreen ROI/profile demand reads.

The renderer still receives CPU-prepared phase/color and scalar intensity tiles. Full GPU-side
complex-to-phase/RGBA generation remains future work.

The VisPy camera follows the PyQtGraph `ViewBox` through a coalesced range/flip sync so pan/zoom does
not synchronously push camera state for every range-change signal.

## Tests required

- Typed tile payload and tiled value-source tests.
- Tile-region provider reads committed tiled payloads before evaluating.
- VisPy direct tiled payloads use one batched GPU layer rather than per-tile image visuals.
- Clean direct tiled commits update zero textures.
- Dirty direct tiled commits update only dirty payload counters.
- Benchmark scenarios for large tiled initial commit, clean flush, dirty tile commit, level preview,
  and pan/zoom no-upload paths.

## Future work

- GPU-side complex scalar to phase/color generation.
- Multi-page, byte-budgeted atlas residency with runtime device limits, explicit viewport-near retention, and eviction diagnostics.
- Production perf gates on target GPU/compositor combinations after collecting stable baselines.
