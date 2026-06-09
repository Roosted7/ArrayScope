# 0022 - Stable Progressive Montage Rendering

## Problem

Interactive montage previously derived the viewport canvas from the loaded tile rectangle. If only a
subset of visible tiles was loaded, the canvas could shrink or move around those tiles, making hover,
profile, panning, and tile edges unstable.

Missing tiles were also represented only by placeholder data/NaN-like behavior, so the UI could not
distinguish gaps from tiles that were still loading or skipped by memory budget. Missing visible tiles
were evaluated as one batch, so users did not see useful progress until the whole batch completed.
Stale montage callbacks could also clear overlays that belonged to a newer render.

## Decision

Montage viewport canvases are now based on the requested viewport rect clipped to full montage bounds,
never on loaded tile bounds. `MontageViewportCanvas` carries per-tile states: loaded, loading, skipped,
and unloaded. Normal visible tiles are scheduled and loaded progressively; skipped is reserved for a
hard per-tile memory-budget refusal and shows a detailed warning. `DisplayGeometry` uses those states
to resolve only loaded tiles to array/profile mappings and to report loading/skipped/gap status for
hover/profile behavior.

Interactive montage still uses one image canvas. A `MontageRenderSession` commits cached tiles
immediately, marks missing tiles as loading, schedules missing visible tiles sequentially through the
one-worker visible controller, and copies completed tiles into the current canvas. Visual commits are
coalesced to roughly 30 Hz. Stale tile callbacks are ignored without mutating the current canvas,
geometry, or overlays. Cached tile values are layout-independent payloads; the current montage plan
binds those payloads to tile placement during composition.

## Consequences

- Canvas coordinates stay stable while tiles load.
- Hover and live profile can distinguish real data from loading/skipped/gap regions.
- Cached tiles appear immediately.
- Exact tile results remain cached per tile.
- Progressive montage feedback works without multiple `ImageItem`s or extra worker pools.
- Full operation stage caching remains deferred to Phase 4g.

## Rejected Alternatives

- Multiple image items per tile: more UI state and more coordinate complexity than needed now.
- Encoding loading/skipped as NaN only: conflates tile state with data/ROI statistics.
- Jumping directly to a stage cache: important, but larger than this montage correctness fix.
- Increasing visible worker count: visible work remains latest-only with one worker for predictable UI behavior.

## Tests Required

- Stable canvas shape/origin while tiles load.
- Tile-state mapping for loaded/loading/skipped/unloaded.
- Hover/profile behavior for loaded, loading, skipped, and gap points.
- Progressive tile scheduling and partial canvas updates.
- Stale montage callbacks do not mutate current UI state.
- Architecture guards for Qt-free montage state modules and no batched missing-tile evaluation.
