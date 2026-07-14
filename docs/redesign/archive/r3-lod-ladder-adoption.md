# R3 — LOD ladder adoption (LOD is first-class, not bolted on)

**Goal:** `render/ladder.py` is the only answer to "what quality next";
`window/montage_lod.py` dissolves; one pyramid store; operations run once
per rung; PyQtGraph reaches parity through capabilities.

**Status (2026-07-07):** code adoption landed. `window/montage_lod.py` is
deleted, pure LOD helpers live in `render/lod.py`, montage sessions own one
`pyramid_cache`, and demanded-level materialization is claimed by
`TileLifecycle` from `MontagePipelineEffects.prepare_rung`. Completed DESIRED
rungs mark lifecycle levels resident, dirty stale presentations, and request
one coalesced pipeline retarget, closing the lost-wakeup class where
`tile_lod_desired_factor` stayed coarser than `tile_lod_applied_factor` at
idle. Level-stat maintenance moved to `render/level_stats.py`; the remaining
timer-to-kernel cleanup stays in R4.

## Steps

1. **Ladder inputs get real.** Replace `LadderPolicy.ops_commute_with_reduction`
   with a per-document query against
   `operations.capabilities` (`_preview_pipeline_commutes_for_display_lod`
   is the reference). `floor_level`/`preview_level` derive
   from `display/pyramid.preview_level_for_tile_shape` instead of constants.
2. **One pyramid store.** Merge the `shared_pyramid` / `preview_pyramid`
   pair into one `PyramidCache` keyed by `PyramidLevelKey` (quality is
   already part of the key via component tags). Claims/ownership stay in
   `TileLifecycle`. Delete `montage_lod.shared_pyramid/preview_pyramid` and
   the renderer-attribute stashing (`_montage_lod_*_cache`).
3. **Replace planning calls.** Desired-level materialization now flows through
   `ladder.plan` + pipeline submission; the old request-list and standalone
   materialization scheduler names are guarded against reintroduction. The
   floor helpers (`best_floor_key`, `floor_can_progress`,
   `ensure_floor_payloads`) become FLOOR-rung effects. Keep
   `viewport_identity`, `pyramid_key_for*`, `texture_source_for_rendered`
   as pure helpers (move to `render/` or `display/`, they are fine).
4. **Level values into the machine** (roadmap X5 queue item 4): move
   per-tile level-value convergence from `PresentationGenerationTracker`
   into `TileLifecycle` so level progress has one owner; the ladder's
   `levels_authoritative_rung` decides when preview-derived samples yield
   to refined ones. This closes cluster D of the map: implement
   `LevelStatsService` (scans as HISTOGRAM_REFINEMENT kernel tasks; the
   GUI thread never scans payload arrays again). R3 moved the service module
   under `render/`; R4 owns the remaining timer-driven drain deletion.
5. **PyQtGraph parity via capabilities.** The ladder emits identical rung
   plans for both backends; `apply_commit` branches on capabilities:
   uniform-level changes on VisPy, bounded CPU redraw batches on
   PyQtGraph. Re-run the Plan 02 A/B (cold settle vs level-change) and
   flip the PyQtGraph resident-LOD default if the ladder's floor-first
   fill fixed the cold-settle regression — record either outcome in ADR
   0050's status table.
6. **Transform-preview queue** (roadmap X5 item 2): non-display transform
   previews submit at `Priority.PREFETCH` on `DISPLAY_PREVIEW` lane so
   they can never compete with exact visible fills. Re-decide
   shared transform preview on a fresh A/B. The `ARRAYSCOPE_SHARED_TRANSFORM_PREVIEW`
   env-var fork is deleted; the remaining decision is policy/evidence only.

## Exit gate

- `window/montage_lod.py` deleted; architecture guard forbids `montage_lod`,
  `pending_lod_requests`, `lod_preview_pyramid`,
  `ARRAYSCOPE_SHARED_TRANSFORM_PREVIEW`, `admit_ingest_reduction`, and
  `admit_preview_reduction` in `arrayscope`.
- Scrub + zoom-threshold benchmarks within bars (README ground rule 1);
  zoom threshold crossings do not rebuild the active set (counter-pinned).
- Exact inspection values still native under every displayed rung
  (existing tests keep passing untouched — they pin a policy constraint).
- One `PyramidCache`; lifecycle claim counters balance to zero at idle.
- PyQtGraph resident-LOD default decision recorded with numbers.
