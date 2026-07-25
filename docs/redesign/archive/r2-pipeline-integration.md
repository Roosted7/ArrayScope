# R2 — MontagePipeline live (dissolve frame_renderer clusters B/C/E)

**Goal:** the montage render path runs Intent → Ladder → Kernel →
CommitBatch → Lifecycle-acknowledged presentation, and the corresponding
frame_renderer clusters are deleted. Follow the
[dissolution map](frame-renderer-map.md); this plan fixes the order and the
tricky seams.

## Order of work (one commit each)

1. **Effects: evaluation.** Create `arrayscope/render/effects.py` with the
   Qt-free rung evaluators ported from frame_renderer:
   `evaluate_target_tile(...)` (level 0 is the native target),
   `evaluate_preview_tile(...)` / `evaluate_shared_preview(...)` (from the
   three preview snapshot evaluators + reduce helpers). They already avoid
   Qt — the port is mostly de-`self`-ing (pass document, view_state,
   caches explicitly). Unit-test against small arrays; compare outputs to
   the old functions before deleting them (golden test in the same commit).
2. **Effects: tile state snapshot.** Implement
   `PipelineEffects.tile_states` from `TileLifecycle` records +
   `PyramidCache` residency (`tile_resident_levels` in montage_lod is the
   reference; port it, delete the original in R3).
3. **Effects: commit application.** Implement `apply_commit(CommitBatch)`
   on top of the existing `setTiledPresentation` delta path (port the
   batching core of `_commit_montage_session_tile_layer` /
   `_commit_montage_tile_delta_direct`; keep the identity-aware ack flow —
   lifecycle `presented` still requires backend acknowledgement, ADR 0051
   is not renegotiated by this redesign).
4. **Wire lifecycle events.** `pipeline._on_rung_done` feeds
   `level_materialized`/residency claims; acknowledgement reports keep
   flowing from the backend adapters into `TileLifecycle` unchanged.
   Delete `_on_montage_tile_done/_error`, `_flush_montage_tile_results`,
   `_apply_montage_tile_result` and their flush timers.
5. **Stage dependencies.** Port stage planning to kernel tasks with
   `deps=` (map cluster C); stage results admit into the stage cache from
   the worker; dependent tile tasks list the stage key in `deps`. Delete
   the stage-wait pump family.
6. **Retarget entry.** `update_image_view` montage branch shrinks to:
   build `RenderIntent` + `LodDemand` (from `display/lod.select_lod_demand`
   + `viewplan`), call `pipeline.retarget`, publish content extent,
   auto-fit decision. Delete `_schedule_montage_tiles`,
   `_schedule_next_montage_tile`, `_dispatch_montage_work`,
   `derive_montage_dispatch` (its decisions are now ladder+kernel), and the
   commit/viewport/priority timer families listed in the map.
7. **Watchdog → assertion probe.** Keep one idle-state consistency check
   behind the diagnostics dock; delete the rescue behavior.

## Seams to be careful with

- **Session reuse/retarget (`montage_session.py`) stays.** The pipeline's
  scope key must incorporate the session key exactly as
  `_montage_session_key_for_view` did, or scrub fast-paths die. Add a
  `RenderIntent.semantic_key` builder that reuses that function before
  deleting it.
- **Histogram/level metadata:** until R3's LevelStatsService lands, keep
  the existing level-stats calls working by leaving cluster D untouched —
  the pipeline commits with `level_metadata=None` and the old path still
  publishes levels. (This is the ONE place a temporary dual path is
  allowed; it is deleted in R3 and tracked in known-red.md if any test
  wobbles meanwhile.)
- **PyQtGraph:** `apply_commit` must route through the shared surface
  contract (`present_tiled`), not grow a backend fork. Run
  `tests/display/test_imagesurface_contract.py` after every commit here.

## R2b — stabilization (added 2026-07 after the first R2 completion attempt)

The first R2 completion attempt shipped four architectural defects (commit
storm, camera-only churn, deps-as-ordering, per-tile-native FFT floors) and
tried to compensate with symptom patches (worker clamps, drain clamps,
dtype sniffing, a weakened exit gate). Those are reverted and fixed at the
root in a5c69487/5866b309. **Binding rules going forward, for every plan:**

1. **Exit gates are hard.** A plan is not done while its gate fails, and
   the gate text is never edited to match a result. Weakness found in a
   gate is itself a change that needs a written decision.
2. **No symptom patches.** Any change whose justification is "it reduces
   the symptom" (clamping workers, shrinking batches, skipping event
   processing in a harness) requires a written root-cause note first. If
   the root cause is unknown, the symptom stays visible.
3. **Deps are not ordering.** Kernel `deps` fail-propagate; use them only
   for real data dependencies (stage keys). Ordering = priorities +
   submission order.
4. **Camera-only invariance is tested.** Any change to task keys,
   supersession values, or replan logic must keep
   `test_camera_only_retarget_never_invalidates_rung_work` green.
5. **Payload semantics ride payload metadata** (`quality`, `texture_kind`,
   levels), never dtype/shape sniffing in a backend.

R2b work items (see [known-red.md](known-red.md) for the failing tests):

1. Window-levels cluster: one root-cause triage of the relative-levels /
   levels-sync breakage in the moved level flow (3 tests).
2. Overlay/ROI timing vs the presentation gate: overlay-clear and hidden-
   ROI sampling must key on committed state, not on pre-gate timing
   (2 tests).
3. Runtime diagnostics formatter → kernel sections (1 test, trivial).
4. Complex windowing seam: either implement skip-redundant-rewindow +
   lazy rgb_base rebuild via payload evaluation-levels metadata, or move
   that optimization claim to R3 and adjust the two tests to pin the
   *contract* (no corruption, rewindow works) rather than the
   optimization.
5. Re-run the montage workflow benchmark (Thomas, onscreen) and compare
   against the gate bars below.

## Exit gate (unchanged — not yet met; do not weaken)

- Montage workflow benchmark (both backends, resident + native-only):
  first-payload / first-complete-fill / settled within ±10% of pre-R2, and
  warm scrub ≤ 15 ms (the Plan 01 bar). Event-loop heartbeat gap ≤ 16 ms
  during fills (the multi-second gaps in the first R2 JSONLs are failures,
  not notes).
- GPU harness green, `stall_assertions==0` outside the diagnostics probe,
  `[DESYNC!]` probes quiet.
- frame_renderer.py shrinks below 2,000 lines with clusters B, C, E gone;
  every deleted method's tests deleted or rewritten against
  pipeline/ladder/kernel counters.
- Zero scheduling `QTimer` in the montage data path (the coalescing
  presentation gate and anti-hang fallbacks are the only allowed timers;
  each carries a category comment).
