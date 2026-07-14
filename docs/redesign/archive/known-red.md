# Known-red ledger (redesign branch)

Rule 5 of the [ground rules](README.md): window/display tests may only be
red if listed here with the plan step that fixes or deletes them. Qt-free
suites are never allowed red.

**Status 2026-07-07 (R2b stabilization closure):** the focused 8-test R2b
ledger is green after moving the tests onto the live pipeline/kernel path and
closing the payload-windowing/sync seams. R2 is **code-complete but NOT
closed** — the benchmark, manual onscreen workflow, and GPU harness gates
still require fresh evidence. Do not weaken the benchmark gate to match
results (that happened once on this branch; the gate text is restored and
binding).

## Active red tests

| test | cause | resolved by |
|---|---|---|
| _none in the focused R2b/R4 ledger_ | The R4 broad-verification reds found on 2026-07-09 are resolved by the preview-to-target wakeup fix, VisPy retained-presentation stats fix, and test modernization onto lifecycle/pipeline observables. | Keep R4 open until benchmark/manual/GPU gates pass. |

Resolved by R2b stabilization:

- `test_relative_window_levels_survive_fast_scroll_with_render_in_flight`
- `test_relative_window_levels_match_for_cached_and_uncached_display_tiles`
- `test_window_levels_sync_between_windows`
- `test_stale_tile_result_does_not_clear_updating_overlay`
- `test_hidden_montage_roi_overlay_does_not_sample_loading_placeholder`
- `test_format_runtime_diagnostics_includes_all_major_sections`
- `test_pyqtgraph_complex_fast_scroll_budget_keeps_presentable_slots`
- `test_large_complex_montage_tile_layer_histogram_drag_does_not_update_base_image_item`

## Scroll-scrub convergence stall (fast index retarget) — RESOLVED (2026-07-07)

**Repro:** `profile_montage_workflow` `montage_scroll_scrub` phase under the
**resident** quality policy (`--montage-quality-policy resident`; this is the
real app default via `normalize_montage_quality_policy_choice`, but the profile
CLI defaults to `native-only`, which does NOT exercise the LOD ladder — always
pass `resident` to reproduce). User-visible as coarse/pixelated tiles among
sharp neighbours after slow scrolls (mixed LODs), and earlier as holes.

**Actual root cause (traced by introspection, NOT the seam hypothesised
below):** `render/effects._resident_levels_from_lifecycle` returned every
RESIDENT level in a tile's lifecycle record **without filtering by the tile's
current source**. `montage_index` is a grid position, so after a scroll each
position points at a new source, but the record still holds the *previous*
source's resident level entries. The ladder (`tile_lod_states` → `LodLadder`)
counted those stale levels, computed `desired_resident=True`, and emitted **no
refinement rung** — while `best_floor_key` (which *does* filter by source) only
found the coarse floor level. Presentation and ladder disagreed: the tile
floored coarse and the ladder believed it converged, so it stuck with no work
in flight (`pending>0, active=0`). Onscreen-only because the residency that
goes stale comes from prior views; the offscreen cold cache had none, so the
ladder always emitted the rung and the scroll settled — which is why the
earlier "retarget↔pipeline `mark_materialized`" hypothesis looked right
offscreen but was wrong: introspection at the stall showed the shared stage
value present (`n_stranded=0`) and the tiles' *current-source* residency was
only the coarse floor level.

**Fix (landed):** scope `_resident_levels_from_lifecycle` to the tile's current
`source_id`/`tile_id` so the ladder and `best_floor_key` agree. Supporting
hardening in the same change: `best_floor_key` now skips floor levels that
cannot actually be drawn (complex/RGB level whose display histogram is not
resident) so a missing finer histogram falls back to a presentable coarser
level instead of a hole; and `request_montage_replan`'s coalesced `fire()`
replans the *current* session rather than bailing when the session id/key
changed under scroll churn (a dropped-wakeup race the busier onscreen event
loop lost). Verified: 5× pyqtgraph + 2× vispy resident scroll-scrub runs settle
(`scroll_fast_settled=True`, `slow_unsettled=0`, 50/50 presented, `stale=0`),
uniform-LOD screenshot, no stall-guard trips.

**Harness:** the montage waits now fail fast — a montage that is not settled
with no kernel work in flight (a lost wakeup) aborts after a short grace window
with the stall signature instead of hanging the full timeout; post-fast-jump
and initial-montage waits were shortened. Use this to catch regressions of this
class quickly.

## Known-slow / wedge evidence (not test failures)

| symptom | evidence | resolved by |
|---|---|---|
| Re-measured after 76860b4b (gate-signature + replan-coalescing fixes): pyqtgraph raw settled 1.87 s (max gap 693 ms), FFT 3.23 s (gap 760 ms), level refinement 3.57 s (gap 107 ms); vispy raw 2.27 s (gap 485 ms — pre-R2 parity), FFT 6.08 s (gap 586 ms, was 42.8 s / 22.8 s), level refinement 1.64 s (gap 127 ms). STALL 0 both backends | `tests/artifacts/r2b-postfix2-{pyqtgraph,vispy}-resident.jsonl` | Remaining gaps are the initial session build + first commit (~0.5–0.8 s single block; `_montage_viewport_plan_ms` ≈ 570 ms is one known piece) and per-commit fixed cost. Still above the 16 ms heartbeat bar → R2b item 5 continues; the *wedges* and multi-second storms are gone |
| VisPy level refinement is 1.6 s where main's uniform-only path did a 272-tile level drag in ~0.26 s | same JSONLs | R3 (level values into the lifecycle machine; uniform-only fast path must not regress through the generic commit) |

Resolved on this branch (for the record):

- R2 commit storm / camera-only churn / dep-parking / per-tile-native FFT
  floors / windowing-metadata — fixed in 4464a6e4 (tests pin the camera
  invariant and the no-deps-ordering rule).
- WIP symptom patches reverted in a23fb2b2 (worker clamps, bridge drain
  clamps, dtype sniffing, harness event-processing edit).
- `test_tile_layer_level_change_uses_governed_presentation_batches`,
  `test_scalar_tile_layer_level_change_uses_governed_batches_without_image_replacement`
  — red on main since 6fa5c758 (legacy cost-signature stubs); fixed.
- `test_vispy_montage_view_range_change_expands_visible_tile_set` — red on
  main; bisected to 2995d039; fixed by plan-time content extent (a3992c8f).
- The R1 vispy/resident FFT transform-preview wedge no longer sticks after
  R2; the GPU harness includes
  `test_fft_preview_refinement_settles_without_stalls` (7/7 green).
