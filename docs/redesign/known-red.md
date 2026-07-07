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
| _none in the focused R2b ledger_ | The former 8-test ledger is resolved by R2b stabilization. | Keep R3 scroll-scrub/lost-wakeup work below; do not mark R2 closed until benchmark/manual/GPU gates pass. |

Resolved by R2b stabilization:

- `test_relative_window_levels_survive_fast_scroll_with_render_in_flight`
- `test_relative_window_levels_match_for_cached_and_uncached_display_tiles`
- `test_window_levels_sync_between_windows`
- `test_stale_tile_result_does_not_clear_updating_overlay`
- `test_hidden_montage_roi_overlay_does_not_sample_loading_placeholder`
- `test_format_runtime_diagnostics_includes_all_major_sections`
- `test_pyqtgraph_complex_fast_scroll_budget_keeps_presentable_slots`
- `test_large_complex_montage_tile_layer_histogram_drag_does_not_update_base_image_item`

## Scroll-scrub convergence stall (fast index retarget) — PRIORITY

**Repro (offscreen, deterministic):** `profile_montage_workflow`
`montage_scroll_scrub` phase — `scroll_fast_settled=False`: a full-window
index jump (100:150 → 150:200) does not settle in 20 s. User-visible as the
"hang with wrong levels, one stuck tile, then all tiles glitch" report and
the single un-converged tile in the 2026-07 screenshot.

**Root cause (traced, not guessed):** the index-scrub fast path
`MontageRenderSession.retarget_index_window` REUSES the session for the new
index window and demotes every tile whose new content isn't cached
(`rendered_tiles.pop` + `loading_tiles.add` + `pending_removals` +
`dirty`, montage_index is grid position so all 50 slots turn over). It then
relies on fresh evaluations to re-materialize them. Under the R2 async
pipeline those evaluations complete in the kernel (`display_preparation`
lane: 50 completed) but do NOT re-enter `mark_materialized` for the reused
session — the tiles stay in `loading` with `rendered_tiles` emptied,
`active_tile_requests=0`, nothing in flight: a lost-wakeup at the
retarget↔pipeline seam. This is the same optimization that used to carry the
`ARRAYSCOPE_DISABLE_SCRUB_FASTPATH` kill switch (removed in redesign
hygiene); its interaction with the pipeline is the defect.

**Fix owner: R3** (viewport/index-scoped retarget through the pipeline). Two
candidate approaches, decide with the benchmark: (a) route index-window
changes through a full session rebuild (correct, slower — measure the cost
on the scroll stage), or (b) make `retarget_index_window` submit its demoted
tiles through `pipeline.retarget` and gate loading→presented on the pipeline
acks like the fresh-session path does. Do NOT reinstate a kill switch.

**Also present (smaller, same class):** even without scrubbing, ~1/120 tiles
can reach a resident demanded level but stay presented at native with no
dirty flag; a forced retarget does not converge it (level-completion
lost-wakeup). Likely the same seam; fix together in R3.

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
  — red on main since 6fa5c758 (stubs missing `work_signature`); fixed.
- `test_vispy_montage_view_range_change_expands_visible_tile_set` — red on
  main; bisected to 2995d039; fixed by plan-time content extent (a3992c8f).
- The R1 vispy/resident FFT transform-preview wedge no longer sticks after
  R2; the GPU harness includes
  `test_fft_preview_refinement_settles_without_stalls` (7/7 green).
