# Known-red ledger (redesign branch)

Rule 5 of the [ground rules](README.md): window/display tests may only be
red if listed here with the plan step that fixes or deletes them. Qt-free
suites are never allowed red.

**Status 2026-07 (post R2 stabilization pass):** suite 1642 passed / 8 red,
GPU harness 7/7 green. R2 is **code-complete but NOT closed** — its exit
gate (benchmark bars) has not been met and must be re-measured after the
root-cause fixes in 4464a6e4. Do not weaken the gate to match results
(that happened once on this branch; the gate text is restored and binding).

## Active red tests

| test | cause | resolved by |
|---|---|---|
| `test_relative_window_levels_survive_fast_scroll_with_render_in_flight`, `test_relative_window_levels_match_for_cached_and_uncached_display_tiles`, `test_window_levels_sync_between_windows` | one cluster: relative window-level semantics broke when R2 moved the levels/auto-window flow into `montage_commit`/`montage_level_stats` (pre-existing on the R2-step commits, NOT caused by the gate fixes) | R2b item 1 — triage as ONE root cause in the level-source flow |
| `test_stale_tile_result_does_not_clear_updating_overlay` | overlay lifecycle vs. the new admission path (stale results now classified by the kernel; the overlay-clear condition still assumes the old controller path) | R2b item 2 |
| `test_hidden_montage_roi_overlay_does_not_sample_loading_placeholder` | ROI sampling reads a placeholder while the gate defers the first commit by one loop turn | R2b item 2 |
| `test_format_runtime_diagnostics_includes_all_major_sections` | diagnostics formatter still expects deleted R1 controller sections | R2b item 3 (trivial: update the formatter/test to kernel sections) |
| `test_pyqtgraph_complex_fast_scroll_budget_keeps_presentable_slots` | pins an unimplemented optimization: skip redundant re-window when an exact payload arrives already windowed at the commit levels, with LAZY rgb_base rebuild on the first level change. The WIP dtype-sniff faked it and broke the rewindow contract | R2b item 4 — implement via payload metadata (evaluation levels on the payload), or move the test to R3 and delete the optimization claim |
| `test_large_complex_montage_tile_layer_histogram_drag_does_not_update_base_image_item` | one image replacement sneaks into a level drag on large complex montages (same windowing seam as above) | R2b item 4 |

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
