# Known-red ledger (redesign branch)

Rule 5 of the [ground rules](README.md): window/display tests may only be
red if listed here with the plan step that fixes or deletes them. Qt-free
suites are never allowed red.

| test | state | cause | resolved by |
|---|---|---|---|
| `tests/ui/test_performance_settings.py::test_selecting_fft_workers_updates_settings` | flaky under `-n 16` only (pre-existing on main) | xdist worker contention on settings store | R4 (governor rescope reworks the settings⇄worker plumbing); passes alone today |
| `tests/ui/test_resource_governor_integration.py::test_resource_governor_applies_worker_and_callback_limits` | flaky under `-n 16` only (pre-existing on main) | same class | R4 deletes the per-controller limits this pins |
| `tests/ui/test_render_scheduler.py::test_compute_policy_configures_stage_and_montage_lanes` | flaky under `-n 16` only (pre-existing on main) | same class | R1 rewrites it against kernel lane quotas |
| `tests/window/test_montage_backend.py::test_montage_ready_display_payloads_commit_immediately` (teardown) | flaky teardown under `-n 16` only | Qt teardown race | R2 replaces the commit path and its fixture |

## Known-slow / wedge evidence (not test failures)

| symptom | evidence | resolved by |
|---|---|---|
| FFT transform-preview montage floor fill takes ~73 s (272 tiles, vispy resident) with event-loop gaps up to 4.7 s; progress happens only via STALL WATCHDOG rescues (`lost wakeup: loading≈267→231, flush_pending=True, final=True`, 4 fires) | `profile_montage_workflow --backend vispy --montage-lod-policy resident`, 2026-07-07 on `redesign` (a3992c8f tip; attribution A/B against `main` 6fa5c758 still owed — the raw montage phase is healthy at 2.4 s, and the refinement phase is 150 ms/0 B uploads, so the wedge is specific to the transform-preview floor pump) | R2 replaces the flush/pump path with kernel completions + capacity waiters (lost wakeups become structurally impossible); R3 owns the transform-preview queue. If a one-line pump re-arm is found earlier, fix on main and record here |
| GPU harness (`tests/gpu_interaction`) asserts `stall_repairs==0` but does not cover the FFT/transform-preview scenario — the 73 s wedge above passes CI | add an FFT-preview scenario to the harness in R2 step 4 | R2 |

Resolved on this branch (for the record):

- `test_tile_layer_level_change_uses_governed_presentation_batches`,
  `test_scalar_tile_layer_level_change_uses_governed_batches_without_image_replacement`
  — red on main since 6fa5c758 (stubs missing `work_signature`); fixed.
- `test_vispy_montage_view_range_change_expands_visible_tile_set` — red on
  main; bisected to 2995d039 (fit-unlock re-ranged around stale
  single-slice bounds while preview-floor gating deferred the first
  commit); fixed by plan-time content extent (a3992c8f).
