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

Resolved on this branch (for the record):

- `test_tile_layer_level_change_uses_governed_presentation_batches`,
  `test_scalar_tile_layer_level_change_uses_governed_batches_without_image_replacement`
  — red on main since 6fa5c758 (stubs missing `work_signature`); fixed.
- `test_vispy_montage_view_range_change_expands_visible_tile_set` — red on
  main; bisected to 2995d039 (fit-unlock re-ranged around stale
  single-slice bounds while preview-floor gating deferred the first
  commit); fixed by plan-time content extent (a3992c8f).
