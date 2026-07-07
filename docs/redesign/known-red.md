# Known-red ledger (redesign branch)

Rule 5 of the [ground rules](README.md): window/display tests may only be
red if listed here with the plan step that fixes or deletes them. Qt-free
suites are never allowed red.

| test | state | cause | resolved by |
|---|---|---|---|
| none in the latest R1 all-core non-GPU run | n/a | n/a | keep this table empty unless a reproducible red test is intentionally carried by a named plan |

## Known-slow / wedge evidence (not test failures)

| symptom | evidence | resolved by |
|---|---|---|
| FFT transform-preview montage floor fill takes ~64 s (272 tiles, vispy resident) with event-loop gaps up to 4.6 s; progress happens only via STALL WATCHDOG rescues (`lost wakeup: loading≈267→231, flush_pending=True, final=True`) | `profile_montage_workflow --backend vispy --montage-lod-policy resident --jsonl tests/artifacts/r1-profile-montage-workflow-vispy-resident.jsonl`, 2026-07-07 on `redesign`; raw montage is ~2.8 s and the later level-only refinement is ~145 ms, so the wedge is specific to the transform-preview floor pump | R2 replaces the flush/pump path with kernel completions + capacity waiters (lost wakeups become structurally impossible); R3 owns the transform-preview queue. If a one-line pump re-arm is found earlier, fix on main and record here |
| GPU harness (`tests/gpu_interaction`) asserts `stall_repairs==0` but does not cover the FFT/transform-preview scenario — the 73 s wedge above passes CI | add an FFT-preview scenario to the harness in R2 step 4 | R2 |

Resolved on this branch (for the record):

- `test_tile_layer_level_change_uses_governed_presentation_batches`,
  `test_scalar_tile_layer_level_change_uses_governed_batches_without_image_replacement`
  — red on main since 6fa5c758 (stubs missing `work_signature`); fixed.
- `test_vispy_montage_view_range_change_expands_visible_tile_set` — red on
  main; bisected to 2995d039 (fit-unlock re-ranged around stale
  single-slice bounds while preview-floor gating deferred the first
  commit); fixed by plan-time content extent (a3992c8f).
- Earlier parallel-only settings/governor/scheduler flakes are no longer
  current R1 known-red entries after the kernel lane-quota and bridge-drain
  rewrite; keep them out of the active table unless they reproduce again.
