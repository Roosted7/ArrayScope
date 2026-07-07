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
| Full montage workflows still show multi-second event-loop gaps even though they complete without stall assertions | R2 JSONL saved in `tests/artifacts/r2-profile-montage-workflow-{pyqtgraph,vispy}-{resident,native-only}.jsonl`: PyQtGraph FFT full ~5.2–5.4 s with max gaps ~2.1–3.9 s; VisPy native-only FFT full ~6.8 s with max gap ~5.2 s; VisPy resident FFT full ~42.8 s with max gap ~22.8 s | R3 removes the remaining `montage_lod.py`/level-stats dual path; R4 audits governor/timer pacing and GUI-thread commit work |

Resolved on this branch (for the record):

- `test_tile_layer_level_change_uses_governed_presentation_batches`,
  `test_scalar_tile_layer_level_change_uses_governed_batches_without_image_replacement`
  — red on main since 6fa5c758 (stubs missing `work_signature`); fixed.
- `test_vispy_montage_view_range_change_expands_visible_tile_set` — red on
  main; bisected to 2995d039 (fit-unlock re-ranged around stale
  single-slice bounds while preview-floor gating deferred the first
  commit); fixed by plan-time content extent (a3992c8f).
- The R1 vispy/resident FFT transform-preview wedge no longer sticks after
  R2: the same workflow completes in the R2 JSONL, and the GPU harness now
  includes `test_fft_preview_refinement_settles_without_stalls`.
- Earlier parallel-only settings/governor/scheduler flakes are no longer
  current R1 known-red entries after the kernel lane-quota and bridge-drain
  rewrite; keep them out of the active table unless they reproduce again.
