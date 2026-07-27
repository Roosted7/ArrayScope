# Frozen performance baselines and acceptance ledger

## Why this file exists

`docs/queue.md` commits to *"benchmark deltas stay within ±10% of the frozen
baseline"*. Until 2026-07-25 that baseline existed only as untracked JSONL under
`tests/artifacts/`, alongside ~24 GB of screenshots and profiler traces. The raw
artifacts have been deleted and purged from git history — **artifacts do not
belong in the repository**. What a bar needs is the *numbers*, and those live
here.

The runs themselves are reproducible: check out the tagged version and re-run the
harness. Re-measuring on today's machine is arguably better evidence than a stale
file, since the comparison then shares one kernel, one GPU driver, and one power
profile.

**Reading rule.** These are historical reference points, not current truth. A
number here is only comparable to a fresh run made the same way — same backend,
same `qt_qpa_platform`, same scenario. When in doubt, re-baseline rather than
trust a row below.

---

## v0.8.0 rendering benchmark — the ±10% reference

Recorded 2026-06-22 at the v0.8.0 release. Regenerate with the command in
[`docs/testing/release-candidate.md`](release-candidate.md).

The VisPy rows below are preserved as release-era evidence only. VisPy is no
longer an executable comparator or a current performance gate. Current changes
must measure WGPU and PyQtGraph together; WGPU is the baseline and PyQtGraph
gets the documented 2× allowance.

**Environment (both runs).** `Linux-7.0.12-zen1-1-zen-x86_64-with-glibc2.43`,
Python 3.12.13, PySide6, Wayland session, `gpu_max_texture_size` 4096. The two
runs differ only in `qt_qpa_platform`: `offscreen` for the `-linux` file, and the
default (real platform) for the other.

`elapsed_ms`, lower is better. `frame_count` was 0 for every scenario in both
runs — these measure commit/upload cost, not presented frames.

| Scenario | Backend | offscreen | default platform |
|---|---|---:|---:|
| `scalar_level_preview` | pyqtgraph | 0.035 | 0.035 |
| `scalar_level_preview` | vispy | 0.042 | 0.036 |
| `complex_tile_level_preview` | pyqtgraph | 0.482 | 0.274 |
| `complex_tile_level_preview` | vispy | 0.094 | 0.121 |
| `tile_level_uniform_update` | pyqtgraph | 2.275 | 2.126 |
| `tile_level_uniform_update` | vispy | 0.327 | 0.338 |
| `clean_tile_flush` | pyqtgraph | 0.136 | 0.208 |
| `clean_tile_flush` | vispy | 0.139 | 0.177 |
| `large_complex_tiled_initial` | pyqtgraph | 36.156 | 41.233 |
| `large_complex_tiled_initial` | vispy | 7.859 | 7.880 |
| `one_dirty_tile_commit` | pyqtgraph | 0.607 | 0.654 |
| `one_dirty_tile_commit` | vispy | 0.829 | 0.873 |
| `pan_zoom_no_upload` | pyqtgraph | 0.112 | 0.104 |
| `pan_zoom_no_upload` | vispy | 0.138 | 0.237 |
| `progressive_tile_stream` | pyqtgraph | 32.116 | 52.280 |
| `progressive_tile_stream` | vispy | 16.460 | 18.598 |
| `warm_residency_queue_scaling` | vispy | 1.199 | 1.294 |

**What the shape says.** VisPy led pyqtgraph by ~4.6x on
`large_complex_tiled_initial` and ~2-3x on `progressive_tile_stream` and
`tile_level_uniform_update` — the upload-bound scenarios. pyqtgraph was mildly
ahead on `one_dirty_tile_commit`. Note the run-to-run spread on
`progressive_tile_stream` (32.1 vs 52.3 ms, a 63% swing between two runs of the
same code): treat single-run deltas on that scenario below ~60% as noise, not
signal. This is precisely why the ±10% bar is applied to a re-measured pair, not
to a lone historical row.

## v0.8.0 diagnostics snapshots

Two summaries were captured 2026-06-22. Both are pyqtgraph; the vispy file
recorded the same field set.

| Field | Short run | 311 s run |
|---|---:|---:|
| Snapshots | 3 | 6 |
| Trace duration | 0.020 s | 311.204 s |
| Expected / median sample interval | 500 / 9.9 ms | 500 / 12.9 ms |
| Maximum sample gap | 13.3 ms | 311164.6 ms |
| Stalls above threshold | 0 | 1 |
| `render_timing.last_render_sync_ms` | 7.786 | 8.330 |
| `render_timing.last_display_commit_ms` | 2.242 | 2.375 |
| `render_timing.last_set_image_ms` | 1.288 | 1.538 |
| `render_timing.last_control_sync_ms` | 1.055 | 1.067 |
| `render_timing.last_worker_queue_wait_ms` | 1.079 | 0.872 |
| `montage_timing.last_viewport_plan_ms` | 1.307 | 1.320 |
| `montage_timing.last_initial_commit_ms` | 1.142 | 1.485 |
| `montage_timing.last_canvas_commit_ms` | 1.099 | 1.433 |

The 311 s run's single "stall" is an artifact of the measurement, not the app:
the 311164.6 ms gap is the whole idle trace between two snapshots, with no
session loaded (`loaded/pending 0/0`). The per-field timings are the durable
part.

---

## Journey-matrix acceptance ledger

33 journey-matrix runs were recorded between 2026-07-17 and 2026-07-21. Only the
per-run pass/fail verdicts are kept here; the diagnosis behind each red lives in
the dossiers under [`docs/redesign/`](../redesign/) and the narrative in
[`docs/queue.md`](../queue.md) row 3 and [`docs/queue-done.md`](../queue-done.md).

This ledger records the historical three-backend decision evidence. The
maintained matrix is now 12 cells: six journeys across WGPU and PyQtGraph.
Historical VisPy rows below remain evidence for the retirement decision, not
cells in the current gate.

Journeys are `cold_fill`, `zoom_in`, `zoom_out`, `scroll_shuffle`,
`index_scroll`, and — from 2026-07-21 — `deep_zoom_far_scroll`. Early
(2026-07-17) runs used a 10-row schema with boolean statuses; later runs use 15
rows (3 backends × 5 journeys) or 18.

### Runs that went fully green

| Run | Ring | Present method | Rows |
|---|---|---|---:|
| `journey-matrix-2026-07-19-v1` | real-wayland | — | 15 |
| `journey-matrix-2026-07-19-v5` | real-wayland | — | 15 |
| `journey-matrix-2026-07-19-v7` | real-wayland | — | 15 |
| `journey-matrix-2026-07-20-integration-audit` | real-wayland | bitmap | 15 |

`v7` on 2026-07-19 is the reference all-green matrix: vispy, pyqtgraph and wgpu
each passed all five journeys.

### Final recorded state (2026-07-21-postfix, 18 rows)

| Backend | cold_fill | zoom_in | zoom_out | scroll_shuffle | index_scroll | deep_zoom_far_scroll |
|---|---|---|---|---|---|---|
| wgpu | pass | pass | pass | **fail** | pass | **fail** |
| pyqtgraph | pass | **fail** | pass | **fail** | **fail** | **fail** |
| vispy | pass | pass | pass | **fail** | **fail** | **fail** |

`deep_zoom_far_scroll` was newly added and red across all three backends — it was
introduced to expose a known gap, so its reds are expected, not a regression.
wgpu was the strongest backend in the final matrix, consistent with the
subsequent backend-retirement decision.

### All runs, chronologically

`ok` is the matrix-level verdict; a run is green only if every row passed.

| Run | Ring | Present | Rows | ok |
|---|---|---|---:|---|
| `journey-matrix-2026-07-17` | real-wayland | — | 10 | no |
| `journey-matrix-2026-07-17-v2` | real-wayland | — | 10 | no |
| `journey-matrix-2026-07-17-v3` | real-wayland | — | 10 | no |
| `journey-matrix-offscreen-2026-07-17` | offscreen-smoke | — | 10 | no |
| `journey-matrix-main-7b6a7e9b-2026-07-17` | real-wayland | — | 10 | no |
| `journey-matrix-main-caa62ca9-2026-07-18` | real-wayland | — | 10 | no |
| `journey-matrix-postpolicy-2026-07-18` | real-wayland | — | 10 | no |
| `journey-matrix-wgpu-2026-07-18` (+ v2, v3, v7, v17, v18, v19) | real-wayland | — | 15 | no |
| `journey-matrix-2026-07-19-v1` | real-wayland | — | 15 | **yes** |
| `journey-matrix-2026-07-19-v2` … `v4`, `v6` | real-wayland | — | 15 | no |
| `journey-matrix-2026-07-19-v5` | real-wayland | — | 15 | **yes** |
| `journey-matrix-2026-07-19-v7` | real-wayland | — | 15 | **yes** |
| `journey-matrix-baseline-smoke-2026-07-19` | offscreen-smoke | — | 15 | no |
| `journey-matrix-blackoutfix-2026-07-19` | real-wayland | — | 15 | no |
| `journey-matrix-glyph-2026-07-19` | real-wayland | — | 15 | no |
| `journey-matrix-2026-07-20-integration-audit` | real-wayland | bitmap | 15 | **yes** |
| `journey-matrix-2026-07-20-post-audit` | real-wayland | bitmap | 15 | no |
| `journey-matrix-2026-07-20-postfix` (+ `-v2`, `-post-fixes`, `-final`, `-final-audit`) | real-wayland | screen | 15 | no |
| `journey-matrix-2026-07-21-managed-weston` | real-wayland | screen | 18 | no |
| `journey-matrix-2026-07-21-postfix` | real-wayland | screen | 18 | no |

---

## What was deleted, and how to get it back

Purged 2026-07-25: ~24.2 GB under `tests/artifacts/` — 25,300 PNG screenshots and
~830 JSONL traces (largest a single 206 MB compositor trace), plus 38 files that
had been force-added past `.gitignore` and were removed from history with
`git filter-repo`.

To reproduce any of it, check out the relevant commit and re-run the harness:
the journey matrix via `tests/app/test_journey_matrix.py`, the rendering
benchmark per `docs/testing/release-candidate.md`, and the wgpu gate evidence per
[`docs/proposals/wgpu-renderer-experiment.md`](../proposals/wgpu-renderer-experiment.md).

Some older documents still cite `tests/artifacts/<run>/…` paths. Those citations
were already dangling before this purge — several referenced runs had been
cleaned up in earlier passes. Treat any such path as a description of how the
evidence was produced, not as a file you can open.
