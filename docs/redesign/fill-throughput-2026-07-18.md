# 272-tile cold-fill "stall" — throughput collapse, not a lost wakeup (2026-07-18)

**Status:** root-caused, fixed, gated. Offscreen repro green on vispy and
wgpu; pyqtgraph completes the fill and unmasks a distinct pre-existing
entry-blackout defect (chip spawned, see Follow-ups).

**Blocks lifted:** this was the blocker for the row-3(d) perf-bars promotion
measurement (`--backend all` perf-bars run) — the fill stages now complete on
every backend, so that measurement is unblocked.

## The repro and the bisect verdict

Deterministic offscreen repro (~2.5 min):

```
QT_QPA_PLATFORM=offscreen python -m arrayscope.tools.profile_montage_workflow \
  --backend vispy --data data/_WIPDelRec-tT2_20260223150234_14.nii \
  --session-fixture "" --stages raw_full_tiled_montage
```

→ `TimeoutError … target_unsettled=271 … gate_backlog armed` for vispy AND
pyqtgraph, offscreen AND real Wayland, on main and the wgpu branch.

The queue's hypothesis was a regression in main `661b6ba5..976ea275`
(G5 measured "PyQtGraph raw 272/272 in ~11.4 s" green on 2026-07-17).
**Bisect verdict: there is no breaking commit.** The stall reproduces at the
G5 merge commit `661b6ba5` itself and at `62904128` (the very commit whose
message records the 11.4 s green run). The G5 evidence was a *real-Wayland
pyqtgraph* run that fit inside the wait budget by luck of phase accounting;
the offscreen invocation used here was never validated green. The
"harness invocation was never re-validated" theory from the queue row is the
correct one — with real product throughput defects underneath it.

## What the trace-walk actually found

Instrumenting the presentation gate, replan gate, and commit rearm
(`presentation_gate` / `replan_gate` / `commit_rearm` trace events, now
permanent) proved **every wakeup fires**: armed-post == fired-commit across
the whole run, `commit_rearm` alternates rearm/no-backlog correctly, and the
last trace event of a failing run is a healthy `REARM rearm → armed-post`
cut off by the harness deadline. This is NOT a member of the deferred-stage
lost-wakeup family ([stale-empty-tiles-2026-07-16](stale-empty-tiles-2026-07-16.md),
[coverage-stall-2026-07-15](coverage-stall-2026-07-15.md)) — the montage
watchdog stays silent because the gates are genuinely live.

The fill was simply **too slow for its budget**, for four compounding
reasons (cProfile, offscreen vispy):

1. **O(tiles²) page-set resolution per fill.** Every replan re-derived
   floors for all 272 tiles (`best_floor_key` → `_page_set_resolution`), and
   every 4-upsert commit triggered a replan: 143,399 `resolved_page_set`
   calls (~78 µs each, 11.2 s cumulative) across 51 replans — ~2,800 per
   replan, mostly re-resolving identical (usually *incomplete/None*) sets.
2. **Latency-governor inversion on the commit cohort.** The tiled commit's
   cost is fixed-dominated (full-plan classify + delta walk + acknowledgement
   ≈ 80–120 ms regardless of item count; marginal per-item cost is a few ms
   and separately byte-capped). The governor saw over-budget callbacks and
   clamped the batch to its minimum (4 for vispy), which cannot shorten the
   callback — it only multiplies the fixed cost: 272 tiles × ~2 commits each
   ÷ 4 per turn ≈ 136 full-plan walks.
3. **O(tiles) linear scans inside the per-tile floor scan.** `best_floor_key`
   scanned `plan.tiles` and `floor_can_progress` scanned `visible_tiles`
   per call — O(tiles²) per ladder-mark pass, ~10 such passes per pyqtgraph
   viewport retarget.
4. **A gesture budget applied to a build.** `_run_phase` clamped the
   completion wait to the 5 s interaction cap for the cold 272-tile fill.
   The churn harness learned this lesson on 2026-07-17 (`62904128`:
   "Build-time keeps a generous fill budget; the 5 s cap stays on
   per-gesture probes") but `profile_montage_workflow` never adopted it, so
   a slow-but-progressing fill was reported as a stall. Pass/fail authority
   for fill *time* is the recorded milestones (perf-bars program), not the
   wait budget.

## The fixes (red-first where testable)

- **Page-set resolution memo** (`arrayscope/display/pyramid.py`):
  `resolved_page_set` results — including the None incomplete-coverage
  verdict, the common case during a fill — are memoized per residency
  revision. Resolution is a pure function of (revision, requested keys); the
  resolver-table snapshot already relied on every mutation bumping
  `_revision`. Red-first:
  `tests/display/test_lod_page_route.py::test_page_set_resolution_is_memoized_per_residency_revision`.
  **Store race (caught by the journey matrix, fixed):** the result is
  computed outside the lock; a worker admit can bump residency mid-compute,
  and if a newer query refreshes the memo epoch first, storing the stale
  verdict poisons the fresh epoch — usually a stale None that keeps
  reporting a now-resident floor as missing until the *next* residency
  change, which at a convergence tail never comes. The first memo version
  turned the green pyqtgraph scroll_shuffle/index_scroll journey rows red
  (three matrix runs, then a one-driver A/B bisect against the base
  product pinned it); the store now requires the revision to be unchanged
  across the whole compute (miss-epoch == live revision == store-epoch).
  Deterministic regression (red-first against the old guard):
  `::test_page_set_resolution_memo_rejects_stale_result_after_concurrent_admit`.
- **Idle backlog cohort — shader-windowing backends only**
  (`arrayscope/window/frame_effects.py`, `_idle_backlog_cohort`):
  non-interactive commits with a backlog deeper than the governed limit take
  a fixed-cost-amortizing cohort (`min(32, backlog)`); the byte cap stays
  authoritative for upload size and the interactive clamp is untouched.
  Scoped to `_persistent_tile_upsert_limits` (vispy/wgpu upload-only
  commits): the CPU-windowed pyqtgraph layer pays a real per-item cost, and
  a first attempt to apply the cohort there turned two green journey-matrix
  pyqtgraph scroll rows red (long idle callbacks delayed the next gesture's
  pixels) while buying nothing — the pyqtgraph fill is evidence-sweep-bound.
  Red-first:
  `tests/window/test_montage_backend.py::test_vispy_idle_upsert_cohort_scales_to_large_backlog`,
  `::test_pyqtgraph_idle_commits_keep_governed_cohort_under_deep_backlog`.
- **Indexed floor lookups** (`arrayscope/render/lod.py`,
  `_plan_tile_for_source` / `_visible_tile_for_number`): identity-keyed
  indexes over `plan.tiles` / `visible_tiles` replace the per-call linear
  scans.
- **Build budget for cold-fill stages**
  (`arrayscope/tools/profile_montage_workflow.py`,
  `COLD_FILL_BUILD_TIMEOUT_S = 120`): the raw/FFT full-montage phases carry a
  build-scale completion budget; gesture stages keep the 5 s cap. A genuine
  wedge still fails fast — the wait loop's stall detector (4 s of unchanged
  no-work signature) is independent of the deadline.
- **Permanent liveness traces**: `presentation_gate` (armed/coalesced/fired),
  `replan_gate` (armed/coalesced/fired), `commit_rearm`
  (rearm/repeat/no-backlog) in `frame_effects.py`/`frame_runtime.py`. A
  future stall trace now *proves* lost-wakeup vs throughput in one look:
  balanced armed/fired + progressing revisions = throughput; armed with no
  fired = lost wakeup; `commit_rearm repeat` + `commit_gate_no_progress` =
  the documented no-progress stop.

## Results (offscreen repro, this hardware)

| Backend | Before | After (final code) |
|---|---|---|
| vispy | TimeoutError, 213–261/272 | **PASS**, 272/272, full refined 8.0 s |
| wgpu | stalled in same stage | **PASS**, 272/272, full refined 9.3 s |
| pyqtgraph | TimeoutError, 0–233/272 | fill completes 272/272 (~24 s); R8 gate red on a **pre-existing** entry blackout (below) |

Real-Wayland vispy: 272/272, full refined 8.9 s; only the known perf-bars
`gui_callbacks_below_50ms` red remains (red in the G5-era green run too).
Full offscreen suite: 2427 passed, 0 failed.

## Journey-matrix cold_fill reds — mechanism confirmed and cleared

Baseline `journey-matrix-wgpu-2026-07-18-v7` (real Wayland): vispy cold_fill
red (`level_converged_within_budget: false`) and pyqtgraph cold_fill red
(`demand_fresh_within_budget: false`, 5.8 s).

Post-fix runs `journey-matrix-fillfix-2026-07-18-v2` and `-v3` (real
Wayland): **cold_fill green on all three backends, twice** — the standing
cold_fill reds were this throughput mechanism. v3 scoreboard: vispy 5/5,
wgpu 5/5, pyqtgraph green except `zoom_in` (`demand_fresh` — the standing
AUTO-camera demand-freshness lane, red in the v7 baseline too; the
`zoom_out` variant flips backends run-to-run).

The v2/v3/v4 runs also turned the previously-green pyqtgraph
scroll_shuffle/index_scroll rows red. A one-driver A/B bisect against the
base product (`tests/artifacts/scroll-control-*`) pinned it to the memo
store race described above (the cohort was scoped to shader-windowing
backends along the way — the CPU-windowed layer pays a real per-item cost,
so the deep-backlog cohort belongs to the upload-only path regardless);
with the race-fixed memo the scroll driver is green again.

Final full matrix `journey-matrix-fillfix-2026-07-18-v5`: vispy 5/5;
pyqtgraph cold_fill + scroll_shuffle + index_scroll green, zoom_in/zoom_out
red; wgpu 4/5 with zoom_out red — the remaining reds are exactly the
documented zoom demand-freshness family (queue row 3 open item, v6/v7-pinned
`first_new_pixels_ms=None` shape, adjudication chip already out), which
predates and is untouched by this work. Net vs the v7 baseline: both
standing cold_fill reds cleared, no new stable reds.

Note for worktree runs: the matrix drivers default to the relative
`data/<nifti>` path — symlink the main checkout's `data/` into the worktree
or every row dies `FileNotFoundError` with `instances: 0`.

## Follow-ups (pre-existing, unmasked or adjacent)

- **pyqtgraph montage-entry blackout — FIXED 2026-07-19** (chip completed):
  the successor's first pixels waited ~7.7 s because CPU windowing was gated
  on the full 272-source level evidence sweep (`commit_bail
  level-evidence-wait` from 0.6→5.3 s), while the montage-axis transition
  blanked the predecessor at entry (`presentation_transition_retention
  reason=montage-axis retained=false`). Three mechanisms, all fixed:
  1. *Full-sweep gate:* `tile_layer_first_pixels_wait_for_level_source` and
     `_publish_first_cpu_histogram` now accept a **provisional refined first
     batch** (≥ `MONTAGE_LEVEL_STATS_FIRST_CPU_BATCH` sources) for scopes
     larger than one batch; rank stays `MONTAGE_VISIBLE_SUBSET` and the
     settled-metadata refresh delivers the single refined re-window
     (contract point 6 amended in `docs/architecture/rendering.md`).
  2. *Evidence queued behind the fill it gates:* the visible-dependency
     evidence producers (payload batch, continuation, semantic sweep,
     histogram aggregate) ran at `VISIBLE_IMAGE`/UNRANKED while cold tile
     evaluations run at `INTERACTIVE` — the 4 ms first batch queued ~2.6 s
     deep. They now run at the same INTERACTIVE priority, rank 0
     (`FIRST_PIXEL_EVIDENCE_SCHEDULING_RANK`).
  3. *Entry blank:* `plan_presentation_transition` now returns an honest
     **montage-axis bridge** (retain, never atomic) when only the montage
     selection differs and the predecessor is settled; a
     `presentation_bridge_pending` flag carries the bridge through rebirths
     that replace a successor before its first commit (the fixed3 trace
     showed session 3 re-blanking at 0.25 s otherwise).
  Result (offscreen repro): R8 gate **PASS**, first pixels 1.98 s (was
  4.3–6.9 s offscreen / ~7.7 s field trace), `blackout_observed=False`,
  `minimum_retained=1`, first visible windowed at provisional rank 3 /
  quality REFINED, final levels converge to the refined full-population
  bounds, fill completes 272/272.
  Real-Wayland journey matrix
  (`tests/artifacts/journey-matrix-blackoutfix-2026-07-19`): 14/15 green,
  no new reds; `first_new_pixels_ms` on pyqtgraph cold_fill improved to
  353 ms (960–1010 ms across every committed incumbent run). The one red is
  the INCUMBENT pyqtgraph cold_fill `demand_fresh_within_budget` at
  incumbent magnitude (5.5 s vs 5.4–6.3 s in v1/v2/v3/v7) — so the earlier
  guess that the blackout was "likely the identity" of that standing red is
  **refuted**: it is the AUTO-camera demand-freshness lane (queue standing
  row), not the entry blackout.
- **Kernel queued-work shutdown drain fixed 2026-07-19:** shutdown now closes
  admission, cancels queued work plus running tokens, and applies one global
  five-second join deadline with loud live-thread/task-scope diagnostics.
  The real-Wayland workflow trace completes the GUI close callback in 56.8 ms
  instead of draining queued work for the recorded ~19–26 s; focused
  cancellation and global-deadline tests are green (`112343f8`). Whole-
  process exit is not yet bounded: a current non-daemon worker evaluation can
  continue after the loud deadline. The active queue retains the <5 s process
  exit gate rather than misreporting the callback result as full closure.
- The commit's fixed cost itself (~80–120 ms full-plan walk per commit) is
  perf-bars territory (50 ms GUI-callback bar is already red and owned
  there); the cohort change amortizes it but does not shrink it.
