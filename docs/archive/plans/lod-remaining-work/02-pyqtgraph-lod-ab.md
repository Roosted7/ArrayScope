# Plan 02 — Re-measure PyQtGraph resident-LOD A/B → default decision

**Status:** landed 2026-07-06. Read `README.md` ground rules first.

## Background

ADR 0050 phase 3 (commit e68788ae) implemented PyQtGraph resident LOD behind
`ARRAYSCOPE_PYQTGRAPH_RESIDENT_LOD=1`, default OFF per the "where measured" rule, because the
first hardware A/B (Wayland, 272-tile montage) REGRESSED: raw settle 1263->1441 ms, FFT settle
2590→4367 ms, level drag 9019→9686 ms.

Since then the **auto-levels wait wedge was fixed** (ADR 0051, commit f8b00bff line; stalls
8–10 → 0 per run). The wedge inflated FFT settle by ~2 s of watchdog rescue — i.e. most of the
FFT regression in the numbers the default-off decision was based on. **The decision must be
re-made on clean numbers.** Remaining named suspects if it still regresses: worker-side
ingest-reduction contention, and the level-drag phase being dominated by the
histogram/full-stats loop rather than per-tile re-window (ADR 0050 adoption-status paragraph).

The clean 2026-07-06 rerun fixed the display-payload bug and kept the PyQtGraph policy opt-in:
native raw/FFT/level medians were 1071/2245/7193 ms; resident medians were
1155/3232/3012 ms with 0 watchdogs and applied factor 4 / level 2. The level loop now wins
by more than 2x, but first-settle still regresses because display reductions are extra work after
native evaluation. The next adoption attempt is Plan 04, not another blind A/B.

## Step 1 — Clean A/B runs

Host conda python, Desktop Commander, foreground + blocking reads; ~30–40 s per run; never
concurrent with the test suite. Check `profile_montage_workflow --help` once for the exact
JSON-output flag.

For each arm, 3 repetitions (report median; if spread >15% investigate before concluding):

- **Native arm:** `~/miniconda3/envs/arrayscope/bin/python -m arrayscope.tools.profile_montage_workflow --backend pyqtgraph --montage-lod-policy native-only` → JSONL `/tmp/lod-baseline/p02_native_{1,2,3}.jsonl`
- **Resident arm:** same with `ARRAYSCOPE_PYQTGRAPH_RESIDENT_LOD=1` and `--montage-lod-policy resident` → `/tmp/lod-baseline/p02_resident_{1,2,3}.jsonl`

Per run, capture stderr and assert **0 `STALL WATCHDOG` lines** (a stall invalidates the run
AND is a new rule-6 bug — triage per README ground rule 2 before continuing).

## Step 2 — Compare

Metrics (per-phase dicts in the JSONL, `montage_lod_*` keys; the tool also prints a summary):
raw settle ms, FFT settle ms, level-drag ms, event-loop gap p95 per phase. Sanity-check the
resident arm actually applied reduction (applied factor 4 / level 2, as in the first A/B).

Decision table:

| Outcome | Action |
|---|---|
| Resident wins or ties every metric (≤ ~5% worse on any, wins on some) | Flip default → resident for PyQtGraph montage/tiled scenes (Step 3) |
| Mixed / still regresses | Keep default OFF; do Step 4 (split measurement) so the next attempt starts from evidence |

## Step 3 — If flipping the default

1. Change the default policy selection for PyQtGraph montage scenes (find it: grep
   `ARRAYSCOPE_PYQTGRAPH_RESIDENT_LOD` — the env check site is the seam). Keep the env var as
   an override/kill switch, now meaning "force"; add the inverse disable if the pattern calls
   for it.
2. Full suite `-n 16` + GPU harness (`stall_repairs==0`) + wedge repro (0 stalls).
3. One more onscreen sanity pass on PyQtGraph backend (open, scrub, level drag — hands off).

## Step 4 — If it still regresses: measure the split (do not guess)

1. **Level drag:** instrument or cProfile the level-drag phase; attribute time between the
   histogram/full-stats loop and per-tile re-window. If histogram/full-stats dominates, the
   expected LOD win cannot materialize there — record that in the ADR so nobody retries LOD
   for it; the fix would be in the stats loop itself.
2. **Settle:** compare per-phase worker-lane utilization native vs resident; ingest-reduction
   contention shows up as visible-lane work queued behind reduction (the class the
   zero-redundant-work pass erased on VisPy — see ADR 0050 "Zero redundant semantic work").
3. Record findings as concrete follow-up items in the ADR adoption-status paragraph.

## Step 5 — Docs + memory

1. ADR 0050 "Adoption status" paragraph: replace the stale first-A/B numbers with the new
   median numbers + date + decision (either way — a re-confirmed OFF is also a result).
2. `docs/roadmap.md`: advance the X5 active queue if the decision changes the next step; update
   `docs/current-state.md` only for the high-level PyQtGraph LOD state.
3. Commit; append to Claude memory (`arrayscope-lod-residency`) with the superseding decision and
   numbers; do not delete historical notes.
