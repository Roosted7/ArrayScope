# Plan 01 — Delta-commit walk cost (warm scrub ~36 ms → target ≤25 ms, stretch ≤16 ms)

**Status:** landed 2026-07-06. Read `README.md` ground rules first.

## Background

Historical dimension-scrub cost on the LOD residency branch (VisPy, 272-tile montage):
~216 ms → ~23 ms (uncached burst, P2-adjacent scrub fastpath) / ~50 ms (cached rebuild) → ~36 ms
(session reuse + index-window retarget, commit 07121fc2). ADR 0051 P2 names the remainder
explicitly: "the remaining cost is the delta-commit walk itself (vispy layer update, overlays,
full-image apply)". That is ~20–30 ms of synchronous work per warm scrub step — above Thomas's
~16 ms interaction bar.

**Goal:** make the per-step commit cost proportional to what actually changed in the step
(the delta), not to scene size — without violating the six machine rules (especially rule 1
no-optimistic-transitions and rule 6 event-driven convergence).

## Step 0 — Baseline (do not skip)

1. Tell Thomas the next run is ONSCREEN and to keep hands off.
2. Run `tmp_probes/profile_cached_rebuild.py` with the arrayscope conda python (foreground DC
   process + blocking read; ~30–60 s). It prints per-step timings and retarget/reject counters.
3. Also run `/tmp/lod-baseline/verify_scrub.py` for the heartbeat-gap histogram.
4. Record: warm scrub ms/step, retargets vs rejects (rejects should be ~0 since the
   level-pending fallback removal), max heartbeat gap. Save output to
   `/tmp/lod-baseline/p01_baseline.log`. If warm scrub is no longer ~36 ms while replaying this
   historical plan, stop and re-check ADR 0051's recorded baseline and kill switches before
   interpreting the result.

## Step 1 — Profile the walk (find the split)

py-spy is out (Python 3.14); use cProfile around the scrub loop:

1. Copy `profile_cached_rebuild.py` to `tmp_probes/p01_cprofile.py`; wrap the scrub-step loop
   in `cProfile.Profile()`; dump with `pstats` sorted by cumulative, top ~40.
2. Attribute time to the three named suspects + anything new:
   a. **VisPy layer update** — backend-side texture/atlas layer apply per commit.
   b. **Overlays** — overlay rebuild per commit.
   c. **Full-image apply** — applying the whole presentation when only a delta changed.
   Also check payload-wrapper construction and any remaining per-commit scans over all 272
   records (grep the hot function names in `presentation/`, `window/`, `backends/vispy/`).
3. Write the measured split into the work log before changing anything. Decide the order:
   biggest wedge first.

## Step 2 — Fix, one lever per commit

For each lever: implement → full suite → GPU harness → re-run Step 0 probes → commit with
before/after ms. Candidate levers, matched to the suspects (verify against the profile — do
not implement a lever whose cost didn't show up):

1. **Delta-proportional walk.** If the commit walk visits every tile record per step, restrict
   it to the machine's dirty/changed set (the machine already knows: emitted-but-unconfirmed,
   dirty payloads, parked→re-armed). The derivation (`derive_montage_dispatch`) may already
   compute the needed sets — reuse, don't duplicate (rule 3 of README ground rules).
2. **Skip unchanged overlays.** Cache overlay state keyed by the inputs that actually change it
   (plan revision, selection, viewport); rebuild only on key change. No optimistic skip: the
   key must cover every input.
3. **Sub-region / identity-gated apply.** If the backend re-applies a full image/layer when the
   acknowledged payload identity is unchanged, gate the apply on identity delta (the identity
   map is already the machine's presented truth — X5b). Never skip an apply the backend hasn't
   acknowledged as already-shown (rule 1).
4. **Bounded wrapper construction** already exists (admission-budget capped); confirm the
   backlog continuation consumes leftovers (rule 6) rather than adding cost per step.

Constraints:
- All state reads/writes via the machine; no new session collections.
- Interaction gating stays: mid-burst steps present floors/cached payloads only; only the
  landing step plans.
- If a lever needs a risky behavior change, add a kill switch env var mirroring the existing
  pattern (`ARRAYSCOPE_DISABLE_...`) and note it in the ADR.

## Step 3 — Verify

1. Full suite `-n 16` green (~35 s; known flakes list in README).
2. GPU harness green, `stall_repairs==0` (~14 s).
3. Wedge repro still clean (0 `STALL WATCHDOG` lines, ~20 s):
   `ARRAYSCOPE_PYQTGRAPH_RESIDENT_LOD=1 ~/miniconda3/envs/arrayscope/bin/python -m arrayscope.tools.profile_montage_workflow --backend pyqtgraph --montage-lod-policy resident`
4. Probe outcomes (onscreen, hands-off): warm cached-rebuild scrub ≤25 ms/step (stretch ≤16),
   uncached burst not regressed (~23 ms), pan max heartbeat gap ~16 ms, no sync step >50 ms.
5. Scrub-back / zoom-back manual sanity: no stale-LOD or wrong-content tiles (if in doubt, the
   GPU-harness content assertions are the arbiter).

## Step 4 — Docs + memory

1. ADR 0051: replace "**P2 remaining:** the per-commit delta walk cost." with a landed
   paragraph in the P2 phase entry (what changed, before/after ms, any new kill switch).
2. `docs/roadmap.md`: advance the X5 active queue if this changes priority; update
   `docs/current-state.md` only if the high-level lifecycle/performance state changes.
3. Commit (style: what + why + numbers). Then update Claude memory
   (`arrayscope-lod-residency`): new tip, new warm-scrub number, new roadmap head, any new
   gotchas/probes.
