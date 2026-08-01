# The level-evidence coverage deadlock — why the FFT scroll never showed a pixel (2026-08-01)

**Status:** root-caused and fixed. Closes the stall named and left in
[fill-bistability-2026-07-31](fill-bistability-2026-07-31.md) §6 and the
`STALL GUARD (4, 0, 60, …)` observed in
[pyqtgraph-commit-cost-attribution-2026-07-31](pyqtgraph-commit-cost-attribution-2026-07-31.md)
§6. Both are one defect. Pre-existing, not a regression from either dossier's
work.

**Environment honesty up front:** every run below shared the machine with other
sessions' suites (11–15 of 16 cores from elsewhere at the worst). The
adjudication rests on structural counters — trace doors, gate outcomes,
pass/fail of a hard timeout — and every test comparison is order-balanced
against a same-binary control arm. Wall-clock is quoted only to size effects.

## 1. The symptom

`profile_montage_workflow --backend pyqtgraph --scroll-max-tiles 60`, managed
Weston, full stage list. **Unmodified main, 4/4 runs show the defect:**

| run | outcome |
|---|---|
| base-1 | `TimeoutError: coverage pass did not close within 10.000s: coverage_open=True` |
| base-2 | same |
| base-3 | `STALL GUARD: montage frozen (no work in flight) — signature stable 2.5s: (4, 0, 60, 0, 0, 0, 1, 0, 0, 0)`, first tile **29 376 ms** |
| base-4 | same guard, first tile **11 462 ms** |

The signature reads: session 4, 0 loading, **60 required target tiles
unsettled**, 0 active requests, 0 dirty payloads, **0 presented**, 0 rendered.
Nothing in flight and nothing on screen. The scalar scroll passes in the same
process, every time.

## 2. The cycle

Three statements form a closed loop for any round whose first display commit is
blocked on round-level evidence:

1. `tile_layer_first_pixels_wait_for_level_source`
   ([frame_effects.py:5602](../../arrayscope/window/frame_effects.py)) refuses
   the first display commit while
   `round_level_evidence_source == "preview-cohort-pending"`.
2. `ProgressiveSchedulingPolicy.observe` closes coverage **only** once
   first-pass pixels are presented.
3. `_schedule_semantic_level_evidence`
   ([level_stats.py:1041](../../arrayscope/render/level_stats.py)) refuses to
   start the fallback semantic sweep while the source is
   `preview-cohort-pending` **and** `coverage_open`.

(3)'s comment says to start the fallback "only after the preview producer is
terminal" and uses `coverage_open` as that test. **`coverage_open` cannot be
it:** coverage closes on first pixels, and first pixels wait on exactly the
evidence the fallback would produce. The test is circular.

The claim is made by `rung_admitted`
([frame_effects.py:1130](../../arrayscope/window/frame_effects.py)) the moment
the **first** FLOOR rung is admitted — before any preview payload exists. The
only thing that can clear it is a *complete* cohort: `frozenset(stats_by_source)
!= frozenset(expected)` returns 0 and leaves the claim latched.

So the round's two producers both die:

- **the cohort** needs all 60 preview payloads at once. Measured on the stalled
  session: previews arrived strictly one at a time — tiles 24, 23, 25, 14, 34
  at 2.27 / 1.96 / 2.02 / 1.51 / 1.36 s each, one worker, never overlapping. 60
  tiles at that rate is ~2 minutes; the harness allows 10 s.
- **the sweep** was submitted at session start (`kernel_submit
  semantic-level-evidence`, lane `display_preview`, t=27.365) and *finished*
  (t=29.922) — but the claim had replaced its target generation ~150 ms after
  submission, so `done()` discarded the result at the
  `current_target.generation != generation` guard **and rescheduled nothing**.
  Every re-drive after that returned at (3).

Result: 10 commits, all `commit_bail outcome='level-evidence-wait'
wakeup='level-scan'`, `pending_upserts` crawling 1→5, 0 tiles presented, until
the 10 s coverage budget expired.

The two target generations are structurally different by construction —
`_ensure_semantic_level_evidence_target` builds a 3-tuple `(session_id,
level_key, expected)` and `_expect_preview_cohort_level_evidence` a 4-tuple
`(…, "preview-cohort")` — so the discard is guaranteed whenever the claim lands
mid-flight, not occasional.

Whether a round falls in is a **race**: the sweep took 23 ms in session 1 and
869 ms in the stalled session. Under load it loses, which is why this
reproduces under the managed compositor and only occasionally on a quiet
session.

### It is backend-independent

On WGPU `visible_dependency` is False, so the sweep is classified
`SchedulingWork.REFINEMENT` and the phase machine refuses it while coverage is
open — a second, independent way to reach the same dead end. That is the
`pending_upserts` up to 179 / `wakeup='level-scan'` shape
[fill-bistability §6](fill-bistability-2026-07-31.md) named. Same defect.

## 3. What was refuted

- **"`evaluate_shared_preview` is dead, so the cohort can never install"**
  (queue row 12's open adjudication). Refuted: `semantic_level_evidence_batch
  source='preview-cohort'` fires in production — 60 sources in session 1, 272 in
  session 2. The cohort is assembled from `session.display_tile_payloads` by the
  recovery path at
  [frame_effects.py:2108](../../arrayscope/window/frame_effects.py), not from a
  shared-preview row payload. That row-payload producer being uncalled is a
  separate question; it is not what starves this round.
- **"The preview cohort is the normal producer."** Refuted: across one healthy
  38 s scroll the sweep submitted **58** times and the cohort installed **0**.
  The cohort is the rare win; the sweep is the de-facto owner. That is what
  makes parking the sweep behind it so expensive.
- **"It is simply slow, not stalled."** Refuted by the signature: 2/4 baseline
  runs show `no work in flight` with 0 presented — zero throughput, not low.

## 4. The fix

Three narrow changes, each mutant-verified.

1. **The claim may not park a producer the round is waiting on.**
   `_schedule_semantic_level_evidence` now hands the round back when the cohort
   has proven it is not atomic *and* first pixels are still owed, instead of
   returning on `coverage_open`. "Proven not atomic" is one short cohort
   admission — the existing `frozenset(...) != frozenset(expected)` branch,
   which now records `preview_cohort_incomplete_observed`.
2. **The hand-back is terminal for the round**
   (`preview_cohort_round_claim_released`). Without it the preview ladder's next
   FLOOR rung re-claims and resets the sweep's cursor on every arrival. It also
   keeps a late cohort from restating levels a CPU backend has already baked
   into pixels (R3).
3. **A discarded batch re-arms its owner.** `done()`'s generation guard now
   re-drives `_schedule_semantic_level_evidence` when the session is still
   current. Discarding the *result* is right; discarding the *continuation* left
   the round's only guaranteed producer dead.

Plus the shader complement: once the round is handed back, preview rows are
routed to `_admit_first_pass_level_evidence_batch` instead of being dropped —
rough evidence is all a shader backend needs to present, and it has no other
producer during coverage.

The healthy atomic case is unchanged: with no short admission the park stands,
which is what the two pre-existing tests
(`test_admitted_preview_claim_prevents_source_sweep_race`,
`test_pending_preview_cohort_parks_sweep_only_while_first_presentation_is_blocked`)
pin, and both still pass untouched.

### New instrument

`level_evidence_gate` names which of the sweep's ten doors each scheduling
attempt died at — `cohort-pending-coverage-open`, `cohort-incomplete`,
`result-discarded-generation`, `submitted`, … A round that produced all its
evidence and a round that produced none previously read identically from
outside; this is what separated them. It is the `rung_prepare` of level
evidence. The stall run showed `cohort-pending-coverage-open` ×9 and
`cohort-incomplete` ×121; the healthy run showed 0 and 4.

## 5. Result

Same command, same box, managed Weston, full stage list:

| | main (4 runs) | fixed (3 runs) |
|---|---|---|
| coverage-pass timeout | **2/4** | **0/3** |
| STALL GUARD | **2/4** | **0/3** |
| fft-scroll first tile | 29 376 / 11 462 ms (the two that got there) | **1 428 / 1 247 / 965 ms** |
| fft-scroll tiles | 60/60 | 60/60 |
| level source rank | 4 (`MONTAGE_COMPLETE`) | 5 (`MONTAGE_SAMPLED_FULL`) |

First pixels ~10× earlier, and the phase stops dying. The rank improves because
the sweep samples at REFINED quality where the cohort carries ROUGH_TARGET.

`slow_scroll_converged` still fails in both arms — the paced-scroll target is
missed on this backend either way (D2 in the commit-cost dossier), and total
convergence is unchanged (44.5–48.5 s fixed vs 49.9–58.8 s main, inside this
box's spread).

## 6. One oracle turns red, and it is not a flicker

`window_level_flicker_free` reads False 3/3 with the fix and True 2/2 on main
(same stage list). Exactly one of its five terms fires:
`level_source_count_regressed`. The two terms that actually measure window
movement are perfectly clean in both arms —
`level_transient_span_dip_ratio == 1.0`, `level_center_excursion_fraction ==
0.0`.

The timelines say why. Both arms contain the identical interlude, one
`level_action_id`, source count `60 → 0 → 60`, the 0 row at rank 2
(`PREVIOUS_COMMITTED`), with `display_levels` **byte-identical across all three
rows** (`[0.0, 32373.916015625]`):

- main (base-4): interlude at 3 452 → 6 088 → 6 918 ms; the successor does not
  become visible until 11 462 ms, so the oracle never sees it.
- fixed: interlude at ~1 428 → 10 400 → 10 500 ms; first pixels land at 1 428
  ms, so it falls inside the visible window.

So the fix does not create the transition; it makes the montage visible 10 s
earlier, which brings a pre-existing, level-stable provenance interlude inside
the oracle's scope. `source_count_regressed` counts a `PREVIOUS_COMMITTED`
interlude as a regression even when the published window does not move by one
bit.

Two follow-ups, neither taken here: eliminate the mid-round drop to
`PREVIOUS_COMMITTED` (it lands at ~10.4 s in all three runs, suspiciously
regular), or make the term not fire when the published window is unchanged.
Deciding which is a contract question about what the count is evidence *of*.

## 7. Gate evidence

`pytest` (affected set, 768 tests): 4 red, all adjudicated.

- Three cleared on `--rerun-reds`.
- `test_single_slice_to_full_montage_presents_around_retained_exact_tile[scalar]`
  — an isolated 3/3-fail vs 3/3-pass looked like a regression and was not. An
  order-balanced 8-run ABBA measured **fix 3 pass / 1 fail, main 3 pass / 1
  fail**. The documented load-flake; identical rate.
- `test_rebind_and_evaluation_paths_settle_identical_levels` — 10-run ABBA,
  **fix 2 pass / 3 fail, main 2 pass / 3 fail**. Identical. This is the R3
  presented-bounds clamp latch, pre-existing at `7d50f5b0`.

**A trap worth recording:** `git stash` is shared across all worktrees of this
repo. Using it to swap arms for an A/B popped a *concurrent session's* stash
into this worktree and dropped mine. Both were recovered from dangling objects,
but the lesson stands — swap arms by copying files to the scratchpad, never via
the stash stack.

## Reproduce

```
env -u WAYLAND_DISPLAY tools/env python -m arrayscope.tools.headless_display -- \
  env QT_QPA_PLATFORM=wayland python -m arrayscope.tools.profile_montage_workflow \
  --backend pyqtgraph --scroll-max-tiles 60 \
  --stages load_data,raw_full_tiled_montage,fft_full_tiled_montage,montage_scroll_fft,montage_scroll_scalar \
  --jsonl out.jsonl --trace out-trace.jsonl
```

The trace ring holds 32 768 events and this stage list overruns it — drop to
`--stages load_data,montage_scroll_fft` when the fft scroll's own trace is what
you need. Entry points: `level_evidence_gate` doors, `commit_bail` rows with
`outcome='level-evidence-wait'`, and `scheduling_phase` `coverage_closed` (its
absence is the stall). Raw artifacts stayed in the session scratchpad per the
no-artifacts rule.
