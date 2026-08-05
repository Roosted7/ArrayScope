# The planner's own convergence gate — and what the `submitted=0` stream actually is

2026-08-05. Branch `claude/ui-performance-smoothness-aade14`. Landed as
`5dc44099`; the latent test fix it depended on is `1271900e`.

## 1. The starting claim, and how much of it survived

The brief was: `montage_retarget` runs 1596 / 2059 times per field session, 92-93 %
of those passes report `submitted=0`, they cost 12.7-16.5 % of wall clock, and the
existing viewport convergence latch (`_try_update_montage_viewport_only`) is the
correct containment — it simply does not guard the replan gate, which was
estimated at "at least 62 % of the calls".

Two of those are confirmed, one is corrected, and one is **refuted**.

**Confirmed — the waste is real and the size is right.** Re-derived from the two
`.traces/` diagnostics files by de-duplicating the governor's 64-observation ring
across every snapshot:

| trace | span | `montage_retarget` | `submitted=0` | total ms | zero ms | zero p50 | passes > 16.68 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| `20260805-121508` (thread) | 150 s | 1596 | 1474 (92.4 %) | 21531 | 19030 | 9.12 | 316 |
| `20260805-123056` (process) | 162 s | 2059 | 1913 (92.9 %) | 29901 | 26703 | 8.96 | 422 |

**Confirmed — the viewport latch is the wrong door.** Every call to
`retarget_frame_pipeline` was attributed to its caller by wrapping the bound class
attribute (zero product change) and running the 272-tile wgpu workflow
(`raw_full_tiled_montage` + `montage_scroll_scalar` + `montage_zoompan_scalar`).
558 recorded calls, 516 of them `submitted=0`, 3108 ms of waste:

| caller | calls | `submitted=0` | ms | zero ms |
|---|---:|---:|---:|---:|
| `request_montage_replan`'s `fire()` | 207 | 183 | 1820 | 1406 |
| `_maybe_retarget_frame_session` | 83 | **83** | 457 | 457 |
| `_try_update_montage_viewport_only` (post-fan-in) | 33 | 32 | 420 | 401 |
| `frame_effects.retarget_on_gui` | 95 | 92 | 399 | 387 |
| `_try_update_montage_viewport_only` (main) | 90 | 79 | 518 | 318 |
| `replan_deferred_interactive_native_quality` | 7 | 6 | 87 | 85 |
| `_try_update_montage_viewport_only` (interactive fan-in) | 37 | 37 | 38 | 38 |
| `_retarget_montage_resize_payloads` | 6 | 4 | 71 | 15 |

**Corrected:** the replan gate is 37 % of the calls and 45 % of the wasted
milliseconds, not 62 %. The *non-viewport* callers together are 70 % / 76 %,
which is the number the brief was reaching for. `_maybe_retarget_frame_session`
has never once submitted — 355 consecutive zero passes across four runs — and is
kept only because its comment names an obligation this workload does not exercise.

**Refuted:** *"the existing containment predicate is correct, it just needs to
guard more doors"*. It is correct, and it does need to guard more doors, but it
cannot reach most of the stream. Counting the gate's own refusal reasons over 575
passes:

| refusal | passes |
|---|---:|
| `arm:epoch_moved` (the session was written since the latch) | 419 |
| **`arm:camera_pending` (a tile the camera reaches genuinely still needs work)** | **409** |
| `obligation:flush_pending` | 253 |
| `obligation:dirty_payloads` | 256 |
| `guard:render_round_id` | 151 |

**71 % of the `submitted=0` passes are not convergent.** They are mid-fill passes
that walk all 272 tiles to rediscover that everything admissible is *already in
flight*. No extension of a convergence predicate can suppress them, because
convergence is precisely what they are not. The ceiling for this family of fix is
therefore around 25-30 % of the passes, not 93 %.

## 2. Where a pass spends its 5.8 ms

Same workflow, 561 calls, 3270 ms total, 2625 ms in passes that submitted nothing:

| | ms | share |
|---|---:|---:|
| `pipeline.retarget` | 1933 | 59 % |
| — `effects.tile_states` | 1246 | 38 % |
| — — `render_effects.tile_lod_states` | 843 | |
| — — `_seed_resident_crop_rebinds` | 389 | |
| — `_drain_pending_admissions` | 419 | |
| — `ladder.plan` | 128 | |
| — `kernel.rerank_unstarted` | 54 | |
| `mark_ladder_swaps_for_current_demand` | 908 | 28 % |
| `_lod_admission_scope` | 80 | |
| `selected_lod_factor` | 79 | |
| everything else | ~190 | |

Two things follow. The brief's "four O(montage) passes inside `pipeline.retarget`
before it discovers it has nothing to submit" is **directionally right but wrongly
attributed**: the diagnostic loops it names (`last_plan_states`, the coarse-rung
refusal scan, the rerank dict) are ~70 ms of 1933 together. One call —
`effects.tile_states` — is 61 % of the planner and 38 % of the whole pass.

And `mark_ladder_swaps_for_current_demand` is 28 % and runs **before** the
planner, so a gate placed inside `pipeline.retarget` cannot reach it. That is why
the containment went to the top of `retarget_frame_pipeline` rather than into
`pipeline.py`.

## 3. What landed

`FrameControllerMixin.montage_planner_pass_is_idle`, consulted first thing in
`retarget_frame_pipeline`. Same two arms as the viewport latch —

* **epoch arm**: same session, same mutation epoch, same invariant key, same
  non-session inputs ⇒ identical inputs to a pass that already returned zero;
* **camera arm**: every tile the camera reaches is drawn at the demanded level
  with nothing outstanding, and the camera terms the tiles cannot answer for
  (chiefly `preview_floor`) are unchanged. This is the arm that can serve the
  replan gate at all, because that caller fires *because* the session was
  written, so its epoch has always moved.

— minus the publication half (`_published_scope_covers_camera`), which is not
owed here: the planner is not the publisher. `retarget_viewport` is, and it runs
in the caller above this gate.

Three things it adds that the viewport latch does not need:

* **`interactive` in the guard.** The only planner input that is not session
  state. The ladder defers native rungs on it and the gesture ENDING writes
  nothing to the session, so without this term the gate would suppress exactly
  the pass `replan_deferred_interactive_native_quality` exists to run.
* **`FramePipeline.has_pending_admissions`.** Chunked admission can leave planned
  steps queued in the pipeline that appear in no session-owned set —
  `loading_tiles` has never heard of them. One attribute read; the gate refuses
  while it is non-empty.
* **the tail runs anyway.** The gate suppresses the *planner*, not the pass.
  Deferred-stage completion, stage-dependent re-arm, the cached-level wakeup, the
  display flush, the shedding release, the session-completion edge and the
  watchdog all still run, at ~0.2 ms.

That last point is the correction that mattered most. The first draft **vetoed**
the gate on those obligations and suppressed 48 of 559 passes; `flush_pending`
alone was the first refusal in 198 of 575. Counting the guard's branch outcomes
before trusting it is what found that. Omitting `rearm_ready_stage_dependents`
from the tail was a genuine lost wakeup and announced itself as a 16-tile fill
that never settled.

## 4. Measured effect on the planner

Same workflow, same instrument, before and after:

| | passes | `submitted=0` | planner ms | zero ms |
|---|---:|---:|---:|---:|
| before | 558 | 516 (92.5 %) | 3811 | 3108 |
| after | 580 | 638 → gate suppressed **103 (17.8 %)** | 2792 | 2277 |

## 5. Adjudication — NOT adjudicated on frame rate

21 runs, 7 per arm, order-balanced (`on off null` on even repeats, reversed on
odd), one fresh process per run, all three arms **in the same worktree, the same
venv and the same commit**. The arms differ only in whether
`montage_planner_pass_is_idle` is allowed to answer, so the worktree-level bias
channel that cost the stage-2 A/B its calibration does not exist here. `null` is
byte-identical to `off`.

Gate behaviour is exactly as designed: **668 of 3959 planner passes suppressed
(16.9 %)** in the `on` arm, 0 of 3900 and 0 of 3905 in the two control arms.

Paired per-repeat deltas against `off`:

| phase | metric | off mean | CoV | **on Δ** | **null Δ (control)** |
|---|---|---:|---:|---:|---:|
| fill | draws/s | 8.52 | 6.9 % | **−4.0 %** | **−3.9 %** |
| fill | tile draws/s | 6.04 | 3.3 % | −3.5 % | +0.8 % |
| scroll | draws/s | 18.83 | 12.5 % | **+4.7 %** | **−4.9 %** |
| scroll | tile draws/s | 18.34 | 10.3 % | +2.2 % | −3.1 % |
| zoom/pan | draws/s | 10.85 | 5.0 % | **−1.0 %** | **−1.0 %** |
| zoom/pan | tile draws/s | 12.71 | 8.5 % | +1.6 % | +1.4 % |

**On every axis the treatment arm moves by the same amount as the same-code
control.** The fill and zoom/pan rows are indistinguishable to the first
decimal; the scroll row's +4.7 % sits inside a control that swung −4.9 % on the
same metric. Event-loop gap p95/max show the same picture (fill +2.1 % on
against −7.4 % null; scroll −0.0 % / +0.1 %).

**Verdict: the change is adjudicated by COUNT and refused by CLOCK.** It removes
16.9 % of planner passes, which is what it claims and what
`tests/ui/test_planner_retarget_suppression.py` pins. It does not move
`presentation_draw_count` per second, and the arithmetic says why: 16.9 % of
passes × 5.8 ms ≈ 600 ms over a ~30 s run ≈ **2 % of wall clock**, against a
per-run spread of 5-12 %. Two percent is not measurable with this harness at any
run count that is reachable in a session.

Caveat on the fill rows: 6 of 21 runs produced no complete
`raw_full_tiled_montage` record, so those two rows rest on n=3-4 pairs. The
scroll and zoom/pan rows are n=7.

**Do not read this as "the gate is worthless".** Read it as: this containment is
worth about 2 % of wall clock, the remaining 71 % of the defect is a different
problem (§6), and any future claim about it must be adjudicated the same way.

## 6. What is still owed

The 71 % that the gate cannot reach is the larger half of the defect and it is a
different problem: a montage mid-fill re-derives the whole tile-state vector on
every completion to conclude that every admissible step is already in flight. The
two costs are `render_effects.tile_lod_states` (843 ms) and
`mark_ladder_swaps_for_current_demand` (908 ms), both O(required tiles) by
construction and both outside this change's file ownership. A memo on the session
mutation epoch would be unconditionally sound for the second, but the epoch holds
only 27 % of the time at that point, so it is worth ~245 ms of 908 — not decisive
on its own.
