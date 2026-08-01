# The bistable fill — which state a refinement lands in, and what selects it (2026-07-31)

**Status:** root-caused and fixed at the governor (three closed feedback doors,
§4); one adjacent silent-stall defect fixed at the pipeline (§5); one
independent stall named, instrumented, and left for its own session (§6).
Follow-up to [per-commit-transaction-count-2026-07-26](per-commit-transaction-count-2026-07-26.md)
§7/§8.3, which made supply starvation nameable and moved this hunt onto the
retarget/lifecycle side — the right direction, wrong floor: the pacing lives in
`ResourceGovernor.decide_render_pass`, the R5 chunk optimizer.

**Environment honesty up front:** every run here shared the machine with at
least one other session's benchmark suite (load average 18 on 16 cores at the
worst; the harness banner reported up to ~29/16 cores from elsewhere). All
conclusions rest on structural counters — commit counts, cohort sizes,
admission verdicts, per-fit model parameters from `governor_details` — not on
wall-clock, and the A/B is order-balanced across that load. Wall-clock numbers
are quoted only to size effects, never to adjudicate them.

## 1. The symptom, re-measured on today's main

The July-26 fast state (a refinement in 2 free-remap commits / 0.28 s) **no
longer exists on main**. Since the R5 render-pass governor took ownership of
commit chunking, the 272-tile refinement always proceeds in governed cohorts —
and the instability moved into the governor's own cost model. Across 21
unmodified passes (probes 0–8), the same 272-tile refinement landed anywhere
between:

- **healthy-adaptive:** ~34 target commits of 8–13 items (`fixed≈33 ms,
  item≈0.7 ms` fitted), full refinement inside the stage window;
- **dribble:** 61–67 commits of 3–5 items (`steering marginal 4.3–4.8 ms/item`
  fitted, on the same workload), refinement incomplete at window end
  (48–168 of 272);
- **batch-1 death:** 12–16 commits of 1–2 items after one poisoned two-point
  fit (`item=3.5 ms` from an 18-item/100 ms outlier pair), 48–53 tiles total;
- **stranded:** zero refinement submissions at all for a whole pass (§5);
- **coverage-stall:** refine never opens because level evidence never lands
  (§6).

The delta between the best and worst *governed* states is the difference
between re-paying the ~25–45 ms whole-montage commit cost 34 times and 90+
times, plus a 64–88 ms median inter-commit gap (p90 up to ~700 ms under load)
multiplied by that count. That is the user-visible "fill rate is unstable,
with drops in throughput".

## 2. What was refuted along the way

- **Worker arrival pacing (again).** In the dribble state all 272 exact
  evaluations were finished by t≈3 s while commits carried 2–5 items each for
  the rest of the window (probe8 run2: 272 finished, 48 committed). The
  supply sat in `pending_payload_upserts`; the cohort cap was the governor's.
- **The commit caps (again).** `admission_limit` read `bytes` against the
  *governor-shrunken* byte cap (`byte_cap = batch × bytes/item`), not against
  the idle 32 MB policy cap. The admission layer faithfully enforced whatever
  cohort the R5 decision asked for; nothing in `TileAdmissionQueue` selects
  the state.
- **"The 60-tile phase refines, so the machinery works."** True and
  irrelevant: the initial zoomed view (60 required tiles) refines fully at
  level 1 in every pass; the full-montage phase (272 required, level 2) is
  where every pathology lives, because its whole-montage fixed commit cost is
  4.5× larger and the cost model is re-learned per round.
- **`item-independent=1` is not used.** The governor already detected that
  commit cost is fixed-dominated (`fixed ≥ 0.75×mean`, and the
  shrink-count-same-elapsed probe) and printed it in `governor_details` —
  and the decision ignored it. The exculpatory evidence existed on every
  slow commit.

## 3. The state selector: three self-confirming doors in one adaptive fit

All three live in `arrayscope/core/resource_governor.py`; each lets a wrong
marginal survive its own falsification, which is exactly the "adaptive
controller with contaminated feedback" hypothesis the brief named.

**Door 1 — the byte-prior echo.** One weakly-independent preview fit
(item/byte design independence 0.185, six samples) split its marginal as
`item=0.05, bytes=4.79 ms/MiB`. The pooled shared-terms median handed that to
every pass under the same structural key as a *prior*; the prior **overwrote**
each later local fit's byte term; the overwritten fits were retained and voted
the same value back into the pool. Result: `bytes=4.14 ms/MiB` frozen for the
rest of the session — even after 24 MB commits measured at *half* what that
term alone predicts (30 items/24 MB predicted ≈121 ms, measured 61 ms). At the
target pass's 1 MiB/item, that is a 4.1 ms/item phantom marginal; under the
R5 latency quadratic (>45 ms priced hard), a 4–5 ms marginal caps the cohort
at 3–5 items. The preview's split was fitted at 0.13–0.25 MiB/item and
transferred to a 1 MiB/item regime it had never seen.

**Door 2 — the forgetful exploration ceiling.** The optimizer prices any
cohort beyond ~1.1× the *observed* maximum with an exponential extrapolation
cost, and "observed" was the 8-sample rolling window (reset per round, per
representation). One early 32-item commit was forgotten eight callbacks later;
after that, 3-item cohorts kept the ceiling at ~3.3, and the ceiling kept the
cohorts at 3. The ratchet's base was being erased faster than the ratchet
could climb.

**Door 3 — the frozen reference slope.** A "stable" reference model promoted
from the previous round anchors intrinsic cost against machine load — but its
item slope is frozen for the whole round, and the promotion bar accepted fits
whose slope was measured across a 1–4 item span. One such reference steered
`item=4.30 ms` for 70 consecutive commits (fix3 run1) while every live sample
sat at 2–5 items, a span at which a 4.3 ms slope and a 1.0 ms slope are
indistinguishable under 5–10 ms of noise. Small cohorts can never falsify a
too-large slope; the too-large slope forbids the cohorts that would.

**Why it is bistable and not merely noisy:** whichever fit forms in the first
~10 commits of a round determines the cohort size; the cohort size determines
what the next observations can identify; small cohorts produce fits that
keep cohorts small, large ones keep them large. Two attractors, selected by
the first seconds of supply arrival and fit noise — 21 recorded passes landed
across both plus the stall states, with no code difference.

## 4. The fix (landed, `65c86f75`)

1. **Pool hygiene:** `_RenderPassCostModel.byte_identified` marks byte terms a
   fit separated from its *own* design; only those vote in
   `_render_pass_shared_terms`. An inherited prior can no longer echo.
2. **Round high-water + evidence-gated exploration:** the round keeps a
   high-water mark of every cohort size it has measured
   (`_render_pass_round_item_max`), and when the already-computed
   item-independence evidence is present, the outward exploration step is 4×
   per decision instead of 1.1× — still never a one-shot jump to an unmeasured
   optimum (the existing two-point-model test pins that), but an escape in
   ~3 commits instead of never. Watched working in the field: a batch-1 lock
   released to an 18-item cohort in one decision (fix3 run0).
3. **Reference promotion span bar:** a slope may only anchor a round if
   measured across a real span (`observed_item_max ≥ 8` and `≥ 4×
   observed_item_min`). Narrow fits still steer their own round from live
   evidence; they just cannot be frozen into the next.

Regression tests (red-first verified against the unfixed build, which
initially exposed that my first test scenario *couldn't* fail — rewritten
until it did):

- `test_fixed_dominated_pass_escapes_a_small_cohort_seed_quickly` — a tight
  fixed-dominated fit over 2–4-item cohorts must reach an amortizing cohort
  (post-fix it reaches 91 of 272 in one decision; the bar is ≥12 and
  no-shrink-back).
- `test_shared_byte_terms_pool_only_locally_identified_byte_fits` — an
  inherited byte value must not vote (pre-fix the pool median was the echo).

### A/B, order-balanced, 12 processes (B F F B B F F B + B F F B) × 3 in-process runs

Base = main `26f952d1`, fix = this branch; 15 runs per arm; run under a
machine load of 20–28 on 16 cores (two other sessions ran suites and their
own benchmarks concurrently — the mirrored schedule is what makes the
comparison survive that). Two processes (one per arm) died at the profile
fixture geometry guard before any 272-phase and are excluded; the
pre-existing coverage/level-evidence stall (§6) took 6 base and 4 fix runs
in both arms alike.

Refinement completeness — exact level-2 tiles committed inside the stage
window, full per-run distribution (not a median):

| arm | per-run exact-2 committed (sorted) | complete (≥272) |
|---|---|---:|
| base | 0×10, 38, 77, 136, 138 — never completes | **0 / 15** |
| fix  | 0×8, 63, 74, 82, 275, 275, 276, 276 | **4 / 15** |

The mechanism metrics — properties of the governor's *decision*, robust to
load in a way wall-clock is not (pooled over every 272-phase target commit):

| metric | base | fix |
|---|---|---|
| decided cohort (`max_upserts`) p10/med/p90/max | 2 / 4 / 23 / 63 | 2 / 8 / 40 / 248 |
| steering marginal ms/item p10/med/p90/max | 0.0 / **3.40** / 5.1 / **21.6** | 0.0 / **1.44** / 5.3 / **11.3** |
| commits steered in the poisoned regime (≥3 ms/item) | 58/89 (65%) | 68/184 (37%) |
| delivered items per target commit, median of per-run medians | 4.4 | 6.4 |

Head-to-head inside the same load window (round 2, adjacent processes):
base p9 refined 136/138/0; fix p10–p11 refined 276/275/0 and 275/74/0.
Transient high-marginal fits still occur in the fix arm — small-span local
fits under 5–10 ms noise — but they now self-correct (max cohort reached
248) instead of locking; the base arm's frozen 21.6 ms/item reading is the
attractor this change removes. A quiet-machine rerun of this table is
worth one process per arm when the machine is actually quiet; the
mechanism metrics are not expected to move against the fix.

## 5. The stranded state: replanning against a dead kernel (fixed, `ddebef39`)

Separately from the governor, whole passes recorded **zero refinement
submissions** with claims taken and no error anywhere: `prepare_rung`
admitted the step (claiming the rung in the lifecycle), and `kernel.submit`
returned `None` at the door because the kernel was **shutting down** — a
session cached across window teardown kept replanning on its coalesced timer
forever. One traced process recorded 1,901 shutdown-refused submissions
(including whole 272-step refine plans) competing for the GUI thread with the
live window's fill. The silent `handle is None` drop cost three full profile
passes to even see.

Fix: `Kernel.is_shutting_down` + `FramePipeline.retarget` refuses to plan
against a closing kernel (no claims taken, no ladder work, no churn), with
`kernel_submit_refused_shutdown` / `submit_step_no_handle` traces naming the
residual race. Red-first test:
`test_retarget_against_closed_kernel_takes_no_claims`.

## 6. Named and left: the level-evidence coverage stall (FIXED 2026-08-01)

**Closed by [level-evidence-coverage-deadlock-2026-08-01](level-evidence-coverage-deadlock-2026-08-01.md).**
It was a closed cycle, not a slow producer: the preview cohort's claim parked
the fallback semantic sweep on `coverage_open`, coverage closes on first
pixels, and first pixels waited on the evidence the sweep would have produced.
The `commit_bail` signature below was the right entry point. The original text
follows unchanged.

Intermittently (3/3 repeats of one process under heavy load, and identically
on unmodified main), the 272-phase never opens refine at all: all 272 preview
evaluations finish, ~11 preview commits land, then every commit bails with
`outcome='level-evidence-wait'`, `pending_upserts` up to 179, `wakeup=
'level-scan'` — and the level scan never completes inside the window. No
first pixels, R4 red, 0/272. This is upstream of everything in §3–§5 and is
not touched by this change; it is spawned as its own task with the
`commit_bail` signature as the entry point. The `rung_prepare` /
`ui_exception` traces added here (`f45fca27`) are the instruments that
separated it from the governor and kernel states.

## 7. Instruments left behind

- `rung_prepare` trace: every prepare verdict with reason, session, renderer —
  a plan with zero submissions now names which of the nine refusal doors each
  step died at (a converged plan and a stranded plan previously read
  identically).
- `kernel_submit_refused_shutdown` / `submit_step_no_handle`: the silent
  drops, named.
- `ui_exception` trace: exceptions swallowed by completion-handler hooks land
  in the same trace file as their consequences.
- Governor regression tests above — the first two fail on the slow attractor's
  mechanics, not on wall-clock.

## Reproduce

Same harness as the July-26 file:

```
env -u WAYLAND_DISPLAY tools/env python -m arrayscope.tools.headless_display -- \
  env QT_QPA_PLATFORM=wayland tools/env python -m arrayscope.tools.profile_montage_workflow \
  --backend wgpu --stages load_data,raw_full_tiled_montage --repeat 3 \
  --jsonl out.jsonl --trace out-trace.jsonl
```

State classification from the trace: `commit_batch/backend_complete` rows with
`required_tile_count==272`, `pass_kind=="target"`; the fitted model per commit
is in `governor_details` (`candidate-fixed/-item/-bytes`, `steering-marginal`,
`item-independent`). Exact-2 completeness = rows of `delta_qualities` with
quality `exact`, level 2. Raw JSONL stayed in the session scratchpad per the
no-artifacts rule.
