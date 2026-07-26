# Rung evaluation that is computed and never shown (2026-07-26)

**Status:** measurement plus instrumentation; **no scheduling behaviour
altered.** Opened by the per-(rung, level) counters added for
[preview-lod-anatomy](preview-lod-anatomy-2026-07-26.md) §10, which showed a
272-tile FFT montage spending ~8 s of worker time on a level the session had
already abandoned. §1–§5 establish where that work comes from and that nothing
is presented from it; **§6 closes the follow-up by declining all three candidate
repairs on evidence** — one of them because there is no product defect to
repair — and by showing that the premise, that this work costs anything
user-visible, is not established.

Two separate wastes, with different causes and different verdicts.

| | level-1 waste | level-2 duplication |
|---|---|---|
| size | 7607 ms, 8 evaluations | ~124 extra evaluations, ~4.5 s |
| outcome | every one **discarded** | every one **committed** |
| cause | a real camera change, correctly planned against | duplicate producer, unproven |
| verdict | **do not fix — §6** | do not fix; not understood (§5) |

Numbers are one run of
`profile_montage_workflow --backend wgpu --stages load_data,fft_full_tiled_montage`
on `data/_WIPDelRec-tT2_20260223150234_14.nii` (`(336, 336, 272)`, 272 tiles of
336²), headless Weston, over `1d7e53ca` plus this dossier's instrumentation.
Stage elapsed 6386 ms. **Quote FFT numbers from single runs**: unlike the raw
stage, `--repeat 3` inflates this stage to 7.0–7.2 s and its per-evaluation
means by ~40%, because three cold FFT fills in one process contend for the same
four workers.

## 1. What the counter now says

`montage_quality_rung_evaluations` gained a `discarded` column — evaluations
that ran to completion and whose result could never be committed:

| rung | level | calls | discarded | total | mean | wasted |
|---|---:|---:|---:|---:|---:|---:|
| DESIRED | 1 | 8 | **8** | 7607 ms | 950.9 ms | **7607 ms** |
| DESIRED | 2 | 396 | 0 | 14 202 ms | 35.9 ms | 0 ms |

**Every level-1 evaluation in the run has its payload discarded**, and each is
~1 s. Against a 6.4 s stage on four workers that is the largest single item the
counters have found — but "largest by this counter" is not "on the critical
path", and §6d measures that it is not. The level-2 pass discards nothing yet
runs 396 times for 272 tiles.

### 1a. The counter was wrong first, and said so loudly

The first version counted a discard whenever a rung was released without
committing. That conflates two different things — an evaluation that ran and
was thrown away, versus a **queued** task the kernel superseded before any
worker started it, which spent nothing. It reported `discarded=32` against
`calls=8`, pricing 45.9 s of waste inside a 5.4 s stage.

The absurdity is the only reason it was caught, which is the argument for
keeping counters in units that can be sanity-checked against the wall clock.
The fix is a per-submission flag set by the worker wrapper and **consumed** when
counted, so `discarded <= calls` holds by construction rather than by the
delivery paths happening to line up (a superseded reusable task can be told
twice). Two tests pin the invariant directly.

## 2. Where the level-1 demand comes from

From the plan trace, the FFT session's plans in order:

```
t=2942ms  demand=1  steps={(DESIRED,1): 88}
t=3097ms  demand=2  steps={(DESIRED,2): 272}
t=3146ms  demand=2  steps={(DESIRED,2): 272}
...
```

The session's **first** plan is made against a demand of level 1 and queues 88
cold `DESIRED(level=1)` steps. 155 ms later the demand becomes level 2 and
supersedes all of them. Eight had already been picked up by workers.

So the level-1 demand is the FFT session's own first demand snapshot, taken
before the viewport settles to the level the stage then holds for its whole
life.

**Do not read that as a defect — §6a measures what moves the viewport, and it is
the harness's simulated user fit, not the application.** The demand at this plan
correctly described the camera the app had.

## 3. Nothing is ever presented from it

Splitting acknowledgements by whether the identity carries the operation key:

| quality | level | carries FFT op | acks |
|---|---:|---|---:|
| exact | 1 | no | 120 |
| exact | 2 | no | 272 |
| preview | 4 | no | 60 |

**No acknowledgement at any level carries the FFT operation at level 1.** The
120 level-1 acks belong to the earlier raw session. The 7607 ms is not "work
that arrived too late to help" — it is work whose output never existed as far
as the screen is concerned.

## 4. Why superseded work runs to completion, and why not to "fix" it

The kernel supersedes correctly: all eight report outcome `stale` and their
payloads are refused. What it cannot do is stop them. Cancellation is
cooperative — `EvaluationCancelled` is raised where an evaluation *polls* its
token — and one cold tile of this pipeline is a single indivisible
`CenteredFFT → FFTShift → CenteredIFFT` over the montage axis. There is no
poll point inside it. Measured: `fn_ns` for these tasks is 1015–1027 ms, and
the supersession lands ~155 ms in.

So "cancel harder" is not available. Three repairs are plausible from here;
**§6 tests each and declines all three**, so the list below is the candidate set,
not a recommendation:

- **Do not admit expensive cold DESIRED work on a session's first plan** until
  the demand has settled. Most direct, and it is exactly the kind of admission
  tuning ADR 0046 forbids without an A/B — with a 4 s first-pixel budget in
  play, delaying the first admission can cost more than it saves.
- **Make the first demand snapshot right.** The demand is level 1 for 155 ms and
  level 2 forever after; the bug may be that the first snapshot is taken before
  the viewport fit is applied. That is the LOD/viewport path, not the ladder.
- **Give the pipeline a cancellable FFT.** Chunking the transform to add poll
  points is a real change to operation evaluation and would slow the common case.

The counter is the deliverable here: whoever revisits this has a
sub-0.5 s-resolution work counter (`discarded` x mean) to argue on, which the
4.0–4.9 s wall clock cannot provide. §6 is what that counter then said.

## 5. The level-2 duplication is not superseded work, and is not understood

396 level-2 evaluations for 272 tiles, **all `completed`, none discarded**. On a
second run: 514 evaluations, 272 distinct tiles, 242 of them exactly twice, 30
once. The two evaluations of a tile are ~1.3 s apart (median), carry the **same**
`source_id`, and fall in two clean waves (first finishing 4072–5128 ms, second
5030–6379 ms).

What rules out the obvious explanations:

- **Not supersession.** Every one completes and commits; `discarded` is 0.
- **Not a fit-stretch replan.** There is **no `pipeline_plan` event at all** in
  the 4900–6500 ms window that contains the entire second wave, while 227
  kernel tasks start and finish inside it. Submission without planning is
  normal — `_drain_pending_admissions` releases one plan's steps 24 at a time
  through a kernel continuation — but that accounts for one evaluation per
  tile, not two.
- **Not a source change.** The `source_id` prefix is identical across the pair.

The shape that fits is a duplicate *producer*: acknowledgements lag evaluations
by ~1.2 s (evaluations finish 4072–5128, `exact@2` acks span 5279–7586), and the
second wave falls inside that lag, so a tile is re-admitted while its first
result is still queued for commit. If that is right, `prepare_rung`'s
in-flight/admitted dedupe and the ladder's `ready_satisfies_display_demand`
guard are both failing to see a result that has completed but not yet been
acknowledged. **That is a hypothesis, not a finding**, and the second wave
partly falls *after* the stage's own completion milestone, so some of it is
post-settlement work rather than fill cost. Not fixed, and not worth fixing
until someone instruments the admitted/ready sets across that gap.

Priced at the level-2 mean, ~124 extra evaluations is ~4.5 s of worker time on
this run, so it is the same order as §1's waste and deserves its own pass.

## 6. Verdict on the three repairs: none of them, and the premise is weaker than it looked

The follow-up asked for the A/B this dossier made possible, with the bar "the
~8 s of discarded work measurably shrinks *without* first pixel or refinement
regressing". Measured answer: **no candidate meets that bar, and the bar itself
rests on an unestablished assumption.**

### 6a. "Fix the pre-settle demand snapshot" — there is no product defect to fix

This looked strongest, including to me. It is not available, and the reason is
worth stating loudly rather than quietly.

The `lod_demand` trace shows the transition on **both** stages identically —
`level 1 @ 3.65 texels/px` then `level 2 @ 5.94` about 135 ms later — so it is
not FFT-specific. A probe that suppresses the harness's `_pulse_fit_stretch`
entirely settles the attribution:

| | fit pulse on | fit pulse suppressed |
|---|---|---|
| demand transitions | L1 → **L2** at +135 ms | **L1, and never leaves it** |
| stage completes | yes, 272/272 | **no record — never settles** |

So the camera change is the **harness's simulated user fit**, and the
application never re-fits a montage on its own here: without the pulse the
camera stays where the entry left it. `AUTO_UNTOUCHED` does own an
`_auto_square_fit`, which is why this was worth checking, but it is not what
moves the camera on this path.

That means the demand at the first plan was **not stale or premature — it was
correct**. It described the camera the app actually had, and 135 ms later a
(simulated) user action changed that camera. There is no wrong snapshot to
repair; there is a camera change, which is a legitimate event arriving at an
inconvenient time. **Do not "fix" this — it would be optimising the fixture.**

What survives is the *mechanism*, which is product-side and user-reachable: any
user who fits, zooms, or scrolls within ~1 s of entering a montage on an
expensive pipeline discards whatever cold uncancellable work is in flight. The
fixture is emulating a real user action, not inventing one. But the fixture's
particular 135 ms is a fixture number.

### 6b. "Delay first-plan admission" — refuted by mechanism, with a counter

Not merely "worse properties": on this pipeline it degenerates into pausing the
whole fill, and the system already demonstrates that every run.

`_defer_native_quality_during_interaction` defers `DESIRED` steps with
`level <= 0` **or** `reduce_from_native=True`. On this pipeline
`reduced_input_available` is False (§10f of the
[preview-LOD dossier](preview-lod-anatomy-2026-07-26.md)), so the ladder sets
`reduce_from_native=True` on **every** DESIRED step — the whole fill is inside
the deferred class, not just the expensive level-1 tail.

The newly surfaced `PipelineCounters` row proves it rather than arguing it:
**`interactive_native_deferred = 3808`** in one run, and the plan trace shows the
guard doing exactly what it says while the (harness) fit gesture is live:

```
t=2876  interactive=False  demand=1  submitted=24   steps={(DESIRED,1): 88}
t=3058  interactive=True   demand=2  submitted=0    steps={(DESIRED,2): 272}
 ... eight consecutive plans, interactive=True, submitted=0 ...
t=3306  interactive=False  demand=2  submitted=24   steps={(DESIRED,2): 272}
```

So "treat montage entry as unsettled" would arm this guard for the entry window
and stop the *whole* fill, which is the first-pixel cost the investigation
exists to avoid. Refuted.

### 6c. "Make the FFT cancellable" — still the only general repair, still out of scope

Unchanged from §4: it would help every long op, and it touches the operation
contract. It is also the only candidate that attacks the real invariant — that a
~1 s indivisible transform cannot honour a cancellation token. Worth scoping;
not startable from admission and demand timing, which is where this round was
scoped to stay.

### 6d. The premise: worker time is not the constraint here

The bar assumed shrinking the discarded work is worth something. That is not
established, and one measurement points the other way.

A configuration that eliminated the level-1 wave completely — zero discards, and
**9.7 s less total evaluation work** (15.5 s → 5.9 s, with the level-2 mean
*improving* 29.2 → 21.6 ms and `retained_stage_decision` going `attached` →
`hit`) — finished **1.35 s slower** (elapsed 3849 → 5099 median, refined 3445 →
5067).

That configuration is confounded: it ran the fit pulse twice, so it both removed
the pre-fit wave and added an interaction. I could not separate the two, because
in this fixture the pre-fit plan and the camera delivery are **the same event** —
`_pulse_fit_stretch` is also how the harness delivers the settled viewport to the
async render, so moving it before the render leaves the stage unable to converge
at all (no record, as in §6a's probe). Reordering is not available.

The confound is itself the finding, though: **one extra interaction cost more
than 9.7 s of saved worker work.** On this stage the constraint is serialization
on the presentation path, not worker throughput — the same conclusion the
op-preview retraction reached from the other direction ("blocked by
serialization rather than by a gate"), and the same shape as the raw montage's
transaction-count result.

**Consequence for the counter.** `discarded` says a payload never reached the
screen. It does **not** say the evaluation was worthless, and it does not say
removing it would make anything faster. The docstring now states that with this
measurement attached, because "7.6 s of waste in a 3.9 s stage" is exactly the
kind of number that gets acted on without being checked — and I nearly did.

## 7. What was changed

Observation only:

- `RungEvaluationTimings.record_discarded` + a `discarded` column on
  `tile_lod_rung_evaluations` / JSONL `montage_quality_rung_evaluations`,
  bounded by `calls` by construction (§1a).
- `LodLadder.coarse_rung_refusal` and cumulative
  `tile_lod_coarse_rung_gates` — see
  [preview-lod-anatomy](preview-lod-anatomy-2026-07-26.md) §10f, which is what
  established that the FFT gate is `no reduced input and no retained floor`.
- `pipeline_plan` trace events gained `policy_floor_level`,
  `policy_preview_level`, `policy_reduced_input`, `demand_level` and
  `coarse_rung_refusals`, which is how §2's demand transition became readable.

## Reproduce

```
ln -sfn /home/thomas/projects/ArrayScope/data data
env -u WAYLAND_DISPLAY python -m arrayscope.tools.headless_display -- \
  env QT_QPA_PLATFORM=wayland python -m arrayscope.tools.profile_montage_workflow \
  --backend wgpu --stages load_data,fft_full_tiled_montage \
  --jsonl fft.jsonl --trace fft-trace.jsonl
```

Waste: `montage_quality_rung_evaluations`, columns `calls` / `discarded` /
`total_ms`; price a discard as `discarded * total_ms / calls`. Demand history:
`kind == "pipeline_plan"`, fields `demand_level` and `steps`. Discard proof:
`kind == "kernel_finish"` with `outcome == "stale"`, `rung`, `level`, `fn_ns`.
Duplication: group `kernel_finish` by tile for `(rung=2, level=2)` and count.

Single runs for this stage; `--repeat` inflates it (see the header).
