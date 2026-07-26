# Rung evaluation that is computed and never shown (2026-07-26)

**Status:** measurement plus one instrumentation change; **no scheduling
behaviour altered.** Opened by the per-(rung, level) counters added for
[preview-lod-anatomy](preview-lod-anatomy-2026-07-26.md) §10, which showed a
272-tile FFT montage spending ~8 s of worker time on a level the session had
already abandoned. This dossier is the follow-up that asks where that work
comes from, whether anything is ever presented from it, and what is safe to fix.

Two separate wastes, with different causes and different verdicts.

| | level-1 waste | level-2 duplication |
|---|---|---|
| size | 7607 ms, 8 evaluations | ~124 extra evaluations, ~4.5 s |
| outcome | every one **discarded** | every one **committed** |
| cause | pre-settle demand snapshot | duplicate producer, unproven |
| verdict | do not fix blind (§4) | do not fix; not understood (§5) |

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

**Every level-1 evaluation in the run is wasted**, and each is ~1 s. Against a
6.4 s stage on four workers that is the largest single item the counters have
found. The level-2 pass wastes nothing by this measure but runs 396 times for
272 tiles.

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

So the level-1 demand is not a stray or a fit-stretch artefact: it is the FFT
session's own first demand snapshot, taken before the viewport settles to the
level the stage then holds for its whole life.

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

So "cancel harder" is not available, and three plausible repairs are all
policy changes this dossier deliberately does not make:

- **Do not admit expensive cold DESIRED work on a session's first plan** until
  the demand has settled. Most direct, and it is exactly the kind of admission
  tuning ADR 0046 forbids without an A/B — with a 4 s first-pixel budget in
  play, delaying the first admission can cost more than it saves.
- **Make the first demand snapshot right.** The demand is level 1 for 155 ms and
  level 2 forever after; the bug may be that the first snapshot is taken before
  the viewport fit is applied. That is the LOD/viewport path, not the ladder.
- **Give the pipeline a cancellable FFT.** Chunking the transform to add poll
  points is a real change to operation evaluation and would slow the common case.

The counter is the deliverable here. Whoever takes the fix now has a
sub-0.5 s-resolution work counter (`discarded` x mean) to prove it on, which
the 4.0–4.9 s wall clock cannot provide.

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

## 6. What was changed

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
