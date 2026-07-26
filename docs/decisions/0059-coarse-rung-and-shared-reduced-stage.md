# ADR 0059: One coarse rung, fed by a shared reduced-input stage

- **Status:** Mechanism accepted and implemented (2026-07-26); unconditional
  product admission retracted after follow-up measurement. The shared
  real-document stage remains the canonical way to produce a reduced rung,
  but whether to schedule that rung is now an empirical utility decision.
  Supersedes the scheduling half of ADR 0050's "reduce-before-ops and
  preview-then-refine" section: the two routes it designed still exist as
  *evaluation* capabilities, but this ADR replaces the shared-transform
  *scheduling* path with the stage cache the native path already uses, and
  collapses FLOOR/PREVIEW into one rung.
- **Number:** 0059 was free at `59592c26`. Parallel worktrees have collided on
  ADR numbers before; renumber on integration if 0059 is taken.

## Context

Three sessions of measurement on the 272-tile FFT montage
(`CenteredFFT(axis=2)` → `FFTShift(axis=2)` → `CenteredIFFT(axis=2)`, display
axes (0, 1), `data/_WIPDelRec-tT2` 336×336×272) converged on a picture that
none of them predicted, and it invalidates the premise of the work queued
behind it.

### The reduced-input preview has not existed since 2026-07-16

`fbbb6f64` ("Migrate live LOD ownership to canonical pages") removed **both**
routes ADR 0050 designed, in one commit, with an empty body:

- it deleted the reduced-input branch of `evaluate_preview_tile`, leaving the
  function an unconditional call to `_evaluate_tile_native_output_preview` —
  the *native-then-reduce* fallback;
- it added `if lod_policy_mode == "resident": return False` to
  `shared_preview_is_useful`, which disables the shared reduced-volume route on
  every montage (`resident` is the montage default).

So `evaluate_at=reduced` has been unreachable everywhere for ten days.
`_evaluate_reduced_preview_volume` is live code behind a shut gate; the
per-tile reduced branch is gone from the tree. Every "the preview should be
cheaper" claim since then has been priced against a capability nothing could
reach.

This corrects [dossier §6a](../redesign/preview-lod-anatomy-2026-07-26.md):
gate 1's per-tile arm cost 65.5 s of worker time because each tile ran a full
**native** evaluation and reduced the output, not because reduced-input
evaluation is expensive per tile. The symptom was reported correctly; the
cause was not.

### The level is a 10× compute lever — and the retention floor already lands past its knee

Measured directly on the dataset (three runs, best of):

| level | reduced input | box-mean reduce | pipeline evaluate | total |
|---:|---|---:|---:|---:|
| 0 (native) | 336×336×272 | — | 573.2 ms | 573.2 ms |
| 1 | 168×168×272 | 230.5 ms | 128.2 ms | 358.7 ms |
| 2 | 84×84×272 | 118.1 ms | 15.9 ms | 134.0 ms |
| 3 | 42×42×272 | 63.9 ms | 3.3 ms | 67.2 ms |
| 4 | 21×21×272 | 35.5 ms | **1.0 ms** | 36.5 ms |
| 5 | 11×11×272 | 15.8 ms | 0.6 ms | 16.3 ms |

Read the table honestly: **level is a large compute lever.** Level 1 costs
358.7 ms and level 4 costs 36.5 ms — 10×, and the reduce column halves per
level (230.5 → 118.1 → 63.9 → 35.5 → 15.8) rather than staying flat. An
earlier draft of this ADR claimed the reduction was level-independent "because
it reads the whole source either way"; that is refuted by its own table. The
reduce cost scales with the *output*, not the input, so a read-bound constant
is exactly what it is not. (Cross-checked on a synthetic `(336,336,272)`
float32 volume: reduce 26.3/17.2/10.6/10.2/7.7 ms for factors 2/4/8/16/32 —
flatter than the numbers above, same 100×+ transform swing.)

What actually removes the compute-side level driver is narrower, and it is a
clamp rather than a physical law. `preview_level_for_tile_shape` ends in
`max(int(min_level), level)`, and `frame_controller` passes
`min_level=PREVIEW_FLOOR_MIN_LEVEL = 4`, so **the retention formula can never
choose a level finer than 4** — and `preview_evaluation_level` only ever
coarsens it further (`max(desired, preview)`). Level 4 is already past the knee
where the transform stops mattering (1.0 ms of a 36.5 ms rung). A compute-aware
level driver would therefore have nothing left to optimise *given today's
floor*.

Stated so a future reader is not misled: **a finer coarse rung is not free. If
`PREVIEW_FLOOR_MIN_LEVEL` ever moves finer, the compute term returns as a ~10×
lever and this decision must be revisited.** The claim below is conditional on
that floor, not on level-independence.

Nor is 35.5 ms a floor on the reduction itself — it is an implementation, not
physics. `render/effects.py:reduce_nd_axis_mean` is pure NumPy and makes **two
separate axis passes** with a dtype conversion, materialising an intermediate;
the accelerated kernel in `display/_numba_pyramid` (njit, `parallel=True`,
`prange`) serves only 2D pyramid pages and returns `None` until its kernels
finish compiling, so the reduced *volume* path never reaches it. That gap is
the likeliest reason these numbers run 3–4× above the synthetic cross-check.

### The serial cost is the fan-out, not the evaluation

Instrumenting `evaluate_shared_preview` in situ on a real run:

| pass | read+reduce | evaluate | **fan-out (272 tiles)** | total |
|---|---:|---:|---:|---:|
| preview, level 4 | 0.8 ms | 137.1 ms | **505.4 ms** | 643.5 ms |
| target, level 2 | 0.7 ms | 45.0 ms | **683.5 ms** | 729.3 ms |

78–94% of each shared task is the per-tile loop that turns one evaluated volume
into 272 display planes. That loop is embarrassingly parallel — every tile
reads the same finished array — and it runs inside a single kernel task
because the shared route is built as one task per pass.

### And the wall clock is dominated by neither

Acknowledgement spans, baseline against the shared route opened
([dossier §6c](../redesign/preview-lod-anatomy-2026-07-26.md) arm B):

| pass | baseline | shared route |
|---|---:|---:|
| FFT preview L4 | — | 3978 → 7148 ms (**3171 ms**) |
| FFT exact L2 | 6387 → 7636 ms (**1249 ms**) | 8570 → 13629 ms (**5059 ms**) |

The shared preview task computes all 272 rows in 643 ms and then takes 3171 ms
to acknowledge them; the shared target regresses the *existing* exact pass 4×
(1249 → 5059 ms) purely by routing it through one serial task instead of 272
parallel rungs. Admission, not compute, is the larger term in both.

## Decision

### 1. One coarse rung, and its level is a retention decision only

Merge FLOOR and PREVIEW into a single rung. ADR 0050 described them as
separate because they answer different questions, and §7 of the dossier
proposed keeping both as parameters:

```
CoarseRung(level, retained: bool, evaluate_at: reduced | native_then_reduce)
```

**The `level` input has exactly one driver: retention.** Keep
`preview_level_for_tile_shape(target_edge=48)` — whole-stack footprint against
spare display budget — which answers its own question well. The proposed
second driver, a compute-derived preview level "adaptive to operation cost per
texel", **is dropped as redundant, not as worthless**: the retention formula's
`min_level=4` clamp already lands past the knee, where the transform is 1.0 ms
of a 36.5 ms rung, so a compute-aware driver could only agree with it. It stops
being redundant the moment that floor moves finer, where level is worth 10×.

This is the simplification the §7 proposal was reaching for. The coarse rung's
real degrees of freedom are **who evaluates it** and **how the result is fanned
out** — not how coarse it is.

Merging also fixes the §2 collapse directly. Today `floor_level` and
`preview_level` are both `session.lod_preview_level`, so the PREVIEW guard
(`preview_level < finest_available()`, counting steps already planned in the
same plan) can never pass once FLOOR is planned: the four-rung ladder runs as
two. One rung cannot shadow itself. The raw path's dominant coarse-rung
refusal (`floor already covers this level`, 2901–3635 tile-plans per run) is
the same collapse seen from the plan side, which is why both halves are decided
here rather than in two worktrees.

### 2. Restore `evaluate_at=reduced` per tile, and let the stage cache share it

Reinstate the branch `fbbb6f64` deleted: when the pipeline supports reduced
display input, the coarse rung evaluates on reduced input instead of
evaluating natively and reducing the output.

The sharing that the shared-transform route was built to provide then comes
free from machinery that already exists — **but only if the rung is keyed on
the region, and this is the detail that sank the deleted code.**

`StageMaterializationManager` keys a stage on `(document_key,
operation_prefix, region, dtype, shape)` and attaches duplicate requests to one
in-flight materialization ("tiles wait for shared stage"). For a montage-axis
transform every tile's reduced read is *identical*:
`CenteredFFT.required_input_region` replaces the montage axis with `ALL`, and
the display-axis regions are common to the montage. Checked directly on an
8-tile montage — all eight tiles return a byte-identical `(4, 4, 8)` reduced
read covering the whole montage axis, while the same check on a raw pipeline
returns a different `(4, 4, 1)` plane per tile. So all 272 tiles do resolve to
one region, one stage, one evaluation, with 271 attaches.

The deleted code would **not** have got that, and restoring it verbatim would
not either. It wrapped each tile's reduced array in a fresh
`ArrayDocument(reduced_base, ...)`, and `stage_document_key` begins with
`id(document.base_data)` — a new array object per tile is a new stage key per
tile, so 272 tiles would have paid 272 × 36.5 ms of reduction (each a full pass
over all 30.7 M source texels) with no sharing at all. **The coarse rung must
be expressed as a subsampled region of the real document, never as a synthetic
document wrapping a pre-reduced array.** That is the binding constraint this
ADR adds, and gate 2 exists to catch its violation.

With it, the rung has the same structure as the native path (`cache_stage=True`
gives one 573 ms native FFT stage plus 272 parallel slices), which is why the
native path is fast: one ~36 ms evaluation plus 272 parallel fan-outs.

### 3. Retire the shared-transform route as a scheduling path

`submit_shared_transform_floor`, `_submit_shared_transform_target`,
`_shared_transform_owns_tile_display_target`, the shared-transform claims, and
the acknowledge-all barrier are a **second implementation** of "evaluate once,
fan out to many" — one that fans out serially inside a single task and
therefore needs a global barrier to stay coherent. With the reduced stage in
place they are duplicate owners of the same truth, which ground rule 2 says to
delete rather than gate.

`evaluate_shared_preview` and `_evaluate_reduced_preview_volume` keep their
value as *evaluation* helpers; what retires is the parallel scheduling path,
its ownership predicates, and the `resident` policy gate that had to exist to
keep the two owners apart.

The shared route's acknowledge-all *presentation* barrier retires with it.
[§6c arm C](../redesign/preview-lod-anatomy-2026-07-26.md) proved that barrier
cannot simply be removed while the old route remains: per-tile targets win the
race and the shared preview presents **0 of 272** tiles while still paying its
full evaluation.

The initial implementation also put a coarse successor's `DESIRED` step on
`DISPLAY_PREVIEW`. The comment explicitly permitted that target to overlap the
remaining coarse fan-out. That made every phase-only experiment incapable of
enforcing the promise: target evaluation consumed preview workers while only
its acknowledgement was delayed. This is why an ACK-only trace could pass
while dogfooding still showed sharp tiles replacing blocky ones mid-pass.

The implemented invariant is stronger: when a FLOOR exists, `DESIRED` is
always `DISPLAY_PREPARATION`, and `COVERAGE` does not close on generic
first-pixel state. It closes only when the current required set owns
backend-acknowledged first-pass payload identities. Thus no target-rung worker
starts before the final preview task finishes or before T1. The profiler tags
kernel work with the scheduling generation so transient automatic-layout
scopes are not mixed with the final required set. This uses the existing
one-way phase owner and bounded presentation cohorts; it does not restore the
retired acknowledge-all scheduler.

### Follow-up: schedule only when T1/T2/B proves delivery utility

The relevant quantities are T1 (every required tile has preview quality), T2
(every required tile has target quality), and B (T2 with FLOOR disabled). A
preview is useful only when T1 is far below B and T2 remains close to B.
Worker totals and `operations/cost.py` cannot answer that delivery question.

With the execution invariant actually enforced, three order-balanced
single-process passes per arm gave:

| backend / pipeline | median T1 | median T2 | median B | verdict |
|---|---:|---:|---:|---|
| WGPU raw | 2139 ms | 5585 ms | 2723 ms | reject: T2 +105%; A missed the 5 s gate 3/3 |
| WGPU FFT | 3041 ms | >5000 ms | 3451 ms | reject: only 410 ms earlier coverage; A missed 3/3 |
| PyQtGraph raw | ≥4710 ms | >5000 ms | ≥5000 ms (one 3861 ms completion) | reject: A reached no T2 in 3/3 |
| PyQtGraph FFT | N/A | >5000 ms | >5000 ms | no reduced RGB format; target path independently broken |

The WGPU FFT rows are single runs, never `--repeat`; process order was balanced
`A, B, A, B, B, A`. PyQtGraph is bistable: the bounded passes report the
incomplete ACK count instead of waiting longer. The complex run can also block
inside the synchronous CPU-composition action, in which case the 8 s process
guard produces no phase record; that is a broken action, not a censored timing.

No current signature qualifies, so the product defaults to target-only.
`--enable-coarse-rung` remains a profiling arm, not a product heuristic.
Future admission requires new order-balanced T1/T2/B evidence for the complete
backend/pipeline/display signature. Unknown signatures skip; no speculative
runtime cost predicate is landed.

## Consequences

Mechanism consequences, each one gated below rather than a universal product
claim:

- The coarse rung's evaluation for the whole montage drops from 272 native
  evaluations to one ~36 ms reduced evaluation plus 272 parallel fan-outs.
- Refinement stays on the per-tile parallel ladder. The original expectation
  that this preserved the exact span is retracted by the follow-up above:
  shared evaluation does not prevent the extra presentation stream from
  delaying complete coverage.
- Three predicates lose their reason to exist
  (`preview_montage_planes_are_independent` and the two shared-transform
  ownership tests), and `shared_preview_is_useful`'s policy gate goes with the
  route it gated.
- Admission remains the dominant wall-clock term (3171 ms to acknowledge 272
  preview rows that took 643 ms to compute). **This ADR does not address it**,
  and no first-pixel claim should be made without measuring it: it is the
  standing "per-commit whole-montage cost" queue item, and the coarse rung
  merely stops being the reason it is not reached.
  [The per-commit dossier](../redesign/per-commit-transaction-count-2026-07-26.md)
  prices it: `apply` is 63–69% of every commit and scales with `presented`
  rather than with the delta, so two commits per fill cost 93.0 and 89.8 ms
  while carrying a *zero* delta at 272 presented. The implementation's
  obligation is therefore narrow but real — **the fan-out must not multiply the
  number of commits.** Parallel fan-out tasks must feed the existing bounded
  commit cohorts, not one commit per task.

- **Non-goal: optimising the reduction.** Even erasing it entirely saves 35.5 ms
  against 643 ms of compute and 3171 ms of admission — 5% of compute, under 1%
  end-to-end. This is the fourth time this investigation has found the
  arithmetic irrelevant beside the scheduling shape, and it is recorded here so
  the numba gap noted above is not reopened as an optimisation.

### Gates

1. Reduced-input evaluation is reachable again, pinned by a test that fails on
   the current tree.
2. On the FFT montage: exactly one stage materialization for the coarse rung
   across all 272 tiles (counter-pinned via the stage manager's
   attach/hit diagnostics), not 272.
3. If the policy schedules a coarse rung, its ACKs carry the operation key at
   `quality="preview"` and, across the required scope,
   `max(coarse ACK) < min(exact ACK)`. Per-tile order is insufficient. In
   addition, no target-rung `kernel_start` may precede the final preview-rung
   `kernel_finish`; this is the clause that catches delayed ACK with concurrent
   target execution.
4. For each admitted signature, full refined stays at or under baseline. Quote
   a **median over at least three order-balanced passes** and say how many: the
   per-commit dossier found the refinement is bistable — one unmodified
   baseline pass took 49 batches over 8.0 s with `fully_visible_ms` 16 015
   against the usual 2 batches over 0.28 s — so a single pass can land on that
   tail and prove nothing. The FFT stage must still come from single runs,
   never `--repeat`, which inflates it 40% through worker contention.
5. `montage_quality_rung_evaluations` prices both alternatives. Raw WGPU is
   expected to skip after the follow-up A/B above; merely removing `floor
   already covers this level` is no longer a success criterion.

### Implementation evidence

Implemented on local `main` at `6ad55232`.

- The reduced-input branch has a direct test, and the 272-tile FFT gate reports
  one stage materialization (`scheduled=1`, `completed=1`) with the other 271
  requests accounted for by stage attach/hit diagnostics. The stored
  `StageValue` is also checked against its subsampled real-document region, so
  a synthetic-document key cannot pass the gate.
- In five WGPU single-run FFT passes, every one of the 272 operation-bearing
  level-4 `quality="preview"` acknowledgements preceded that tile's
  operation-bearing level-2 exact acknowledgement. This was the original
  gate; the product regression above shows why it was too weak.
- Gate 4 used five passes per arm in the order-balanced sequence
  `baseline, branch, branch, baseline, baseline, branch, baseline, branch,
  branch, baseline`. Full-refined medians were **5388.3 ms baseline** and
  **5332.1 ms branch** (−56.2 ms). These are single-run FFT stages; `--repeat`
  was not used.
- The branch's WGPU counter gained
  `{level: 4, representation: complex_rg32f, uploads: 272}`. That is the first
  reduced operation page population; it proves the exact native-plane warm
  declined rather than substituting a full native FFT plane. The physical page
  allocation is still 256×256 per 21×21 logical tile, as expected.
- The same five-pass branch cohort's median complete coarse-floor fill was
  **5321.4 ms**, only 10.7 ms before the full-refined median. This is not a
  first-pixel speedup claim: admission remains dominant, and the original
  per-tile overlap violated the whole-montage ordering promise. The 21×21
  source is visibly blocky when drawn at roughly 57 px, and the now-default
  minification filter correctly does not engage while that source is
  magnified.
- Red-first bounded 32-tile traces on the old successor rule failed the worker
  clause on both maintained backends and both raw/FFT stages: target work began
  before preview work finished even where the ACK clause passed. After
  `DESIRED` left the coverage lane and phase closure used acknowledged
  first-pass identities, the final explicit A-arm check passed both clauses:
  WGPU raw T1/T2 287/548 ms, WGPU FFT 651/999 ms, and PyQtGraph raw
  718/1041 ms (32/32 ACKs in every pass). PyQtGraph FFT reports
  `no-preview-pass`, not a vacuous ordered result, because its CPU-composited
  RGB payload has no reduced-page format; its 32-tile target-only T2 was
  2047 ms.
- Full-grid A-arm traces also exposed a profiler oracle bug: one action can
  publish transient 66/84-tile automatic-layout generations before its final
  272-tile generation. Kernel traces now carry `scheduling_generation`, and
  T1/T2 use each required tile's first qualifying ACK. This prevents an older
  generation's target start from being compared with the final generation's
  preview pass.
- The profiler's five-second cold-fill budget now includes the synchronous
  action itself. A timeout still closes the trace and writes a structured
  incomplete record with requested/presented/payload/ACK counts; it does not
  erase the evidence by raising before JSONL publication.
- The final three-pass WGPU A/B is the T1/T2/B table above. With product
  target-only policy, a subsequent full-grid run completed raw in **3261 ms**
  (`T2=2909 ms`) and FFT in **4205 ms** (`T2=3930 ms`). PyQtGraph raw completed
  in **4151 ms** (`T2=4111 ms`). PyQtGraph FFT did not return from its
  synchronous CPU-composition action before an 8 s process guard and emitted
  only the load row; its growing-set CPU mapping remains a separate queue item.
- A three-pass in-process raw run (`--repeat 3`) reported only
  `tile already has committable coverage` and `allow_preview false` as coarse
  refusal reasons; `floor already covers this level` is gone.

All baseline and branch WGPU passes still failed the pre-existing
`gui_callbacks_below_50ms` performance oracle. That is not reclassified as
green by this ADR.

## Alternatives rejected

- **Keep the shared route and parallelise its fan-out.** Workable — the
  fan-out is 78–94% of the task and is trivially splittable, since batching the
  *fan-out* re-reads nothing (the constraint that batching means duplication
  applies to the *evaluation*, which stays single). Rejected because it leaves
  two owners of "evaluate once, fan out", keeps the global barrier, and
  reproduces by hand what the stage cache does by construction.
- **Adaptive compute-driven preview level** (§7's proposal). Rejected as
  redundant *under today's floor*: `PREVIEW_FLOOR_MIN_LEVEL = 4` means
  retention never picks finer than level 4, where the transform is 1.0 ms of a
  36.5 ms rung, so a compute-aware driver has nothing to add. Not rejected on
  the grounds that level is cheap — level 1 costs 10× level 4, and this bullet
  reverses if the floor ever moves finer.
- **Keep FLOOR and PREVIEW separate and fix the guard.** Rejected: their levels
  come from one value today, and the merged rung needs one level anyway once
  the compute driver is dropped.
- **Split the FLOOR/PREVIEW merge into its own task.** Rejected deliberately:
  the raw path's dominant refusal and the op path's missing preview are the
  same collapse, and designing the rung twice in two worktrees would guarantee
  a semantic merge conflict.
