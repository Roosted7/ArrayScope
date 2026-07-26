# ADR 0059: One coarse rung, fed by a shared reduced-input stage

- **Status:** Proposed (2026-07-26). Supersedes the scheduling half of ADR
  0050's "reduce-before-ops and preview-then-refine" section: the two routes it
  designed still exist as *evaluation* capabilities, but this ADR replaces the
  shared-transform *scheduling* path with the stage cache the native path
  already uses, and collapses FLOOR/PREVIEW into one rung.
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

The global barrier retires with it, and this is the part worth stating
carefully, because [§6c arm C](../redesign/preview-lod-anatomy-2026-07-26.md)
proved a naive removal fails: with the shared preview and the per-tile target
running concurrently, the per-tile rungs win the race and the preview presents
**0 of 272** tiles while still paying its full evaluation. The barrier is what
made the shared route coherent. It is not needed here because the ladder
already enforces the same ordering *per tile* — coarse rung before DESIRED for
each tile independently — which is strictly weaker than a global barrier and
therefore does not serialize. Ordering was never the reason for the barrier;
having a fan-out that could not be interleaved was.

## Consequences

Expected, and each one is a gate below rather than a claim:

- The coarse rung's evaluation for the whole montage drops from 272 native
  evaluations to one ~36 ms reduced evaluation plus 272 parallel fan-outs.
- Refinement stays on the per-tile parallel ladder, so the 1249 ms exact span
  is preserved and ground rule 3's 5 s limit is not at risk from this change.
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
3. Coarse-rung acks carry the operation key at `quality="preview"`, and every
   tile's coarse ack precedes its exact ack.
4. Full refined stays at or under baseline (~5.2 s). Quote a **median over at
   least three order-balanced passes** and say how many: the per-commit dossier
   found the refinement is bistable — one unmodified baseline pass took 49
   batches over 8.0 s with `fully_visible_ms` 16 015 against the usual 2
   batches over 0.28 s — so a single pass can land on that tail and prove
   nothing. The FFT stage must still come from single runs, never `--repeat`,
   which inflates it 40% through worker contention.
5. `montage_quality_rung_evaluations` shows the coarse rung's cost, and the
   ladder gate counters no longer report `floor already covers this level` as
   the dominant refusal on the raw stage.

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
