# Preview LOD — what the rung actually costs (2026-07-26)

**Status:** measurement only, no code changed. Answers the standing
`TODO(redesign R3)` in [`render/ladder.py`](../../arrayscope/render/ladder.py)
("re-derive `preview_level` bounds from fresh A/B evidence before changing
defaults") with the evidence it asks for.

Two findings, in increasing order of importance. On a **raw** wgpu montage the
preview rung buys nothing and a lower or adaptive preview *level* is measured
to be the wrong lever (§1–§5): the cost is commit transactions, not data. On a
montage with **expensive operations** the preview rung is worth an order of
magnitude — and the system currently produces none at all (§6). §7 proposes the
FLOOR/PREVIEW merge that follows from both, §8 answers the upload questions,
§9 records that the levels/histogram merge into the pyramid already exists.
§10 replaces this dossier's two inferences with counters (both held) and adds
the per-rung evaluation cost: a 16× input reduction buys only ~2.5× on raw
data, and on the FFT stage — now that its stall is fixed — the counters find
8 s of FFT evaluation computed and discarded per run, with the shared
reduced-input preview refused 23 times.

Field report that opened this: *"loading the NIfTI with a full montage over
the third dimension takes many seconds. It often runs two quality levels, but
the first is already very high and therefore slow, and the quality upgrade is
barely even visible. The preview should have been a lower LOD."*

Every number below is one run of
`profile_montage_workflow --backend wgpu --stages load_data,raw_full_tiled_montage`
on `data/_WIPDelRec-tT2_20260223150234_14.nii` (`(336, 336, 272)`, 272 tiles
of 336², desired level 2 at 5.94 source texels/pixel), headless Weston, plus
two field diagnostics JSONLs from the same dataset.

## 1. The preview level is not fixed, and it is not high

`frame_controller` derives it per session from the tile shape
(`preview_level_for_tile_shape(tile_shape, target_edge=48, min_level=PREVIEW_FLOOR_MIN_LEVEL=4, max_level=6)`)
and `render.lod.selected_lod_factor` then coarsens it per demand to
`max(base, desired_level)`. On this dataset that is **level 4 — factor 16,
21×21 texels per 336² tile, 1.7 KiB per tile, 0.48 MB for the whole
272-tile montage.** The demanded level 2 is 84×84 per tile, 7.7 MB total.

So the first rung is already 16× coarser than the target. The premise that it
is "already very high" does not hold; the premise that it is fixed does not
hold either.

## 2. FLOOR and PREVIEW are the same rung

`frame_runtime._frame_pipeline_for_session` builds the ladder policy with

```python
floor_level=max(1, int(getattr(session, "lod_preview_level", 0) or 0)),
preview_level=max(1, int(getattr(session, "lod_preview_level", 0) or 0)),
```

Both come from the same value, so once the FLOOR step is planned at level 4
the ladder's PREVIEW guard (`preview_level < finest_available()`, and
`finest_available()` counts steps already planned in this same plan) can never
pass. The four-rung ladder documented in `ladder.py` runs as **two** rungs.
The trace agrees — every acknowledgement in the run is one of:

| quality | level | acks |
|---|---:|---:|
| preview | 4 | 332 |
| exact | 2 | 272 |
| exact | 1 | 60 |

and every planned pre-native step in `pipeline_plan` is `rung=0` (FLOOR) at
level 4. `pipeline_plan` never emits a `rung=1` step.

Corollary: `tile_lod_preview_reduced_scheduled` / `_blocked` / `_failures` were
**dead diagnostics**. `diagnostics_snapshot.py` read
`window.renderer._montage_preview_reduced_*`, which was assigned nowhere in the
tree, so all three reported 0 forever — exactly the three counters one would
reach for to ask "did the reduced-input preview path run?".

**Now wired** (see §10) to `submit_shared_transform_floor`, plus a
`tile_lod_preview_reduced_last_gate` string so a 0 says *why*. On this raw
montage they read `0/0/0` with gate `per-tile rungs own reduced input` — the
per-tile ladder rungs are the owner, and §10's rung timings measure them. Two
siblings in the same family are **still dead** and were left alone:
`_montage_quality_ingest_reductions` and
`_montage_quality_stage_hits_serving_derivations` are assigned nowhere either.

## 3. The preview rung is four times slower than the target it precedes

Acknowledgement timeline, relative to the first kernel submit:

| pass | data per tile | 272 tiles acked over |
|---|---:|---:|
| preview, level 4 | 1.7 KiB | 2.22 s → 6.14 s (**3.92 s**) |
| exact, level 2 | 28 KiB | 7.68 s → 8.63 s (**0.95 s**) |

**16× more data settled in a quarter of the time.** The stage's own milestones
say the same thing more bluntly: `first_preview_floor_fill_ms = 5300.97` and
`fully_visible_ms = 5300.97` — whole-montage preview coverage and whole-montage
refined coverage land in the same millisecond. The preview buys no earliness
at all here. (`first_visible_tile_ms = 871.8`, so *some* pixels are early; the
*fill* is not.)

## 4. On wgpu the coarse commit uploads the native plane — but that is not the cost

From the commit-batch trace, split by pass:

```
2228.3 ms  upserts= 32  uploads=128  bytes=33 554 432  {(preview,4): 32}
5693.6 ms  upserts= 32  uploads=128  bytes=33 554 432  {(preview,4): 32}
...
7703.7 ms  upserts= 91  uploads=  0  bytes=         0  {(exact,2): 91}
8649.5 ms  upserts=181  uploads=  0  bytes=         0  {(exact,2): 181}   elapsed=178.1 ms
```

**1 MiB of texture upload per tile during the preview pass, zero during the
refinement.** 1 MiB is exactly four full 256×256 float32 pages — the *native*
page grid of a 336² plane (`plan_source_grid_pages` yields 4 pages at
reduction (0,0), and 1 page at reduction (2,2) or (4,4)). Session totals
confirm the scale: `wgpu_uploads_total = 1088` (= 272 × 4) and
`wgpu_active_resident_bytes = 285 212 672` — **285 MB resident to display 7.7 MB.**

(§10a instruments this directly: the per-level upload counter reports **one
row**, `(level 0, scalar_r32f)` = 1088 uploads / 285 212 672 bytes, and no
level-2 or level-4 row at all.)

This is deliberate and documented.
`wgpu_imageview2d._wgpu_reusable_native_texture` says so:

> A zoomed-out full montage commonly presents LOD1/2 while the payload still
> owns its exact semantic plane. Uploading that plane through the same bounded
> tile commit populates the already-budgeted L0 pages once; a later
> displayed-axis crop then becomes only a source-window rebind.

and `frame_effects.wgpu_payload_upload_nbytes` states the substitution
outright: the warm path "uses the exact native pages **instead of also**
uploading the redundant reduced/cropped payload".

Two independent multipliers stack on top of it:

- **Every page write is a whole page.** `_coerce_payload` requires exactly
  `(PAGE, PAGE)` = (256, 256) and `write_texture` writes extent
  `(PAGE, PAGE, 1)`. A level-4 page carries `stored_rect_yx=(0,21,0,21)` —
  441 valid texels in a 65 536-texel write, **0.7% payload efficiency**. Level
  2 carries 84×84 → 10.8%.
- **The draw is a point sample.** The raw display path is `textureLoad`, one
  texel per fragment, no filtering (only the BC codec path has a sampler, and
  it is pinned `nearest` on purpose to match `textureLoad`). At 5.94 source
  texels per screen pixel a native-page draw shows 1 of every ~35 texels.

The point sampling explains the other half of the field report directly: when
the preview draw and the refined draw are both point samples of the same
resident native texels at different rates, *of course* "the quality upgrade is
barely even visible".

### 4b. Provenance: this is `e266260`, deliberately

`git log -S _wgpu_reusable_native_texture` returns exactly one commit:
**`e266260` (2026-07-23) `fix(wgpu): reuse source pages across cropped montage
scrolls`**, extended twice on 07-25 (`6c9a602`, `6843a04`) for born-cropped
views. The zoomed-out full montage was not collateral — the docstring names it
as the target case, and the trade is stated: pay the native upload once in the
coarse commit so a later displayed-axis crop is only a source-window rebind.
The same commit added the bounded-cohort guard (`batch_limit = min(2, ...)`,
`byte_cap = 3 MiB`) whose over-broad qualification became the 15.4 s cold fill
fixed by `f450660`
([dossier](montage-cold-fill-cohort-2026-07-25.md)). Its own acceptance gate —
zero WGPU uploads on the X/Y crop stages — passes; the cost landed on a stage
that gate does not watch.

### 4c. A/B: removing the warm makes everything worse — the bytes are not the cost

Same tip, same command, `_wgpu_reusable_native_texture` short-circuited to
`None` behind a temporary env flag (reverted; the crop stages' zero-upload gate
would fail with it off, which is why this is a probe and not a proposal):

| | warm ON (default) | warm OFF |
|---|---:|---:|
| uploads / bytes | 1088 / 285 MB | 564 / 158 MB |
| resident pages | 1088 | 272 |
| **preview ack span** | **4.21 s** | **4.24 s** |
| exact-2 ack span | 0.95 s | 2.78 s |
| `fully_visible_ms` | 5301 | 8068 |
| stage elapsed | 5328 | 8239 |

**The preview pass is invariant: 4.21 s vs 4.24 s across a 127 MB upload
swing.** The upload traffic §4 measures is real, and it is not what makes the
preview rung slow — an earlier draft of this dossier attributed it that way and
was wrong. Meanwhile the warm's benefit is large and confirmed beyond its own
crop-scroll case: without it the refinement costs 2.78 s instead of 0.95 s and
the stage costs **2.9 s more**. `e266260`'s trade is paying off here too.

What the same A/B *does* isolate is transaction count:

| window | warm ON | warm OFF |
|---|---|---|
| preview | 18 batches / 392 upserts / 3.06 s | 21 / 392 / 2.63 s |
| refine | **2** batches / 272 upserts / **0.33 s** | 15 / 272 / 1.77 s |

Mean cost per commit transaction is **120–170 ms almost regardless of how many
tiles it carries** — two batches moved 272 tiles in 326 ms, while eighteen
moved 392 in 3062 ms. Uploads add ~45 ms per batch (170 vs 125), second-order.
The refinement gets 2 transactions because everything it needs is already
resident; the preview gets 18–21 because tiles trickle in from workers under
byte/item cohort caps. **That is the whole 3 s.**

## 5. What this rules out, and what it points at

**Ruled out twice over — lowering or adapting the preview *level* on raw
data.** At level 4
the payload is already 0.7% of the transfer, *and* §4c shows the preview pass
does not move when the transfer changes by 127 MB. Level 5 or 6 divides
1.7 KiB of an invariant, and only makes the first frame blurrier. Device- or
dataset-adaptive level selection tunes a term that is negligible twice.
Do not spend the R3 A/B on it in this shape.

**Also ruled out: touching the native-plane warm.** It is a −2.9 s net win on
this stage for +127 MB. `e266260` was right.

**Where the cost actually lives**, in the order the evidence supports. Items 1
and 2 are the two separate prizes: §6's is the larger one and is about work
never done, §5.2's is about work done too many times.

1. **Give the preview rung back to expensive pipelines** (§6). An FFT montage
   currently produces *no* preview at all and shows its first pixel at 6.7 s.
   Two gates: make `pipeline_commutes_for_display_lod` axis-aware (an FFT along
   a non-display axis commutes *exactly*), and re-examine
   `shared_preview_is_useful`'s blanket `resident` refusal. This is the only
   item on the list that changes an order of magnitude rather than a constant.
   **Gate 1 is now built and measured (§6a): necessary, but it moves nothing on
   its own, and routing the preview per-tile instead of through the shared
   route costs 18.7× the worker time. Gate 2 is where the prize is.**
2. **Cut the number of commit transactions the preview pass needs** (18–21 → 2,
   worth ~2.7 s here). This is the queue's standing "per-commit whole-montage
   cost" item, and §4c hands it the lower-variance repro it was waiting for: the
   preview ack span and per-window batch count are direct work counters, stable
   across a 127 MB upload swing, so a fix shows up on them without fighting the
   4.0–4.9 s wall-clock spread. First question to answer: why does an
   incremental worker-fed pass take 18 transactions where a
   fully-resident pass takes 2 — cohort byte/item caps, or arrival pacing?
3. **Pack coarse tiles into shared pages** — 21×21 tiles tile a 256² page
   12×12 = 144 per page, so the whole 272-tile preview montage is 2 pages
   instead of 272. Still the strongest structural idea, but for the
   **transaction-count** reason, not the byte reason: few pages means few
   upserts. It is the expensive option: page identity (`content_key`,
   `SourceGridPageIdentity`) and the executor's `_plane_grids` /
   `_flat_indices` are per-plane, so a montage-level page is a new identity
   family, not a parameter change.
4. **Sub-page writes** (write `stored_rect`, not the whole page) — demoted to
   opportunistic. It would cut preview bytes ~148× and level-2 bytes ~9×, but
   §4c prices the whole byte term at ~45 ms/batch, so it is worth doing for
   memory (285 MB resident to display 7.7 MB) rather than for latency. If
   attempted, verify the row-pitch rule: `queue.write_texture` is not supposed
   to carry `copyBufferToTexture`'s 256-byte `bytes_per_row` alignment, but
   wgpu-native may stage through a buffer; if it does, pad rows to 64 texels.
5. **Preview blur / proper minification (independent, cheap, do it anyway).**
   `textureLoad` point sampling is why the preview aliases *and* why the
   refinement is invisible. A preview-only multi-tap read is O(1) per fragment
   and, with the native plane already resident, costs no bandwidth — it
   improves preview quality *and* signals "still working" as the field report
   asks. This is the one item that addresses the felt complaint without
   touching the fill at all. Precedent and machinery exist: the Stage A shader
   flags live in spare `Mapping` uniform words with a CPU mirror in
   `display/shader_mapping.py`, and shipped default-off to avoid a forbidden
   display-oracle rebaseline. Same discipline applies.
6. **Adaptive preview level — only in the §7 form.** Adaptive to *operation
   cost*, as the compute half of a merged coarse rung; and for the retention
   half, "the coarsest level whose whole visible set fits one transaction".
   Neither driver is device speed or screen-fill rate.

**Backend note.** §4 is wgpu-specific: on PyQtGraph a coarse payload really is
a smaller raster, and preview-first keeps its plain advantage there. §4c's
transaction-count finding is not backend-specific — the per-commit cost is
shared-path work. Nothing below argues for weakening the preview rung; §6
finds the regime where it is worth far more than anyone was claiming.

## 6. Ops change the answer completely — and there the preview is switched off

Everything above is the **raw** montage, where the "operation" is a copy. The
preview's second and much stronger justification is compute: an operation
evaluated on 16×-reduced input is ~16× cheaper, and that saving lands *before*
any commit or upload exists to be counted.

Measured on `fft_full_tiled_montage` (`CenteredFFT(axis=2)` → `FFTShift(axis=2)`
→ `CenteredIFFT(axis=2)`, display axes (0, 1)), splitting every acknowledgement
by whether its identity carries the operation key:

| quality | level | pipeline | acks |
|---|---:|---|---:|
| exact | 2 | **FFT** | **271** |
| exact | 1 | raw | 120 |
| preview | 4 | raw | 60 |

**Zero previews on the FFT pipeline.** Every preview in the run belongs to a
raw tile from an earlier phase. Planned steps for the FFT fill are 5944 ×
`(rung=2, level=2)` — DESIRED only. First FFT pixel lands at **6664 ms**, and
the stage did not finish: it hit the 4 s stall guard at 271/272 tiles, twice,
identically. **Triaged and fixed 2026-07-26** — a leaked page-pool layer, not
an LOD or preview defect; see
[wgpu-pool-layer-leak-2026-07-26.md](wgpu-pool-layer-leak-2026-07-26.md). The
stage now completes 272/272, so the §6 numbers below (which were taken from
the truncated runs) should be re-measured on the fixed tip. **§10e re-measures
them:** "zero previews" survives unchanged, the first FFT pixel moves to
2564 ms, and the rung counters find 8 s of FFT evaluation discarded per run
that a truncated stage could not have shown.

Two independent gates each suffice to cause it:

1. **`pipeline_commutes_for_display_lod` is axis-blind.** It asks each op's
   `capabilities(shape, dtype).lod_commuting`, a per-op constant that
   `CenteredFFT.capabilities` never sets — *regardless of which axis the FFT
   runs on*. So `reduced_input_available` is False and the ladder's PREVIEW
   rung is off (`ladder.py:172`).
2. **The shared transform-preview path is disabled for every montage.**
   `shared_preview_is_useful` (`effects.py:976`) opens with
   `if str(getattr(session, "lod_policy_mode", "")) == "resident": return False`,
   and `resident` is the montage default. That path is the one ADR 0050
   designed for exactly this case ("preview-then-refine for expensive
   commuting pipelines"), and `_evaluate_reduced_preview_volume` implements it.
   §10e measures the refusal: **23 blocked calls per run, last gate
   `resident lod policy mode`** — this gate is not merely reachable, it fires
   on every retarget of the stage.

Gate 1 is worth spelling out, because the case is stronger than "accept a
quality compromise". This pipeline's FFT axis is the **montage** axis, not a
display axis. Box-mean reduction along axes 0 and 1 commutes *exactly* with a
linear transform along axis 2 — reduce-then-FFT and FFT-then-reduce are the
same array, to float error. The system is paying 16× for an identical result
because the commuting question is asked of the operation instead of the
operation *and* the display axes.

The machinery to ask it properly already exists:
`pipeline_windowable_display_axes` in the same module reasons per-axis over
`blocking_axes` / `expands_request_axes` / op kind for the window-shift fast
path. An axis-aware `pipeline_commutes_for_display_lod(..., display_axes=...)`
modelled on it would classify `CenteredFFT(axis=2)` with display axes (0, 1)
as exactly commuting. (Note `pipeline_supports_reduced_display_lod` already
takes a `display_axes` parameter and never reads it.)

And where the FFT *is* on a display axis, the field intuition is right and the
ADR already licensed it: reduced-input FFT is a legitimate *preview*, not a
result. It stays `quality="preview"`, semantic consumers keep refusing it, and
exact inspection keeps coming from native — that contract is already enforced.

### 6a. Gate 1, built and measured: necessary, and not sufficient on its own

Gate 1 is closed. `pipeline_commutes_for_display_lod` now takes `display_axes`
and answers per-axis: a stage passes on `lod_commuting` (the axis-blind
pointwise licence) or on a new `OperationCapabilities.real_linear` declaration
when its declared axes are disjoint from the display axes. `CenteredFFT(axis=2)`
→ `FFTShift(axis=2)` → `CenteredIFFT(axis=2)` under display axes (0, 1) is now
classified exactly commuting, and a unit test pins that reduce-then-evaluate and
evaluate-then-reduce agree to float32 round-off through the production box mean.
(`pipeline_supports_reduced_display_lod`'s unread `display_axes` was removed
rather than wired up: its permissiveness is axis-independent *by design*, and
keeping the argument left two apparent owners of the axis question.)

Turning that answer loose on `frame_runtime`'s `reduced_input_available` — the
literal reading of gate 1 — was then measured on the 272-tile FFT montage
(wgpu, `fft_full_tiled_montage`, acks split by whether the identity carries the
operation key; two runs per arm):

| arm | FFT acks | first FFT pixel | fill ends | tile worker time |
|---|---|---|---|---|
| baseline | 271 × `exact` L2 | 5477 / 9747 ms | 8.3 / 11.8 s | 3.51 / 4.79 s |
| naive gate 1 | **271 × `preview` L4, zero exact** | 6427 ms | **84.8 s** | **65.53 s** |
| shipped | 271 × `exact` L2 | 7206 / 8727 ms | 10.7 / 10.6 s | 3.98 / 3.97 s |

**The naive arm produces the previews and is much worse.** The 18.7× worker
time is the ADR 0050 warning arriving on schedule: a per-tile rung for a
montage-axis transform re-runs the whole transform once per tile, so first
pixels land *later* than the old exact ones and the exact rung never lands at
all. Previews appearing was never the goal; previews appearing *cheaply* was.

So the per-tile gate is now asked the question it always claimed to ask.
`frame_runtime`'s comment already said "pipelines whose display-LOD result is
independently tileable"; the code asked whether the pipeline commutes, and
until the predicate became axis-aware the two happened to agree. They are now
separate predicates in `render/effects.py`:

- `preview_pipeline_is_tile_local` — feeds `reduced_input_available`. True when
  no stage expands any request axis, so one tile's cost does not scale with the
  volume. Reproduces the pre-change answer on every existing pipeline.
- `preview_montage_planes_are_independent` — feeds the shared route's
  montage-axis narrowing. Reducing the display axes is legal for a
  montage-axis FFT while reading only the requested planes is not; conflating
  them returned the spectrum of 3 planes where 272 were meant (caught by
  `tests/render/test_effects.py`).

Measured verdict: **the gate that matters for this montage is gate 2**, the
`resident` refusal at `effects.py:976`. The shared route computes the reduced
transform once and fans out planes, which is the only shape in which a
reduced-input FFT preview is cheap. Gate 1 was the prerequisite — without an
axis-aware commuting answer the shared route has no licence to reduce first —
but on its own it moves nothing, which the shipped arm confirms (every metric
inside baseline run-to-run spread).

Two findings in passing:

- The 4 s stall guard at 271/272 fires in **all five runs**, baseline included,
  so it is not attributable to anything here. Its character differs by arm:
  baseline and shipped stall with `target_unsettled=1` (one tile short of its
  exact target), the naive arm with `target_unsettled=272` and
  `fallback_shown=271` (every tile stuck on a preview). Still untriaged.
- Routing FFT pipelines per-tile crashed `summarize_chunk`: a near-constant
  preview plane bounded `(3.8782985, 3.8782990)` spreads 1.2e-7 of relative
  width over 64 bins, and consecutive float32 edges collapse.
  `_histogram_edge_bounds` guarded `high == low` but not "narrower than float32
  can resolve". Fixed and pinned separately; the defect is latent on main.

### 6b. How to subsample, and what a crop does to it

The crop bookkeeping the field intuition worries about ("crop to the centre 40,
so reduce to 10") is already implemented in
`effects.read_reduced_preview_base_and_state`:
`_display_axis_region_for_preview` takes the crop region first,
`_sample_preview_axis_region` multiplies a `SLICE` region's step by the factor
(`0:end:1` → `0:end:4`) or keeps every factor-th entry of an `INDICES` region,
and `_reduced_axis_length` is `ceil(len(region) / factor)`. Crop-then-reduce
composes correctly today.

What is *not* settled is which of the two reduction modes an op preview should
use, and for FFT it matters:

- `sample_display_axes=True` — **decimation** (`0:4:end`). Cheapest I/O, reads
  1/16 of the rows. Aliases: high spatial frequencies fold into the preview
  spectrum, so an FFT preview shows structure that is not in the data.
- `sample_display_axes=False` — read the region, then
  `reduce_array_display_axes` **box mean**. Full I/O, but box mean is a
  low-pass before decimation, so the preview spectrum is the true
  low-frequency content.

For an FFT preview the box mean is the right default and the decimation is the
I/O-bound fallback — a lazy 4 GiB source may prefer aliasing over reading
everything. That choice deserves to be explicit and recorded, not implied by a
default argument.

## 7. Why FLOOR and PREVIEW were ever separate, and how to merge them

ADR 0050's "retained preview level" section is the origin, and it describes two
different things that both happen to make coarse pixels:

- **FLOOR is a retention decision.** "A fixed coarse preview level per dataset
  (chosen from data size and the memory budget…) is materialized speculatively
  for the whole montage/stack — not just the viewport — … and is exempt from
  normal eviction while the dataset is open", so that "dimension scrolling and
  viewport jumps present instantly". Its level is driven by **stack footprint
  vs spare memory** — which is exactly what `preview_level_for_tile_shape`'s
  `target_edge=48` encodes, and it is the right formula for that question.
- **PREVIEW is a compute decision.** "Preview-then-refine for expensive
  commuting pipelines presents the preview-input result tagged
  `quality="preview"` while the exact native pipeline computes." Its level
  should be driven by **operation cost vs the first-pixel budget**.

They are orthogonal, so merging them into one rung is right — but only if the
merged rung keeps both as *parameters*, not if it keeps one *level*. Collapsing
them onto a single level is precisely what `frame_runtime` does today
(`floor_level = preview_level = session.lod_preview_level`, §2), and it broke
both halves at once: the retention level (4) is imposed on the compute preview,
and the compute preview's guard can then never fire.

The merged shape that keeps both strengths:

```
CoarseRung(level, retained: bool, evaluate_at: reduced | native_then_reduce)
```

with two independent level inputs feeding one rung:

- `retention_level` — coarsest level whose whole-stack footprint fits the spare
  display budget. Keep today's formula; it answers its question well.
- `preview_level` — coarsest level at which the pipeline's estimated cost drops
  under the first-pixel budget. Degenerates to "no preview rung" for a raw
  pipeline (nothing to save, §5) and to something aggressive for an FFT stack.

**This is the honest form of "adaptive preview quality."** Not adaptive to
device speed or dataset size or screen-fill rate — adaptive to **operation cost
per texel**, in the one regime where the preview has something to buy.
`operations/cost.py` already has `estimate_pipeline_cost`, but it models bytes
and peak memory, not time; a time term (or a measured per-op throughput,
already tracked per lane) is the missing input and is a bounded addition.

## 8. Answers to the upload questions

- **Are uploads slow?** Directly, no: ~45 ms per 33.5 MB batch (~745 MB/s),
  second-order against a 120–170 ms transaction. Indirectly, yes: bytes buy
  transactions. The byte cap splits a byte-heavy pass into more commits, and
  each commit costs whole-montage work. That is the whole explanation of why
  removing 127 MB of uploads (§4c) made the run *slower* — the refinement went
  from 2 transactions to 15.
- **"If we upload whole planes at full resolution anyway, is there a point in
  showing lower quality first?"** On wgpu with a raw pipeline: measured no
  (§3, §4c) — preview coverage and refined coverage land in the same
  millisecond. On PyQtGraph: yes, plainly. With expensive ops on any backend:
  yes, and it is currently missing (§6). The point of the preview was never the
  upload; it is the *work before* the upload.
- **"Upload a low-quality version first, then the high one, then drop the
  low."** This is what the §4c probe does, and it measured **+2.9 s**. Dropping
  the low page is not free either — it is another page-table edit and another
  transaction. Keeping the native pages and varying only the sampling rate is
  strictly better on wgpu, which is what `e266260` already does.
- **"A reduced upload conflicts with cropped indexing — sub-pixel shifts in the
  shader?"** It does conflict, and with the native warm the conflict does not
  arise: a crop is a source-window rebind on canonical pages, which is exactly
  what `e266260` bought and what its zero-upload crop gate certifies. A reduced
  upload would reintroduce the misalignment that the sub-pixel shift would then
  have to correct. Do not spend a shader shift (image-domain interpolation or
  an FFT-domain phase ramp) to buy back a property the current design already
  has — spend the shader budget on §5.5 instead, where a multi-tap read
  improves every coarse draw.

## 9. The levels/histogram merge is already built

The proposal to fold level/histogram sampling into the LOD pyramid — "our
sampled levels are kind of a very low LOD of the data" — describes the
existing design:

- `MaterializedLodPage.__post_init__` computes a
  `ChunkHistogramSummary` for **every** materialized page, weighted by each
  bin's source extent so coarse levels aggregate correctly
  (`pyramid.py:238-251`).
- `gpu/chunk_summary.py` aggregates them over a non-overlapping frontier
  (`chunk_summary_frontier` requires rank-2 keys, `aggregate_chunk_summaries`
  rebins), and `representative_sample_from_histogram` turns an aggregate back
  into a sample.
- Level evidence is ranked by pyramid provenance, not sampled separately:
  `LevelEvidenceQuality.ROUGH_PREVIEW < ROUGH_TARGET < REFINED`, resolved in
  `level_stats._rendered_level_evidence_quality_for_session`, which promotes a
  preview-quality tile to `ROUGH_TARGET` when its LOD level already meets the
  demand.
- The 2026-07-23 histogram-evidence work already routed normal wgpu frames
  through materializer summaries and reserved resident GPU histograms for
  resident-only content
  ([dossier](histogram-evidence-pipeline-2026-07-23.md)).

And there are no cycles to reclaim: `last_level_stats_ms = 12.3`,
`last_levels_histogram_ms = 0.32`, and the `histogram_refinement` lane ran
924 admitted / 924 completed / **0 blocked by quota** in the field session.
This run's first visible levels came from evidence quality 1
(`ROUGH_PREVIEW`) at rank 3 over 92 sources — i.e. the coarse-pyramid-as-
histogram path is what already publishes the first window.

The one real gap on this axis is *earliness*, not cost:
`first_nondefault_levels_ms = 5299.4`, which is the fill-completion time
again. Auto-levels are late because the preview pass is slow, so they are
gated behind §5.1–5.3 and should not be attacked separately.

## Reproduce

```
ln -sfn /home/thomas/projects/ArrayScope/data data
env -u WAYLAND_DISPLAY python -m arrayscope.tools.headless_display -- \
  env QT_QPA_PLATFORM=wayland python -m arrayscope.tools.profile_montage_workflow \
  --backend wgpu --stages load_data,raw_full_tiled_montage \
  --jsonl base.jsonl --trace base-trace.jsonl
```

Ack timeline: group `kind == "backend_ack"` by `(quality, level)` over
`ts_ns`. Transaction and upload attribution: `kind == "commit_batch"`,
`phase == "backend_complete"`, fields `delta_qualities` / `delta_upserts` /
`uploads` / `upload_bytes` / `elapsed_ms`. Run-to-run spread on this stage is
4.0–4.9 s (±10%; measured 3879–4976 over five repeats in §10c), so nothing
under ~0.5 s can be shown end-to-end here — the per-pass ack spans and the
per-window batch counts are the direct work counters to move.

Since §10 the same questions read straight off the JSONL, no trace parsing:
`wgpu_uploads_by_level` (uploads and bytes per page key) and
`montage_quality_rung_evaluations` (calls, total_ms, max_ms per rung and
level). Add `--repeat N` for the spread. Per-task provenance rides the trace's
`kernel_start` / `kernel_finish` as `rung` / `level` / `fn_ns`.

**A/B discipline this dossier learned the hard way** (§10d): alternate which
side runs first. The second run of any back-to-back pair is ~700 ms slower on
this machine, so two same-order pairs manufactured a phantom 1 s regression
that vanished under six order-balanced pairs.

The §6 FFT numbers come from
`--stages load_data,fft_full_tiled_montage`; split acknowledgements by whether
`identity` contains the operation name to separate op-pipeline tiles from raw
ones, and read planned rungs from `kind == "pipeline_plan"`, field `steps`
(`[tile, rung, level]`). That stage hit the 4 s stall guard at 271/272 on both
runs; that stall is now fixed
([dossier](wgpu-pool-layer-leak-2026-07-26.md)), so the §6 measurement can be
redone against a stage that actually completes.

The §4c probe was a temporary `return None` at the top of
`_wgpu_reusable_native_texture` behind an env flag, reverted after the run.
Both configurations are n=1; the 0.7% preview-span difference is inside noise
(which is the point — it does not move), while the 2.9 s stage difference and
the 2-vs-15 refinement batch count are far outside it.

## 10. The two inferences are now counters (2026-07-26, later the same day)

Everything above was derived from trace events. Two of its steps had to be
inferred because the counter did not exist. Both now do, and both inferences
held.

### 10a. Upload attribution by page key — §4's inference confirmed

`wgpu_executor` splits `uploads_total` / `texture_upload_bytes_total` by
`(DataChunkKey.lod.level, representation)`; the rows reach
`MontageRuntimeDiagnostics.wgpu_uploads_by_level` and JSONL
`wgpu_uploads_by_level`. On this stage, on every one of five in-process
repeats:

| level | representation | uploads | bytes |
|---:|---|---:|---:|
| 0 | `scalar_r32f` | 1088 | 285 212 672 |

**One row. Every upload is a native page.** No level-2 row and no level-4 row
exist — the reduced and preview payloads are never uploaded at all, they are
substituted by the native plane exactly as
`_wgpu_reusable_native_texture` documents. §4 inferred this from `272 × 4`;
it is now read off a counter, and §5.4's memory case (285 MB resident to
display 7.7 MB) is now measured rather than deduced.

### 10b. Per-(rung, level) evaluation cost — and the 16x claim does not hold on raw data

`FramePipeline` wraps each rung's worker function and accumulates calls, total
and max wall time per `(rung, level)` into `RungEvaluationTimings`, surfaced as
`tile_lod_rung_evaluations` / JSONL `montage_quality_rung_evaluations`. The
trace carries the same provenance per task: `kernel_start` and `kernel_finish`
gained `rung` / `level` (−1 for non-ladder work) and `kernel_finish` gained
`fn_ns`, the function-body duration — no new event kinds.

Five in-process repeats of the raw stage (`--repeat 5`, §10c):

| rung | level | calls | total ms | per call |
|---|---:|---:|---:|---:|
| FLOOR | 4 | 272–373 | 2329–3139 | 8.0–8.7 ms |
| DESIRED | 2 | 272 | 5944–11298 | 21.9–41.5 ms |

Two rungs, never a `PREVIEW` row — §2's FLOOR/PREVIEW collapse, confirmed from
the evaluation side rather than from acknowledgements.

**FLOOR/DESIRED total-time ratio: median 0.40, range 0.24–0.50.** A 16× input
reduction buys a **~2.5×** cheaper evaluation, not 16×. That is the expected
shape for a raw pipeline — §5 already ruled out the preview level on raw data
for two other reasons, and this adds a third: the saving the preview rung is
supposed to bank is not there to bank. §10e takes the same counter to the
*operation* pipeline §6 is about, where the answer is different again.

Absolute times here are wall time per evaluation under whatever load the
machine has, not CPU time — the five repeats span 2329–3139 ms for the same
272-call FLOOR pass. Compare ratios across a batch, not absolutes across
sessions.

**New, unexplained:** FLOOR ran **272–373** times for 272 tiles. Up to 37% of
the preview pass is re-evaluating a floor some tile already had. That is
squarely inside §5.2's territory and is the cheapest lead on this list.

### 10c. Variance is now visible instead of assumed

`profile_montage_workflow --repeat N` runs the selected stages N times in one
process and prints per-run values plus median/min/max. Each pass builds a fresh
window and session, so montage stages stay genuinely cold (1088 uploads on
every pass). **`load_data` does not** — its file read is OS-page-cache warm
from pass 2 (190 ms, then 44–53 ms); read that row as warm.

Five repeats, `raw_full_tiled_montage` elapsed: **3879 / 4167 / 4266 / 4625 /
4976 ms**, median 4266, spread 1098 ms (26%). The 4.0–4.9 s band this dossier
warns about is now a measurement, not folklore.

### 10d. The counters do not perturb what they measure

Six order-balanced A/B pairs (`--trace` on, alternating which side ran first —
the second run of any pair is systematically ~700 ms slower, which made two
unbalanced pairs look like a 1 s regression):

| | counters off | counters on |
|---|---:|---:|
| stage elapsed, median | 4409 ms | 4517 ms |
| **preview ack span, median** | **3348 ms** | **3272 ms** |
| uploads / bytes | 1088 / 285 MB | 1088 / 285 MB |
| commit batches | 21–23 | 22 |
| exact-2 acks | 272 | 272 |

The work counters are identical, and the preview ack span — the low-variance
counter §4c nominated — is 76 ms *lower* with the instrumentation on, i.e.
inside noise in the favourable direction. Cost per upload is two integer dict
bumps; cost per rung evaluation is one lock and three dict bumps against a
multi-millisecond evaluation.

### 10e. The FFT stage, re-taken over the stall fix

`ccb951fc` closed the 271/272 stall, so §6's stage finishes and its numbers can
be taken from a completed run instead of a truncated one. One run,
`--stages load_data,fft_full_tiled_montage`, stage elapsed **5595 ms**
(three repeats: 5632 / 5646 / 5803, spread 170 ms — six times tighter than the
raw stage's 1098 ms, so this stage does not need `--repeat` to be trustworthy).

**§6's gate 2 is now a counter, not a code reading.**
`tile_lod_preview_reduced_*` reports **scheduled 0, blocked 23, failures 0,
last gate `resident lod policy mode`**. Twenty-three times per run the pipeline
asked for the shared reduced-input preview and `shared_preview_is_useful`'s
opening `resident` clause refused it. This is what the three dead counters
should have been saying all along.

**Uploads, all at level 0 again** — and 2.2× the raw stage's bytes:

| level | representation | uploads | bytes |
|---:|---|---:|---:|
| 0 | `complex_rg32f` | 1088 | 570 425 344 |
| 0 | `scalar_r32f` | 240 | 62 914 560 |

633 MB of native-plane uploads, no reduced row anywhere. `complex_rg32f` is
8 B/texel, which is where the factor over §10a's 285 MB comes from.

**Rung evaluation, split by lane and outcome** (from the trace's `rung` /
`level` / `fn_ns`; the diagnostics rows aggregate the same work):

| rung | level | lane | outcome | calls | mean | total |
|---|---:|---|---|---:|---:|---:|
| FLOOR | 4 | display_preview | completed | 62 | 5.3 ms | 326 ms |
| DESIRED | 2 | display_preview | completed | 514 | 21.6 ms | 11 098 ms |
| DESIRED | 1 | display_preparation | completed | 60 | 21.6 ms | 1 293 ms |
| **DESIRED** | **1** | **display_preview** | **stale** | **8** | **1000.6 ms** | **8 005 ms** |

Three things fall out of that table.

1. **Eight seconds of FFT evaluation is computed and thrown away, every run.**
   Eight cold `DESIRED(level=1)` tasks at almost exactly 1 s each, all
   `stale` — superseded before they could commit. Inside a 5.6 s stage on four
   workers, that is the single largest lever these counters found, and it is
   precisely the waste ADR 0050's shared reduced-input preview exists to
   prevent — the path the counter above shows refused 23 times per run.
2. **Input size is not what sets evaluation cost here.** The *same*
   `(DESIRED, level 1)` costs 21.6 ms in one lane and 1000.6 ms in the other, a
   **46× spread at identical input size**. The cheap ones hit a warm stage
   cache; the expensive ones are the cold first-and-only presentable rung for a
   tile with nothing (`has_first_pixel` False routes DESIRED to
   DISPLAY_PREVIEW). And level 1 carries 4× the texels of level 2 yet costs the
   same 21.6 ms when warm.
   So "reduced input is ~16× cheaper" cannot be read off this stage: the
   reduced-input evaluation never runs, and the term the claim scales — texels
   in — is not the term that dominates. Cold-vs-warm stage cache is.
3. **Still zero previews on the FFT pipeline**, over the fix: the only FLOOR
   rows are 62 level-4 raw tiles inherited from the earlier phase, and the
   stage reports `first_preview_floor_fill_ms = n/a` with the first FFT pixel at
   2564 ms. §6's finding survives the stall fix intact.

## Open questions, left open deliberately

- **Why 18 transactions for the preview and 2 for the refinement.** This is
  §5.1 and the only thing worth measuring next. Candidates: the cohort byte cap
  (`_idle_backlog_cohort` / `_persistent_tile_upsert_limits`), the item clamp's
  interactive arm, or simply worker arrival pacing forcing a commit per
  completion wave.
- **Why FLOOR evaluates up to 373 times for 272 tiles** (§10b). New, cheap to
  chase, and the same pass §5.2 is about.
- ~~**The `fft_full_tiled_montage` stall at 271/272**~~ — **CLOSED
  2026-07-26.** A/B'd back to `51b826a` (2026-07-23) and root-caused to a
  page-pool layer leaked at construction, unrelated to LOD or preview policy:
  [wgpu-pool-layer-leak-2026-07-26.md](wgpu-pool-layer-leak-2026-07-26.md).
  What it leaves behind for this dossier is that **§6 was measured on a stage
  that never finished**, so its FFT numbers are a lower bound on work done and
  need re-taking. §10e re-takes them over the fix.
- ~~**Which page keys the 1088 uploads belong to**~~ — answered by §10a.
- **Is reduced-input evaluation ~16× cheaper for op pipelines?** Still open, and
  §10e explains why it cannot be measured here: the reduced-input preview never
  runs on this stage (23 refused calls per run, all `resident lod policy mode`),
  so there is no reduced-input evaluation to time. What §10e does refute is the
  claim's *premise* — input size is not what sets evaluation cost on this
  pipeline; cold-vs-warm stage cache is, by 46×.
- **The 8 s of discarded FFT evaluation per FFT stage** (§10e). Newly visible
  and the largest thing the counters found.
