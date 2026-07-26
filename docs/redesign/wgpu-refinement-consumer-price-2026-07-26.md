# What the pixel-identical WGPU refinement buys

**Date:** 2026-07-26

**Scope:** `raw_full_tiled_montage`, WGPU, zoomed out far enough for the
native-plane warm to apply

**Base:** local `main` at `feeea32a` (`1771e118` is an ancestor)

**Outcome:** diagnosis only; no production code changed

## Verdict

The CPU desired rung is not a WGPU display refinement on this stage. It does
three jobs which the current pipeline has coupled:

1. **Target-quality level evidence — real and useful.** The 272 desired
   materializations provide weighted histogram summaries. They move the
   stage-visible evidence from `ROUGH_PREVIEW`, source rank 2, to
   `ROUGH_TARGET`, source rank 4, and move the auto-level upper bound from
   `42571.27734375` to `46329.37890625` (8.1% below the target result when the
   desired work is removed). This result does not require presenting or
   acknowledging the desired payload: an evidence-only probe reproduced the
   baseline level bounds with zero exact payloads and zero exact backend ACKs.
2. **Lifecycle settlement — real and ACK-dependent.** Removing the desired
   payload/ACK leaves `required_target_settled() == false`,
   `is_complete() == false`, and 272 preview payloads unrefined. Evidence-only
   publication does not close that lifecycle. Because semantic `REFINED`
   evidence is scheduled only after `is_complete()`, it is blocked too.
3. **Zoom readiness — no measured physical benefit here.** A later zoom into
   the same montage produced its first physical WGPU draw with zero additional
   uploads in both arms. The median was 55.6 ms after normal refinement and
   42.1 ms after suppression over three order-balanced observations per arm.
   This small sample does not establish that suppression is faster, but it
   rules out the proposed warm-page latency benefit at the measured scale.

The presentation portion of the desired rung therefore buys no source-pixel
change and no measured zoom warm-up. It buys an exact-target lifecycle token.
The useful level evidence comes from the worker result and can be published
without making that result a display rung.

Do **not** simply suppress the rung. That leaves coarser auto levels, an
incomplete frame, and no path to semantic `REFINED` evidence. The recommended
direction is to separate target evidence from presentation, then let WGPU
settle an unchanged physical target through an explicit equivalence contract.

## Why this is physically the same draw

Three independent observations agree with the finding in the
[preview/LOD anatomy](preview-lod-anatomy-2026-07-26.md) and the refutation in
[the shader-legibility proposal](../proposals/wgpu-shader-legibility.md):

- [`WgpuImageView2D._wgpu_tile_instances`](../../arrayscope/display/wgpu_imageview2d.py)
  emits every `TileInstance` with LOD `0`.
- Both native binding paths set `lod_level=0`. When
  `_wgpu_reusable_native_texture` succeeds, the reduced payload is replaced by
  the exact semantic plane before upload/binding.
- Every measured arm reported exactly
  `wgpu_uploads_by_level = [{level: 0, representation: scalar_r32f,
  uploads: 1088, bytes: 285212672}]`. Normal refinement retained CPU level-2
  payloads; suppression and evidence-only retained CPU level-4 payloads. None
  of those reduced pages appeared in the WGPU upload counter.

In the pixel-oracle pass, the normal arm's FLOOR and FINAL PNGs had the same
SHA-256:

```text
1be1a2468671392faed1824be30dcfc70b0784374e61ac002396dbb50f087fc4
```

The files had zero differing pixels. This proves that accepting all 272
desired payloads did not alter that arm's framebuffer. It does not mean that
removing all desired work is semantically safe: the desired results also carry
the level summaries which change the shader's mapping uniform.

## Probe

The temporary probe lived under
`tests/artifacts/refinement-price-2026-07-26/` and remains gitignored. It
changed behavior only while `raw_full_tiled_montage` was active:

- **baseline:** unmodified pipeline;
- **suppressed:** reject every raw-stage `DESIRED` submission and observe for
  750 ms after all 272 preview payloads are physically present, without
  pretending the session is complete;
- **evidence-only:** run the 272 desired workers, feed their prepared weighted
  summaries to the existing level tracker as `ROUGH_TARGET`, then discard
  their display payloads and backend ACKs.

The suppression and evidence-only comparisons each used six interleaved
in-process passes, three per arm, in order-balanced sequences:

```text
baseline, probe, probe, baseline, baseline, probe
```

The normal cost reference used an additional unmodified `--repeat 3` run.
Those invocations used the requested tool-managed headless compositor and
Wayland Qt platform. Sequential fresh-process A/B timings were not used.

The refinement is bistable. The interleaved baseline cohort landed in a slower
state than the independent repeat-three cohort, so stage wall-clock deltas are
not treated as savings. The consumer conclusions below come from ACK,
materialization, evidence, residency, upload, and commit counters. Medians are
quoted over at least three passes wherever a timing comparison is made.

`montage_quality_coarse_rung_gates` contained only the expected policy reasons
(`floor already covers this level` and `allow_preview false: tile is covered
or too few missing`) in every arm. Their occurrence counts followed replan
frequency rather than physical work. The decisive rung counter was
`montage_quality_rung_evaluations`: suppression had no desired row;
evidence-only and baseline each had 272 desired calls per pass. In the
order-balanced evidence run their median summed worker times were 3557.5 and
3716.3 ms respectively, confirming that evidence-only retained rather than
optimized the desired worker work.

## Consumer matrix

| Arm | Desired worker results | CPU resident payloads | Exact backend ACKs | Stage-visible evidence | Auto levels | Canonical completion |
|---|---:|---|---:|---|---|---|
| baseline | 272 | `[[2, 272]]` | 272 | `ROUGH_TARGET` (2), rank 4, 272 sources | `[0, 46329.37890625]` | true |
| suppressed | 0 | `[[4, 272]]` | 0 | `ROUGH_PREVIEW` (1), rank 2, 272 sources | `[0, 42571.27734375]` | false |
| evidence-only | 272 summaries; payloads discarded | `[[4, 272]]` | 0 | `ROUGH_TARGET` (2), rank 4, 272 sources | `[0, 46329.37890625]` | false |

All three arms uploaded the same 1,088 level-0 WGPU pages and 285,212,672
bytes. Thus:

- the **summary result**, not the exact display ACK, is what buys the
  target-quality auto levels;
- the **exact display ACK**, not the summary result, is what currently buys
  target settlement;
- neither result changes which WGPU source pages are resident for this draw.

### Level evidence

[`MaterializedLodPage`](../../arrayscope/display/pyramid.py) constructs a
source-extent-weighted `ChunkHistogramSummary`. The quality order in
[`montage_levels.py`](../../arrayscope/display/model/montage_levels.py) is
`ROUGH_PREVIEW < ROUGH_TARGET < REFINED`.

The suppression probe settled only the first step of that order:
`ROUGH_PREVIEW`, rank 2. The evidence-only probe reproduced the normal arm's
`ROUGH_TARGET`, rank 4 result exactly, including all 272 sources and both level
bounds, while keeping the displayed CPU payload population at level 4.

The raw stage did not itself reach semantic quality `REFINED` (3). Normal
target ACKs let `is_complete()` become true, after which
[`_finish_frame_session_if_complete`](../../arrayscope/window/frame_controller.py)
starts the separate semantic evidence pass. Suppression and evidence-only
leave completion false, so they indirectly block that pass. The current
display ACK is consequently a scheduling prerequisite for refined semantic
evidence, not the evidence's data source.

### Lifecycle settlement

[`FrameSession.is_complete`](../../arrayscope/window/frame_session.py) requires
both `required_target_settled()` and no unrefined preview payloads. In every
suppressed and evidence-only pass, the trace observation after full preview
coverage was:

```text
complete=false
target_settled=false
preview_payloads=272
exact_payloads=0
```

This is a real dependency. It is also why the profile record's generic
`complete` field must not be used for this counterfactual: the probe returned
from its bounded observation window so the process could continue, while the
canonical session remained incomplete.

An unchanged WGPU draw cannot be settled by accepting a preview ACK as exact;
that would violate target identity. The owner needs a new honest statement:
the target's physical binding is equivalent to the already drawn native
binding, and its separate target evidence is ready.

### Later zoom

The zoom probe changed the same montage's view after either normal settlement
or full suppressed preview coverage:

| Prior arm | First physical draw, ms | Additional WGPU uploads |
|---|---:|---:|
| baseline | 55.6 median (`75.8`, `48.6`, `55.6`) | `0, 0, 0` |
| suppressed | 42.1 median (`42.1`, `44.5`, `38.0`) | `0, 0, 0` |

There is no sign that the level-2 CPU pages warm the physical WGPU zoom. Native
level-0 pages already service it.

The synthetic zoom did not reach its requested CPU level-0 lifecycle target
within the five-second observation in either arm: both reported no pending
materializations but retained twelve coarser and three level-0 payloads. That
makes target-lifecycle zoom settlement inconclusive. It does not invalidate
the physical first-draw and zero-upload result, but it is not evidence that
the later zoom's whole lifecycle is healthy.

## Price

### Desired worker work

In the unmodified `--repeat 3` run, the desired rung performed 272 evaluations
per pass. `montage_quality_rung_evaluations` reported summed worker wall times
of:

```text
3114.4 ms, 3246.0 ms, 2802.6 ms; median 3114.4 ms
```

This is the sum across tasks running on four workers, not 3.1 seconds of
serial critical path. Suppression removes all 272 evaluations. Evidence-only
retains all 272 and therefore prices the present evidence implementation at
essentially the same worker work. A source/stage-summary implementation could
remove that work only if it proves identical final evidence.

### Target presentation transactions

For the healthy unmodified repeat-three cohort, the exact level-2 ACK payloads
landed in one or two `backend_complete` commit transactions:

| Pass | Transactions | Commit elapsed total | `resident_rebinds` fields |
|---:|---:|---:|---:|
| 1 | 2 | 285.5 ms | `272 + 272` |
| 2 | 2 | 278.7 ms | `272 + 272` |
| 3 | 1 | 176.7 ms | `272` |

The median is two transactions and 278.7 ms. Each transaction enumerated the
entire 272-binding resident set even when its exact delta contained only a
subset; summing the trace fields gives 544 binding visits in the median pass,
not 544 unique tiles. Upload count and upload bytes were zero for these
resident target commits.

The slower order-balanced suppression cohort showed the bistability rather
than a stable A/B: its baseline exact commits took 634.2, 472.8, and 471.8 ms
(median 472.8 ms) in 3, 2, and 2 transactions. This is why no stage-time
"speedup" is claimed.

The final exact transaction is charged at the end of the harness's
`fully_visible_ms` interval because that milestone includes target settlement.
The healthy repeat-three `fully_visible_ms` values were 4312.4, 4139.3, and
4356.4 ms (median 4312.4 ms). In pass 1 the final exact transaction began at
4167.0 ms, ran for 160.1 ms, and straddled that 4312.4 ms milestone. The
physical WGPU source binding was already the native level-0 binding; this tail
closes lifecycle truth, not a new draw.

### Level metadata transactions

Target summaries also caused two empty-delta level/histogram publications in
each healthy pass. They cost 167.9, 194.4, and 182.3 ms in total (median
182.3 ms). Every such transaction again reported `resident_rebinds=272`.

The evidence-only order-balanced passes likewise produced exactly two
metadata-only commits with zero payload deltas and zero exact ACKs. Their
totals were 283.4, 266.6, and 282.6 ms (median 282.6 ms), again with two full
272-binding visits. This isolates a second owner problem: decoupling evidence
from the display payload is not sufficient while a metadata publication still
republishes the whole montage.

These metadata and target-payload commit totals overlap worker arrival and
other stage work, so they must not be added mechanically to predict stage
wall time.

## Recommendation

**Make the WGPU desired display rung conditional on a physical draw change,
but retain its two non-display contracts through separate owners.**

The bounded implementation target is:

1. **Evidence:** generate target-quality source/stage summaries and publish
   them through the existing `MontageLevelTracker` path without constructing
   or presenting a redundant WGPU display payload. The first acceptance gate
   is exactly the evidence-only row above: quality 2, rank 4, 272 sources,
   `[0, 46329.37890625]`, and the same final histogram evidence.
2. **Settlement:** add a canonical target-equivalence acknowledgement whose
   identity says that the requested target resolves to the already committed
   native WGPU binding. Admit it only after target evidence is ready. Do not
   relabel preview acceptance as exact, and do not let backend drawability
   become semantic target truth.
3. **Presentation:** when page keys, representation, sampling, geometry, and
   mapping are physically unchanged, skip the tile-rebind transaction. A
   level/histogram-only update must update its uniforms/metadata without
   re-enumerating 272 tile bindings. **Follow-up on `6ad55232`: the literal
   payload-object predicate does not hold today.** Both final metadata-only
   commits carried 272 fresh `DisplayTilePayload` wrappers because the new
   level generation is part of each wrapper's required presentation identity,
   even though sampled wrappers retained the same image object, source ID, and
   LOD. A one-pass predicate probe reported 272/272 object mismatches on both
   commits, so no fast path landed; see per-commit dossier §8.4. The missing
   prerequisite is a canonical physical-binding identity separate from the
   level-bearing wrapper, owned across payload construction/lifecycle rather
   than inferred in the WGPU backend.
4. **Fallback:** retain the existing CPU display rung for PyQtGraph and for
   WGPU paths where the renderer proves that the target changes the physical
   binding. ADR 0059 work may change that predicate; rerun this gate after it
   lands.

No production patch is proposed here because the evidence/settlement split
crosses the active ADR 0059 ladder work. The diagnosis identifies the owner
seams and the counters a later change must hold.

## Is the CPU LOD pyramid's WGPU display role dead?

**Scoped answer: yes for this zoomed-out raw WGPU montage on the measured
tip.** The level-4 floor and level-2 desired CPU pages never become WGPU pages;
the renderer binds level-0 native pages for both. The CPU pyramid's observed
roles on this path are:

- weighted histogram/level summaries;
- preview/target lifecycle and settlement tokens;
- CPU display payloads for the PyQtGraph backend.

That makes retention the next question. The stage reported 14,930,496 bytes
and 604 CPU pyramid entries while the desired reduced population itself is
about 7.7 MB. WGPU reported 285,212,672 active resident bytes and a
287,047,680-byte allocated pool, all for level-0 pages. A WGPU-only retention
audit should therefore price CPU evidence retention separately from CPU
display retention and GPU page allocation; it must not assume the CPU desired
payload prevents a later GPU upload, because the measured zoom did not upload
anything.

This claim does **not** cover:

- cropped scrolling;
- a zoomed-in single image;
- PyQtGraph;
- any post-ADR-0059 path where a reduced page is actually uploaded.

The synthetic zoom-into-montage physical probe is included above, but its CPU
target-lifecycle settlement was inconclusive. These boundaries should feed
queue row 4's large-data retention audit rather than being generalized into a
global pyramid-removal proposal.

## Reproduction record

The unmodified reference used:

```bash
env -u WAYLAND_DISPLAY \
  /home/thomas/miniconda3/envs/arrayscope/bin/python \
  -m arrayscope.tools.headless_display -- \
  env QT_QPA_PLATFORM=wayland \
  /home/thomas/miniconda3/envs/arrayscope/bin/python \
  -m arrayscope.tools.profile_montage_workflow \
  --backend wgpu \
  --stages load_data,raw_full_tiled_montage \
  --repeat 3 \
  --jsonl tests/artifacts/refinement-price-2026-07-26/baseline/out.jsonl \
  --trace tests/artifacts/refinement-price-2026-07-26/baseline/trace.jsonl
```

The temporary probe used the same compositor, Python, backend, dataset, and
stage, with its arm order supplied by environment variables. The evidence
quoted from JSONL was:

- `montage_quality_rung_evaluations`;
- `montage_quality_coarse_rung_gates`;
- `montage_quality_resident_tile_levels`;
- `wgpu_uploads_by_level`;
- `level_evidence_quality`, `level_source_rank`, and `level_source_count`;
- `histogram_data_bounds` and `display_levels`.

The trace evidence was:

- accepted `backend_ack` rows by level and quality;
- `commit_batch` rows with `phase == "backend_complete"`;
- bounded probe observations of canonical completion and target settlement.

Artifacts are intentionally not committed.
