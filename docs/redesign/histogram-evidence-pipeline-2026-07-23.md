# Histogram evidence pipeline investigation — 2026-07-23

## Question and verdict

The histogram stack had four legitimate lifecycle layers but two competing
first-pass producers. Normal WGPU frames recomputed 64-bin histograms over
resident pages even though the CPU page materializer had already attached
immutable summaries to those exact payloads. The result was semantically
correct eventually, but it added submit/fence/readback/sample-reconstruction
work and repopulated levels a few sources at a time. During crop and axis
changes the user could see the window and histogram contract move through
partial populations.

The route is now evidence-driven, not backend-driven:

1. reuse prepared page/tile summaries whenever present;
2. otherwise sample accessible CPU semantic values on the kernel;
3. otherwise dispatch over resident GPU pages.

The GPU route remains required for genuinely GPU-generated/resident-only
content. It is no longer mandatory merely because the surface is WGPU.

## What exists and why

| Layer | Canonical owner | Stored form | Purpose |
|---|---|---|---|
| page materialization | `MaterializedLodPage` / `chunk_summary` | 64-bin immutable summary + representative sample | compute once alongside values and reuse across views/backends |
| semantic source tracking | `LevelStatsService` / `MontageLevelTracker` | one quality-ordered `TileLevelStats` per semantic source | generation safety, population/rank truth, cross-window retention |
| resident-only fallback | WGPU surface/executor | fenced histogram readback keyed by physical page frontier | evidence when no CPU summary/value source exists |
| visible plot | histogram controller / `histogram_plot` | bounded aggregate sample, adaptively rebinned for widget pixels | UI resolution and interaction, independent of evidence storage |

These locations should not be collapsed into one mutable histogram: page
summary identity, semantic population identity, physical residency, and widget
view resolution differ. The defect was choosing the physical producer before
checking whether stronger reusable evidence already existed.

The old WGPU flow was:

`materialize values + summary → upload pages → GPU bins → CPU readback →`
`representative samples → montage aggregate → widget bins`

The normal flow is now:

`materialize values + summary → upload pages → montage aggregate → widget bins`

The longer flow is retained only as the resident-only fallback.

## Benchmark coverage

`arrayscope.tools.histogram_pipeline_benchmark` is the proper differential
matrix. Its default `representative` CPU suite covers:

- `uint8`, `int16`, `uint16`, `float32`, `float64`, and `complex64`;
- contiguous arrays, non-contiguous strided views, and `.npy` mmap sources;
- gradients, extreme outliers, and NaN/Inf populations;
- singleton, 60-source, and 272-source populations;
- exact finite-bound containment, complete source coverage, populated widget
  output, and atomic cross-semantic window/level transitions.

The `exhaustive` suite runs the full Cartesian product. GPU cells use the real
NIfTI dataset and execute on requested physical low-power and high-performance
adapters. The real-window `profile_montage_workflow` remains the flicker,
physical-pixel, crop/scroll, and GUI-callback gate; the algorithm benchmark is
not a substitute for it.

Commands used:

```bash
python -m cProfile -o /tmp/histogram-pipeline-cpu.cprofile \
  -m arrayscope.tools.histogram_pipeline_benchmark \
  --suite representative --engines cpu --shape 336x336 --repetitions 1 \
  --output /tmp/histogram-pipeline-cpu.json

python -m arrayscope.tools.histogram_pipeline_benchmark \
  --suite smoke --engines wgpu-low-power,wgpu-high-performance \
  --repetitions 1 --output /tmp/histogram-pipeline-gpu-both.json
```

## Measurements on this machine

Host: Intel i7-11850H; Intel UHD Graphics TGL GT1 integrated Vulkan adapter;
NVIDIA RTX A2000 Laptop discrete Vulkan adapter.

The 14-cell representative CPU matrix at 336×336 was correctness-green.
Across cells, median production evidence construction was 11.8 ms. The worst
cell was the 272-source strided float32 population at 65.9 ms. Reusing and
installing all already-prepared stats cost at most 0.48 ms; aggregate sampling
cost at most 0.92 ms and widget binning at most 0.50 ms. The cProfile confirms
that source fixture construction and exact-reference validation dominate the
whole benchmark; production `sample_tile_level_stats` totaled about 0.25 s
over 993 sources.

Resident GPU evidence remained much more expensive for the same 272-source
rough obligation:

| Adapter | rough 272 total | submit | resolve | GPU compute |
|---|---:|---:|---:|---:|
| Intel integrated | 327.7 ms | 139.9 ms | 127.5 ms | 51.7 ms |
| NVIDIA discrete | 364.7 ms | 127.5 ms | 189.5 ms | 18.1 ms |

Exact 272-source evidence was 523.0 ms integrated and 411.0 ms discrete. The
discrete GPU makes compute faster, but Python submission/readback and bounded
sample reconstruction still dominate enough that device class alone cannot
select the best route. Evidence availability selects it correctly on both.

## Visible correctness gate

The profile montage harness now samples histogram population, applied semantic
key, evidence quality, and displayed levels through nested crop and scroll
event loops. It rejects:

- an empty histogram or `(0, 1)` fallback after successor pixels appear;
- a source-count regression within one semantic action;
- a transient range collapse or center excursion within one semantic action.

A single complete range change between two different crop/channel semantics is
valid and is not called flicker. `WindowLevelController` retains a complete,
meaningful predecessor's visuals while current-successor metadata progresses,
then switches once when the successor population is complete at target
quality. First-ever display still accepts partial evidence rather than
retaining the numeric fallback.

The initial WGPU reproduction had 41 observable transitions, source coverage
cycling 50→0 and a transient span ratio of 0.11. Prepared-summary reuse removed
the incremental 1→3→7→…→50 build and reduced the trace to about 30 transitions.
The final individual real-Wayland X/Y WGPU runs emitted zero
`wgpu_histogram_dispatch` events, were green for the continuity and
first-visible-evidence gates, and settled in 7.13/7.51 s. PyQtGraph X/Y was
likewise continuity-green at 7.22/6.72 s. Both backends remain red on the
independent 50 ms GUI callback bar (roughly 115–120 ms observed), which this
slice does not conceal.

## Addendum 2026-07-24 — resident crop-rebind cannot re-anchor from the GPU histogram

The resident crop-window rebind (`FramePipelineEffects._seed_resident_crop_rebinds`,
commit 38c51806) is default OFF because a rebind step reuses the predecessor
window's auto-level evidence instead of re-anchoring. Enabling it by default was
scoped to feed the GPU-computed resident histogram evidence for the new window
into the post-first-pass montage level tracker. That is **blocked by a
granularity mismatch**, not merely unwired.

Measured ground truth (row-gradient 336x336x272 montage, 100 tiles, no
operations; wgpu backend, resident policy):

- The ordinary evaluation settles **window-exact** levels — window `[40:240]`
  gives `(118.9, 713.9)`, window `[120:320]` gives `(358.1, 953.0)`. Auto-levels
  track the cropped rows' value range.
- With the rebind on, both windows show `(118.9, 713.9)` — the predecessor's
  levels held across the scrub. On a gradient this is a large, visible error.
- The rebound `[120:320, 66:266]` tile binds to **four full-plane 256x256
  canonical pages spanning source rows `[0:336]`**. The scrub re-addresses the
  same whole-plane pages at a shifted origin (this is precisely why they are
  resident and the rebind is free).

`DispatchHistogram` (`arrayscope/gpu/command_protocol.py`) histograms whole
`DataChunkKey` blocks (`chunk_origin : chunk_origin + chunk_shape`) and carries
no source sub-rectangle. A histogram over the rebound frontier is therefore the
**whole-plane** range (~`(0, 1000)` here), window-independent, and cannot
reproduce the window-exact levels the crop-parity physical-truth gate asserts.
Routing it in would either fail to re-anchor (gradient gate still red) or
re-anchor to wrong whole-plane levels (a worse regression). The resident GPU
histogram is the correct fallback for resident-only *content*, but it is not a
window-scoped evidence source.

Window-exact evidence without a display evaluation would require one of:

1. a sub-rectangle histogram dispatch — a new GPU capability (shader +
   `DispatchHistogram` + executor change), beyond wiring existing pieces; or
2. CPU sampling of the cropped source values per tile on the kernel (the doc's
   route 2) — window-exact and off the `display_preparation` lane, but a fresh
   evidence route that must match the evaluator exactly across LOD / operations
   / complex mode / canonical orientation / scaled sources or it ships a levels
   regression.

Until one of those lands the rebind stays default OFF (opt-in via the raw
`resident_crop_rebind` QSettings flag), keeping every crop-parity gate green.

## Invariants

- Evidence quality and availability choose the producer; backend name does
  not.
- Prepared immutable summaries are reusable computation, not stale widget
  state.
- A physical histogram is a fallback for resident-only values, never proof
  that existing CPU evidence should be discarded.
- Histogram/level presentation changes at most once per completed semantic
  successor; partial evidence may prepare off-screen but does not replace a
  complete incumbent.
- Timing gates remain device evidence. CI asserts matrix shape and
  correctness, never wall-clock thresholds.

## Addendum 2026-07-25 — the semantic evidence owner as route 2, measured

Route 2 above (kernel-side CPU sampling of the cropped window) does not need to
be built: `LevelStatsService._ensure_semantic_level_evidence_target` +
`evaluate_level_evidence_snapshot` already are it. They sample the source
through the operations evaluator on `WorkLane.HISTOGRAM_REFINEMENT`, bounded to
`REFINED_TILE_SAMPLE_LIMIT` pixels per source, and they key on
`session.level_key`.

**Keying, checked first.** `montage_level_key` folds the whole scope `ViewState`,
and the crop window lives in `axis_range_indices`, so every crop window gets its
own tracker entry. `_montage_level_family_key` strips only the montage
population, so the per-source memory cache is window-scoped too. New-window
evidence therefore supersedes the predecessor summary by *arriving under its own
key*, not by outranking anything. There is no key collision to resolve.

**The collision was inside the key, not between keys.** A rebound wrapper is a
`replace()` of its predecessor, so it carries that window's `level_stats`, and
`_queue_montage_level_stats_for_payloads` merged them under the NEW key at
target quality. Because a scrub clones forward from one ancestor, every step
republished the FIRST window's bounds: on a 50-tile row gradient, `10:50`'s
`(112, 590)` survived `11:51`, `12:52` and `13:53` unchanged — the display even
stepped *backwards* after a re-anchor had already corrected it.

The fix is a demotion, not a drop: a rebound wrapper is stamped
`level_evidence_window_stale` and its evidence is admitted at
`ROUGH_PREVIEW`. The successor population is then COMPLETE but IMMATURE, which
is the maturity contract's own signal to hold the predecessor's window until one
target-quality population lands — so the re-anchor happens once, atomically.
`FrameSession.level_evidence_reanchor` arms the owner at the same seam and
promotes it from the two-source background trickle to the blocking batch size.

Measured, 50-tile 64x64 row gradient, wgpu, `resident_crop_rebind` on:

| | settle | producers | levels at settle | window-exact at |
|---|---|---|---|---|
| rebind, before | 125-171 ms | 0 | ancestor window | 155-186 ms after settle |
| rebind, after | 100-135 ms | 0 | predecessor window | 80-100 ms after settle |
| ordinary evaluation | 570-750 ms | 50 | window-exact | at settle |

Both paths settle on identical window-exact levels (`caf435b7` path
independence). The 100-tile 336x336x272 field repro is unchanged in per-step
cost within run-to-run spread (28 baseline steps mean 449 ms, 40 changed steps
mean 480 ms, one outlier run carrying all of the difference; excluding it,
448 ms), still zero `display_preparation` producers, with a ~200-400 ms
statistics-lane tail after the last step.

### Why the default stayed OFF — and what it actually was

The route did not reach **operation-pipeline montages**. Under
`CenteredFFT(axis=2)` the owner is armed but never admitted — zero completed
batches — with or without the rebind; under `CenteredFFT(axis=0)` the content
key changes per window so the rebind declines outright and the owner is never
even armed (`blocking_reason: inactive`). A crop scrub on such a montage could
only be as good as its demoted placeholder: it held the predecessor's window and
kept a histogram source, which is honest, but it is not the window-exact
re-anchor the plain-montage case gets.

Attributing that to the `verdict` / `_montage_side_work_visible_settled`
refusals was reading the refusals a probe happens to catch rather than the one
that persists. Both gates are **open** by the time such a session settles
(`verdict: refine`, side-work settled, `first_pass_histogram_published: True`).
The refusals seen are the ordinary early ones, and neither protects an operation
budget. See the closing addendum.

## Addendum 2026-07-25 (second) — the op-pipeline gap was a stale deferral backlog

`_schedule_semantic_level_evidence` is level-triggered: it re-checks its gates
only when something calls it. After a session settles, the sole caller that can
still re-arm it is `_finish_frame_session_if_complete`, which bails on
`FrameSession.is_complete()`.

`is_complete()` reads `deferred_missing_tiles` directly. The index-window
retarget (`_maybe_retarget_frame_session`) recorded that backlog
**unconditionally** while setting its `stage_planning_deferred` flag
conditionally. The backlog is only meaningful with the flag — it is the argument
`complete_deferred_stage_fan_in` replans from, and that owner returns early
unless the flag is set. A retarget that found an existing stage plan therefore
left a backlog nobody would ever drain.

Finding an existing stage plan is exactly what an **operation-pipeline** montage
does once its shared stage is warm, and exactly what a raw montage never does.
That is the whole of the raw-versus-op asymmetry: on the FFT montage every
retarget parked 50 undrainable tiles, `is_complete()` stayed False forever after
its pixels settled, and every completion continuation behind it never ran.

The fix pairs the two, as the session-build path already did:

```python
session.deferred_missing_tiles = (
    tuple(missing_tiles) if session.stage_planning_deferred else ()
)
```

**Scope is wider than the rebind.** Measured on the 50-tile 64x64 row gradient
under `CenteredFFT(axis=2)`, wgpu: before the fix the displayed auto levels were
frozen on the FIRST window for the whole scrub on **both** paths — the rebind
path AND the ordinary per-tile evaluation (`blocking_reason: inactive`,
`ROUGH_TARGET`). An operation-pipeline montage simply never refined its levels.

| `CenteredFFT(axis=2)`, 50 tiles | settle/step | producers | evidence batches | window-exact levels |
|---|---:|---:|---:|---|
| rebind, before | 90-139 ms | 0 | 0 | never |
| evaluation, before | 313-346 ms | 0 | 0 | never |
| rebind, after | 102-107 ms | 0 | 4 | 198-323 ms after settle |
| evaluation, after | 305-352 ms | 0 | 25 | 174-269 ms after settle |

Both paths settle on identical levels, as on raw data. Visible settle per scrub
step is unchanged within run-to-run spread on every cell; the evidence lane is
`WorkLane.HISTOGRAM_REFINEMENT` work admitted only after the visible plan
settles, and it stays 62-113 ms of worker time per step across the bounded
batches. The raw-montage cells are untouched by the fix (130-143 ms rebind,
550-770 ms evaluation, both anchoring ~220 ms after settle).

Other readers of `is_complete()` were mis-served by the same residue on every
operation-pipeline montage: the montage watchdog kept re-arming, the loading
overlay stayed eligible, and `_montage_render_active` reported the render busy
forever, holding the memory policy in its active-render branch.

### The default

The capability stays opt-in in this step; flipping it is a separate decision
with its own gates.  Nothing is left of the caveat that gated it, though: the
rebind now settles the same auto levels as the ordinary evaluation on raw and
operation-pipeline montages alike.

Standing reds observed while gating this, neither caused by it (both reproduce
on the unmodified tip): `tests/ui/test_montage_scroll_level_retention.py::test_vispy_one_tile_scroll_retains_level_population`
times out settling, and `tests/app/test_memory_stress.py::test_montage_tile_residency_rss_stays_bounded`
is flaky at roughly the same rate either way (8/12 red unmodified, 7/12 red with
the capability forced on, with LARGER overshoots unmodified — 17.1 MB against
15.3 MB), so its accounted-bytes gate is not evidence about this capability.
