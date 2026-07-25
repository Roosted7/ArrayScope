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
