# GPU engine implementation plan (G-program)

**Date:** 2026-07-15. **Branch:** `codex/gpu-engine`.
**Decision record:** [ADR 0055](../decisions/0055-view-tiles-data-chunks-residency-pages.md)
(three-way separation of view tiles, data chunks, residency pages).
**Status (2026-07-15):** G1 and the G2 first slice are implemented (gpu
package + atlas on the page table). G3 is implemented offscreen-first:
windowability predicate (`pipeline_windowable_display_axes`), source-anchored
single-image planning (`ANCHORED_CHUNK_SHAPE` 256², `source_rect`,
buffer-independent content source_ids, gpu_atlas-gated), TileDrawPart
multi-quad/UV-crop substrate, and the residency gate tests (boundary-only
uploads on ±1 shift, zero-upload scroll-back, FFT negative control). The G3
exit gate still requires the real-display harness half. Known G3b-1 limit:
windows narrower than ~3 chunks contain no interior chunks and still
re-materialize fully — lifted by G3b-2 (full-chunk materialization with
window-as-camera clipping, design below).

## User problem

Interactions that change *which resident values are shown* — same-extent
slice-window shifts, fixed-index scrolls, axis reversals, montage reshuffles —
today cost full CPU re-materialization plus full GPU re-upload, because view
tiles, data chunks, and residency slots are bound 1:1. Only montage index
scroll has a hand-built fast path (`retarget_index_window`). The GPU sits
mostly idle as a texture cache; it should be doing the indexing, mapping, and
eventually reduction work. The endpoint: on minimal hardware, over small and
large datasets, redisplay-of-resident-data is a descriptor update, and
first-image latency never regresses.

## Non-negotiable constraints

- **One semantic pipeline.** Axis/index semantics (`core/view_state.py`),
  montage layout, viewport/visibility, quality and generation convergence,
  levels/histogram meaning (ADR 0054), and interaction stay shared between
  backends. Backends compile the same logical view into different execution
  plans.
- **No scheduler beside the kernel** (ADR 0053). Residency and upload work is
  kernel work; the page table is passive state.
- **Time-to-first-image is a gate on every stage.** Whole-array residency is
  background convergence (priority: level-evidence samples → visible →
  adjacent → remainder), never a first-frame prerequisite.
- **PyQtGraph stays simple and correct** — CPU slice → CPU map → `ImageItem`.
  It is the portability fallback and the semantics oracle for backend
  conformance tests.
- **Performance claims need harness evidence on real hardware**
  (`docs/redesign/README.md` bars; upload counts and heartbeat gaps recorded
  before/after, both backends where applicable).

## Architecture target

```
shared semantic layer (view state, sessions, pipeline, lifecycle, levels)
        │  logical view + ViewTileKeys + DataChunkKeys
        ▼
arrayscope/gpu/            ← Qt-free, VisPy-free engine
    keys.py                ← ViewTileKey / DataChunkKey / representation
    page_table.py          ← chunk → (pool, page, slot) | missing; generations
    pool.py                ← physical pools: fixed-size slots, eviction policy
    (later) transfers, repack, compute, compression
        │  residency plan (uploads, remaps, evictions)
        ▼
display/backends/vispy/    ← canvas/context/shader integration only
display/backends/pyqtgraph ← CPU materialization path (unchanged contract)
```

A `DataChunkKey` is
`(document_generation, operation_key, lod, chunk_origin, chunk_shape, dtype,
representation)` — deliberately parallel to the evaluator's cache keys so the
two derivations can be property-tested against each other (the
`montage_tile_key_batch` self-check pattern).

## Stages and exit gates

### G1 — Concepts and software page table (no behavior change)

Introduce `arrayscope/gpu/` with keys, page table, and pool model; pure
Python, fully unit-tested, imported by the import-health guard. Express the
current world as the degenerate mapping (one view tile ↔ one chunk ↔ one
slot) so later stages are data-shape changes, not rewrites.

*Gate:* model unit tests including property tests for key stability;
`test_import_health` passes; zero diff in rendering behavior/tests.

### G2 — Atlas becomes the first page pool

Re-express `TextureAtlasPool`/`TextureAtlasPage` residency bookkeeping
(`tile_slots`, `_resident_key`, shape-class pages, LRU/reclaim) on the G1
page table. The atlas keeps its texture mechanics; the *bookkeeping* moves to
the engine so PyQtGraph's resident-item pool can later share eviction policy.

*Gate:* existing VisPy tiled-renderer suite green; page-table diagnostics
exposed in `presentation_diagnostics`; upload counting instrumented (needed
for every later gate).

### G3 — Chunk-granular sourcing: the window-shift fast path

Break tile↔chunk 1:1. Per-tile data identity normalizes `axis_range_indices`
into (chunk grid ∩ window) references, so a same-extent shift
(`X=100:200 → 101:201`) resolves to already-resident chunks plus at most a
boundary strip. Requires: texcoord/offset indirection in the tile visual
(sample the atlas with a sub-window), and delta planning that requests only
missing chunks through the existing pipeline.

**Correctness rule — key it, don't special-case it.** Operations stay on the
CPU (2026-07-15 scope decision): the GPU engine consumes *evaluated* planes,
so its chunk space is the evaluated-value space at the current operation
revision, and "ops output" is indistinguishable from "no ops" on the GPU
side. Chunk reuse across a window shift is then a pure keying question:

- when the operation chain provably does not consume the display-axis window
  (v1: the chain is empty / range-inert — the raw-view case), chunk identity
  is source-anchored and shifts reuse resident chunks;
- otherwise the window stays folded into the chunk key exactly as the
  evaluator keys it today, and a shift naturally misses — correct by
  construction, no windowability oracle to get wrong.

An FFT along a displayed axis is the canonical must-miss case and joins the
gate tests as the negative control. Moving simple ops (flip, crop,
conjugate) onto the GPU is a *later* G6 experiment, done only if measured
performance justifies duplicating semantics.

*Gate:* harness scenario — repeated ±1 window shift over a resident frame
uploads only boundary chunks (recorded upload counter), pixel-identical to
the PyQtGraph oracle; the FFT-along-displayed-axis case takes the full
re-evaluation path and stays pixel-correct; first-image latency on cold
window unchanged (±10%).

**G3b-2 — full-chunk residency, window as camera (design).** G3b-1 clips
regions to the window, so small windows have no shift-stable interior
chunks, and every shift still re-evaluates the (cheap for raw views) CPU
plane. The completion: materialize *whole* source-aligned chunks for the
chunk-expanded window (union of chunks intersecting it), let world
coordinates equal source coordinates, and present the user window as a
camera rect over the tiled plane — a shift becomes a camera move plus
boundary-chunk requests, with zero re-materialization of resident chunks.
This subsumes the UV-crop path for interiors (whole quads, camera-clipped
edges) and is the natural place to relax the `"no-axis"` session rebirth in
`frame_controller` for anchored plans. Requires the display-coordinate
mapping (probes, ROI, camera fit) to learn the window origin offset.

### G4 — Small-array whole residency + GPU indexing

After first presentation, opportunistically upload the whole (small) source
array as chunks at background priority. The shader (or per-instance
descriptors) then serves: fixed-dimension index movement, subset offset
changes, axis reversal, simple strides, montage instance remapping — all
without new uploads. This generalizes `retarget_index_window` from a montage
special case into the engine's normal mode, and gives non-montage views the
same cheapness (today they always rebirth).

*Gate:* harness — fixed-index scroll and montage shuffle over a resident
small array performs zero uploads and holds the 16 ms heartbeat; FFT-scroll
benchmark improves toward the scalar rate (the #1 throughput target).

### G5 — N-D chunk pool and streaming for large arrays

Chunked/lazy sources (ADR 0049) stream source chunks; optional
staging-repack into view-optimized bricks when displayed axes are highly
strided. Size-tiered policy: small = whole-array resident; medium =
active-XY repacked layout; large = streamed bricks. Predictive residency
reuses the existing prefetch/priority machinery.

*Gate:* out-of-core scroll scenario sustains interactivity with bounded VRAM
(pool budget respected, evictions traced); no shared-layer code path
distinguishes backends.

### G6 — GPU compute consumers

Histogram/level reductions over resident chunks (sharing the shader's
complex-mapping function, keeping ADR 0054 evidence phases); GPU LOD
generation. Operation kernels (flip/crop/conjugate-class only) are an
opt-in experiment at the end of this stage: CPU evaluation remains the
correctness path, and a GPU op lands only with harness evidence that it
beats the CPU+upload route — semantics duplication has to buy real
performance.

*Gate:* histogram-from-chunks matches CPU histogram within documented
tolerance on both real-hardware backends; levels convergence behavior
unchanged from the ADR 0054 contract.

### G7 — Compressed transport and optional hardware sparse

Codec-aware chunk transport (compressed host cache → GPU decode → raw hot
cache), dtype/error-policy driven (ZFP-class codecs first candidates;
nvCOMP as optional NVIDIA acceleration). Hardware sparse textures only where
they measurably beat the software page table.

*Gate:* benchmark matrix proving the compression inequality
(compress + transfer + decompress < raw transfer) per dtype/scenario before
any default flips on.

## Cleanup that accompanies the program

- Fold legacy `display/colormaps.py` into `colormap_library` (3 import
  sites) — G1.
- `core/scheduler.py` is a misnamed shared-types module (post-kernel);
  rename/relocate its dataclasses — G1/G2 window.
- The three drifted viewport-distance rankers are V2 work on the main
  branch; do **not** duplicate that here (parallel-cowork collision rule).
- As G2 lands, `display/backends/vispy/tiles.py` (2938 lines) splits into
  engine bookkeeping (moves to `arrayscope/gpu/`) and thin GL mechanics.

## Why this is a proposal and not yet roadmap

The V/P programs own the main branch's queue until the performance bars are
met. The G-program runs on `codex/gpu-engine` behind its own gates; it enters
`docs/roadmap.md` when G1–G3 demonstrate the window-shift gate on real
hardware without regressing first-image latency.
