# GPU engine implementation plan (G-program)

**Date:** 2026-07-15. **Branch:** merged to `main` 2026-07-16.
**Decision record:** [ADR 0055](../decisions/0055-view-tiles-data-chunks-residency-pages.md)
(three-way separation of view tiles, data chunks, residency pages).
**Status (2026-07-16):** G1–G4 are merged; the G5 canonical source-grid page
route, reducer families, shared bounded cache, producer migration, and both
backend consumers are implemented on the landing candidate. Final G5
real-Wayland/stress acceptance remains row 1. The physical-presentation-truth
invariant is standing; field defects were root-caused from live traces and
fixed with failing-pre-fix gates. **The remaining G-steps are ordered in
[`../queue.md`](../queue.md)** (the only active queue); this file stays the
design/stage record. Historical status detail: this file's git history and
the [continuation brief](gpu-port-continuation.md).

**Current renderer note (2026-07-27):** the initial VisPy integration steps
below are historical. WGPU is the maintained GPU executor, PyQtGraph is the
CPU/headless/remote executor, and VisPy was retired by
[ADR 0061](../decisions/0061-retire-vispy-rendering-backend.md).

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

## Physical presentation truth (landed 2026-07-15, field-driven)

A live field defect (orange tiles on montage scroll; dossier:
[`../redesign/coverage-stall-2026-07-15.md`](../redesign/coverage-stall-2026-07-15.md))
proved that acknowledgement-only presentation could re-present physically
divergent GL state invisibly: identity deliberately excludes levels/LUT/
scale, and nothing pinned the per-quad mode buffer. The engine now audits
every active page visual (mapping key + derived uniforms, levels, mode
buffer) before ANY re-present, repairs divergence, counts repairs
(`physical_repairs`), and a live framebuffer gate asserts zero-magnitude
complex data never renders the PAL-relaxed LUT[0] orange. This is a
standing engine invariant: every later stage (pyramid pages, compressed
backing, new executors) inherits the rule that acknowledgement requires
physically verified state, never identity equality alone.

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

**G4c montage warming slice landed (2026-07-16).** The existing stage-aware
montage prefetch completion now hands evaluated payloads to the common
`warmTiledResidency` backend seam for both CPU-item and GPU-atlas residency.
VisPy enqueues the payload on its standing bounded GL continuation; warm
content remains source-keyed, non-presenting residency until an ordinary
typed commit promotes it. Montage range changes feed the existing
`SliceScrubMomentum` policy, and idle candidate order prefers the motion side
inside each canonical viewport-priority band. This does not create a second
scheduler or speculative lifecycle rows. In particular, constructing
future-window payloads with the session's presentation helper was rejected:
the warm path consumes only results the existing prefetch owner evaluated and
admitted.

### G5 — Sparse virtual multiresolution pyramid (ADR 0056)

**Canonical route contract (2026-07-16):**
[`g5-source-grid-pyramid-2026-07-16.md`](../redesign/g5-source-grid-pyramid-2026-07-16.md)
defines the one source-grid reduction route, boundary/gutter identity,
best-resident-ancestor binding, owner-scoped pin, and physical-truth rules for
every remaining G5 slice. Ingest, ladder, CPU cache, and later GPU generation
must consume that model; none may independently bin from a viewport origin.

The LOD ladder's members become immutable logical data chunks: anisotropic
`reduction_vector` + reducer identity in chunk keys; uniform plane-pixel
pages shared across LODs (closes the reduced-LOD re-upload gap by
construction); page-table entries resolving to the best resident ancestor
with generations; pinned coarse coverage with never-black eviction order;
mode-specific derived LOD families over shared raw L0 (lazy, budgeted);
sufficient statistics only where exact reducers demand them. Streaming for
large/lazy sources (ADR 0049) rides the same page model; brick depth along
non-displayed axes stays backend policy.

*Gate:* mixed-LOD scene with injected missing fine pages renders from
pinned coarse coverage (no black tile, explicit missing-value only when no
ancestor exists); zoom across LOD thresholds swaps page-table entries with
zero re-uploads of already-resident levels; complex magnitude/phase views
use their own reducer families (unit-verified against phase-cancellation
fixtures); out-of-core scroll sustains interactivity with bounded VRAM.

**Slice 1 landed (2026-07-15): uniform plane-pixel pages across LODs.**
`_payload_chunked_eligible` no longer gates on factor 1: an exact,
gutter-free reduced plane whose shape is exactly the isotropic box
reduction of its anchor rect (`ceil(extent/factor)` per axis, verified
against the payload's `LodInfo`) chunks into the SAME origin-anchored 256²
plane-pixel slots native planes use — a chunk slot holds 256² *stored
samples* at any LOD, so mixed factor-1/factor-4 planes share one shape
class and no per-plane-size class explosion or cross-class eviction occurs.
Chunk keys keep NATIVE source rects (anchor origin + plane rect × factor,
clipped) plus the LOD triple: identical revisits at the same LOD reuse
every chunk (zero uploads); draw-part world rects apply the same uniform
stretch as the classic single reduced quad (exactly `factor` when the
extent divides), so placement is pixel-identical. Live wiring: exact
reduced planes (ingest-reduced payloads and pyramid-materialized floor
payloads) now carry the source anchor sized by their LOD's NATIVE source
shape, so a zoomed-out window shift presents its reduced exact target
through chunked residency (live gate in
`tests/ui/test_window_shift_live_path.py`). HONEST LIMIT (tested): at
factor>1 the reduction bins are anchored to the window origin, so a ±1
NATIVE-pixel shift resamples the plane — every chunk key changes and the
plane correctly re-uploads whole; anisotropic reductions and any
plane/native mismatch fall back to the classic path. True shift-reuse at
factor>1 needs source-anchored reduction binning — the ladder-migration
slice that moves reduction bins onto the source grid (with the gutter
story) remains open G5 work, together with ancestor resolution, pinned
coarse coverage, and reducer families.

**Slice 2 landed (2026-07-16): pure source-grid model and page resolution.**
`PageTable.resolve` now selects the finest compatible covering resident page
and returns its actual key/LOD, slot, target-sample scale/offset, and binding
generation. Compaction/reuse refreshes generations; owner-scoped atomic pin
sets preserve shared coarse coverage and deny refinement rather than evicting
an all-pinned fallback. `reduce_source_grid_mean` is the first consumer of the
canonical route: anisotropic bins are globally anchored, shifted windows share
full interiors, clipped boundaries retain distinct identities, and recursive
reduction is accepted only from an aligned input grid. This slice was pure
model only; later slices below attach it to the live ladder and both backends.

**Slice 3 landed (2026-07-16): VisPy resolved-page consumption seam.**
Anchored atlas chunks now carry canonical `DataChunkKey` identities. The pool
resolves logical targets once on the CPU, owner-pins actual coverage, rebinds
fine arrival/removal without a resolution upload or black intermediate, and
reports actual coarse LOD/fallback quality plus binding generation in physical
truth. Every atlas eviction route respects those pins. This is deliberately a
pool/presentation seam, not a second layer update or scheduler. The later
live-cutover slice replaces the former whole-plane ladder members with logical
page targets.

**Slice 4a landed (2026-07-16): live-ladder logical target planning.**
The render-LOD layer now decomposes desired native source rectangles into
canonical `DataChunkKey` pages on the global reduction grid as a pure value
transform. Shifted factor-2 windows share aligned interiors and keep clipped
boundaries distinct; desired target identity remains separate from physical
fallback. Wiring is intentionally deferred until materialization stops using
window-local `reduce_box_mean`: assigning global keys to those current texels,
or uniformly stretching a page whose partial boundary bins have unequal source
width, would violate semantic/coordinate truth. Materialization plus boundary
draw geometry is the next atomic slice.

**Slice 4b landed (2026-07-16): source-grid page value/geometry partition.**
The canonical reducer output can now be partitioned into uniform stored-page
classes without discarding per-sample native coverage. Partial edge bins keep
their exact widths, draw spans cover the valid source rectangle exactly once,
and shifted windows share only complete interior page identity and values.
The spans coalesce into at most 3-by-3 uniform draw blocks per page (one for a
fully aligned interior), bounding later quad count. This remains pure and
backend-free. The live-cutover slice below carries these pages through ladder
cache/payload state and constructs grouped backend geometry from the recorded
spans.

**Slice 5 implemented (2026-07-16): canonical live-page cutover.** One
immutable page plan now owns source-grid bins, clipped footprints, draw blocks,
anisotropic reduction, reducer lineage, dtype/representation, and value-source
identity. Native, mean, mean-absolute, power, RMS, and circular phase-vector
families materialize checked pages; only aligned complete mean routes recurse.
The renderer-shared bounded cache is keyed directly by `DataChunkKey`, with
page-set singleflight claims and exact producer admission. Ingest, rung,
retained/floor, preview, and prefetch paths request those same plans. Typed
page-backed payloads carry requested geometry separately from resolved actual
pages; PyQtGraph assembles exact bounded draw blocks, while VisPy uploads
canonical pages and resolves complete target sets through `PageTable` without
inventing reduced keys. Legacy `PyramidLevelKey`/`PyramidCache` live ownership
and factor>1 backend chunk inference are structurally forbidden. Final
acceptance is the focused/broad/stress and real-Wayland gate matrix in the G5
contract; it is not a timeout-widening or performance-optimization phase.

### G6 — GPU compute consumers

Histogram/level reductions over resident chunks (sharing the shader's
complex-mapping function, keeping ADR 0054 evidence phases); GPU LOD
generation. Operation kernels (flip/crop/conjugate-class only) are an
opt-in experiment at the end of this stage: CPU evaluation remains the
correctness path, and a GPU op lands only with harness evidence that it
beats the CPU+upload route — semantics duplication has to buy real
performance.

**First landing (branch candidate, 2026-07-17):** canonical materialized
pages now retain fixed-size min/max/histogram summaries weighted by their
exact clipped source-bin areas. The Qt-free frontier holds a coarse parent
until complete child coverage can replace it atomically, and the existing
phase-1 preview worker derives rough `TileLevelStats` from that frontier via
the shared shader mapping without replacing stronger semantic evidence already
attached to a preview. PyQtGraph's CPU-LUT single-pass path continues to skip
rough publication. Refined semantic sampling, GPU workgroup reduction, and
GPU-from-resident-page LOD generation remain subsequent G6 slices.
Presentation construction binds the prepared shader batch to the accepted
global level generation before draw, and rejects a stale crossing without
repeating materialization or upload.
Admission also distinguishes exact supplied-page residency from fallback
coverage: a coarser resolvable ancestor cannot make finer page uploads free or
let them bypass the per-frame work cap.
The first numeric oracle uses 64 local bins and requires aggregate normalized
histogram L1 error at or below 5% against the direct CPU histogram on its
deterministic multi-chunk fixture; levels preserve exact finite min/max.

**Live wgpu consumer (2026-07-18; real-Wayland acceptance pending):** phase-1
rough evidence now issues `DispatchHistogram` over each committed plane's
best resident ADR 0056 frontier. Dynamic finite bounds and bin counts are
computed on the GPU; the coverage worker fences the submission token before a
small deferred readback and installs the reconstructed bounded sample through
the existing level tracker and first-pass publication path. Scalar, complex,
and the scalar signal packed with windowable RGB stay GPU-resident throughout;
display-ready RGB has no level obligation. Incumbent backends retain the CPU
chunk-summary path. The scheduling-policy owner's evidence barrier keeps the
session in coverage until the matching GPU evidence is installed and published,
so the existing `no_phase2_submit_during_coverage` trace invariant remains
authoritative.


**Endpoint scope (Thomas, 2026-07-18 — binding for G6 completion):** the
point of G6 is that level/histogram SAMPLING itself runs on the GPU over
resident pages, not only that CPU summaries ride along with
materialization. Three explicit ambitions beyond the first landing:

1. **GPU-side sampling/reduction shaders — implemented, acceptance pending**
   over resident chunk pages
   (sharing the shader's complex-mapping function). Runtime honesty:
   VisPy/gloo is GL-3.3-era — no compute shaders. A fragment-pass
   reduction ladder over resident textures is the legitimate interim; true
   workgroup reduction arrives with the wgpu runtime (queue row
   "Renderer protocol + wgpu Experiment A") and G6's shader work should be
   written against the backend-neutral command protocol so it ports
   rather than being rebuilt. Coordinate the two rows; pulling Experiment
   A forward is on the table if the GL interim proves awkward.
2. **Evidence-gated design question — collapse rough→refined:** if GPU
   sampling of the full resident population is fast enough, the two-stage
   level phasing may become a single exact pass on shader backends.
   This is a MEASUREMENT decision, not a theory decision (ADR 0046;
   graveyard discipline): keep the ADR 0054 phases until a harness shows
   full-population GPU stats inside the phase-1 budget. The CPU-LUT
   single-pass rule for PyQtGraph (rendering.md contract rule 6) is
   unaffected either way.

   **Measured verdict (2026-07-18): NO; keep ADR 0054 phasing.**
   `arrayscope.tools.g6_histogram_benchmark` compared the live rough shape
   (retained L2 population, exact bounds, 64 bins/512 representatives) with
   the strongest refined-grade candidate (native L0 resident population,
   exact bounds, the histogram UI's 500-bin cap, and 8,192 representatives),
   using the live four-source commit cap. GPU timestamp queries cover the
   bounds+histogram passes; GUI submit, worker fence/readback/reconstruction,
   total publication, and a Qt heartbeat are separate fields. Five warm
   measurements on the real 336×336×272 float64 T2 and Intel TGL Vulkan gave:

   | Session | Real Wayland rough → exact GPU p95 | Exact heartbeat max | Exact verdict |
   |---|---:|---:|---|
   | single slice (L0 → L0) | 4.37 → 2.57 ms | 1.49 ms | eligible; fits |
   | 60 tiles (L2 → L0) | 29.56 → 110.49 ms | 9.28 ms | **ineligible population** |
   | 272 tiles (L2 → L0) | 163.78 → 382.99 ms | 4.08 ms | **ineligible population** |

   Offscreen repeated the montage cost direction (60 tiles: 31.70 → 136.61
   ms GPU p95; 272 tiles: 164.73 → 401.14 ms). The compute-only exact pass is
   heartbeat-safe, but the measurement also makes the decisive eligibility
   fact explicit: representative phase 1 owns retained L2 pages, whereas
   refined semantic evidence samples the native population. Promoting the L2
   mean distribution would contradict ADR 0054; making L0 resident before the
   barrier would move target work into coverage and contradict the binding
   progressive contract / ADR 0053. The singleton cannot set a backend-wide
   policy while both montage populations require that forbidden phase change.
   Publication therefore continues to mark the phase-1 evidence rough and the
   existing phase-2 semantic refinement remains real work; no scheduling order
   or `ProgressiveSchedulingPolicy` decision changed. Raw evidence:
   `tests/artifacts/g6b-histogram-collapse-2026-07-18/{offscreen,wayland}.json`.
3. **VisPy histogram widget:** replace/replicate the PyQtGraph histogram
   UI with a VisPy-rendered equivalent fed straight from chunk/GPU
   summaries — parity first, then improvements. This also retires the
   standing risk that the histogram adapter depends on private PyQtGraph
   API (current-state.md). PyQtGraph backend keeps a working histogram
   throughout (both-backends rule).

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

**Live-benefit audit (2026-07-22; supersedes the product verdict below).** The
codec ratios and sampler closure remain valid component evidence. They did not
measure the live topology: production allocated full raw and compressed pools,
CPU-encoded and reference-decoded every cold page synchronously, and read
compressed LOD sources back to the CPU. The host benchmark used one split budget
while production added a second full budget, retained aliased arrays, double-
counted overlapping keys, and substituted a stage FFT for a display-cache miss.
The [matched live review](../reviews/2026-07-22-compression-live-benefit-review.md)
measured cold submission 8.5–138× slower and LOD 8.4–42.5× slower; configured
pool bytes grew 9–18%. RAW/OFF defaults are restored and G7 closes with a
measured NO. It is revived only by evidence of a real capacity/I/O bottleneck;
the tables below are codec/component history, not proof that AUTO wins.

The same audit found retention ownership questions that precede codec choice.
The evaluator display cache feeds every backend; its separately budgeted region
cache serves exact ROI demand and is not GPU page storage. VisPy/wgpu page pools
and PyQtGraph raster backing are alternative physical backend owners. The
cross-session retained-payload store really was entry-bounded but byte-unbounded
and now has a tile-residency byte cap. The only plausible host compression
continuation is raw-hot/compressed-cold demotion at the actual expensive
`StageCache` owner, off-thread and under one total budget; it is an ideas-list
experiment until real pressure/miss telemetry triggers it.

*Status (2026-07-22, first ZFP-class step).* Landed the codec abstraction
(`arrayscope/gpu/chunk_codec.py`: `raw`/`zfp`/`blosc2`, lossless-by-default,
dtype-driven `resolve_codec` that falls back to `raw` for dtypes a codec
cannot represent exactly), the opt-in host-cache path (`CompressedChunkCache`),
and the `chunk_transport_codec` setting **defaulting to `raw` (off)** — the GPU
upload path in `wgpu_executor` is intentionally untouched, so the production
transport is byte-for-byte unchanged. Benchmark
(`arrayscope/tools/g7_transport_benchmark.py`, 200 × 256² chunks of a real
336×336×272 volume):

| dtype     | codec  | ratio | comp ms | decomp ms | break-even GB/s | inequality @12 GB/s PCIe |
|-----------|--------|-------|---------|-----------|-----------------|--------------------------|
| float32   | zfp    | 1.71  | 245.7   | 294.0     | 0.04            | no |
| float32   | blosc2 | 2.29  | 1147.5  | 161.7     | 0.02            | no |
| complex64 | zfp    | 1.68  | 425.7   | 556.1     | 0.04            | no |
| complex64 | blosc2 | 2.39  | 2022.0  | 207.8     | 0.03            | no |
| int16     | zfp    | 1.88  | 299.4   | 178.3     | 0.03            | no |
| int16     | blosc2 | 2.56  | 968.6   | 119.5     | 0.01            | no |

**Verdict: the inequality does NOT hold for CPU decode on this hardware.**
Real data compresses well (1.7–2.6×), but the break-even link bandwidth
(~0.01–0.04 GB/s — the speed below which saved bytes outweigh codec CPU cost)
sits three orders of magnitude below a real PCIe link (~12 GB/s) or even host
memcpy (~26 GB/s). Every round-trip is lossless-exact. So compression's win
here is **host RAM** (more chunks cached per byte budget), not transfer time.
The default stays `raw` (off) — the gate's "prove before flipping" is met with
a truthful *no win yet*. The transfer-side win requires a **GPU-side decoder**
(nvCOMP or a wgpu compute decode) so the decompress cost leaves the CPU
critical path; that is the G7 follow-up.

*Phase A (2026-07-22, host-RAM win, topology-aware).* Cashes the host-RAM win
the transport gate identified, as a **two-level cache**: a large **compressed
backing tier** under the small raw byte-bounded cache. Seam — the raw cache's
one shared `BoundedCache` core gained an optional `on_evict(key, value, nbytes)`
hook (default `None` → eviction byte-identical for every existing caller);
`operations/compressed_tier.py::TwoLevelArrayCache` wires that hook to
`CompressedBackingTier`, so a value the raw cache *evicts* is stored compressed
instead of dropped, and a raw-miss is served by a **decode** (µs) rather than a
**recompute/re-read** (an FFT / a disk page, ms–s). With no tier the wrapper is
a straight pass-through — the default path is byte-identical to today. The tier
is value-generic (`to_array`/`from_array`) so `StageCache` can adopt the same
tier later; wiring it *into* `StageCache` (retention scoring, in-flight-compute
claims, lock-free resident snapshots) was judged riskier than a standalone tier
the cache delegates to, so Phase A ships the standalone tier.

Topology (`gpu/device_topology.py`): reports integrated/discrete/unknown +
unified-memory from the wgpu adapter info, **reusing the shared device's adapter**
(no new adapter/device) or a single cheap probe; any failure → safe `unknown`
(treated as integrated/RAM-only). Import never touches wgpu. Both integrated
(Intel UHD, unified) and discrete (RTX A2000, PCIe) benefit on the RAM axis, so
topology does **not** gate engagement; the discrete device sets a
`discrete_transfer_candidate` flag that is the **Phase B seam** (compressing
host→VRAM transfer bytes once a GPU-side decoder exists — not Phase A).

Adaptive policy (`gpu/cache_policy.py`): engages the tier **only under RAM
pressure** (working set > cache budget — the only regime where evictions, hence
expensive misses, actually happen), picks the best **lossless** codec per dtype
(zfp's transform for float32/complex64/int16; blosc2's byte codec for dtypes zfp
declines, e.g. uint8), and stays **off** when data already fits so tiny
workloads pay nothing. It never selects a lossy codec, and `resolve_codec` is
the final gate (degrades to `raw` rather than lose pixels).

Benchmark (`arrayscope/tools/g7_cache_benchmark.py`, real 336×336×272 volume,
256² chunks, revisit/scroll workload under a fixed RAM budget):

| dtype     | zfp ratio | fit raw → 2-level (same RAM) | recomputes raw → 2-level | crossover recompute cost |
|-----------|-----------|------------------------------|--------------------------|--------------------------|
| float32   | 2.10×     | 40 → 91 chunks               | 520 → 387                | ~2.2 ms                  |
| complex64 | 2.09×     | 40 → 91 chunks               | 520 → 388                | ~4.5 ms                  |
| int16     | 2.21×     | 40 → 91 chunks               | 520 → 357                | ~5.4 ms                  |

The original model suggested a RAM/eviction win. The **modelled time** result is
miss-cost dependent: for a *cheap* 256² fft2 miss (~8 ms) the tier wins RAM but
not wall time (encode+decode > the tiny recompute); above the crossover
recompute cost (~2–5 ms) it wins both. In the owner's target regime — a
**large-matrix** stage miss (1024² fft2, ~50–64 ms, or a disk re-read) — the
modelled at **2.0–2.7×** by charging a 1024² FFT to each cache miss. The live
audit found that expensive work belongs to the separate StageCache, so this does
not justify AUTO. The repaired wrapper now uses one total budget and preserves
real payload aliases, but remains explicit until off-GUI work at the actual miss
owner passes the queue gate.

*Phase B (2026-07-22, component transfer/page-size experiment — native
block-compressed textures).*
The transfer-side win does **not** come from a compute decode of a byte-stream
codec (that keeps decode near the CPU critical path and, byte-stream-in-VRAM,
never reaches the sampler). It comes from a format the GPU decompresses **for
free in the texture sampler**: a native **BC / ASTC** texture. There is **no
decode pass at all** — the compressed bytes cross PCIe, stay compressed resident
in its codec pool (an 8× page-size cut vs r32float for BC4), and
the hardware sampler returns usable values at sample time. The cost is *lossy*
compression, acceptable on the display path (window/level is applied at sample
time regardless) and **measured**, never assumed.

What landed (all opt-in, default OFF; the wgpu render path is byte-identical):

- **Our own BC4/BC5 codec** (`gpu/bc_codec.py`, pure numpy): BC4 (8 B / 4×4 block,
  scalar tiles) and BC5 (16 B / block, two channels). A tile holds RAW values and
  is normalized to [0,1] per channel (the shader rescales at sample time). Complex
  tiles store **(real, imag)** — matching the existing `rg32float` complex pool —
  *not* (magnitude, phase): re/im are smooth and signed with no ±π wrap, so a
  small block-compression error stays small and isotropic in both derived
  magnitude and phase, whereas storing phase would smear across the wrap.
  A per-tile PSNR/max-abs plan (`bc4_plan`) **declines to raw** when the loss
  exceeds threshold, so quality is never silently sacrificed.
- **On-GPU WGSL BC4 encoder** (`gpu/bc_gpu.py`): a compute shader, one 4×4 block
  per invocation (per-block min/max endpoints + nearest-index), quality-equivalent
  to the CPU encoder — lets GPU-resident derived tiles (LOD/compute) be compressed
  on-device for host spill without a CPU round-trip. Verified against the real
  **hardware sampler** (render BC4 → r32float → readback): the sampler reproduces
  the reference decode (57.5 dB on NVIDIA, 101.7 dB on Intel — residual is the
  r32float target precision, not a codec disagreement).
- **ASTC on Intel** (`gpu/astc_codec.py`, via `astc_encoder`, optional dep):
  flexible block size = a real quality/size knob and better two-channel handling,
  hardware-sampled on Intel (which advertises `texture-compression-astc`; NVIDIA
  does not — BC only there).
- **Topology-aware policy** (`gpu/cache_policy.py::decide_texture_codec`): discrete
  (NVIDIA) → BC4 scalar / BC5 complex; integrated (Intel) → ASTC when available
  else BC; **default OFF** (must opt in — the path is lossy).
- **Unified layer** (`gpu/chunk_texture_codec.py`) + dual-adapter benchmark
  (`tools/g7_gpu_texture_benchmark.py`).

Dual-adapter benchmark (256² tile of a real T2 volume; raw = r32float scalar /
rg32float complex; PSNR/max-abs are the [0,1]-field scalar quality, magPSNR /
phase-err are complex **display-domain** magnitude PSNR and magnitude-weighted
phase RMSE):

| device | kind | format | enc | raw B | vram B | VRAM ratio | quality |
|--------|------|--------|-----|-------|--------|-----------|---------|
| NVIDIA A2000 (discrete) | scalar | bc4-r-unorm | CPU | 262144 | 32768 | **8.0×** | 40.7 dB, max-abs 0.070 |
| NVIDIA A2000 | scalar | bc4-r-unorm | **GPU** | 262144 | 32768 | 8.0× | 40.7 dB (== CPU) |
| NVIDIA A2000 | complex | bc5-rg-unorm | CPU | 524288 | 65536 | **8.0×** | magPSNR 41.5 dB, phase-RMSE 0.027 rad |
| Intel UHD (integrated) | scalar | bc4-r-unorm | CPU | 262144 | 32768 | 8.0× | 40.7 dB |
| Intel UHD | scalar | astc-4x4 | ASTC | 262144 | 65536 | 4.0× | **53.0 dB** |
| Intel UHD | scalar | astc-6x6 | ASTC | 262144 | 29584 | **8.9×** | 42.0 dB (beats BC4 on *both* axes) |
| Intel UHD | complex | astc-4x4 | ASTC | 524288 | 65536 | 8.0× | magPSNR 42.3 dB, phase-RMSE 0.034 rad |
| Intel UHD | complex | astc-6x6 | ASTC | 524288 | 29584 | **17.7×** | magPSNR 36.1 dB, phase-RMSE 0.068 rad |

**Component verdict (not live pool capacity).** On the **discrete A2000 (PCIe)**
one BC page is 8× smaller and the hardware decode is correct; BC is
NVIDIA's only compressed-texture family. On **integrated Intel (unified memory)**
there is no distinct PCIe transfer to shrink; **ASTC's block-size knob** can beat
BC for an isolated page (astc-6x6 gives higher PSNR
*and* a better ratio than BC4). Complex tiles use BC5/ASTC-2ch holding (re,im)
with **no phase-wrap artifact** (magnitude-weighted phase error ≤ 0.07 rad). The
live executor uses ASTC 4×4 and retains a full raw fallback pool, so isolated
page bytes are not allocated-pool bytes. The render path stays byte-identical
(default OFF); a lossless narrow-int `bitpack`
codec is retained in `chunk_codec.py` as the documented lossless fallback (CPU
round-trip tested). **Follow-up:** wiring the atlas/page textures in
`wgpu_executor` to sample BC/ASTC (tile-pipeline integration — the encoders,
policy seam, GPU encoder and measured quality land here as the demonstrator);
BC6H (half-float, less loss for float tiles) whose encode is harder than BC4/BC5;
and a possible lossless raw-buffer transfer on NVIDIA via CUDA↔Vulkan interop
(out of scope — the native BC/ASTC textures are the display-path win).

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
