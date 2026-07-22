# G7 compression live-benefit review — 2026-07-22

## Verdict

The codec components are real and mostly correct, but the previous evidence did
not establish a product benefit. The live AUTO defaults were premature.

- Lossless ZFP/Blosc2 compress real chunks well, but CPU encode + decode still
  loses the transport inequality by orders of magnitude.
- BC/ASTC reduce the bytes of an *active compressed page*, and the hardware
  sampler decodes them correctly. The audited executor initially allocated full
  raw and codec arrays; the follow-up replaces that eager mirror with separately
  demand-sized arrays. Compression still needs a real pressure/reuse win.
- Cold WGPU submission originally paid CPU encode plus reference decode for
  every page. Scalar BC quality is now accumulated inside the GIL-free encoder
  without decoding; compressed-source LOD still leaves the GPU for readback,
  CPU decode/reduce, and re-upload.
- The host tier can retain more keys under one byte budget, but production had
  been given the full raw budget plus an equally large tier, real display aliases
  kept the primary array alive, and the benchmark charged a synthetic FFT to the
  wrong cache owner.

This review restores host `RAW` and texture `OFF` as defaults. Explicit codec
choices remain as experimental mechanisms. G7 closes with a measured NO rather
than consuming the active queue: compression is revived only when telemetry
shows a capacity or I/O bottleneck that it can actually remove.

### Follow-up measurement correction: tile arrival was not 8–138× slower

The original cold-submit ratios below are real, but their scope was too easy to
misread. `g7_live_compression_benchmark` submitted 16 pages in one synchronous
executor call. It did **not** present a frame, acknowledge a draw, run the live
per-source rough histogram topology, or run post-visible exact semantic
evidence. It therefore measures codec batch throughput on the GUI-thread seam,
not perceived time to the first useful tile.

The user's observation that tiles remained reasonably quick while “semantic
evidence” drained slowly exposed a separate owner-level bug. Sparse exact
evidence requests have two sampled image axes followed by one point-selected
montage axis. The shared reader applied `np.take` left-to-right, so it copied a
roughly 90×336×272 intermediate before selecting one source plane. Basic point
and slice indexing now runs first, in one view-like operation, before the two
sparse gathers. Eager region evaluation delegates to the same canonical reader.

On the real 336×336×272 volume, the historical one-source selection took 163.5
ms median versus 0.22 ms after the fix (744× in the regression probe). The full
production-shaped raw sweep—272 sources, 8,192 pixels/source, two sources per
batch—now takes 156 ms median eager and 163 ms lazy, with a 2.50/2.63 ms maximum
worker batch. This defect predates G7 and exact semantic evidence does not read
either compressed cache; compression can delay when the sweep starts, but not
its document/stage evaluation once running.

## Pipeline diagnosis: the wrong work at the wrong seam

The codecs are not intrinsically absurd. They were inserted into a dynamic
scientific-display pipeline as though fewer stored bytes implied lower latency.
That implication is false here:

- ArrayScope's current product problem is bounded first-pixel and interaction
  work. The live benchmark did not show PCIe saturation, GPU-memory exhaustion,
  or display-cache misses dominated by expensive recomputation.
- Host AUTO compressed the *display-result* cache synchronously under its cache
  lock. The expensive reusable operation results belong to the separate
  `StageCache`, so the tier paid encode/decode while usually avoiding only a
  cheap payload rebuild.
- Texture AUTO generated BC/ASTC from every dynamic tile on the CPU, decoded it
  again for a quality check, and kept a complete raw fallback pool. Those two
  implementation defects are now removed for scalar BC: quality is fused into
  encode, and physical raw/codec arrays follow measured demand. Synchronous
  encoding and compressed-source LOD remain on the wrong latency/dataflow seam.
- The histogram and window/level expansion was mostly **format plumbing tax**.
  Once lossy normalized texture pages became another physical representation,
  histogram compute, auto-level bounds, LOD, page identity, and diagnostics all
  had to understand it. It did not improve histogram or level semantics. Scalar
  histogram EMD then exceeded the experiment's own visible-drift threshold, so
  exact semantic refinement still had to remain a separate owner.

In other words, some defects were ordinary implementation bugs (edge padding,
NaN admission, aliases, budgets). The extreme timings come from the larger
design mismatch: CPU encoding and verification were placed on the latency path
to address capacity/transfer limits that were not established as live
bottlenecks.

## How conventional pipelines place compression

Conventional pipelines separate four jobs that this experiment coupled:

1. **Storage and network:** scientific pyramids are chunked, compressed, and
   multiscale *before* viewing. OME-Zarr, for example, defines independent chunk
   arrays per resolution and permits storage codecs at that boundary
   ([OME-Zarr 0.5](https://ngff.openmicroscopy.org/0.5/)). The viewer reads the
   resolution and chunks demanded by the viewport; it does not recompress every
   display tile on admission.
2. **Static 3D assets:** GPU texture compression is normally prepared offline or
   transcoded once from a delivery format, then uploaded directly in the native
   sampling format. Khronos' KTX/Basis sample describes exactly that path
   ([KTX/Basis Vulkan sample](https://github.khronos.org/Vulkan-Site/samples/latest/samples/performance/texture_compression_basisu/README.html));
   NVIDIA's texture tools likewise target asset pipelines
   ([NVTT](https://docs.nvidia.com/texture-tools/index.html)). WebGPU exposes the
   BC/ASTC *formats and samplers*, not a free runtime encoder
   ([WebGPU specification](https://gpuweb.github.io/gpuweb/)).
3. **Dynamic LOD:** derived mip levels stay on-device. Vulkan's conventional
   runtime path blits or runs a custom shader from one GPU mip to the next and
   can schedule transfer work asynchronously
   ([Khronos runtime mipmap sample](https://github.khronos.org/Vulkan-Site/samples/latest/samples/api/texture_mipmap_generation/README.html)).
   GPU→CPU readback, decode, reduction, and re-upload is the opposite dataflow.
4. **Scientific meaning versus presentation:** window/level is a transform from
   semantic input values to displayed output values; it does not redefine the
   input evidence. The DICOM VOI LUT model makes that separation explicit
   ([DICOM C.11.2](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.11.2.html)).
   Exact bounds/histograms should therefore come from source/stage summaries or
   exact resident values, not be reconstructed from a lossy presentation cache.

High-end compressed transfer can be worthwhile when I/O really is the
bottleneck, but those systems remove CPU bounce/copy work and overlap a fast
GPU or hardware decompressor with transfer. GPUDirect Storage and nvCOMP are
examples of that shape
([GDS](https://docs.nvidia.com/gpudirect-storage/),
[nvCOMP](https://developer.nvidia.com/nvcomp)). They do not justify synchronous
per-page CPU compression for this pipeline.

The appropriate ArrayScope shape is therefore: compressed/chunked source data;
exact raw hot values and exact semantic summaries; dynamic LOD generated on
GPU; and an optional lossy presentation cache only when it is prepared off the
GUI thread, physically replaces raw capacity, never drives semantic truth, and
passes a measured capacity or transfer gate.

## Memory and very large data: retain value, not just bytes

The hot-path NO does **not** mean compression has no place in a large-data
viewer. It means a cold tier must be designed as a demotion policy under real
pressure. The audit first had to disambiguate three similarly named owners:

- `_display_cache` is the upstream CPU display-payload cache that can feed every
  backend. `_region_cache` is a separate exact ROI-demand cache used by
  `TileDataProvider`; it is **not** the GPU source-grid/page store. Both can be
  populated in the same GPU-backed viewer, and each is currently allowed the
  entire `display_cache_budget_bytes`. That possible additive retention remains
  visible rather than being hidden by merging their different eviction lives.
- VisPy/wgpu source-grid pages are backend device residency. PyQtGraph instead
  keeps CPU `ImageItem`/tile backing below the shared region-presentation seam.
  These are alternative backend mechanics and must not be added together for
  one viewer.
- `RetainedTiledPayloadStore` kept up to 4,096 acknowledged payloads but charged
  every one as zero bytes. It could keep arrays alive after their evaluator
  cache entry was evicted. It is now byte-accounted and capped by the session's
  tile-residency budget.

`python -m arrayscope.tools.memory_retention_audit` makes the configured owners
visible. On this 64 GiB host, with the balanced profile, wgpu, and a logical
4 GiB input, its conservative host-ownership envelope is 18.75 GiB for a lazy
source or 22.75 GiB for an eager source, plus 0.75 GiB of visible/prefetch
transient allowance and a separately reported 0.5 GiB device-residency bound.
This is deliberately not an RSS prediction: retained-payload arrays may alias,
allocators may retain freed arenas, and GPU/driver overhead is additional. The
important finding is the topology:

| Owner | Why it exists | Compression posture |
|---|---|---|
| source | authoritative data | prefer lazy, chunked, compressed-at-rest input |
| display materialization | upstream display-payload reuse for every backend | raw LRU; normally cheap to rebuild |
| ROI region demand | exact inspection subregions, independent of rendering backend | separate raw LRU; measure reuse before resizing |
| profile/scalar | exact inspection | keep small and raw |
| `StageCache` | avoids reusable FFT/operation-prefix recompute | best lossless cold-tier candidate |
| LOD pages | coverage and zoom stability | source-provided or generated/retained on GPU |
| retained payloads | cross-session successor continuity | strict byte cap; evict by value |
| PyQtGraph raster or GPU page pools | selected backend's physical storage | alternative host/device owners, never summed together |

The 80-page pressure cell also puts the host-cache trade-off in scale. Under
one budget equal to 20 raw pages, three fresh runs were deterministic:

| Data | RAW retained / revisit | AUTO retained / revisit | AUTO admit | AUTO revisit |
|---|---:|---:|---:|---:|
| scalar | 20 / 20 | 26 / 26 | 98.1 ms | 54.2 ms |
| complex | 20 / 20 | 24 / 24 | 192.1 ms | 115.9 ms |

So lossless compression really can retain more—but only 30% more scalar pages
and 20% more complex pages in this real display-payload cell, while adding
synchronous work. If a recovered display page avoids a cheap wrapper rebuild,
that is a poor exchange. If the same exact tier demotes a costly, likely-to-be-
reused `StageCache` result off-thread, it may be excellent.

The candidate policy is therefore **raw hot + compressed cold**, not
compressed everywhere:

1. Lazy/chunked sources prevent the authoritative dataset from becoming an
   unconditional RAM copy. Source prefetch should retain native compressed
   chunks only behind an explicit transport-cache owner, never an invisible
   adapter cache.
2. Visible exact stages and pages stay raw. Pressure, not idleness alone,
   triggers demotion; idleness merely supplies safe CPU time.
3. A low-priority kernel task compresses evicted `StageCache` candidates from
   immutable values, outside the cache lock and GUI thread, and yields
   immediately to visible work.
4. Admission ranks `probability of reuse × measured recompute/read cost ÷
   compressed bytes`. Cheap display derivatives and never-reused stages are
   discarded, not compressed.
5. Raw and compressed entries share one physical budget and one key owner. A
   key is hot or cold, not double-resident by policy; recovery promotes it and
   makes the resulting eviction explicit.
6. The gate measures RSS/owned bytes, prevented real StageCache misses, decode
   latency, callback/heartbeat impact, and allocator release—not just codec
   ratio or cache entry count.

## Correctness findings and fixes

1. Native UNORM codecs cannot represent NaN or infinity. The previous quality
   comparison produced NaN and `NaN < threshold` is false, so non-finite pages
   were silently accepted and changed to finite values. They now fall back to
   the exact raw pool.
2. Boundary pages are zero padded to 256². The previous codec affine and quality
   gate included that invalid padding, so a valid 16² tile in 100..101 was
   normalized as 0..101. Codec normalization and quality now use only the key's
   valid extent; padding is edge-filled in normalized space.
3. Real shader payloads alias `data`, `semantic_data`, `lod_source_data`, and
   sometimes `level_data`. The old template nulled only one field, retaining the
   uncompressed storage and breaking alias identity after recovery. All aliases
   of the selected primary are now stripped and rebuilt together.
4. `TwoLevelArrayCache.max_bytes` now means one total budget. Production uses a
   25% raw hot tier and 75% compressed backing tier, resizes both within the same
   total, reports combined use, and preserves active memory-policy budgets across
   a live codec switch.
5. WGPU physical truth now reports compressed pages as lossy, and command reports
   record actual raw/compressed upload bytes. The executor exposes active page
   bytes separately from configured raw+codec pool allocation.
6. The cross-session retained-payload store now has a physical byte cap instead
   of only an entry cap. Display materialization, ROI demand regions, and
   backend page/raster storage remain distinct owners and are reported as such.

## Benchmark method

`python -m arrayscope.tools.g7_live_compression_benchmark` exercises the actual
cache wrapper and `WgpuPlaneExecutor`. Each result below is the median and range
of three fresh processes on AC power, native Wayland, the real 336×336×272 T2
volume, 16 × 256² pages, and an eight-raw-page total host budget. OFF/RAW and
AUTO/AUTO are separate processes. Intel selects live ASTC 4×4; the RTX A2000
selects BC. Timings are milliseconds. “Cold submit” is the one synchronous
16-page batch described above; it is not a full-viewer latency measurement.

| Adapter / data | Mode | cold submit | histogram | LOD | host admit | host revisit |
|---|---:|---:|---:|---:|---:|---:|
| Intel scalar | OFF/RAW | 14.2 [12.1, 25.9] | 136.1 [123.2, 140.2] | 5.0 [4.4, 5.6] | 0.1 [0.0, 0.1] | 0.0 |
| Intel scalar | AUTO/AUTO | 499.8 [485.4, 511.4] | 129.0 [127.6, 132.9] | 42.2 [41.6, 45.7] | 20.9 [20.1, 21.9] | 20.0 [19.5, 21.8] |
| Intel complex | OFF/RAW | 14.2 [13.4, 15.3] | 132.8 [131.4, 136.3] | 7.4 [7.2, 8.1] | 0.1 | 0.0 |
| Intel complex | AUTO/AUTO | 1956.8 [1949.6, 2287.4] | 143.5 [130.5, 163.6] | 59.6 [56.9, 66.0] | 39.0 [38.0, 49.0] | 44.1 [37.3, 45.3] |
| A2000 scalar | OFF/RAW | 13.8 [13.2, 14.3] | 130.1 [121.5, 136.3] | 1.1 [1.0, 1.5] | 0.0 | 0.0 |
| A2000 scalar | AUTO/AUTO | 116.7 [112.8, 132.9] | 138.4 [136.9, 139.4] | 32.2 [30.4, 35.4] | 21.5 [21.2, 24.3] | 24.5 [20.0, 25.3] |
| A2000 complex | OFF/RAW | 15.7 [14.1, 15.8] | 132.8 [124.3, 133.3] | 1.1 [1.0, 1.2] | 0.1 | 0.0 |
| A2000 complex | AUTO/AUTO | 274.6 [266.0, 357.8] | 147.7 [128.5, 159.4] | 46.8 [43.6, 49.8] | 38.2 [37.7, 44.1] | 37.0 [36.4, 38.7] |

Resident histogram dispatch is approximately codec-neutral. Cold submission is
8.5–138× slower for the 16-page synchronous cell and compressed-source LOD is
8.4–42.5× slower. Intel complex AUTO reaches roughly 2.0 seconds for that batch,
which proves that a large callback would consume the interaction target; it
does not prove that incremental tile presentation is 138× slower.

The extended benchmark explains why individual tiles can still feel reasonable:
Intel OFF/AUTO submission was 12.3/21.6 ms for one page, but 13.8/507.7 ms for
16. The fixed raw overhead is nearly constant while ASTC work scales per page.
The discrete-GPU premise is nevertheless correct. An ABBA cell that excludes
encoding and quality measured A2000 `write_texture` plus completion as follows:

| Accepted pages | Raw → BC bytes | Raw → BC transfer | Speedup | Absolute saving |
|---:|---:|---:|---:|---:|
| 1 | 256 → 32 KiB | 0.13 → 0.09 ms | 1.48× | 0.04 ms |
| 4 | 1,024 → 128 KiB | 0.24 → 0.14 ms | 1.73× | 0.10 ms |
| 14 | 3,584 → 448 KiB | 0.67 → 0.31 ms | 2.18× | 0.36 ms |
| 59 | 15,104 → 1,888 KiB | 3.18 → 1.29 ms | 2.47× | 1.89 ms |

Fewer bytes therefore cross faster, but the roughly 0.03 ms/page saving is much
smaller than current preparation. Threads already help native/NumPy work. The
optional byte-identical Numba BC4 path releases the GIL, fuses scalar quality,
and improves the live 16-page A2000 AUTO submit from 127.7 to 36.5 ms; raw is
still roughly 12 ms. Its 357–403 ms prewarm is registered with the shared Numba
runtime and runs off the visible path; AUTO stays raw until it is ready. Only
that safe infrastructure lands: per-page artifacts and raw-to-compressed
demotion remain experiments requiring cancellation and shared byte accounting.

### Later montage pages: the opportunity is real, but it is mostly capacity

The full montage does provide preparation time: a replay of 544 matched
materialization/upload windows found 19.2 ms minimum and 2.01 s median lead.
One worker prepared every one of the 89 pages that passed 40 dB before its
deadline; peak queued artifacts were 68 pages / 2.23 MB. That validates the
prefetch premise, but chiefly as a future retention tool.

The earlier 538.71 MB OFF / 605.81 MB AUTO allocations were an allocator bug,
not a WebGPU requirement. Formats need separate arrays; their extents need not
match. Demand-sized arrays now grow independently while preserving layer
indices. On the same A2000 workload OFF allocates 150.21 MB; AUTO allocates
146.54 MB, owns 122.26 MB active, and copies 4.33 MB across two growths. A shared
ResourceGovernor byte cap and optional idle compaction remain future work.

The profiler also now records **physical draw edges** and page-backed tile rows,
separate from histogram/semantic/level settlement. In an interleaved
OFF/AUTO/OFF/AUTO fresh-process set, medians were:

| Mode | first physical tile | all 272 preview tiles drawn | new-tile rate | final LOD transition | active / allocated |
|---|---:|---:|---:|---:|---:|
| OFF | 0.587 s | 3.503 s | 93.0 tiles/s | 8.646 s | 143.13 / 150.21 MB |
| AUTO 40 dB | 1.063 s | 4.582 s | 77.5 tiles/s | 10.417 s | 122.26 / 146.54 MB |

These samples are recorded in the WGPU draw callback after presenting the page
table, excluding histogram and semantic settlement. AUTO is about 17% lower in
preview throughput, not 8–138× slower. Its first-tile penalty precedes any
compressed page, pointing to eager codec resource/pipeline activation. A future
artifact race must therefore prove avoided eviction/re-upload, not merely move
roughly 3 ms of aggregate A2000 transfer saving to another thread.

Lowering the gate was tested rather than assumed. The trace curve admitted
89/544 pages at 40 dB, 152/544 at 38 dB, and 408/544 at 35 dB. Under the
representative full-volume auto window those sets measured 54.84/48.57/42.69 dB
display PSNR, with 0.18%/1.24%/6.07% of pixels differing by more than four
display levels. More decisively, a deterministic 39.99 dB page that 38 or 39
would admit rendered at only 39.91 dB in the physical framebuffer under a valid
window, below the existing 45 dB oracle. The global gate therefore remains 40
dB. A better endpoint-search encoder or a carefully specified display-aware
policy—not a silent threshold reduction—is the route to more accepted pages.

The production `low-power` choice selects Intel/ASTC on this machine; one matched
montage was 8.91 s OFF versus 15.29 s AUTO. The profiler now records and can
select the adapter in a fresh process, keeping A2000 evidence distinct from the
default topology.

The old approximately 130 ms histogram column was dominated by first-use GPU
pipeline compilation. The corrected cold first-source cell is 121–130 ms for
both OFF and AUTO; a warm aggregate over all 16 sources is only 7–8 ms. Live
code, however, requests one dynamic-bounds histogram per source. Warm 1/4/16
source totals were 5.5/21.9/89.2 ms OFF and 6.9/18.9/77.1 ms AUTO. For the
16-source raw cell, submit/fence/readback were 10.3/23.5/55.3 ms. The linear
per-source submissions and two readbacks—not the codec—are the remaining rough
evidence scaling problem. A G6 timestamp-query stress run makes that shape even
larger, but adds a third readback and pre-residents all 272 sources, so it is
protocol evidence rather than a production-latency headline.

An isolated replay of the exact CPU phase, using the same production reader and
level-stat functions, gave the following results in addition to the 156–163 ms
raw measurements above:

| Document case | Sources / batch | Full sweep | Max batch | Stage/RSS interpretation |
|---|---:|---:|---:|---|
| eager raw | 272 / 2 | 156 ms median | 2.50 ms | 17.8 MB sampled slabs; ~1.7 MB first-run RSS delta |
| lazy raw | 272 / 2 | 163 ms median | 2.63 ms | same work; no eager-copy penalty in this cell |
| FFT over image axis | 60 / 2 | 403 ms median | 24 ms warm-run max | one distinct retained stage/source |
| FFT over image axis | 272 / 2 | 3.10 s | 136 ms | 272 stores, 208 evictions under the bounded StageCache |
| FFT over montage axis | 60 / 2 | 277 ms median | 178–334 ms first batch | one 35.6 MB stage then 59 cache hits |

The indexing bug is fixed, but the operation-backed rows identify a legitimate
architectural cost: exact levels currently re-evaluate the full selected
population after visible pixels settle. Conventional large-image pipelines
precompute or incrementally retain exact min/max/histogram summaries beside
source/stage chunks, then combine that small metadata tree for the requested
population. ArrayScope should move toward that shape. It should not infer exact
semantic truth by rereading lossy display textures, and it should not keep every
display payload merely to avoid summary recomputation.

## Byte and residency interpretation

The small 16-page table below records the audited eager-allocation implementation
and is retained to explain the defect. It is not the current allocator result.

| Adapter / data | Mode | submitted / active bytes | configured pool bytes | unique host keys |
|---|---:|---:|---:|---:|
| Intel scalar | OFF/RAW | 4,194,304 | 7,077,888 | 8 |
| Intel scalar | AUTO/AUTO | 1,048,576 | 8,388,608 | 9 |
| Intel complex | OFF/RAW | 8,388,608 | 12,058,624 | 8 |
| Intel complex | AUTO/AUTO | 1,048,576 | 13,369,344 | 8 |
| A2000 scalar | OFF/RAW | 4,194,304 | 7,077,888 | 8 |
| A2000 scalar | AUTO/AUTO | 983,040 (14 compressed, 2 raw fallback) | 7,733,248 | 9 |
| A2000 complex | OFF/RAW | 8,388,608 | 12,058,624 | 8 |
| A2000 complex | AUTO/AUTO | 1,507,328 (15 compressed, 1 raw fallback) | 13,369,344 | 8 |

The active-page saving was real; the eager configured-capacity result was not a
fundamental limit. Demand-sized physical arrays now remove that full+full mirror,
as the montage result above demonstrates. They remain separate because WebGPU
textures have one immutable format, and they retain independent high-water marks;
future shrink/compaction and one shared physical-byte cap remain open. Host AUTO
retained only zero or one extra unique key while adding synchronous work; a
larger, compressible working set can improve retention, but the prior `40→91`
figure double-counted raw/tier overlap and its time win substituted a synthetic
FFT for a display-cache miss while the expensive StageCache remained separate.

The lossless transport microbenchmark still gives a measured NO: on the same real
volume, 80 chunks compressed 2.69–5.23×, but break-even bandwidth was only
0.03–0.12 GB/s versus a roughly 12 GB/s PCIe link. The resident histogram
experiment found raw and compressed dispatch times similar, but scalar histogram
EMD was about 0.055–0.058, above that tool's own 0.02 visible-drift threshold.
Exact settled semantic evidence must therefore remain the owner.

## Revival gate

Compression returns to the active queue, and AUTO may flip, only after telemetry
first demonstrates repeated GPU-pool pressure, host-stage eviction cost, or a
remote/storage bandwidth limit. A bounded design must then satisfy all of these:

1. five fresh-process ABBA repeats on Intel and discrete NVIDIA, scalar and
   complex, with fixed power/viewport and identical total host/pool budgets;
2. real-Wayland `profile_montage_workflow` and journey-matrix pixels, final LOD,
   histogram, and levels equal to OFF/RAW within an explicitly accepted lossy
   display contract;
3. cold first pixels and each interaction settle within 2 s (hard failure 5 s),
   GUI callback under 50 ms, heartbeat gap at or below 16 ms;
4. current physical pool allocation, growth-copy bytes, and high-water slack—not
   encoded payload length—demonstrate a memory win; mixed raw fallbacks cannot
   strand required pages;
5. resident-to-resident LOD stays on GPU with zero readback;
6. host compression runs off the GUI thread and proves that tier recoveries avoid
   misses at the actual expensive owner under one measured RSS budget.

If revived, candidate work belongs in this order: consolidate the live
codec-policy owner; lazily activate codec pipelines after first pixels;
prepare/encode payloads off the GUI thread; add compressed-pool sampling to GPU
LOD; make the demand-sized arrays obey one ResourceGovernor byte cap and compact
at safe idle points; then evaluate a compressed tier at the StageCache/
materialization owner. Until a trigger exists, deeper optimization would solve
a hypothetical problem and compete with the measured first-pixel/promotion work.

## Validation record

- Default offscreen suite: 2,916 passed, 64 skipped, 1 expected xfail.
- Focused real-device GPU suite: 170 passed, including Intel ASTC, NVIDIA BC,
  compressed histogram/levels, LOD, physical-quality, and byte-accounting
  oracles.
- The full journey matrix was attempted on the logged-in Wayland compositor and
  in the repository's managed Weston compositor. No journey instance started:
  the unchanged session fixture requires a 1400×940 window *and* a 739-pixel
  viewport. The logged-in compositor produced 1400×948/739; decoration-free
  Weston produced 1400×940/731. All three backends failed the same fixture-
  restore precondition before rendering. This is a loud harness-geometry blocker,
  not pixel evidence for or against this change; the revival gate therefore
  continues to require a successful matrix before any AUTO claim.
