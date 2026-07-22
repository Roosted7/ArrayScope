# G7 compression live-benefit review — 2026-07-22

## Verdict

The codec components are real and mostly correct, but the previous evidence did
not establish a product benefit. The live AUTO defaults were premature.

- Lossless ZFP/Blosc2 compress real chunks well, but CPU encode + decode still
  loses the transport inequality by orders of magnitude.
- BC/ASTC reduce the bytes of an *active compressed page*, and the hardware
  sampler decodes them correctly, but the live executor also allocates the full
  raw fallback pool. Configured pool memory therefore grows rather than shrinks.
- Cold WGPU submission pays CPU encode plus a reference decode/quality pass for
  every page. Compressed-source LOD additionally leaves the GPU for readback,
  CPU decode/reduce, and re-upload.
- The host tier can retain more keys under one byte budget, but production had
  been given the full raw budget plus an equally large tier, real display aliases
  kept the primary array alive, and the benchmark charged a synthetic FFT to the
  wrong cache owner.

This review restores host `RAW` and texture `OFF` as defaults. Explicit codec
choices remain as experimental mechanisms. G7 closes with a measured NO rather
than consuming the active queue: compression is revived only when telemetry
shows a capacity or I/O bottleneck that it can actually remove.

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
  again for a quality check, and kept a complete raw fallback pool. It optimized
  encoded page length while worsening the two resources that matter now: the
  callback/first-pixel critical path and configured physical allocation.
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
selects BC. Timings are milliseconds.

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

Resident-hot histogram dispatch is approximately neutral; it was the only GPU
timing in the previous histogram tool. Cold submission is 8.5–138× slower and
compressed-source LOD is 8.4–42.5× slower. Intel complex AUTO reaches roughly
2.0 seconds for only 16 pages, consuming the whole interaction target before
presentation work is counted.

## Byte and residency interpretation

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

The active-page saving is real. The claimed 8× live VRAM-capacity saving is not:
full raw fallback arrays remain allocated beside full codec arrays, producing a
9–18% configured-pool increase in these cells. Host AUTO retained only zero or
one extra unique key while adding synchronous work; a larger, compressible
working set can improve retention, but the prior `40→91` figure double-counted
raw/tier overlap and its time win substituted a synthetic FFT for a display-cache
miss while the expensive StageCache remained separate.

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
4. configured pool allocation, not encoded payload length, demonstrates a memory
   win; mixed raw fallbacks cannot strand required pages;
5. resident-to-resident LOD stays on GPU with zero readback;
6. host compression runs off the GUI thread and proves that tier recoveries avoid
   misses at the actual expensive owner under one measured RSS budget.

If revived, candidate work belongs in this order: consolidate the live
codec-policy owner; prepare/encode payloads off the GUI thread; add
compressed-pool sampling to GPU LOD; replace parallel full pools with one
physically bounded capacity design; then evaluate a compressed tier at the
StageCache/materialization owner. Until a trigger exists, optimizing these
mechanisms would solve a hypothetical problem and compete with the measured
first-pixel/promotion work.

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
