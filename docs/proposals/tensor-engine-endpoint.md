# Tensor exploration engine — endpoint architecture

**Date:** 2026-07-15. **Status:** direction record (user brainstorm, digested);
the executable program stays [`gpu-engine-plan.md`](gpu-engine-plan.md)
(G-series). This document holds the endpoint the G-stages converge toward
and the ideas banked for later stages, so they are designed-for rather than
retrofitted.

## The endpoint in one sentence

A real-time, deadline-driven tensor exploration engine whose CPU, GPU,
memory, storage, and UI layers cooperate under one scheduler: the GUI
submits immutable view intent; a backend-independent resource broker
compiles it into deadline-aware CPU, transfer, compute, and render work.

**Repo constraint:** ADR 0053 forbids scheduling systems beside the kernel.
The broker is therefore specified as the *kernel's evolution* — its
priorities/lanes/supersession enriched with deadlines, quality-per-
millisecond, and cost/locality estimates — never a parallel scheduler.
GUI-thread pacing and heartbeat work is owned by the main-branch P-program;
this branch designs for it but does not duplicate it.

## Architecture target

```
Qt GUI thread — input, controls, latest immutable view state,
                composition of the latest completed frame ONLY
      ↓ latest-state mailbox (replaces obsolete requests; no queues)
Frame planner / resource broker (the evolved kernel)
      — frame deadlines, quality policy, task graph,
        CPU/GPU/I-O/memory budgets, CPU-token grants
      ↓                       ↓
CPU execution            GPU execution
(native threads,         (render/compute thread, transfer
 worker processes,        queues, virtual residency)
 shared arenas)
      └────── chunk store (compressed/raw backing) ──────┘
```

- The GUI receives one statement per frame — "generation N has a newer
  presentable snapshot" — not per-tile signals.
- Tasks carry descriptors: result key, view epoch, deadline, quality gain,
  cost estimates (CPU/GPU/bytes), locality key, dependencies, cancellation
  scope. Scheduling ≈ urgency + visible quality per ms + reuse probability
  + locality − memory pressure − eviction cost.
- Cancellation is hierarchical (document → operation → view epoch → frame /
  quality / refinement / speculative scopes) and **cancels presentation
  ownership, not useful computation**: a stale-view task producing a
  content-keyed chunk keeps the chunk (this already works today via
  content-keyed residency; the scopes formalize it).

## Operation classification (identity now, execution later)

Every operation is classed — the class drives *where* it executes:

| Class | Examples | Execution |
|---|---|---|
| coordinate metadata | crop, flip, permute, stride, index offset | coordinate transform; never materializes |
| shader-on-read | conjugate, component, magnitude, phase, log, levels, LUT | fragment/compute on read |
| derived-chunk compute | LOD reduction, smoothing, masking | compute into content-keyed chunks |
| reduction | mean, RSS, histogram, ROI stats | compute reduction, small readback |
| global transform | FFT, large transpose | cost model chooses CPU/GPU |
| opaque | arbitrary Python callback | CPU materialization |

Chains fuse lazily (`conjugate → magnitude → log → levels` = one shader);
materialize only at barriers (reuse, reduction, opaque follower, locality),
decided by `on-read cost (pixels × op cost × redraws)` vs `materialization
cost (reads + compute + writes + storage)`. A flip must never copy an array.

## Frame-quality controller

Each frame maximizes quality before its deadline instead of draining queues:
Q0 reprojected previous frame (world-anchored tiles + camera transform
already give most of this) → Q1 coarse valid coverage → Q2 target LOD,
rough levels → Q3 refined levels → Q4 finer-than-screen/analysis-ready.
Frame-budgeted submission caps upload bytes, page-table updates, dispatches,
and planning time per frame; near-deadline background submission stops.
Adaptive-bitrate rules apply: never start refinements that will be obsolete
before completing; degrade aggressively during motion; recover on settle.

## CPU execution domains

- Native/GIL-releasing numerical work → threads over immutable chunks.
- Python-heavy work → persistent processes receiving arena descriptors
  (arena id, offset, shape, strides, dtype, generation) — never pickled
  arrays. A chunk-aware arena allocator sits above shared memory.
- Free-threaded Python (3.14+): keep compatible, benchmark, do not require.
  Subinterpreters: isolated plugin work only (pickle boundary).
- **CPU-token broker** prevents nested oversubscription (workers × FFTW ×
  BLAS × codec threads); it chooses "one 8-thread FFT" vs "eight 1-thread
  FFTs" per mode (interactive / idle-convergence / server-NUMA). The
  `fft_workers_tile=1` finding on the reference laptop is the seed datum.
- Endpoint: a small native data plane (Rust/C++/C) owns arenas, hot
  queues, kernels, GPU submission, page tables, compression, telemetry;
  Python submits coarse batched plans. Never one native call per tile.

## GPU endpoint items (feed G5/G6)

Virtual-texture feedback buffers (GPU appends missing-page requests at
tile/workgroup granularity; CPU schedules next frame); one batched
instanced/indirect montage draw with GPU-side culling and page resolution;
GPU histogram via workgroup-local bins → merge (never per-fragment atomic
contention), aggregated from the ADR 0056 coverage frontier; ROI/profile
reductions returning small results; GPU-generated LOD from resident chunks.

## Precision tiers (extends ADR 0056 keys)

L0 semantic source: exact dtype. Fine display cache: FP32. Coarse display
LOD: FP16/bounded quantization. Histogram summaries: level-selection
accuracy. ROI/export/analysis: exact source or explicitly-accurate derived.
Always explicit in the chunk `representation`/reducer key, always testable,
never silently substituted into scientific results.

## Adaptive compression (feeds G7)

Measure hardware topology once per device/driver (memcpy, upload/readback
bandwidth, codec throughput, CPU/GPU contention; the roadmap X5e evidence
gate is the natural home) and choose per chunk from a codec ladder: raw →
reduced precision → fast lossless → sparse/constant/run → error-bounded
(ZFP-class) → GPU-native backing. Bypass when raw wins. UMA policy: raw
shared pages until memory pressure; compression contends for the same DRAM
bandwidth as rendering. Discrete policy: compress when ratio × throughput
beats PCIe; best fused into passes that already touch the data. nvCOMP-class
vendor codecs stay optional accelerations behind a portable codec interface.

## Fragmentation by design (feeds G5/G7 physical layer)

Lifetime-segregated fixed-slot pools (pinned coarse fallback / hot visible /
speculative / transient scratch; by storage format, never by LOD);
persistent staging rings fenced per frame-in-flight; generational indirect
handles; page-table indirection makes idle, bandwidth-budgeted compaction
possible (allocate → copy → atomic entry swap → retire after fence). CPU
side mirrors it: few large arenas, recycled slots, NUMA-aware placement.

## Renderer strategy (updated 2026-07-15 after the Datoviz/alternatives study)

Keep VisPy as the present integration layer; keep engine logic out of it
(ADR 0055 §5). The renderer decision is now gated by a three-gate framework
proven in the Datoviz study: (1) GPU-native Qt composition (no framebuffer
readback), (2) custom shaders/compute, (3) zero-copy uploads
(caller-owned/persistently mapped staging; no per-request internal copy).

**Study verdicts:**

- **Datoviz** — shaders GO (low-level DRP, precompiled SPIR-V; avoid the
  high-level visuals), Qt backend NO-GO today (synchronous
  `server_grab()` CPU readback inside Qt event handlers; "experimental
  (and slow)" by its own label), uploads FAIL gate 3 today (documented
  internal copy per request), compute not yet in the public protocol.
  WAIT on upstream interop answers (the eight questions are recorded in
  the study); its request-protocol philosophy still aligns with ours.
- **wgpu-py + rendercanvas** — best immediate experiment: real
  `QRenderWidget` for PySide6, WGSL render+compute pipelines, binding
  arrays/non-uniform indexing/timestamp queries; no hardware sparse
  binding (software page table only — which we already prefer, ADR 0056);
  must field-test `present_method="screen"` vs bitmap on Wayland/docks.
  Use wgpu-py directly, NOT Pygfx (no second scene graph; the VisPy
  lesson).
- **Qt 6 QRhiWidget + native runtime** — likely production architecture:
  Qt owns window/composition/resize/DPI into a composited texture;
  `beginExternal()`/`endExternal()` lets a native ArrayScope runtime
  record Vulkan/Metal/D3D commands inside Qt's pass. Costs: C++/Rust
  extension work, and QRhi classes are Qt::GuiPrivate (no minor-version
  compatibility guarantee — per-Qt-version builds). Sparse via native
  external commands is a gate to prove, not assume.
- **Direct Vulkan + Qt** — the escape hatch with the highest ceiling
  (sparse binds, external memory, timeline semaphores, transfer queues,
  CUDA interop). Pay this tax only when measurement proves those needs;
  never raw Vulkan calls from Python — Python submits declarative batches
  to a native data plane.
- **ModernGL / direct OpenGL** — pragmatic low-risk fallback AND the
  control implementation: if a bespoke GL backend matches wgpu/Datoviz on
  the same architecture, the abstraction layer was the problem, not the
  API. Rejected: bgfx (game-oriented, non-turnkey Qt), Pygfx (scene graph
  we don't need), VTK (wrong abstraction level). CUDA/CuPy/Taichi are
  optional compute executors behind the planner, never the display
  architecture.

**The load-bearing decision — a backend-neutral semantic command protocol**
that never assumes WGSL, QRhi resources, Datoviz IDs, GL texture names, or
one-physical-texture-per-tile:

| Protocol command | Existing G-program seam |
|---|---|
| ensure chunk resident | `ChunkStore.ensure` / pool chunk plan (G1/G3b-2) |
| update page mapping | `PageTable` bind/remap (G1/G2) |
| update tile instances | draw parts / quad emission (G3c) |
| set display mapping | shader-mapping uniforms + physical-truth audit |
| dispatch histogram | G6 reduction (planned) |
| present generation | `TileCommitReport` acknowledgement + physical audit |

Formalizing this table into an explicit protocol module is the bridge from
the current VisPy executor to Experiments A/B and is a G-program stage in
its own right.

**Combined plan:** Experiment A = wgpu-py vertical slice (raw complex
chunks, page-table lookup, mixed-LOD fallback, compute histogram, instanced
montage, Qt pacing on Wayland). Experiment B = QRhiWidget narrow slice
(composited texture, one raw texture + shader + compute reduction, async
state handoff, native-commands coexistence). Multi-window shared residency,
DLPack zero-copy ingestion, and the GPU-service/remote split remain
endpoint items on whichever executor wins.

## The steering questions (apply at every design review)

View transform or data transform? Presentation precision or analysis
precision? Recompute or retain (include eviction + transfer costs)? Does
this task improve the next visible frame? Could the data already live
somewhere more useful? Are screen tiles / storage chunks / compute chunks
accidentally identical? Does the op justify GPU materialization? Are CPU
compression and GPU rendering fighting for the same DRAM? Does every queued
task remain useful after the latest interaction? Would this mechanism
survive VisPy's disappearance?

## Progression (mapped to the G-program)

1. **Near term (G4–G5):** operation classification in capability identity
   (landed with this document); batched per-frame completion signaling and
   staging rings as G5 physical-layer items; frame deadlines/quality
   controller specified as kernel enrichment (coordinate with the main-branch
   P-program before touching pacing).
2. **Next (G5–G6):** GPU as compute-aware raw-data backend — feedback
   buffers, batched instanced draw, GPU histogram/levels, GPU LOD pages over
   the virtual page table with pinned coarse fallback.
3. **Then:** renderer-protocol vertical slices (VisPy / Datoviz / wgpu or
   Vulkan), chosen by benchmark on integrated + discrete + low-power +
   workstation hardware.
4. **Endpoint:** native deadline-aware data plane; software-defined sparse
   virtual tensor; optional hardware-sparse allocator; adaptive compression;
   CPU/GPU cost-based planner with learned statistics; shared multi-window
   and external-GPU residency; optional remote renderer.
