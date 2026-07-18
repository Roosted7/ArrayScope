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


## Renderer ranking revision (2026-07-18 — supersedes the 2026-07-15 WAIT on Datoviz)

The graveyard retry condition on Datoviz resolved: **v0.4-dev renders
directly into a Qt-created Vulkan surface** (Qt owns the window/event loop,
Datoviz owns rendering; no framebuffer readback), exposes the canvas
draw-callback with vklite + runtime shaderc from the wheel, and DRP2 models
the render/compute vocabulary our software virtual tensor needs. Full scout
report reviewed 2026-07-18; corrected facts: ArrayScope runs **PySide6**
(not PyQt5), so the Qt-bridge question is PySide6-vs-PyQt6 adaptation; and
we composite Qt overlays over the canvas today (tile-truth overlay, ROI
handles, coach marks), so the native-child-window matrix must include them.

Revised ranking: **(1) Datoviz v0.4-dev** for the real vertical slice,
**(2) wgpu-py** as the comparison implementation, **(3) direct Vulkan**
only if Datoviz extension boundaries obstruct. Run (1) and (2) as PARALLEL
judged slices with identical scenario + measurements (judge-panel pattern),
not serially. QRhiWidget+native remains recorded but is no longer the
presumed production candidate.

**The Datoviz experiment (10 steps, one worktree, pinned v0.4-dev commit):**
PySide6/PyQt6 shell branch → native-window container in the real dock/tab
layout → Wayland/xcb, floating docks, focus, cursors, multi-window, AND our
Qt-overlay stack → empty figure + replace canvas callback via
dvz_view_canvas()/dvz_canvas_set_draw_callback() → one instanced tiled
surface with a custom shader → RG32F complex upload → magnitude/phase/real/
imag + levels switching WITHOUT re-upload → one compute histogram →
measure GUI-thread time, submit time, upload time, offscreen→swapchain blit.

> **Superseded 2026-07-18 (later the same day):** gate A ran on branch
> `codex/datoviz-v04-renderer-gate-a` and FAILED the composition and
> upload-lifetime gates (Qt overlays cannot paint above the native child;
> no completion tokens) while passing shaders/compute/never-black at the
> vklite level — see that branch's copy of this document for the full
> verdict. Combined with the cost side (Datoviz modifications, C++ bridge,
> our own binary distribution before any architectural answer), Thomas
> inverted the ranking: **wgpu-py is the primary experiment**, per the
> tiered plan in [wgpu-renderer-experiment](wgpu-renderer-experiment.md).
> Datoviz is parked with retry conditions (supported hosted frame
> producer; per-submission completion tokens; a proven overlay-migration
> architecture; packaged PySide6 provider) — it remains the route if wgpu
> fails composition/uploads or external-memory/explicit-queue control
> becomes a measured requirement. The 10-step Datoviz experiment below is
> retained for the record.

**Known gaps to validate as blocking-vs-nice-to-have** (we would not lose
these over VisPy either — judge accordingly): supported hosted-canvas /
frame-producer insertion point (vs overwriting the scene callback);
submission-completion tokens (page-cache reuse needs them); persistent
mapped upload arenas (current APIs copy); compute breadth (storage-texture
writes, barriers, atomics, multi-pass reductions); GUI-thread render_once
cost; native-child-window compositing limits; Qt-bridge distribution.
Upstream-proposal order if the slice succeeds: hosted canvas/frame
producer, submission tokens, upload arenas, compute-to-texture, packaged
Qt provider, threaded hosted provider. **Application semantics never
upstream**: tile identity, N-D chunking, LOD families/reducers, complex
mappings, page policy, coverage frontier, histogram refinement, prefetch,
compression, operation planning — Datoviz is the execution substrate,
ArrayScope stays the tensor engine.

### Experiment B findings (2026-07-18) — wgpu-py: renderer gates pass at experiment scale

**Verdict: GO — write G6 shader work against wgpu via the backend-neutral
command protocol.** All three renderer gates pass on real hardware (Intel
TGL iGPU, native Wayland, PySide6 6.11 / wgpu-py 0.31.1 / rendercanvas
2.7.0, pure-Python wheels, no compiled code). Full tiered plan, critical
review of the scout research, and per-tier evidence:
[wgpu-renderer-experiment](wgpu-renderer-experiment.md); machine-readable
artifacts in `tests/artifacts/wgpu-gate-b/`.

| Gate | Result | Evidence / reason |
|---|---|---|
| Qt GPU composition | **PASS (bitmap), escape hatch proven (screen)** | Bitmap keeps ALL Qt overlays working by construction; measured end-to-end GUI-thread frame ~4.0 ms p50 / 8.7 ms p95 at 1300×650 through docks/tabs/resize/two windows on native Wayland. Screen mode: native-Wayland presentation from pure Python (winId-as-wl_surface + QNativeInterface wl_display + Vulkan-only instance), 425/425 acquires clean across the journey — but Qt overlays do not composite above the native child (same signature as Datoviz gate A), so screen requires the GPU-overlay migration. 4K bitmap readback is 26 ms — bitmap fails 60 Hz at 4K; that is the boundary where screen+GPU-overlays becomes the committed follow-up. |
| custom shaders + compute | **PASS** | Tier-2 virtual tensor (page pool array texture + page-table storage buffer + ONE instanced draw, WGSL): mode/levels switches, +1px window shift, and montage scroll all ZERO uploads and pixel-exact vs the CPU reference (tolerance 2/255, max diff 1); absent page renders the coarser resident ancestor (black fraction 0.0), refill = exactly 1 upload then exact. Tier-3: GPU 2×2-mean LOD reduction into the pool (storage-texture write, disjoint subresource views) max err 4.8e-7; two-pass 64-bin workgroup-local histogram EXACT — 1,048,576/1,048,576 samples, max bin diff 0, 3.2 ms wall including readback. |
| upload + lifetime | **PASS** | 16-page RG32F burst (8 MB) via `write_texture`: 2.6 ms mean fenced (3.2 GB/s) — inside a frame slice; per-page-fenced is the recorded anti-pattern (0.14 GB/s). `on_submitted_work_done` completion token round-trips in 0.19 ms after a burst — the recycling contract Datoviz lacked. `mappable-primary-buffers` direct writes: 12.4 GB/s (2× the CPU memcpy baseline) — the UMA zero-copy page-pool candidate. |
| never-black / physical truth | **PASS for the executed slices** | Every Tier-2 oracle compares the rendered framebuffer against the CPU mirror; ancestor-fallback black fraction 0.0; upload counter is the residency oracle throughout. |

**Standing caveats (recorded, not blocking):** `winId == wl_surface*` is
undocumented Qt behavior — pin per Qt minor and propose the surface hook
upstream to rendercanvas (which also force-sets `QT_QPA_PLATFORM=xcb` at
import time on Wayland — an integration hazard ArrayScope must guard);
screen-mode Fifo acquire blocks the GUI thread ~15 ms/frame (Mailbox or
off-thread acquire before any production screen use); compositor-side
overlay capture is impossible on this GNOME Wayland session, so the
Wayland overlay-stacking claim rests on protocol semantics plus the xcb
evidence; all numbers are experiment-scale on the Intel iGPU (NVIDIA
enumerates but was not driven); journey-matrix integration remains the
production gate.
