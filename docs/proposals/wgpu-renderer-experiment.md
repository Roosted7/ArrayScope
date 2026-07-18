# wgpu renderer experiment (gate B) — tiered plan

**Date:** 2026-07-18. **Status:** active experiment; this document is the
plan and the evidence record. **Decision context:** Thomas, 2026-07-18 —
the Datoviz route requires Datoviz modifications, C++ bridge code, and our
own binary distribution before it answers ArrayScope's architectural
questions; wgpu-py answers them from pure Python wheels now. wgpu-py is
therefore promoted to the **primary** renderer experiment. Datoviz is
parked (not buried): gate A ran to completion and its verdict is recorded
on branch `codex/datoviz-v04-renderer-gate-a` in
`docs/proposals/tensor-engine-endpoint.md` (§ Experiment A findings) —
composition (Qt overlays over the native child) and upload-lifetime
(no completion tokens) gates FAILED; shaders/compute/never-black passed at
the vklite level.

The three renderer gates are unchanged
([tensor-engine-endpoint](tensor-engine-endpoint.md)): (1) GPU-native Qt
composition including our overlay stack, (2) custom shaders + compute
including a two-pass histogram reduction, (3) upload path with a
completion/lifetime contract. Plus the standing invariants: never-black,
physical truth, journey-matrix non-regression.

## Critical review of the scout research (what we kept, what we corrected)

The 2026-07-18 wgpu scout report is a good starting point; four of its
claims did not survive contact with this machine or with the Datoviz gate-A
evidence:

1. **"The current rendercanvas screen path falls back to X11/XWayland" —
   true for stock rendercanvas, but NOT a platform wall.** Measured Tier-0
   (2026-07-18, `tests/artifacts/wgpu-gate-b/probe-native-wayland.json`):
   a wgpu Vulkan surface presents on Qt's **native Wayland** window from
   pure Python, 30/30 frames SuccessOptimal, both top-level and as a
   paint-less native child inside a layout. Ingredients: PySide6 6.11
   binds `QNativeInterface.QWaylandApplication.display()` (the real
   in-process `wl_display*`); under the wayland QPA `QWidget.winId()`
   returns the `wl_surface*`; the wgpu instance must be masked to Vulkan
   (`set_instance_extras(backends=["Vulkan"])`) because the GL backend's
   EGL re-init on Qt's display drew a fatal
   `wp_linux_drm_syncobj` protocol error from the compositor; and the
   widget must be strictly paint-less (`WA_PaintOnScreen` + null
   `paintEngine`) or Qt's SHM backing-store commits fight the Vulkan
   swapchain on the same surface. Caveats that keep this an experiment
   result, not a shipped fact: `winId == wl_surface*` is undocumented
   Qt behavior (verify per Qt minor); rendercanvas's Qt backend also
   force-sets `QT_QPA_PLATFORM=xcb` at **import time** on Wayland
   systems — an integration hazard ArrayScope must guard regardless
   (import order / pre-set env), and a small upstream patch opportunity.
2. **The dock warnings are mostly noise for us; the real gate-1 risk is
   the overlay stack.** We never undock the main viewer (the Datoviz
   harness confirmed the viewer stays a fixed tab while Inspection/
   Operations panels float). But Datoviz gate A *proved* the actual
   killer: Qt widgets/QGraphics items cannot composite **above a native
   child** (xcb compositor capture: 0 overlay pixels vs 4,867 in the
   backing store; same stacking on Wayland). `present_method="screen"`
   sets `WA_PaintOnScreen` → implies `WA_NativeWindow` → the wgpu canvas
   has the same problem — on Wayland the child is a `wl_subsurface`
   stacked above the parent's buffer. So the Tier-1 question is not
   "do docks glitch" but: **which presentation mode keeps
   tile-truth/ROI/coach-mark overlays working, at what cost?** Three
   candidate answers, in preference order: (a) screen mode + overlays
   migrated into the wgpu scene (same migration Datoviz would force —
   but in WGSL/Python, not C++); (b) screen mode + Qt overlays in a
   sibling transparent native child... (on Wayland a transparent
   wl_subsurface above wgpu's; speculative, test only if cheap);
   (c) bitmap mode — overlays keep working exactly as today because the
   canvas is an ordinary QWidget painted with QPainter; costs one
   GPU→CPU readback per frame, which Tier 1 must price at real sizes.
3. **"Bitmap presentation is not the endpoint for 60–120 Hz" — assumed,
   not measured, and UMA changes the arithmetic.** This laptop's iGPU
   (and every UMA machine) does the readback over shared DRAM, and the
   adapter exposes `mappable-primary-buffers`. At our actual canvas size
   (~1300×650) the copy is ~3.4 MB; VisPy's whole pipeline today spends
   ~10 ms per scrub-step render. Tier 1 measures bitmap end-to-end
   before anyone declares it disqualifying. A 4K stress point bounds
   the discrete-GPU/external-monitor case.
4. **"272 draw calls bad" — already our design.** The G-program's page
   table + draw-parts substrate is exactly the "one instance buffer, one
   draw" shape; the protocol table in tensor-engine-endpoint maps 1:1.
   Tier 2 exists to prove the shape on wgpu, not to discover it.

Kept from the research unchanged: capability tiers (portable / fast-native
/ future-native-Vulkan), WGSL as the protocol shader source (also DRP2's
mandatory format — preserves a Datoviz/Vulkan migration path), software
page pool over hardware sparse (ADR 0056 already prefers software
indirection), the upload-belt/submission-serial design sketches, and the
warning that external-memory/CUDA interop is a fork-level wall in wgpu —
if that ever becomes a core requirement it is a point for the parked
Datoviz/native-Vulkan route (graveyard retry condition).

## What wgpu must beat

VisPy today: GPU levels/LUT/complex-mode via uniforms (zero-upload mode
switches already work), chunked content-keyed residency, but gloo has **no
compute** (G6's GL interim is a fragment-pass ladder), no binding arrays
(per-tile texture binds), readback-free QOpenGLWidget composition, and no
explicit submission/completion contract. wgpu's offer: real compute, one
instanced draw over a page pool, timestamp queries, an explicit queue —
*if* composition and uploads hold up. Datoviz gate A set the comparison
bar: five mapping/levels switches with `texture_uploads == 1`, an exact
two-pass 65,536-sample histogram, 0.0 black fraction throughout.

## Environment

conda env `arrayscope` (`/home/thomas/miniconda3/envs/arrayscope`):
`wgpu 0.31.1` (wgpu-native via wheel, no compiler), `rendercanvas 2.7.0`,
PySide6 6.11.1, Qt platform wayland (xcb comparisons via XWayland).
Harness: `experiments/wgpu_gate_b/`; evidence:
`tests/artifacts/wgpu-gate-b/` (JSON committed, PNGs as noted).
GPUs: Intel TGL UHD (default; UMA) and NVIDIA RTX A2000 Mobile
(`__NV_PRIME_RENDER_OFFLOAD` — known slower to first frame here).

## The tiers

Each tier has a hard exit gate; a tier that fails stops the ladder and the
failure is recorded here (a killed experiment gets a graveyard row citing
this document).

### Tier 0 — adapter + native-Wayland probe  ✅ PASSED 2026-07-18

Gate: the Vulkan adapter exposes the features the design needs, and a
screen-presentation path exists on this machine that does not regress the
app off native Wayland.

Result (`probe-native-wayland.json`, `probe_adapters` output in this doc's
commit): Intel TGL Vulkan exposes texture-binding-array,
partially-bound-binding-array, sampled-texture-and-storage-buffer-array-
non-uniform-indexing, mappable-primary-buffers, multi-draw-indirect-count,
shader-f16, float32-filterable, timestamp-query (+ inside encoders and
passes), subgroup{,-barrier,-vertex}, push-constants; limits: 16384 max
tex dim, **2048 texture-array layers**, 2 GiB buffers, 1024 workgroup
invocations. Native-Wayland screen presentation: see critical-review §1.
NVIDIA A2000 enumerates as a second Vulkan adapter (multi-adapter
selection stays a Tier-1 measurement, not a default).

### Tier 1 — Qt presentation + overlay gate (the decisive tier)  ✅ PASSED 2026-07-18 (bitmap; screen = priced escape hatch)

Question: on real Wayland, inside a dock-and-tab layout with ArrayScope's
overlay reality (tile-truth boxes, ROI handles, coach marks) composited
over the canvas, which presentation mode is viable and what does each cost?

Matrix (one harness, `run_gate_b.py`, mirroring the gate-A journey where
it applies — viewer never floats; panels dock/undock around it):

| Axis | Values |
|---|---|
| present mode | `screen` (native child) / `bitmap` (QPainter) |
| session | native Wayland / xcb (XWayland) for the screen mode |
| canvas size | ~1300×650 (real layout) and 3840×2160 stress |
| overlay | Qt overlay widgets over the canvas; do they show? |
| journey | tab hide/restore, panel float/re-dock, resize, second window |

Measurements per cell: GUI-thread paint/present time (mean/p95), bitmap
readback time (map wait + copy) at both sizes, dropped/error frames,
overlay-pixels-visible (screenshot oracle, the gate-A method), resize
stalls.

Exit gate: at least one mode passes ALL of — overlays visible OR a priced,
bounded overlay-migration story; steady GUI-thread cost within the 16 ms
heartbeat bar at real size; no black/stale frames across the journey.
Record the honest comparison: if only bitmap passes composition, its
readback price at 4K decides whether "screen + GPU overlays" becomes the
committed follow-up work.

### Tier 2 — minimal virtual tensor (the architecture proof)  ✅ PASSED 2026-07-18

RG32F page pool as one 2D-array texture; integer page table in a storage
buffer; per-tile instance buffer; ONE instanced draw; WGSL shader doing
page lookup → complex mapping (mag/phase/real/imag) → levels → LUT.
Verify, with upload counters as the oracle (gate-A parity):

- mode and levels switches: **zero** uploads, distinct rendered output;
- montage index scroll within resident pages: instance/page-table buffer
  writes only, zero texel uploads;
- window shift 100:200 → 101:201 with resident chunks: zero texel uploads;
- absent page → coarser resident ancestor renders (never black), page
  fills later without re-binding the world.

Exit gate: all four oracles green + a frame is one encoder with ≤3 passes.

### Tier 3 — compute (the G6 unblocking tier)  ✅ PASSED 2026-07-18

64-bin workgroup-local histogram over resident pages → merge pass → small
readback (gate-A parity: exact count, two passes); LOD reduction
(mean/max) from L0 pages into a coarser pool level via compute. Measure
pass GPU time (timestamp queries) + readback latency; verify bins match
the CPU reference exactly (integer source) / within float tolerance.

Exit gate: exact histogram in ≤2 passes inside a frame budget slice;
reduction writes a usable coarser page; G6 shader work can be written
against this shape via the backend-neutral command protocol.

### Tier 4 — upload paths (the gate-3 tier)  ✅ PASSED 2026-07-18

Compare on the iGPU (UMA), per batch shape (many 256² pages vs contiguous
batches vs whole plane): `queue.write_texture`, staging-buffer ring →
copy-to-texture, and `mappable-primary-buffers` direct writes; measure
effective GB/s, CPU copy time, submission time, and frame-time
disturbance while presenting. Prove a completion contract:
`on_submitted_work_done` → staging-slot + page-slot recycling (the thing
Datoviz could not give us).

Exit gate: an upload path exists that sustains scroll-rate page traffic
inside the frame budget WITH a working completion token, or the measured
shortfall is recorded as the wgpu route's gate-3 failure.

## Go/no-go (adapted from the scout report to our gates)

Choose wgpu as the production direction when: a Tier-1 mode passes
composition at bounded cost; Tiers 2–4 gates green; Python records only a
few passes/frame (no per-tile calls); no current requirement needs
external GPU memory. Fall back to parked Datoviz/native-Vulkan when:
composition fails in both modes; uploads or readback dominate at real
sizes; or explicit memory/queues/external interop become measured
requirements (those remain wgpu fork-level walls).

Non-goals now (endpoint items, designed-for only): multi-window shared
residency, DLPack/CUDA ingestion, remote rendering, hardware sparse,
compression (G7). The engine model stays renderer-neutral: everything
lands behind the semantic command protocol
([tensor-engine-endpoint](tensor-engine-endpoint.md) table); WGSL is the
shader source; a Datoviz/Vulkan executor stays constructible.

## Evidence log

All artifacts under `tests/artifacts/wgpu-gate-b/`; the gate-level verdict
table lives in [tensor-engine-endpoint](tensor-engine-endpoint.md)
(§ Experiment B findings). All measurements: Intel TGL iGPU, experiment
scale, single machine.

- 2026-07-18 Tier 0: adapter feature/limit probe + native-Wayland screen
  presentation PASS (`probe-native-wayland.json`,
  `experiments/wgpu_gate_b/probe_native_wayland.py`).
- 2026-07-18 Tier 1 (`run_gate_b.py`; `tier1-*.json` + PNGs):
  - **bitmap / native Wayland**: steady GUI-thread frame 3.96 ms p50 /
    8.74 ms p95 / one-time 447 ms first-frame pipeline compile at
    1300×650; dock float/re-dock 4.1–4.2 ms; tab restore 4.5 ms; resize
    4.1 ms; two windows 6.3 ms; overlays present in the composited
    backing store (5,482 magenta px — for bitmap the backing store IS the
    composited buffer). Offscreen readback price: 7.0 ms @ 3.4 MB
    (1300×650), 26.0 ms @ 33.2 MB (4K) → bitmap is comfortable at real
    size, fails 60 Hz at 4K.
  - **screen-native / native Wayland**: full journey 425/425 acquires
    SuccessOptimal, zero surface errors incl. tab hide/restore and a
    second surface on the same device; encode+submit 0.6 ms, present
    0.08 ms; Fifo acquire blocks ~15 ms (vsync pacing — Mailbox or
    off-thread acquire before production use).
  - **screen-stock / xcb**: journey clean; overlay discrepancy
    reproduced (backing store 5,416 px vs 0 on-screen — the Datoviz
    gate-A signature). Compositor-side capture is impossible on this
    GNOME Wayland session (shell screenshot API denies external callers;
    X grabs return black under rootless XWayland), so the Wayland
    overlay-stacking claim rests on wl_subsurface protocol semantics
    plus this xcb evidence and Datoviz gate A's captures.
- 2026-07-18 Tier 2+3 (`virtual_tensor.py`; `tier23-virtual-tensor.json`):
  ALL oracles green — physical truth vs CPU mirror (max diff 1/255);
  mode/levels/window-shift/montage-scroll all ZERO uploads; ancestor
  fallback black fraction 0.0; refill exactly 1 upload then exact; GPU
  LOD reduction max err 4.8e-7 (in-pool storage write via disjoint
  subresource views); two-pass histogram EXACT 1,048,576 samples, max
  bin diff 0, 3.2 ms wall incl. readback; one render pass per frame.
- 2026-07-18 Tier 4 (`upload_bench.py`; `tier4-upload-bench.json`):
  16-page burst 2.6 ms mean / 8 MB fenced (3.2 GB/s); per-page-fenced
  anti-pattern 0.14 GB/s; whole-plane 2.4 GB/s; staging-ring re-map wait
  0.11 ms p50 after fence; `on_submitted_work_done` token round-trip
  0.19 ms; mappable-primary direct write 12.4 GB/s (2× CPU memcpy
  baseline) — the UMA zero-copy candidate. Completion contract: PROVEN.

## Verdict (2026-07-18)

**GO.** All four tiers passed their exit gates; the three renderer gates
pass at experiment scale (gate table in tensor-engine-endpoint.md). G6
shader work is unblocked against wgpu through the backend-neutral command
protocol. Production adoption still requires: journey-matrix integration;
a bitmap-vs-screen policy (bitmap default at real sizes; screen + GPU
overlays when readback cost bites, e.g. 4K); the standing caveats above
(winId contract pinning, rendercanvas import hazard, Mailbox acquire,
NVIDIA/discrete measurements).
