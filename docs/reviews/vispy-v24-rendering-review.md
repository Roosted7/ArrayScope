# ArrayScope v24 rendering review

> **Historical review.** This document records a dated evidence set. Use [`../current-state.md`](../current-state.md), [`../architecture.md`](../architecture.md), and [`../roadmap.md`](../roadmap.md) for current guidance.

## Executive verdict

The project has changed course in the right direction.  v24's typed tile payloads and one batched
atlas visual are a much better foundation than either a monolithic montage texture or one VisPy
visual per tile.  The correct response is **not** to throw away the VisPy work and **not** to make it
the only backend yet.  Keep both renderers while moving toward a shared semantic display model and
thin backend adapters.

The VisPy path is not universally better today because it is still a hybrid renderer.  PyQtGraph
owns pointer interaction and much of the widget shell while VisPy owns pixels and mirrored overlays.
That preserves behavior, but it keeps two scene systems active, sends camera changes across a bridge,
and leaves final Qt OpenGL composition on the GUI-facing surface.  GPU use alone does not guarantee
lower latency.

This review branch fixes several correctness and memory problems, improves requested interaction
parity, makes tiled presentations first class, adds useful diagnostics, and replaces misleading
submission-only benchmark conclusions with frame-latency and event-loop measurements.  It does not
claim that the remaining architecture is finished.

## What v24 got right

### Typed tile payloads are the right shared unit

`DisplayTilePayload` carries tile identity, source identity, display pixels, and optional scalar
histogram data.  Both backends can consume that contract.  Hover, ROI, and profile reads can use a
committed `TiledValueSource` rather than pretending that a giant montage canvas exists.

That is the most important architectural improvement in v24.  It separates semantic data ownership
from a renderer's storage strategy.

### One batched VisPy visual is the right immediate renderer shape

Replacing one visual per tile with one atlas-backed batched visual removes per-object scene-graph
management and makes clean commits capable of doing no texture work.  It also creates a path toward
persistent residency, page-level draw calls, and shader-side display mapping.

### The hybrid interaction strategy was a sensible experiment

Keeping PyQtGraph interaction avoided a risky rewrite before VisPy's value was known.  That decision
was reasonable.  It should now be treated as a migration scaffold, not as the final architecture.

## Critical findings in the received v24 code

### 1. Atlas slots could display the wrong tile

Slots were reassigned from sorted active tile order on each commit, but unchanged tiles were not
re-uploaded when their slot changed.  For example, changing the active set from `{1, 2}` to `{2, 3}`
could make tile 2 point at the old contents of tile 1's slot.

The branch replaces positional reassignment with stable tile-to-slot ownership and LRU eviction of
inactive tiles only when capacity is required.

### 2. Progressive loading repeatedly rebuilt the atlas

Atlas capacity followed the number of currently supplied payloads.  Progressive batches therefore
allocated a larger texture, discarded residency, and re-uploaded existing tiles repeatedly.  The
renderer now reserves the complete non-skipped visible set, including tiles that are still
`unloaded`, and uses amortized growth only when no complete plan exists.

### 3. The “GPU atlas” retained full CPU shadow atlases

The original pool kept full float scalar and RGB CPU atlases in addition to source tile payloads and
GPU textures.  This was a direct memory regression.  Storage is now allocated by shape on the GPU and
only a tile-sized staging array is created when an input plane is incompatible or non-contiguous.

### 4. Both full texture planes were allocated even when one was unused

Scalar tiles do not need an RGB atlas.  Already-windowed RGB tiles do not need a float scalar atlas.
The pool now has scalar, color, and scalar-plus-color storage modes.  This changes common storage from
7 bytes per atlas pixel to 4 or 3 bytes.

### 5. Residency was cleared instead of retained

Inactive slots and source identities were discarded during normal active-set changes.  The revised
pool separates draw visibility from storage residency, retains offscreen tiles, and reports LRU
evictions explicitly.

### 6. The direct path was unnecessarily VisPy-specific

PyQtGraph already understood typed tile payloads, but planning selected the direct path by backend
name.  Both backends now use the same tiled presentation contract, selected through declared backend
capabilities.

### 7. Tiled presentations were still represented as fake raster images

A broadcast one-pixel placeholder preserved a legacy widget call shape, but allowing it into the core
presentation model made tiled semantics fragile.  Raster and tiled presentations are now distinct
classes.  A placeholder exists only at the old widget boundary where required.

### 8. Progressive commits recreated metadata wrappers

Every batch rebuilt source identities and typed payload wrappers for all loaded tiles.  The montage
session now retains immutable wrappers and invalidates only replaced tile entries.  This is still not
the final delta protocol, but it removes avoidable UI-thread object churn without duplicating pixel
arrays.

### 9. HUD and ROI panels could sit behind the OpenGL surface

Status, evaluation, and ROI widgets were parented to the hidden PyQtGraph surface.  With the VisPy
canvas as a sibling, especially under Wayland composition, they could be obscured.  They are now
parented and mapped through the stacked display container.

### 10. Existing interaction features were incomplete rather than entirely absent

Line endpoints existed in ROI data but were ignored by hover hit testing.  Handles were repeatedly
destroyed and recreated.  Freehand drawing had no backend preview hook.  The branch adds shared,
Qt-free hit testing and reusable VisPy visuals for:

- live-profile center marker;
- rectangle resize corner;
- line endpoint handles;
- polyline/freehand vertex handles;
- handle-before-outline hit priority;
- hover highlighting and cursor changes;
- real-time drag/draw visual updates;
- reusable freehand/polyline preview;
- stale-hover cleanup when tools change.

The status HUD and ROI information panel already existed in shared code; the stacking fix makes them
usable above the VisPy surface.

## Performance assessment

### Why VisPy can still feel slower

The present path is not “the UI thread plus an independent renderer thread.”  It is a Qt widget with a
VisPy OpenGL surface whose final update and window composition remain coordinated through Qt.  Pixel
computation can happen elsewhere and shared contexts can prepare resources, but the visible widget
still has GUI/render-loop scheduling and compositor constraints.

At the same time, the hybrid view keeps PyQtGraph's interaction scene alive.  Panning therefore
involves PyQtGraph range changes, coalesced camera synchronization, a VisPy redraw request, and Qt
surface composition.  PyQtGraph can be surprisingly competitive because `ImageItem` has mature
fast paths and optional display downsampling.

A reported 50% “Renderer/3D” value is not a useful optimization target by itself.  It can represent
frame pacing, vertical synchronization, compositor behavior, command-submission limits, texture
bandwidth, or an otherwise idle GPU waiting for the next frame.  Optimize missed frame deadlines,
first-frame latency, interaction gaps, upload bytes, and draw submissions—not a utilization gauge.

### What should move to shaders next

The highest-value display-only move is complex presentation:

1. Upload display `complex64` tiles as an `RG32F` texture.
2. Compute real, imaginary, magnitude, phase, log/symlog intensity, and LUT lookup in the fragment
   shader.
3. Keep exact semantic values on the CPU/evaluator side for hover, ROI statistics, exports, and any
   operation requiring full precision.
4. Change window, brightness, contrast, phase palette, and display component through uniforms.

This avoids repeated CPU phase/color preparation without coupling the operation engine to a GPU
library.  FFTs and reductions are a different problem: moving them is worthwhile only with a coherent
CuPy/other device execution backend that keeps intermediate arrays resident end to end.

### Level of detail is required for very large views

The proposed “roughly twice screen resolution” idea is sound, but it should be implemented as a tile
pyramid or mipmapped representation rather than downsampling the complete montage every frame.
Choose a level near one to two source texels per display pixel, retain full-resolution source/value
semantics, and promote to finer tiles as zoom increases.  Atlas gutters or duplicated edge texels are
required before linear interpolation or mipmapping to avoid tile bleeding.

### The next residency design

The current pool is deliberately one page.  Production design should add:

- actual runtime texture-size limits rather than a hard-coded 8192 assumption;
- an explicit GPU byte budget derived from device capability and user policy;
- multiple atlas pages, with one batched draw per page initially;
- viewport-near retention and a wider hysteresis ring;
- visible, near, warm, and evictable residency classes;
- page/slot LRU diagnostics and upload-reuse counters;
- optional texture arrays where supported and proven beneficial.

Do not aggressively unload merely because a tile left the viewport.  Hide it from the draw list and
keep it resident until memory pressure or a higher-priority tile requires the slot.

### Progressive commits still need a delta protocol

Wrapper reuse is a useful intermediate fix, but every batch still builds active dictionaries and
several consumers scan the current set.  The next contract should be revisioned persistent state:

```text
TilePresentationDelta
  structure_revision
  upserts: tile_id -> payload
  removals: tile_id[]
  active_tiles: tile_id[] or visibility revision
  level_revision
  histogram_revision
  viewport_revision
```

The renderer would apply only the delta, while committed semantic state keeps a persistent map.  This
is more important for UI fan-in than adding more worker threads.

## Benchmark review

The old benchmark timed how quickly a Python setter returned.  VisPy defers GL work, so that number
could make a large texture upload appear almost free.  The revised harness reports separately:

- submission time;
- first observed draw/paint callback;
- event-loop drain time;
- frame count;
- maximum Qt heartbeat gap;
- texture/vertex submissions and uploaded bytes;
- resident/capacity/rebuild/eviction counters;
- estimated GPU storage and CPU shadow storage.

It also includes clean commit, one dirty tile, pan/zoom without upload, level preview, initial tiled
load, and progressive stream scenarios for both backends.

The former stress function loaded and transformed a NIfTI array, deleted it, and then ran unrelated
small synthetic scenarios.  It now generates the motivating production-scale workload directly:
272 tiles of 336×336 pixels, streamed in batches.  Run it with presented-frame measurement on every
supported platform/GPU combination; tune thresholds only after collecting repeatable distributions.

Example:

```bash
ARRAYSCOPE_RUN_STRESS=1 \
ARRAYSCOPE_BENCH_PRESENTED=1 \
python -m arrayscope.display.rendering_benchmarks
```

The size, count, columns, and batch size are overrideable with `ARRAYSCOPE_STRESS_*` variables.

## Recommended abstraction

Do create backend-specific folders eventually, but put only backend implementation there.  Do not
create VisPy and PyQtGraph copies of essential rendering, ROI, profile, or presentation files.

Recommended ownership:

```text
display/model/            semantic presentation, geometry, viewport, interaction state
display/backends/base.py  capabilities and narrow renderer protocol
display/backends/pyqtgraph/{surface,raster,tiles,overlays}.py
display/backends/vispy/{surface,raster,atlas,shaders,overlays}.py
display/widget.py         histogram/HUD/ROI panel/signals shared by both
```

The backend protocol should express ArrayScope intent, not mirror graphics-library APIs.  Examples:
`present_raster`, `apply_tile_delta`, `set_levels`, `set_viewport`, `set_overlay_state`, and
`diagnostics`.  It should not contain generic `add_visual`, `set_image_item`, or backend object access.

The current inheritance (`VisPyImageView2D(ImageView2D)`) can remain during migration, but composition
is the target: one shell owns semantic state and contains one pixel backend.

## Alternatives

### PyQtGraph only

Best current interaction maturity and a strong baseline.  It is simple to maintain, has optimized
image paths, and can downsample for display.  Its limits are CPU-side display preparation, texture or
QImage upload behavior, and less direct control over persistent tiled GPU storage and custom shaders.
Keep it as the reference/fallback backend until measurements justify removal.

### VisPy

Best fit for the current incremental GPU experiment.  It provides programmable visuals and direct GL
texture control while integrating with Qt.  Its deferred command model makes benchmarking subtle, and
the hybrid scene currently pays synchronization/stacking cost.  It remains the preferred backend for
pursuing atlas residency and display shaders.

### napari

Useful as an architecture reference: separate models/state from Qt widgets and VisPy rendering, and
keep rendering state testable.  Adopting napari itself would bring a much larger application/plugin
model than ArrayScope needs and would not remove the need to solve workload-specific FFT, montage,
and memory policy.

### Qt Quick scene graph / QQuickRhiItem

The strongest candidate if a true render-thread-oriented shell becomes necessary.  Qt Quick can use a
threaded scene-graph render loop, and `QQuickRhiItem` separates GUI-side item state from renderer-side
resources.  This is a substantial UI/surface rewrite, so first clean the semantic backend boundary and
then build a focused spike.

### wgpu / pygfx

Attractive for a modern explicit GPU API and portability.  Resource upload and lifetime semantics are
clearer, but adopting it means replacing the current renderer, Qt integration details, shaders, and
interaction bridge.  Evaluate it behind the same backend contract rather than rewriting the viewer
around it first.

### Custom OpenGL / ModernGL / Vulkan

Offers maximum control and maximum maintenance burden.  ArrayScope should not own this complexity
unless VisPy or Qt RHI proves unable to meet measured requirements.

## Roadmap

### Stage 1 — Stabilize and measure (this branch)

- Correct atlas identity/residency and eliminate full CPU shadow atlases.
- Make storage mode-aware and reserve the complete visible plan.
- Make tiled presentations first class and capability-selected.
- Restore basic ROI/profile hover and live visual parity.
- Expose upload/residency/storage diagnostics.
- Measure presented latency and event-loop starvation.

### Stage 2 — Persistent tiled state

- Introduce tile deltas/revisions rather than full active mappings per batch.
- Add multi-page, byte-budgeted residency and viewport-near retention.
- Query device limits and provide graceful page/fallback behavior.
- Add production benchmark baselines per platform, compositor, and GPU class.

### Stage 3 — LOD and complex shaders

- Add tile pyramids/mip selection with gutters and one-to-two texels per display pixel.
- Add complex `RG32F` display shaders and uniform-driven component/LUT/window controls.
- Retain exact CPU semantic value access for inspection and export.

### Stage 4 — Shared interaction controller and backend composition

- Extract pointer state, hit testing, capture, drag, and cursor decisions from QGraphics items.
- Make PyQtGraph and VisPy overlay renderers consume the same interaction state.
- Replace backend inheritance with one shared widget shell plus a backend adapter.
- Run one semantic conformance suite against both.

### Stage 5 — Decide with evidence

- Keep both if they win different deployment classes and maintenance stays low.
- Prefer one only when it has feature parity and wins representative latency/memory workloads.
- Spike Qt Quick/RHI or wgpu only if the remaining Qt/VisPy surface model is a measured blocker.

## Remaining risks

- The atlas is still one page and assumes a configured maximum texture size.
- Initial storage is reserved for the full visible montage; low-power devices need an explicit GPU
  budget before very large sessions are allowed to reserve indiscriminately.
- The VisPy canvas is still mouse-transparent and camera-following, not the native interaction owner.
- Complex phase/color data is still prepared on the CPU.
- Presented callbacks are better than setter timing but are not GPU timestamp queries or compositor
  scan-out measurements.
- Real Qt/Wayland/macOS/Windows behavior must be checked on systems with full dependencies.

## Validation performed in the review environment

- Python compilation of `arrayscope` and `tests`.
- Qt-free atlas, display contract, presentation, geometry, diagnostics, montage session, and hit-test
  tests.
- No real Qt/VisPy GUI execution was possible because those optional GUI packages were unavailable.

## Reference points

- PyQtGraph `ImageItem` performance and `autoDownsample` documentation.
- VisPy `ImageVisual`, `Texture2D`, canvas update, and complex image implementation documentation.
- Qt Quick threaded scene graph and `QQuickRhiItem` documentation.
- napari architecture and rendering-state testing documentation.
- wgpu queue texture-upload documentation.
