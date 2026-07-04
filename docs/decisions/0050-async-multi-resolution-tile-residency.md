# 0050 — Asynchronous multi-resolution tile residency

**Status:** Accepted (2026-07). Implements the design separated by ADR 0041; first slice targets
VisPy montage/tiled scenes.

## Context

ADR 0041 split LOD into demand selection, materialization, and residency, and froze production at
`native-only` until all three existed. Demand selection has been live since: every montage commit
computes a `LodDemand` with hysteresis, and diagnostics report desired versus applied factor.

A baseline on real hardware (2026-07-04, Wayland/Intel-NVIDIA, `data/_WIPDelRec-tT2` 336×336×272
f64, 272-tile montage in a 1400×900 window) shows what native-only costs at this scale:

| Metric (vispy unless noted) | Native-only baseline | At desired factor 4 |
|---|---|---|
| Desired / applied LOD factor | 4 / 1 (every montage phase) | 4 / 4 |
| Estimated tile-layer GPU bytes (FFT scene) | ~280 MB | ~17.5 MB |
| Direct tile compute, raw montage | 541 ms (pyqtgraph 944 ms) | ~1/16 of texel work |
| Event-loop gap p95 during montage settle | 74–98 ms | bounded by smaller payloads |
| pyqtgraph 10× level drag on FFT montage | 8.36 s | scales with presented texels |

The montage shows ~5700×5400 source texels through ~1.3 M viewport pixels: sixteen times more
texel work than the display can express, in compute, conversion, upload, residency, and level
re-window cost. The 2026-06-19 synchronous prototype failed for placement reasons (pyramids built
in the presentation commit), not because the goal was wrong.

Prerequisite state: X5a telemetry exists; acknowledged residency (X5b) holds on the montage commit
path (`DisplayCommitter` commits through the backend, ADR 0044); montage viewport retargeting
exists. Viewport-scoped retargeting for internally tiled normal images (X5c) and region-first
materialization (X5d) remain open, so this ADR scopes enablement to montage/tiled-scene
presentations and keeps normal-image LOD gated behind X5c/X5d.

## Decision

Implement ADR 0041's three stages as separately owned components, VisPy first, sharing the
semantic planner and materializer across backends.

### Pyramid materialization (`LodMaterializer`, Qt-free)

A shared materializer produces reduced tile payloads outside any GUI or GL callback:

- **Key:** `(semantic source id, region/tile id, component representation, level_xy, algorithm
  version)`. The payload source-id scheme in `MontageRenderSession._payload_source_id` already
  carries `(lod factor, level, gutter, content token)`; materialized levels extend the same
  identity rather than inventing a parallel one.
- **Reduction:** power-of-two box mean per level, float32 accumulation, computed level-from-level
  (level *n+1* from *n*); per-axis levels so extreme aspect ratios reduce anisotropically.
  Complex payloads reduce components separately (RG32F planes), preserving the shader mapping
  contract. The algorithm version is part of the key.
- **Execution:** workers consume immutable snapshots; duplicate requests singleflight; requests
  are cancellable and supersedable; completed levels enter a bounded cache charged to the
  existing memory budget. Source-provided pyramids (ADR 0049 chunked/mmap sources) are preferred
  over recomputation when a source can serve a level directly.
- **Scheduling:** demanded levels for visible tiles run on the visible lanes with the frame
  deadline (`quality="preview"` for a coarser-than-desired level, `"exact"` once the demanded
  level is resident). Adjacent-level and near-viewport speculation runs on
  `SPECULATIVE_RESIDENCY`: bounded before admission, ordered coarse-to-fine outward from the
  viewport (and from the pointer anchor during zoom gestures), always superseded by visible work.

### Presentation policy (progressive, retained)

`native_lod_policy` gains a sibling `resident_lod_policy`: applied level = the resident level
closest to the demanded level within `acceptable_levels`, never blocking on materialization.

- The presented level stays presented until the replacement is backend-acknowledged resident
  (ADR 0044 semantics); a pending coarser request never clears a valid finer tile.
- Zoom-in shows the retained coarser level immediately while finer levels stream in per tile
  through the ordinary upsert path; zoom-out prefers an already-resident finer level over
  waiting (GPU filtering handles minification, as today).
- Promotion/demotion keeps the existing asymmetric hysteresis; repeated threshold crossings hit
  the pyramid cache instead of rebuilding (ADR 0041 gate 4).
- A level change with unchanged content never re-uploads source pixels for already-resident
  levels (gate 6).

### Residency (backend-owned)

VisPy: atlas pages are classed by `(level, texture shape, format, gutter)`. A reduced tile never
enters a native-shaped slot; each class accounts its own GPU bytes and eviction. Retaining one
adjacent level per tile is allowed when the budget permits. Backend-native mipmap generation
remains a later, gated option (edge handling, complex mapping, and memory accounting must be
proven first).

PyQtGraph: same planner and materializer; reduced `payload.image` is applied only where measured
scene savings exceed replacement cost. The first measured target is montage level re-window,
whose cost scales with presented pixels (8.36 s baseline above).

### Operations integration

Operations already declare capabilities and region behavior (ADR 0025/0026). Capabilities gain a
display-LOD contract:

- **`lod-commuting`** (pointwise maps, magnitude/phase/component selection, window/level): the
  evaluator may feed reduced input to produce a reduced display payload directly, cutting raw
  work by roughly the squared factor. Only display payloads may take this path.
- **`lod-transforming`** (crop, transpose, axis permutations): the demanded level passes through
  the region planner with mapped geometry.
- **`lod-opaque`** (FFT and other domain transforms): input reduction changes the result, so the
  stage runs at native resolution as today; the materializer reduces the *stage output* on the
  stage lane. LOD still removes the conversion/upload/residency cost even when it cannot remove
  the transform cost.

Exact-value consumers are untouched: hover, histograms, profiles, and ROI statistics read exact
or explicitly qualified sources (ADR 0041), through the existing region-limited evaluation paths.
A full-resolution region computed for an ROI or profile lands in the stage cache and is reusable
by later exact rendering, but display LOD never silently substitutes approximate values into
semantic inspection.

### Diagnostics and rollout

Existing desired/applied diagnostics stay; residency gains per-level counters (resident tiles per
level, pyramid cache bytes/hits, level transition traces, upload bytes per level). The policy is
selectable (`native-only` remains the fallback and the default for non-tiled scenes), and every
ADR 0041 acceptance gate maps to a test or a hardware trace before `resident` becomes the montage
default.

Rollout order: (1) materializer + pyramid cache, Qt-free, fully tested with the applied factor
still 1; (2) VisPy montage residency + progressive presentation behind the policy flag, validated
against the ADR 0041 gates on hardware; (3) PyQtGraph montage adoption where measured; (4)
`lod-commuting` evaluator input reduction and pointer-anchored speculation; (5) internally tiled
normal images once X5c/X5d land.

## Consequences

Positive:

- Zoomed-out tiled scenes stop paying texel costs the display cannot express; the reference
  montage's texture residency drops ~16× and level re-window work scales with what is visible.
- Zoom-in feels continuous: retained coarser levels present immediately, refinement streams in
  without blocking the event loop, and speculation can hide much of the transition.
- The pyramid cache makes threshold crossings cheap in both directions.
- Ops save real compute for commuting chains; opaque chains still save presentation cost.

Costs:

- A second resident level per tile raises GPU memory within budget caps.
- Pyramid production adds background CPU work that must stay subordinate to visible lanes.
- More storage classes complicate atlas accounting and eviction.
- Two applied-LOD policies exist until native-only can be retired for tiled scenes.

## Alternatives considered

- **Revive the synchronous prototype.** Rejected; ADR 0041 documents the stalls and identity churn.
- **Backend mipmaps only.** Rejected as the first step: no CPU-side savings for compute or
  upload, unproven edge/complex handling, and PyQtGraph gets nothing.
- **Screen-space downsampling after full evaluation.** Rejected: it preserves the dominant costs
  (native compute, conversion, residency) and only shrinks the final blit.
- **Enable for all presentations at once.** Rejected: normal-image tiling lacks viewport
  retargeting (X5c); enabling there could reveal non-resident tiles without scheduled work.

## Related records

ADR 0021 (scheduler v2), 0025/0026 (operation capabilities), 0027 (stage cache), 0039 (deadline
scheduler), 0041 (LOD separation), 0044 (acknowledged residency), 0046 (evidence-first strategy),
0049 (out-of-core sources).
