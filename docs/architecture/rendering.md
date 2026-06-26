# Rendering

Rendering is split into semantic presentation and concrete backend mechanics.

## Semantic inputs and outputs

An evaluation result supplies data plus semantic metadata such as texture kind, histogram/value source, component, and exact/degraded status. Presentation planning combines that with:

- committed geometry;
- levels and histogram domain/source;
- LUT and scale mapping;
- viewport intent;
- tiled region payloads;
- dirty/retained/presented tile state.

The resulting display presentation is backend-independent. A backend adapter translates it into concrete widget/texture calls and reports commit work/acknowledgement.

## Committed frame

A frame is the visible semantic truth. It owns:

- document/semantic/presentation revision;
- geometry used to draw the pixels;
- value source used for hover/inspection;
- level source and mapping;
- visible tile/payload identity;
- exact/degraded and dirty/acknowledged state.

Hover and ROI/profile mapping use this frame, not whatever `ViewState` happens to be queued next. That prevents a pointer value from being read from new state against old pixels.

## Geometry

`display.geometry` maps among:

- backend world/ViewBox coordinates;
- canvas-local coordinates;
- montage tile-local coordinates;
- array indices and profile states.

Geometry is committed with the frame. Normal and montage paths must not invent separate coordinate conventions inside event handlers.

## Levels and histograms

Three concepts are separate:

1. semantic values/pixel source;
2. level source/coverage used for automatic windowing;
3. detailed histogram plot data shown to the user.

A progressive tile can be shown before a high-detail plot is complete, but automatic levels for that tile must be based on semantic coverage that includes it. User-locked levels are not overwritten by later refinement. Preview drags/manual edits may update pixels immediately while only final edits emit the semantic level-change signal.

## Unified Tiled Surface

ArrayScope presents normal images, large planes, and montages through one semantic tiled image
surface. A small/stable image may collapse to one raster commit or one backend texture/item; a large
single plane may use internal tiles; a montage uses multiple semantic tile regions. Those are
optimizations and layouts inside one presentation model, not separate semantic renderers.

A tiled presentation is a set of semantic regions and payloads. PyQtGraph uses persistent per-tile
image items; VisPy uses atlas/texture-backed visuals. Tile identity is based on materialized data and
compatible physical representation, not levels/LUT. Level/window/LUT changes are presentation
updates, preferably shader/uniform updates where the backend supports them, and do not imply new
source pixels.

Presentation-generation and admission state are Qt-free. `PresentationGenerationTracker` owns the
latest level target, revision, active coverage, pending work, and acknowledgement state.
`TileAdmissionQueue` owns priority/aging/item/byte/deadline admission without knowing array semantics.
`LevelConvergenceStrategy` keeps PyQtGraph progressive tile redraws and VisPy uniform updates behind
one semantic convergence contract. `WorkGraph` sits above these component models and owns lane-level
frame-value admission/counters for visible planning, materialization, display preparation, GUI fan-in,
backend commit, side analysis, stage materialization, and speculative residency. Work visibility is
carried by target quality as well as lane: exact visible stage materialization is visible work, while
retained stage warmup is optional and subject to available-budget admission.

A montage is one reason to have semantic regions, but not the only one. Internal tiling of one huge
plane is implemented without inventing a montage axis; `FramePlan` region bounds and data slices are
the source of truth for backend tile placement.

Montage resize and column reflow use shared viewport policy. `ViewportController` preserves manual
screen zoom across widget resize; then the Qt-free `montage_viewport` policy handles montage-specific
layout retargeting. The renderer supplies the current Fit/near-auto facts and applies the returned
`MontageViewportReflow`. Manual views do not refit because the viewport grew, shrank, changed aspect,
or temporarily intersects zero tiles. Near-auto re-entry requires all four current edges to be near the
next fitted auto range. When the same source indices are laid out in different columns, manual reflow
translates by `source_index` and tile-local focus without applying another zoom change. When the tiled
dimension scrolls to a different source set, the range remains in world coordinates and samples the
new content. ROI geometry follows the same source-local rule through canonical
`RoiSelection` remapping. See
[ADR 0042](../decisions/0042-montage-viewport-reflow-and-roi-ownership.md).

### Multi-resolution

Production residency currently favors native-resolution tiles. Arbitrary CPU-reduced sizes must not be placed into fixed atlas slots whose sampling assumes one tile shape. Future multi-resolution storage requires compatible classes: separate pages per LOD/shape, arrays grouped by dimensions, or a virtual texture/page table.

## Backend contract

Shared code asks for capabilities such as:

- raster presentation;
- direct tiled payload presentation;
- persistent residency;
- shader windowing for scalar/complex data;
- native pointer/viewport interaction;
- diagnostics and acknowledgement.

It must not branch on `isinstance(...VisPy...)` to decide semantic meaning.

Concrete backend code may own:

- image items, visuals, buffers, textures, atlas pages;
- upload/window preparation specific to the library;
- shader sources/uniforms;
- camera synchronization and scene redraw mechanics;
- resource/context-loss handling.

It may not own:

- what constitutes a frame target;
- which ROI/profile target wins;
- whether levels are user-locked;
- cache/document identity;
- the meaning of a viewport request;
- montage reflow or ROI source-local remapping semantics.

## Current backends

### PyQtGraph

The default path is mature and provides the complete feature baseline. Its tiled implementation avoids rebuilding a full montage canvas, but large item counts and per-item updates can become GUI/scene-graph bottlenecks. Warm item visibility/geometry changes should not be reported as cold CPU windowing/upload.
It accepts typed tiled presentations for internally tiled single planes as well as montages.

### VisPy

VisPy supports shader mapping and persistent tiled residency with atlas-backed drawing. It can avoid
repeated CPU windowing and reduce many-item overhead. `VisPySurface` now reaches presentation commits
through the shared `ImageSurface` contract rather than inheriting the PyQtGraph concrete surface.
The VisPy canvas remains passive and mouse-transparent; the shared pointer interaction controller
owns drag/capture semantics while the VisPy surface mirrors hover and capture visuals.
VisPy is the preferred backend for sustained large tiled rendering, pending small-view latency and
platform validation. Its active visible commit should be a coherent GPU presentation transaction:
admitted payloads are acknowledged only after texture data, atlas/page geometry, visibility, and draw
invalidation are consistent.
Its atlas/quad path uses frame-plan tile geometry for internally tiled single planes and montage
geometry for montage presentations.

Widget close stops warm-tile work, cancels queued histogram refresh, and closes the VisPy canvas.

## Presentation performance contract

- Keep the last valid frame until a replacement is usable.
- Reject stale commits by revision/key.
- Timer callbacks carry explicit session/revision work tokens; timers reschedule bounded work but do
  not establish semantic order.
- Do not clear because an identity is merely unknown.
- Apply backpressure before visible admission; once admitted, visible payloads commit coherently or
  the previous placeholder/retained frame remains in force.
- Bound cold preparation/upload by items, bytes, and elapsed time.
- Do not count a batch of many tiles as one feedback item.
- Separate submission time, preparation time, upload bytes/time, queue delay, and first-frame/presented age.
- Publish work-graph counters by lane so dropped, superseded, reusable, and budget-blocked work are
  observable separately from backend physical work.
- Run local prefetch/warmup gates before graph admission; rejected optional work must not appear as
  active admitted work.
- Changes to levels/LUT/scale should update uniforms or prepared display state without re-materializing unchanged source pixels.

## Migration direction

`ImageViewShell` is the shared widget contract for controls, histogram, HUD, viewport intent,
interaction state, and display timing. PyQtGraph and VisPy expose concrete `ImageSurface`
implementations with declared capabilities; `DisplayCommitter` commits semantic raster/tiled
presentations directly to that surface contract. The contract also covers camera application, overlay
coordinate mapping, diagnostics, context-loss reset, teardown, interaction-state visual sync, and
declared shared interaction-event ownership.
