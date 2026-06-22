# Rendering

Rendering is split into semantic presentation and concrete backend mechanics.

## Semantic inputs and outputs

An evaluation result supplies data plus semantic metadata such as texture kind, histogram/value source, component, and exact/degraded status. Presentation planning combines that with:

- committed geometry;
- levels and histogram domain/source;
- LUT and scale mapping;
- viewport intent;
- raster or tiled payloads;
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

## Storage strategies

### Raster

A small/stable image can use one PyQtGraph `ImageItem` or one VisPy texture. This is simple and efficient until plane size, update rate, texture limits, or preparation cost make one full upload undesirable.

### Tiled

A tiled presentation is a set of semantic regions and payloads. PyQtGraph uses persistent per-tile image items; VisPy uses atlas/texture-backed visuals. Tile identity is based on materialized data and compatible physical representation, not levels/LUT.

A montage is one reason to have semantic regions, but not the only one. The target architecture also permits internal tiling of one huge plane without inventing a montage axis.

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
- the meaning of a viewport request.

## Current backends

### PyQtGraph

The default path is mature and provides the complete feature baseline. Its tiled implementation avoids rebuilding a full montage canvas, but large item counts and per-item updates can become GUI/scene-graph bottlenecks. Warm item visibility/geometry changes should not be reported as cold CPU windowing/upload.

### VisPy

VisPy supports shader mapping and persistent tiled residency with atlas-backed drawing. It can avoid repeated CPU windowing and reduce many-item overhead. It remains experimental because `VisPyImageView2D` still subclasses the full PyQtGraph widget, so two scene/event systems and lifecycle models coexist.

Widget close now stops warm-tile work, cancels queued histogram refresh, and closes the VisPy canvas. This is necessary cleanup, not the final composition architecture.

## Presentation performance contract

- Keep the last valid frame until a replacement is usable.
- Reject stale commits by revision/key.
- Do not clear because an identity is merely unknown.
- Bound cold preparation/upload by items, bytes, and elapsed time.
- Do not count a batch of many tiles as one feedback item.
- Separate submission time, preparation time, upload bytes/time, queue delay, and first-frame/presented age.
- Changes to levels/LUT/scale should update uniforms or prepared display state without re-materializing unchanged source pixels.

## Migration direction

The destination is a shared `ImageViewShell` containing controls, HUD, viewport, interaction controller, and an `ImageSurface` implementation. PyQtGraph and VisPy surfaces should implement the same semantic conformance tests. Remove backend inheritance only in incremental steps that preserve a runnable default path.
