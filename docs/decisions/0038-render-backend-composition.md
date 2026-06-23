# 0038 — Compose rendering backends behind shared presentation semantics

## Status

Accepted for incremental migration.

## Context

ArrayScope needs to retain PyQtGraph and VisPy rendering while the GPU path is measured and brought
to feature parity.  The current VisPy experiment inherits the full PyQtGraph `ImageView2D` and places
a mouse-transparent VisPy canvas under the existing interaction surface.  That was a pragmatic way to
preserve histogram, ROI, profile, context-menu, and viewport behavior, but it also means that two scene
systems remain alive and backend-specific conditions have spread into shared display code.

Creating parallel copies of the viewer, presentation pipeline, ROI controller, and montage renderer in
`pyqtgraph/` and `vispy/` directories would make ownership look clear while actually duplicating the
hardest code.  The copies would drift in coordinate semantics, commit ordering, hover validity,
window/level behavior, cancellation, and diagnostics.

## Decision

Keep one semantic display pipeline and compose it with backend adapters. The abstraction preserves
shared meaning, not identical physical commit style.

The shared side owns:

- tiled presentation models, including one-tile and small-tile cases;
- display geometry and viewport intent;
- level and histogram decisions;
- committed value-source semantics for hover, ROI, and profile reads;
- pointer/interaction state, ROI geometry, hit testing, and tool lifecycle;
- scheduling, coalescing, cancellation, and diagnostics vocabulary.

A rendering backend owns only:

- surface creation and lifecycle;
- one-tile/small-tile upload and presentation mechanics;
- tiled storage, residency, and draw submission;
- camera/range application;
- backend-specific overlay visuals;
- concrete upload/draw diagnostics.

Capabilities, not backend names, select optional paths. Both PyQtGraph and VisPy consume the same
typed `DisplayTilePayload` values and must pass one semantic conformance test suite.

The protocol must expose semantic commonality:

- tile identity;
- value source;
- level/window/LUT semantics;
- ROI/profile semantics;
- viewport and committed-frame identity.

The protocol must also allow backend-specific mechanics:

- PyQtGraph may use item-oriented update semantics, including per-item visibility, geometry, and
  `ImageItem` data updates;
- VisPy may use coherent atlas/page transaction semantics, where texture writes, geometry/page
  metadata, visibility, and draw invalidation are acknowledged as one presentation commit for the
  admitted payloads.

Shared semantics do not require a shared physical commit model. A backend protocol that forces VisPy
to look like PyQtGraph item mutation, or forces PyQtGraph to fake VisPy atlas transactions, is the
wrong abstraction.

The intended package shape is:

```text
arrayscope/display/
  model/                 # presentation, geometry, viewport, overlay/interaction state
  backends/
    base.py               # protocol and capabilities
    pyqtgraph/
      surface.py
      raster.py
      tiles.py
      overlays.py
    vispy/
      surface.py
      raster.py
      atlas.py
      shaders.py
      overlays.py
  widget.py               # shared Qt shell: histogram, HUD, ROI info, signals
```

This is a destination, not a flag-day move.  Existing public widget methods remain the migration
boundary while implementation is extracted by responsibility.  `VisPyImageView2D(ImageView2D)` may
remain temporarily, but the end state is a shared shell containing an `ImageRenderBackend`, not a
backend inheriting another backend.

## Consequences

Positive:

- PyQtGraph and VisPy share all domain semantics without copied pipelines.
- Backend-specific optimizations remain possible without contaminating render orchestration.
- Feature parity is testable at the semantic contract rather than by maintaining two implementations
  of ROI/profile logic.
- A future Qt Quick/RHI, wgpu, or custom OpenGL backend can be evaluated against the same contract.

Trade-offs:

- The current hybrid surface remains during migration and still pays some dual-scene and Qt stacking
  cost.
- The backend protocol must stay semantic while still permitting different mechanics; exposing every
  VisPy or PyQtGraph primitive through it would create a lowest-common-denominator abstraction.
- Native VisPy pointer interaction is deferred until the shared interaction controller no longer
  depends on `QGraphicsItem` event ownership.

## Migration sequence

1. Keep canonical display presentation objects as the only pixel commit inputs while converging normal
   image and montage presentation on tiled regions.
2. Move capability lookup, tiled payload commit, camera application, and overlay drawing behind a
   small backend protocol.
3. Extract pointer hit testing and drag state from Qt graphics items into one shared controller.
4. Replace backend inheritance with a shared widget shell plus backend composition.
5. Run the same semantic and benchmark scenarios against every backend.
6. Select or retire a backend only after production workload measurements and parity checks.

## Rejected alternatives

- **Duplicate PyQtGraph and VisPy viewer trees.**  This duplicates semantic state and guarantees drift.
- **Drop PyQtGraph immediately.**  The VisPy path is not yet measurably superior across workloads and
  still lacks mature interaction behavior.
- **Keep only PyQtGraph.**  That abandons programmable display shaders and persistent GPU tile
  residency before they have been implemented and measured properly.
- **Wrap every backend API in a common façade.**  The abstraction should describe ArrayScope display
  intent, not mirror two unrelated graphics APIs.
- **Rewrite immediately in Qt Quick/RHI.**  It may ultimately provide a better render-thread model,
  but it is a larger shell replacement and should be evaluated after the semantic boundary is clean.

## Required tests

- One conformance suite for tiled levels, geometry, value lookup, viewport preservation, dirty tile
  semantics, overlays, and interaction state.
- Backend-specific tests for upload counts, residency, storage rebuilds, evictions, and shader mode.
- Backend-specific commit tests for PyQtGraph item updates and VisPy coherent atlas/page
  transactions.
- Presented-frame benchmarks that measure first-frame latency and Qt event-loop starvation separately
  from setter submission time.
- Manual parity checks for hover, cursors, ROI/profile drag feedback, status HUD, and ROI information.
