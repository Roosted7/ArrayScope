# Current project status

**Snapshot:** v30 rendering-consistency review, 2026-06-24. For the concise live maturity map, read
[`current-state.md`](current-state.md). For evidence and recommendations, read the
[v30 rendering-consistency audit](reviews/v30-rendering-consistency-audit.md).

## What is solid today

- N-dimensional semantic state and reversible operation documents.
- Operation-declared shape/dtype/region/cost behavior, runtime planning, slab/chunked evaluation, and
  stage materialization/singleflight.
- Separate caches and explicit memory/compute/resource policy.
- Backend-independent display frames, geometry, value sources, level sources, and tile payloads.
- Progressive montage with retained last-valid pixels, stale-result guards, viewport/hover priority,
  and persistent backend items/residency.
- Production PyQtGraph fallback plus an experimental VisPy raster/tiled path with shader-based levels.
- Profiles, hover, ROI statistics/histograms, and demand evaluation outside the visible viewport.
- Deterministic rendering counters, callback observations, diagnostics snapshots, JSONL traces, and a
  substantial pure/Qt test suite.

## Recent rendering review

The recent optimization sequence improved callback budgets, active-plus-latest scheduling, stage-plan
caching, tile priority, direct deltas, and VisPy residency. It also exposed a control-plane flaw:
retained visible tiles were treated as proof that a current replacement had committed. On PyQtGraph,
a level target may require a bounded CPU/item redraw for every active tile, while VisPy can normally
change uniforms immediately.

The v30 review repairs the immediate level/histogram/benchmark issues and records two durable choices:

- [ADR 0040](decisions/0040-backend-aware-presentation-convergence.md): one semantic presentation
  generation with backend-specific convergence and exact acknowledgement;
- [ADR 0041](decisions/0041-lod-selection-materialization-and-residency.md): production remains
  native-only until LOD demand, asynchronous materialization, and compatible residency are separate.

## Main risks

### Rendering control-plane concentration

The main risk is no longer missing abstractions in the evaluation core. It is the concentration of
many lifecycle state machines in `MontageRenderSession` and `montage_renderer.py`: compute/stage
queues, payload admission, viewport/residency hints, semantic level coverage, level convergence,
acknowledgement, committed frames, and timers. The roadmap now places Qt-free state-machine extraction
before more renderer unification.

### Histogram backend coupling

The numerical/refinement direction is sound, but the current PyQtGraph binding manually manages
private `HistogramLUTItem` details and signal connections. Isolate this behind a compatibility adapter
or replace the widget shell before adding more histogram features.

### LOD availability

LOD demand selection runs, but applied factor is intentionally one. The old implementation built CPU
pyramids in a GUI presentation callback and used incompatible dimensions with a fixed-slot atlas.
Diagnostics now report desired/applied factor and policy rather than silently appearing inactive.

### Backend migration and hardware evidence

`VisPyImageView2D` still subclasses the full PyQtGraph view. Real OpenGL, Wayland/X11, Windows/macOS,
high-DPI, context-loss, texture-limit, frame-pacing, and interaction evidence is incomplete. Keep
PyQtGraph as the safe default until conformance and platform traces justify a change.

## Release blockers

1. Complete the N5 semantic convergence matrix and run the full clean test suite.
2. Perform real-platform PyQtGraph and VisPy manual regression with diagnostics traces.
3. Verify wheel installation, version/tag/package identity, and CI publication evidence.
4. Keep LOD native-only and VisPy experimental in release messaging unless their explicit gates pass.
5. Do not merge further cross-cutting rendering features before the N6 ownership extraction starts.

## Scale and test posture

The largest source files are currently about 3,070 lines (`montage_renderer.py`), 2,230
(`imageview2d.py`), 2,202 (`vispy_imageview2d.py`), 2,075 (VisPy tiled backend), and 2,028 (profile
workflow). These sizes are evidence of ownership concentration, not an instruction to split files
mechanically. Extract generation, admission, stage fan-in, histogram model/adapter, and backend
mechanics along the boundaries in the roadmap.
