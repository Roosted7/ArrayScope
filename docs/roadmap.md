# Roadmap

This roadmap is ordered by risk reduction, not by feature excitement. Historical phase checklists are archived at [`archive/roadmaps/phase-roadmap-through-v28.md`](archive/roadmaps/phase-roadmap-through-v28.md).

A roadmap item is complete only when its exit gate is met. “Code exists” is not completion.

## Now — stabilize the v28 foundation

### N0. Correctness and reproducibility gate

**Goal:** establish one trustworthy `0.8.0` release-candidate baseline before expanding the rendering architecture.

Work:

- Choose release/version semantics after the ArrayScope rebrand; align package metadata, changelog, tags, and build workflow.
- Keep the `v0.8.0` tag, package metadata, runtime `arrayscope.__version__`, changelog, and dedicated ArrayScope repository identity aligned before publication.
- Run the broad test matrix on supported Python/Qt platforms and record known skips.
- Add focused regression around the latest slicing/range, histogram, viewport, and tile-priority changes.
- Keep strict stale-commit/document-revision checks enabled in all visible paths.
- Remove test-order dependence and lifecycle leaks as they are found.
- Reproduce both RC artifact types from documented commands:
  `python -m arrayscope.tools.release_diagnostics --jsonl tests/artifacts/v0.8.0-diagnostics-pyqtgraph.jsonl --backend pyqtgraph`,
  `python -m arrayscope.tools.release_diagnostics --jsonl tests/artifacts/v0.8.0-diagnostics-vispy.jsonl --backend vispy`,
  `python -m arrayscope.core.diagnostics_trace tests/artifacts/v0.8.0-diagnostics-pyqtgraph.jsonl`,
  `python -m arrayscope.core.diagnostics_trace tests/artifacts/v0.8.0-diagnostics-vispy.jsonl`,
  and `python -m arrayscope.display.rendering_benchmarks --runs 1 --jsonl tests/artifacts/v0.8.0-rendering-benchmark-linux.jsonl`.

Exit gate:

- clean checkout builds and launches on Linux, macOS, and Windows CI targets;
- all supported automated suites pass or have documented platform skips;
- package/version/repository identity is unambiguous;
- a v28 diagnostics trace and benchmark artifact can be reproduced from documented commands without mixing their JSONL schemas.

### N1. Enforce GUI callback budgets

**Goal:** make responsiveness a contract instead of a convention.

Work:

- Instrument every Qt/OpenGL mutation callback with elapsed time, item count, and bytes.
- Bound stage-wait release, ready-tile fan-in, presentation upserts, histogram refresh, and visibility/geometry updates by time as well as items/bytes.
- Split or reschedule any callback that can traverse a user/data-sized collection.
- Record request-to-first-frame and maximum event-loop gap in benchmark output.
- Treat 4 ms interactive, 8 ms idle, and 16 ms warning thresholds consistently.

Exit gate:

- stress scenarios show no unbounded callback path;
- deterministic tests verify partial progress/rescheduling;
- real traces identify work class/backend for every callback over 16 ms;
- continuous pan, zoom, slicing, and level drag do not freeze the event loop on reference datasets.

### N2. Progress-preserving visible scheduling

**Goal:** prevent latest-only cancellation from repeatedly discarding useful near-complete work.

Work:

- Introduce explicit presented, active, and latest-queued visible targets.
- Define supersession keys separately for semantic, viewport, and presentation changes.
- Let cheap-to-finish or reusable active work complete when restart cost is higher.
- Keep exact-visible progress moving during continuous input while still rejecting stale commits.
- Feed cancellation cost and reusable output into diagnostics/feedback.

Exit gate:

- a continuous-input regression always reaches useful progressive/exact frames;
- obsolete queued targets collapse to one latest target;
- no stale frame can commit;
- traces show less repeated/cancelled CPU work than the current latest-only baseline.

### N3. Dynamic tile priority without mouse-event sorting

**Goal:** make active work follow viewport/hover value safely.

Work:

- Represent visible/near/waiting tile queues with stable priority keys or buckets.
- Coalesce viewport-center and hover updates to a bounded cadence.
- Reprioritize only affected queue metadata; never sort/materialize the full set in each mouse callback.
- Define priority aging so distant tiles eventually complete.
- Keep stage-attached waiters bounded when released.

Exit gate:

- hover or camera movement changes the next scheduled tile within one bounded update interval;
- mouse-move callbacks remain under the interactive budget;
- starvation tests complete lower-priority visible tiles;
- priority changes do not invalidate cache/residency identity.

### N4. Histogram and level refinement discipline

**Goal:** preserve responsive level control while maintaining semantic correctness.

Work:

- Measure adaptive histogram sampling/binning and manual edit paths on large/complex data.
- Move expensive refinement off-thread when it exceeds budget; retain immediate cheap committed-frame feedback.
- Keep level coverage separate from detailed plot completion.
- Verify user-lock, auto-window, restore, partial montage, and backend parity transitions.

Exit gate:

- level drag/manual edit previews remain responsive on reference datasets;
- no deleted-widget timer callback is possible;
- automatic levels never use incomplete semantic coverage for displayed pixels;
- PyQtGraph and VisPy conformance tests agree on level/value semantics.

## Next — converge the architecture

### X1. Unified frame planner and storage strategy

**Goal:** normal images and montages become one semantic presentation pipeline.

Work:

- Introduce explicit `FrameTarget`/quality and a storage-neutral region/tile model.
- Move normal and montage planning behind one `FramePlanner`.
- Choose raster, internally tiled large-plane, or montage-tiled storage from dimensions, limits, update rate, and residency.
- Make one-tile montage and normal image share level/value/cache/scheduling tests.
- Generalize `DisplayTiledPresentation` so montage geometry is optional.

Exit gate:

- no semantic branch depends on “normal versus montage” when storage strategy is the real distinction;
- a huge single plane can use internal tiling;
- conformance tests pass across raster/tiled and both backends;
- existing public interactions remain available throughout migration.

### X2. Backend composition

**Goal:** replace backend inheritance with a shared shell and thin image surfaces.

Work:

- Define `ImageViewShell` ownership of controls, histogram, HUD, viewport, and semantic signals.
- Define `ImageSurface` protocol for raster/tiled commit, camera, overlay state, pointer conversion, diagnostics, and teardown.
- Move remaining PyQtGraph/VisPy mechanics to their backend packages.
- Retire compatibility shims only after internal imports and tests use canonical paths.

Exit gate:

- `VisPyImageView2D` no longer subclasses the full PyQtGraph view;
- only one active scene/event system owns image interaction per backend;
- backend replacement/context loss has explicit lifecycle tests;
- feature-parity tests target the surface contract rather than widget class internals.

### X3. Shared pointer capture and drag lifecycle

**Goal:** one semantic interaction controller governs both backends.

Work:

- Move pointer capture, press/move/release, handle drag, cancellation, and cursor policy out of backend event handlers.
- Keep backend hit primitives/coordinate conversion mechanical.
- Define deterministic priority among ROI handles, bodies, profiles, pixel hover, and camera gestures.
- Test drag interruption by mode change, frame replacement, window deactivation, and widget close.

Exit gate:

- both backends execute the same interaction state-machine tests;
- no duplicate semantic ROI/profile drag logic remains;
- pointer loss cannot leave a stuck active tool or cursor.

### X4. Hardware evidence and residency policy

**Goal:** base GPU decisions on real device behavior.

Work:

- Record queried texture/format limits and proven allocation outcomes.
- Separate estimated GPU residency from CPU caches and track eviction/reupload.
- Build Linux X11/Wayland, Windows, and macOS reference traces on integrated and discrete GPUs.
- Decide whether/where VisPy becomes default from measured latency, stability, memory, and parity—not theoretical throughput.
- Design multi-page/multi-resolution residency only after scheduler metrics are stable.

Exit gate:

- published benchmark matrix includes request-to-frame, event-loop, RSS, residency, and upload counters;
- no fixed assumed max texture size drives policy;
- context loss and allocation failure recover without semantic corruption;
- backend default decision has documented evidence.

## Later — product capabilities that fit the mission

These are candidates after the foundation gates, not parallel commitments.

### Linked windows and inspection groups

Adopt ArrayShow’s useful synchronized-window idea through explicit group objects and typed messages, never a global workspace registry. Support selected dimensions, levels, cursor, ROI, or operation recipe links independently. Prevent feedback loops with origin/revision IDs.

### Focused compare mode

Provide side-by-side or overlay comparison with shared coordinates/levels and a small set of difference views. Keep registration/segmentation pipelines outside the core product unless they become narrow inspection adapters.

### Rich axis metadata

Surface axis names, units, coordinates, spacing, and orientation without making every data source conform to a medical-imaging model. Continue the `AxisInfo` proposal incrementally.

### Out-of-core and lazy sources

Add a source protocol for memory-mapped/chunked arrays and explicit region reads. Keep request planning, cancellation, and memory budgets above the source adapter so “lazy” does not mean unbounded transport or decoding.

### Invocation adapters

Improve Jupyter and editor launch routes only when they call one stable semantic API. Avoid duplicating a frontend/state machine per host.

### Multi-resolution storage

Add explicit compatible LOD pages/arrays or virtual textures after N1–X4. Do not revive mixed-size tiles in fixed-size atlas slots.

## Explicitly not now

- General plugin marketplace/layer ecosystem.
- Broad segmentation, registration, qMRI, or vector-field workbench.
- Remote multi-user server/collaboration architecture.
- Destructive workspace-style operations.
- Another large renderer rewrite without incremental conformance tests and traces.
