# Ideas

This is a parking area for useful possibilities that are not active commitments. Items move to the [roadmap](roadmap.md) only when they have a clear user problem, owner, dependencies, and measurable exit gate.

## Validate soon

### Selection grammar preview

The range parser currently supports Python-like forms and compatibility fallbacks. Add an inline normalized preview such as “indices 0, 2, 4” and explicit error/repair messages so users do not have to infer whether a form used inclusive/exclusive or MATLAB/Python ordering.

### Callback-budget trace overlay

Expose the last few >4/8/16 ms GUI callbacks with lane, item count, bytes, and cause in developer diagnostics. This would make event-loop regressions visible during ordinary interaction.

### Queue visualization

A developer-only panel could show presented/active/latest targets and visible/stage/speculative work counts. It must read typed scheduler state, not become another controller.

### Benchmark fixture datasets

Store small deterministic generators plus metadata for representative scalar, complex MRI-like, high-dimensional, large-plane, and many-tile workloads. Keep large binary data outside normal source archives when possible.

## Architecture experiments

### Indexed tile priority

Compare heap-with-version, bucketed distance rings, and ordered visible/near queues. Requirements: bounded retarget cost, starvation prevention, stage-wait integration, stable IDs, and no full sort on mouse move.

### Active-plus-latest scheduler

Prototype a pure Qt-free model before wiring workers. Simulate input rates, cost estimates, cancellation delays, cache reuse, and deadline misses to choose completion-versus-restart policy.

### Storage-neutral region presentation

Represent a raster as one region and a huge plane/montage as several regions, independent of montage-axis semantics. Validate that geometry, levels, values, dirty state, and backend commits use the same model.

### Surface composition seam

Extract one narrow capability at a time from `ImageView2D`: teardown, camera, raster commit, tiled commit, overlays, pointer mapping. Avoid creating a giant new abstract base class that merely mirrors both current widgets.

### Off-thread histogram refinement

Measure whether bounded sampling/binning actually violates budget. When it does, move only the expensive refinement to an immutable worker request; preserve immediate level feedback and target-key guards.

### Real GPU budget probe

Record device limits, attempt representative allocations conservatively, and cache proven-compatible texture classes for the session. Treat allocation failure as recoverable evidence, not a crash.

## Product candidates

### Linked viewer groups

Typed links for cursor, slice, levels, ROI, or recipe. Default to no link. Explicit group object, origin/revision guard, and per-channel enablement; no global `asObjs`-style registry.

### Compare/difference inspection

A narrow compare workflow with shared viewport/levels, absolute/signed difference, and ROI statistics. Resist turning it into a full registration tool.

### Dimension presets

Named view recipes for common axis/channel/operation selections, stored separately from raw array data and portable across compatible shapes/metadata.

### Axis metadata surface

Human-readable axis labels, units, physical coordinates, and spacing in dimension controls, profiles, export, and hover. Missing metadata remains a valid simple array.

### Lazy source adapter

Protocol for shape/dtype/metadata plus cancellable region reads. First adapters: NumPy memmap and chunked HDF5/Zarr-like sources. Evaluate dependencies and remote semantics separately.

### Editor/Jupyter integration

Single semantic launch/session protocol that hosts can invoke. Learn from ArrayView’s broad reach but avoid maintaining divergent WebSocket/stdio/browser state machines unless real usage justifies them.

## UI polish parking lot

- Better empty/loading/degraded/error visual hierarchy.
- Searchable command palette generated from the same command registry as menus/shortcuts.
- Compact per-axis labels/units and normalized range preview.
- Optional pixel grid and crosshair at high zoom.
- ROI naming/group visibility with simple bulk actions.
- Persisted workspace presets without persisting stale document identities.
- Accessible keyboard traversal and contrast checks.

## Scientific/MRI parking lot

- Coil/channel quick presets and root-sum-of-squares operation recipe.
- Phase/magnitude paired inspection and phase-circle profile view.
- K-space/image-space linked recipe, implemented as linked views rather than destructive toggling.
- Orientation/spacing adapters for NIfTI/DICOM metadata while keeping the core array model generic.
- Export of ROI/profile measurements with source/operation/view provenance.

## Avoid

Do not pursue these as shortcuts:

- global viewer registries or workspace scanning;
- backend-specific semantic state;
- destructive default operations;
- one giant self-contained frontend/module for deployment convenience;
- duplicated state representations that rely on perpetual reconciliation;
- mixed-size LOD images in fixed-shape atlas slots;
- debounce timers as the only scheduling policy;
- feature modes whose interactions with every existing mode are undefined;
- wall-clock benchmark claims that do not distinguish CPU submission from GPU presentation.
