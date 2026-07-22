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

### Region presentation

Represent a small image as one tile and a huge plane/montage as several regions, independent of montage-axis semantics. Validate that geometry, levels, values, dirty state, and backend commits use the same model.

### Surface composition seam

Extract one narrow capability at a time from `ImageView2D`: teardown, camera, tiled commit, overlays, pointer mapping. Avoid creating a giant new abstract base class that merely mirrors both current widgets.

### Source/stage summaries and batched evidence readback

Prototype mergeable source/stage summaries for exact evidence and one packed
WGPU submission/readback for rough evidence. Gate both on identical final
levels/histograms, less work, and no unbounded display-payload cache. Baseline:
[`2026-07-22 compression follow-up`](reviews/2026-07-22-compression-live-benefit-review.md).

### Real GPU budget probe

Record device limits, attempt representative allocations conservatively, and cache proven-compatible texture classes for the session. Treat allocation failure as recoverable evidence, not a crash.

### Capacity-triggered compression

Keep the G7 codecs as experimental mechanisms, not an active optimization.
Revive them only when field diagnostics repeatedly show one of three user
problems: GPU pool exhaustion/eviction churn, costly `StageCache` misses under a
measured RSS cap, or remote/storage bandwidth dominating first pixels. The
candidate must attack that owner directly: source chunks compressed at rest,
an off-thread lossless tier under `StageCache`, or GPU-native/pre-encoded
presentation pages with on-GPU LOD and one physically bounded pool. Lossy
display pages never own exact histogram, levels, or cursor values. The
[2026-07-22 live review](reviews/2026-07-22-compression-live-benefit-review.md)
is the baseline and revival gate. The active retention audit must keep the
evaluator's exact ROI-demand cache distinct from GPU page storage and compare
only the physical owner selected by the current backend. If StageCache pressure
triggers the work,
prototype raw-hot/compressed-cold demotion on the kernel's lowest-priority lane:
compression happens from an immutable evicted value outside the cache lock,
is cancelled by visible demand, and admits by expected recompute latency saved
per compressed byte. Idle CPU alone is not an admission signal.

Already-encoded BC transfers faster, but current encoding costs more than the
saved upload time. If retention pressure later justifies compression, race one
bounded off-thread artifact against the raw upload and never delay visible work;
measure prevented evictions/re-uploads, wasted preparation, physical tile rate,
and byte-capped residency. Prefer pre-encoded or on-device paths. Keep the 40 dB
gate unless a display-aware quality contract passes physical-framebuffer tests.
Detailed baseline: [compression follow-up](reviews/2026-07-22-compression-live-benefit-review.md).

### Elastic preview residency

Experiment with a retained preview LOD tier that uses otherwise-unused CPU/GPU memory for nearby
or whole-dataset preview tiles, especially on low-end devices where exact work is slow. This is not
the active preview-then-refine contract in the roadmap; it is the more aggressive policy question
after that contract exists.

Questions to answer before it can move to the roadmap:

- whether the preview tier should have a fixed budget or borrow only memory not needed for exact
  visible work;
- whether preview residency belongs in GPU memory, stage memory, or both;
- how far speculative coverage should extend beyond the viewport without starving visible/exact
  refinement;
- how to evict preview data without causing black tiles, re-upload churn, or misleading diagnostics;
- how source-provided pyramids and chunked/lazy sources should feed the preview tier.

Important distinction from the 2026-07-07 Plan 05 work: retained preview planes are display
previews (`lod_preview_pyramid`) and should be used for fast first pixels / future offscreen GPU
warming. They are not a replacement for the stage cache, which remains the reusable operation
intermediate cache. A later roadmap item should decide how the two cooperate without letting
speculative preview uploads steal visible or exact-refinement bandwidth.

## Product candidates

### Linked viewer groups

Typed links for cursor, slice, levels, ROI, or recipe. Default to no link. Explicit group object, origin/revision guard, and per-channel enablement; no global `asObjs`-style registry.

Shipped for slice/levels/ROI/recipe links in the default group ([ADR 0048](decisions/0048-linked-window-sync.md), `arrayscope/sync/`). Still open here: cursor links, viewport links, and user-visible named groups.

### Compare/difference inspection

A narrow compare workflow with shared viewport/levels, absolute/signed difference, and ROI statistics. Resist turning it into a full registration tool.

### Dimension presets

Named view recipes for common axis/channel/operation selections, stored separately from raw array data and portable across compatible shapes/metadata.

### Axis metadata surface

Human-readable axis labels, units, physical coordinates, and spacing in dimension controls, profiles, export, and hover. Missing metadata remains a valid simple array.

### Lazy source adapter

Largely landed via [ADR 0049](decisions/0049-out-of-core-lazy-sources.md): source protocol, budgeted
read seam, and NumPy memmap adapters. Still exploratory here: chunked HDF5/Zarr-like adapters,
chunk-aligned request planning, and remote-source semantics/dependencies.

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

## 2026-07-19 brainstorm (course-review follow-up)

Creative candidates from the [2026-07-19 course review](reviews/2026-07-19-course-review.md)
session. Same contract as the rest of this file: parking, not commitment.
Starred items were judged highest-leverage.

### Trust & explainability UX

- ★ **"Why this pixel" provenance receipts.** Click a pixel → source
  value(s), op chain + parameters, LOD/interpolation actually used, window
  mapping, colormap bin. Ground rule 8 already forces the engine to know
  presentation-qualified vs exact; no other viewer has that distinction to
  expose. Mostly a popover over existing facts.
- ★ **"Explain the wait."** Busy indicator → "waiting on FFT of a 1.2 GB
  slab (will be cached), stage 2/6." Renders existing trace-bus/stage
  facts as first-person UI copy.
- **Freshness veil.** Subtle visual treatment on preview-grade tiles, gone
  when exact — the progressive contract as visible trust (productized
  tile-truth overlay).
- **Session → code / provenance-carrying figures.** Export the current
  view as runnable Python; embed the recipe JSON in exported PNG metadata.

### Complex-native visuals

- ★ **Phase spinner.** Circular slider multiplying display by e^{iθ} —
  one shader uniform on GPU backends, zero re-evaluation. First concrete
  consumer of `SHADER_ON_READ` ([tensor-ops T0](proposals/tensor-ops-g8.md)).
- **Domain coloring** channel mode (hue=phase, lightness=magnitude).
- **Phasor glyph lens** at high zoom (complex value as geometry; rides the
  wgpu instanced-overlay pass).

### Compare 2.0 (beyond queue rows 5–7)

- ★ **Chunk-hash structural diff.** The chunk store is content-keyed:
  comparing two recon variants, identical chunks are *known identical at
  residency level* before any difference pixel is computed. Entry view of
  compare = a change-map; identical chunks dedupe GPU memory across
  windows. Signature-feature potential.
- **Magic lens.** Hold a key → cursor becomes a movable lens rendering B
  (or A−B) inside A. Scissored second draw over shared residency.
- **Blink comparator** (radiology flicker; presentation-identity swap over
  shared residency) and **correlation cursor** (live A-vs-B scatter of ROI
  values with identity line).

### The pipeline as a place

- ★ **Pipeline time-travel.** Click any op-stack step → view the array at
  that stage (StageCache already holds the intermediates); two-pane
  before/after any op.
- ★ **ROI teleportation.** Map an ROI through the region algebra
  (`required_input_region`) to any pipeline stage — "show this artifact in
  k-space." Exposes the most sophisticated machinery in the codebase.
- **Dimension grammar bar.** Einops-style one-liner
  (`coil:rss echo:2 x,y:image rep:scrub`) as an alternate head for the
  dimension strip; doubles as human-readable recipe serialization.

### Reach

- ★ **Command protocol as wire protocol.** ADR 0057 can run kernel+engine near
  the data and stream commits to a thin client. G7's current CPU codecs are not
  yet the wire format: the live-benefit audit measured a NO for CPU transport and
  parked encoding until a real remote-bandwidth trace triggers the bounded design;
  remote viewing falls out of architecture already being built (browser/
  WebGPU client is the far end). Reframes "PyQtGraph targets remote."
- **Watch mode / recon debugger.** `--watch out.cfl` or
  `viewer.push(x, iteration=n)`: iteration as a growing scrubbable
  dimension; chunk-hash diff highlights what moved per iteration.
- **Anomaly navigator.** "Jump to next NaN/Inf/zero-block/max" — add
  NaN/Inf counts to the existing per-chunk summaries; navigation becomes a
  residency-metadata query, not a scan.
- **Zero-copy launch.** shm handoff instead of npy write for
  `arrayscope(x)` on multi-GB arrays.

### Engine & testing internals

- ★ **Deterministic kernel simulation.** The kernel is Qt-free, one
  lock/condvar: run it under a simulated clock with virtual workers and
  seeded schedule exploration (FoundationDB-style). Attacks the dominant
  lost-wakeup bug family in ring 0 by enumeration instead of by
  real-Wayland luck.
- **Command-stream record/replay.** Record semantic command streams in
  dogfooding; replay deterministically against any backend; backends
  become differential oracles for each other. Upgrades the flight-recorder
  idea from screenshots to semantics.

### UI feel

- **Filmstrip scrub** (LOD-preview thumbnails along the scrub dim,
  video-editor style; preview pyramid exists).
- **Controls that recede** (panels fade to edge-hints until hover — the
  array is the UI).
- **Vim-for-arrays** (dimension letters + counts, `3e` = echo 3, on the
  existing command registry).

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
