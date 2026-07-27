# ADR 0057: Backend-neutral renderer command protocol

- **Status:** Accepted and implemented through the live backend
  (2026-07-18): the wgpu executor commits every ArrayScope payload shape
  behind the protocol with physical page-table acknowledgements, the LOD
  ladder and montage sessions run on `BindContentPlanes`, and both G6 compute
  consumers (resident-page histograms and reducer-honest resident-page LOD
  generation) are live. Histogram frontier integrity added 2026-07-19 (live
  dogfood crash: a submission's own ensures LRU-evicted a snapshotted
  frontier page and the executor's loud `KeyError` killed the whole commit
  mid-batch): the fix is deliberately **executor-internal**, not new
  `PinChunks`/`UnpinChunks` protocol commands — an ordered submission already
  tells the executor everything a later `DispatchHistogram` will sample, so
  it pre-scans and shields those keys from its own eviction for the
  submission's duration (scoped pin owner, always released). That keeps the
  protocol free of residency-pinning plumbing (backend-neutral: an executor
  without eviction needs nothing) and keeps eviction honest: when pool
  pressure exceeds the shield, the executor yields the shielded page and
  reports it in `FrameReport.histogram_missing` instead of failing the
  submission; consumers treat such evidence as unsatisfied and retry.
  Dynamic histogram readback batching was added 2026-07-23: every
  `DispatchHistogram` still owns an independent result and bounds, while the
  executor packs all deferred outputs in one `FrameSubmission` into one
  staging buffer and one queue read. This is executor-internal synchronization
  policy, not a new protocol command. Native flat overlay geometry is also live through
  `UpdateOverlayGeometry` + the uniform-only `SetOverlayCamera`: ROIs,
  profile cursor geometry, and tile-status geometry draw after tiles in the
  same pass. Native overlay TEXT landed 2026-07-19 and closed the last
  overlay gap: `OverlayPrimitive` grew screen-space-sized `screen_rect` /
  `glyph_quad` kinds (world anchor + physical-pixel offset/size + normalized
  atlas UVs) and one new `UpdateGlyphAtlas` command carries the CPU-baked
  glyph alpha atlas (`arrayscope/display/glyph_atlas.py` — QPainter/
  QFontMetrics bake off the frame path, cached by (font-key, pixel-size,
  glyph), DPR in the cache key, bounded growth with loud
  `wgpu_glyph_atlas_evicted` eviction). Glyph quads join the SAME flat
  instanced overlay pipeline (one extra sampled `r8unorm` binding, nearest
  `textureLoad` for crispness); `FrameReport.glyph_atlas_uploads` is the
  zero-per-frame-upload oracle. Tile-truth labels render natively in the
  wgpu view (QLabels replaced — Qt widgets cannot composite over a native
  child), which unblocked the screen-present-mode experiment. The same
  constraint later forced the FLOATING chips (first-run hints, evaluation
  indicator, pixel HUD, ROI info panel) into the frame as well: a
  `widget_quad` kind plus an `UpdateWidgetAtlas` command carry them as
  straight-RGBA rasters of Qt's own painting
  (`arrayscope/display/backends/wgpu/chip_compositor.py`), sampled through a
  second `rgba8unorm` binding that REPLACES the primitive colour rather than
  masking it, with `FrameReport.widget_atlas_uploads` as the
  zero-per-frame-upload oracle. Unlike every other kind, `widget_quad` is
  camera-independent — chips are window furniture and must not pan with the
  image. The two alternatives are closed, not merely unattractive: a native
  *child* window gets no ARGB visual from Qt (`alphaBufferSize() == 0`, so
  translucent rounded chips flatten into opaque boxes), and the swapchain
  subsurface cannot be restacked under the window because `QWindow.lower()`
  emits no `wl_subsurface.place_below`. Chips stay ordinary Qt widgets and
  keep painting normally, so a chip overhanging the canvas (the hints chip
  overlaps the histogram) still shows there, from the same rendering. Screen
  presentation landed 2026-07-19 behind the `wgpu_present_method` setting
  (bitmap default, explicit opt-in): a bare `QWindow` embedded through
  `QWidget.createWindowContainer`
  (`arrayscope/display/backends/wgpu/screen_canvas.py`) drives its own
  swapchain from the gate-B recipe (QNativeInterface wl_display +
  winId-as-wl_surface, Vulkan-only instance, Mailbox present mode when the
  surface offers it), bypassing rendercanvas and the per-frame bitmap
  readback. The container shape is load-bearing (2026-07-20 glitch fix): a
  native child *widget* drags its ancestor chain — and without
  `AA_DontCreateNativeWidgetSiblings` every sibling — into native windows,
  shattering the top-level into desynchronized wl_subsurfaces (white/hole
  regions, hidden overlays, resize flicker). The embedded window parents
  directly to the top-level: exactly one subsurface. Because that
  subsurface composites above all Qt-painted pixels, floating overlay
  chips (first-run hints, busy label, pixel HUD, ROI info panel) opt into
  their own top-level-parented native windows via
  `_prepare_display_overlay_widget`, and the canvas presents immediately
  on embedded-window resize (a subsurface's footprint IS its latest
  buffer, so paced redraws read as old/new-size flicker). The presentation path is deliberately OUTSIDE the protocol: the
  executor still only receives `PresentGeneration` plus a target texture
  view — whether that view is a rendercanvas bitmap target or an acquired
  swapchain texture is view-side plumbing, so the protocol stays free of
  windowing vocabulary. Draw acknowledgements key on the real swapchain
  present edge, and everywhere the screen path cannot exist (offscreen,
  xcb) the view falls back to bitmap with a loud recorded reason.
  Promotion vs VisPy is evidence-gated in queue row 3.
- **Date:** 2026-07-18
- **Branch note:** authored on `codex/wgpu-renderer-gate-b`; renumber on
  integration if a parallel branch claimed 0057.
- **Related:** ADR 0053 (one scheduler), 0055 (tile/chunk/page split),
  0056 (sparse pyramid); `docs/proposals/tensor-engine-endpoint.md`
  (renderer strategy + command table),
  `docs/proposals/wgpu-renderer-experiment.md` (gate-B evidence).

**Current backend note (2026-07-27):** WGPU is the maintained GPU/rendering
executor. The VisPy strangler discussion below is preserved as migration
history; VisPy was retired by
[ADR 0061](0061-retire-vispy-rendering-backend.md).

## Context

The renderer experiments settled the strategy question with evidence:
Datoviz gate A failed the composition and upload-lifetime gates; wgpu gate B
passed all three renderer gates at experiment scale (Experiment A/B findings
in tensor-engine-endpoint.md). The remaining structural risk is the one the
endpoint document has warned about since 2026-07-15: every migration plan
dies if engine semantics are expressed in a renderer's private vocabulary.
VisPy's executor today *is* that vocabulary (gloo buffers, per-backend
upserts); moving G6 shader work onto wgpu without a seam would just create a
second one.

## Decision

1. **`arrayscope/gpu/command_protocol.py` is the only seam renderers
   implement.** Frozen command dataclasses — `EnsureChunkResident`,
   `EvictChunk`, `GenerateLodPages`, `UpdateTileInstances`,
   `UpdateOverlayGeometry`, `UpdateGlyphAtlas`, `UpdateWidgetAtlas`,
   `SetOverlayCamera`,
   `SetDisplayMapping`, `DispatchHistogram`, `PresentGeneration` — carried
   by an ordered `FrameSubmission`, answered by an auditable `FrameReport`
   (uploads, overlay-buffer writes, glyph-atlas uploads, evictions,
   histogram results, completion token). Commands speak ADR
   0055/0056 identities (`DataChunkKey`, `ChunkLod`) and normalized
   geometry; nothing in the protocol may name WGSL, GL objects, Qt, Datoviz
   IDs, or one-texture-per-tile.
2. **The protocol schedules nothing** (ADR 0053): a submission is
   already-ordered work; the kernel owns priority, supersession, and pacing.
   The report's `wait_completed` token is how page/staging recycling fences
   GPU work — renderer gate 3's contract, now explicit.
3. **`arrayscope/gpu/wgpu_executor.py` is the first implementation**
   (`WgpuPlaneExecutor`): bound 2-D plane pyramids, format-honest page pools,
   `PageTable` bookkeeping, GPU-side ancestor-fallback lookup, one instanced
   tile draw followed by one flat instanced overlay draw in the same render
   pass, two-pass G6 histogram, and in-pool component-mean LOD generation.
   Multiple dynamic histogram commands in one submission share one batched
   staging-buffer readback without merging their evidence identities or
   values.
   Reducer family is physical binding identity: recursive GPU generation is
   accepted only for `mean`; mode-specific `mean_abs`/`phase_vector` families
   remain on their honest CPU route. Default-ring tests
   (`tests/gpu/test_wgpu_command_protocol.py`) hold the gate-B oracles —
   zero-upload mode/levels/shift/scroll, pinned-ancestor fallback
   (never-black), exact histogram, completion token — and skip cleanly
   where no adapter exists.
4. **Backends are migrated by strangulation, not rewrite.** VisPy remains
   the production backend and its executor is progressively re-expressed as
   a protocol implementation at the existing seams (payload upsert →
   `EnsureChunkResident`, draw parts → `UpdateTileInstances`, shader
   mapping → `SetDisplayMapping`, commit acknowledgement →
   `PresentGeneration` report). Promotion of the wgpu backend is an
   evidence decision (journey matrix + perf bars on real data), never a
   flag-day switch; PyQtGraph keeps its first-class headless/remote role
   either way.

## Consequences

- G6 shader/compute slices are written once, against the protocol, and run
  on the wgpu executor now (gloo has no compute); they become portable to
  any future executor for free.
- The protocol's report is the natural hook for the physical-truth and
  zero-upload invariants that today live in backend-private stats.
- A second protocol implementation (VisPy) will surface any accidental
  wgpu-isms in the seam early, while the surface is still small.
- The seed executor's known limits are recorded in its docstring (single
  plane pyramid, complex-only pool, magnitude histogram); expanding any of
  them is ordinary queue work behind tests, not a design change.
