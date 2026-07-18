# ADR 0057: Backend-neutral renderer command protocol

- **Status:** Accepted and implemented through the live backend
  (2026-07-18): the wgpu executor commits every ArrayScope payload shape
  behind the protocol with physical page-table acknowledgements, the LOD
  ladder and montage sessions run on `BindContentPlanes`, and the first
  compute consumer (G6a resident-page histograms) is live. Promotion vs
  VisPy is queue row 3 (evidence-gated).
- **Date:** 2026-07-18
- **Branch note:** authored on `codex/wgpu-renderer-gate-b`; renumber on
  integration if a parallel branch claimed 0057.
- **Related:** ADR 0053 (one scheduler), 0055 (tile/chunk/page split),
  0056 (sparse pyramid); `docs/proposals/tensor-engine-endpoint.md`
  (renderer strategy + command table),
  `docs/proposals/wgpu-renderer-experiment.md` (gate-B evidence).

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
   `EvictChunk`, `UpdateTileInstances`, `SetDisplayMapping`,
   `DispatchHistogram`, `PresentGeneration` — carried by an ordered
   `FrameSubmission`, answered by an auditable `FrameReport` (uploads,
   evictions, histogram results, completion token). Commands speak ADR
   0055/0056 identities (`DataChunkKey`, `ChunkLod`) and normalized
   geometry; nothing in the protocol may name WGSL, GL objects, Qt, Datoviz
   IDs, or one-texture-per-tile.
2. **The protocol schedules nothing** (ADR 0053): a submission is
   already-ordered work; the kernel owns priority, supersession, and pacing.
   The report's `wait_completed` token is how page/staging recycling fences
   GPU work — renderer gate 3's contract, now explicit.
3. **`arrayscope/gpu/wgpu_executor.py` is the first implementation**
   (`WgpuPlaneExecutor`): one 2-D plane pyramid, one rg32float page pool,
   `PageTable` bookkeeping, GPU-side ancestor-fallback lookup, one instanced
   draw, two-pass G6 histogram. Its scope is deliberately the gate-B
   harness's; growth is queue-gated. Default-ring tests
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
