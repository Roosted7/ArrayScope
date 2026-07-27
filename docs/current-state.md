# Current state

**Snapshot: `main`, 2026-07-27.** One line of development. WGPU and
PyQtGraph are the maintained rendering backends: WGPU is the GPU/rendering
certification path and PyQtGraph is the CPU/headless/remote path. The legacy
VisPy renderer was retired after its engine mechanisms moved behind the
backend-neutral command, residency, and presentation contracts
([ADR 0061](decisions/0061-retire-vispy-rendering-backend.md)). Keep this
file a *short* snapshot —
history belongs to the archives, direction to [`queue.md`](queue.md).
Update by replacement, not by layering dated correction blocks.

## Architecture (what stands)

- One execution kernel (`arrayscope/kernel/`), one render pipeline
  (`arrayscope/render/`), one tile lifecycle machine (ADR 0051);
  orchestration on `RenderOrchestrator` (ADR 0045); **one
  scheduling-phase owner** (`render/progressive_scheduling.py`).
- The GPU engine (ADR 0055/0056, G1–G6): Qt-free `arrayscope/gpu/` chunk
  keys/grid, page table, chunk store; the ADR 0057 renderer command
  protocol with the WGPU backend — native GPU
  overlays incl. glyph text, GPU histogram/LOD compute, screen
  presentation opt-in.
- Visible-truth machinery: schema-v1 trace bus; `trace_verify`
  invariants; the 12-cell maintained-backend journey matrix (six journeys
  across WGPU and PyQtGraph); framebuffer-to-CPU pixel oracles on both backends
  with fault injection.

## Known open work

Direction and exit gates: [`queue.md`](queue.md). Headlines: WGPU
field evidence (callback bars, dogfood hours, the FFT-scroll
headline), retention truth, then the product turn (compare,
plugin ops/sigpy/BART, ingestion — queue rows 5–10). Standing lane:
demand-freshness unit fixture, offscreen cold-tail stall, bounded process
exit, R8 continuity-gate adjudication.

## Material risks

1. **Complexity debt**, still: `FrameSession` ~100 fields;
   residency/visibility facts retain multiple owners. Every fix should
   reduce owner count; the "presentation clock" close-out (Program A)
   is the structural answer.
2. **Acceptance is machine-bound.** Rings 3–4 and the journey matrix run
   only on this machine by hand; CI is offscreen and cannot certify physical
   WGPU/Vulkan presentation. Whoever
   changes a display/render/kernel/window lane runs them
   ([testing/README.md](testing/README.md)).

## What is working well

- The Qt-free semantic core, operations pipeline, slicing, profiles, ROI,
  linked-window sync; suite ~2,490 tests in ~2 min parallel, with
  doc-to-test traceability and trajectory-level oracles.
