# Current state

**Snapshot: `main`, 2026-07-19.** One line of development. The 2026-07-19
wave is merged: G6 GPU compute, native wgpu overlays + glyph text, opt-in
screen presentation (`wgpu_present_method`), the fill-throughput and
demand-freshness live-path fixes, the codex post-merge review fixes, and
the course reshape ([reviews/2026-07-19-course-review.md](reviews/2026-07-19-course-review.md):
Programs A–F, queue Next section, Done ledger split to
[`queue-done.md`](queue-done.md)). Keep this file a *short* snapshot —
history belongs to the archives, direction to [`queue.md`](queue.md).
Update by replacement, not by layering dated correction blocks.

## Architecture (what stands)

- One execution kernel (`arrayscope/kernel/`), one render pipeline
  (`arrayscope/render/`), one tile lifecycle machine (ADR 0051);
  orchestration on `RenderOrchestrator` (ADR 0045); **one
  scheduling-phase owner** (`render/progressive_scheduling.py`).
- The GPU engine (ADR 0055/0056, G1–G6): Qt-free `arrayscope/gpu/` chunk
  keys/grid, page table, chunk store; the ADR 0057 renderer command
  protocol with the wgpu backend live behind an explicit pin — native GPU
  overlays incl. glyph text, GPU histogram/LOD compute, screen
  presentation opt-in. wgpu leads fast-scroll; promotion evidence is
  queue row 3d.
- Visible-truth machinery: schema-v1 trace bus; `trace_verify`
  invariants; the journey-matrix trajectory gate (first full 15/15
  reached 2026-07-19); framebuffer-to-CPU pixel oracles on the backends
  with fault injection.

## Known open work

Direction and exit gates: [`queue.md`](queue.md). Headlines: wgpu
promotion evidence (row 3d — callback bars, dogfood hours, the FFT-scroll
headline), G7 compressed transport, then the product turn (compare,
plugin ops/sigpy/BART, ingestion — queue rows 5–10). Standing lane:
demand-freshness unit fixture, offscreen cold-tail stall, bounded process
exit, R8 continuity-gate adjudication.

## Material risks

1. **Complexity debt**, still: `FrameSession` ~100 fields;
   residency/visibility facts retain multiple owners. Every fix should
   reduce owner count; the "presentation clock" close-out (Program A)
   is the structural answer.
2. **Acceptance is machine-bound.** Rings 3–4 and the journey matrix run
   only on this machine by hand; CI is offscreen software-GL. Whoever
   changes a display/render/kernel/window lane runs them
   ([testing/README.md](testing/README.md)).

## What is working well

- The Qt-free semantic core, operations pipeline, slicing, profiles, ROI,
  linked-window sync; suite ~2,490 tests in ~2 min parallel, with
  doc-to-test traceability and trajectory-level oracles.
