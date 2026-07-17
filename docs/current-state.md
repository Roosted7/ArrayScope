# Current state

**Snapshot: `main`, 2026-07-18.** One line of development; the G5 sparse
pyramid, the four 07-17 chip landings (retained-satisfaction trace, key-owner
consolidation, VisPy+PyQtGraph pixel oracles, ImageViewShell dedup), the
journey-matrix trajectory gate, and the `ProgressiveSchedulingPolicy`
phase owner are all merged. Keep this file a *short* snapshot — history
belongs to the archives, direction to [`queue.md`](queue.md). Update by
replacement, not by layering dated correction blocks.

## Architecture (what stands)

- One execution kernel (`arrayscope/kernel/`), one render pipeline
  (`arrayscope/render/`), one tile lifecycle machine (ADR 0051);
  orchestration on `RenderOrchestrator` (ADR 0045).
- The GPU engine (ADR 0055/0056, G1–G5): Qt-free `arrayscope/gpu/` chunk
  keys/grid, page table, chunk store; one source-grid route for anisotropic
  reduced pages and exact clipped draw geometry; both backends consume
  checked `DataChunkKey` materializations; physical presentation truth is a
  standing audited invariant.
- **One scheduling-phase owner** (2026-07-18):
  `arrayscope/render/progressive_scheduling.py` —
  `ProgressiveSchedulingPolicy` owns the COVERAGE→REFINE verdict, the
  lifecycle first-pixel close predicate, and the refinement replan edge;
  ladder, admission, level/histogram work, atomic handoffs, and commit
  batching read it (progressive presentation contract in
  [architecture/rendering.md](architecture/rendering.md)).
- Visible-truth machinery: schema-v1 trace bus; `trace_verify` invariants
  incl. `no_phase2_submit_during_coverage`, retained-satisfaction closure,
  identity-rejection, ack-churn and bail-loop limits; the journey-matrix
  trajectory gate (`arrayscope.tools.journey_matrix`); framebuffer-to-CPU
  pixel oracles on BOTH backends with fault injection.

## Verified behavior (real display, 2026-07-17/18)

- Cold raw fills settle 60/60 exact targets on both backends with zero
  trace violations; VisPy submits zero phase-2 jobs before coverage closes;
  PyQtGraph first fill arrives in bounded ≤12-item batches.
- Chunked residency: ±1 window shift uploads boundary strips only;
  revisit/scroll-back = 0 uploads; never-black fine↔coarse transitions with
  fault-injected proof.
- complex64 PyQtGraph native convergence restored (stress-matrix row is a
  hard pass).

## Known open work

Direction and exit gates: [`queue.md`](queue.md). Headlines: the red
journey-matrix cells — AUTO-camera demand freshness (dead gesture edge;
strict xfail pin `tests/ui/test_lod_demand_freshness.py`), priority order
through commit construction, in-flight re-rank on camera re-anchor (in
flight on `codex/camera-reanchor-rerank`) — then G6. The complex VisPy
phasing oracle has one pre-existing red on main (payload-level identity vs
shader-uniform mismatch, under investigation). Perf bars stay parked per
Thomas 2026-07-17 (act only on true stalls).

## Material risks

1. **Complexity debt**, still: `FrameSession` ~100 fields; the policy owner
   removed the phase/first-pass duplicates, but residency/visibility facts
   retain multiple owners. Every fix should reduce owner count.
2. **Acceptance is machine-bound.** Rings 3–4 and the journey matrix run
   only on this machine by hand; CI is offscreen software-GL. Whoever
   changes a display/render/kernel/window lane runs them
   ([testing/README.md](testing/README.md)).

## What is working well

- The Qt-free semantic core, operations pipeline, slicing, profiles, ROI,
  linked-window sync; suite ~2,300 tests in ~2 min parallel, with
  doc-to-test traceability and trajectory-level oracles.
