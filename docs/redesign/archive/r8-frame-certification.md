# R8 — Adversarial correctness and performance certification

**Goal:** close R2–R4 performance/evidence debt against the final frame
architecture, not against an intermediate implementation.

## Required matrix

- PyQtGraph and VisPy; one-region and multi-region frames.
- Cold, cached, resident, and memory-pressure conditions.
- Slice/scrub, pan/zoom, levels/LUT, operation change, reload, resize, restore,
  cancellation, stale completion, partial ack, and teardown.

## Hard invariants

- Compatible acknowledged pixels never become black while replacement waits.
- Stale work never commits; camera-only work never reevaluates source data.
- Exact inspection stays native under every display LOD.
- Visible settlement leaves no visible obligation or unacknowledged commit.
- GUI callbacks stay below 50 ms; heartbeat targets approximately 16 ms.
- First VisPy pixels carry rough semantic levels and publish the same rough
  sample in the histogram; first PyQtGraph pixels wait for refined levels.

## Exit gate

- Full default-parallel non-GPU suite green.
- Real-hardware serial GPU harness green with zero stall assertions.
- Workflow benchmark records first-pixel, visible/exact settlement, callback
  gaps, initial level/histogram validity, restored viewport geometry, and
  cold/warm work separately for both backends. It starts from the checked-in
  portable session fixture through the production restore path.
- Existing R2–R4 bars are met without weakening them.

## 2026-07-14 R8C certification evidence

- The 20-source committed-manual-camera transition is green on real Wayland
  for PyQtGraph and VisPy at 1200×820 with a measured image viewport.
- PyQtGraph retains predecessor geometry until refined full-population evidence
  exists. VisPy can commit rough evidence and converges to the same population.
- Operation, channel, complex-mode, axes, and viewport-retarget transitions are
  green on both live backends.
- Core/operations and display/window default-parallel slices are green.
- The repository-wide UI/app exit gate remains open on clean-branch failures
  outside R8C: stale `_montage_session` tests and an independent diagnostics
  cache assertion. R8D and the marathon salvage audit remain blocked.
