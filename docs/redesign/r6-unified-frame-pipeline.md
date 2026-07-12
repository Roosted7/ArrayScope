# R6 — Unified frame pipeline

**Goal:** a small image and a montage differ in region geometry, not in render
semantics or scheduling ownership.

## Landed

- `FramePipeline` owns the generic kernel scope and processes `FramePlan`
  regions through the LOD ladder, bounded commit batches, and lifecycle acks.
- `FrameSession` is the shared live frame context; a normal image is a
  one-region plan and a montage is a multi-region plan.
- Camera and presentation identities stay separate from semantic work keys.
- Normal-image interactive requests now supersede stale presentation work
  through the same semantic-key test as montage requests while retaining the
  16 ms input coalescer.

## Exit gate

- No `MontagePipeline` or `MontageRenderSession` type remains.
- One- and multi-region frames use the same planner, lifecycle, surface commit,
  and committed-frame semantics.
- Backend differences remain capability-selected physical mechanics.
