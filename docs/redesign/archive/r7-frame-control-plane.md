# R7 — Frame control-plane dissolution

**Goal:** control flow names and module boundaries describe the shared frame
transaction; old montage-era modules cannot attract new ownership.

## Landed

- `window/frame_controller.py` owns thin frame entry/retarget coordination.
- `window/frame_effects.py` implements worker evaluation and bounded commit
  effects for `FramePipeline`.
- `window/frame_runtime.py` owns interaction/runtime hooks and the
  diagnostics-only stall probe.
- `window/frame_session.py` owns the Qt-free frame context.
- The former `frame_renderer.py`, `montage_commit.py`, `montage_runtime.py`,
  and `montage_session.py` modules were deleted rather than kept as shims.

## Remaining exit evidence

- Finish the R8 full-suite and hardware gates.
- Continue deleting lifecycle-backed mutable collection facades when callers
  are migrated; do not create additional session-local queues or maps.
