# ADR 0043: File-Session Restore Boundaries

Status: Implemented

## Context

Saved file sessions restore operation recipes, display settings, ROIs, viewport range, and window layout. These targets are owned by different systems. Treating them as one after-show corrective step caused startup races: restored view ranges could be overwritten by the first operation-backed presentation, and saved viewport shape was incorrectly used as an outer-window sizing input.

Montage restore has a related constraint: tile payloads may stream after the layout exists. The restore range must be applied when the committed montage scene/plan exists so placeholders appear in the right world location; it must not wait for every tile to finish rendering.

## Decision

File-session restore is represented as a restore transaction with separate consumers:

- Layout consumes only the saved dockless outer window size, before first show.
- Viewport consumes only the saved viewport mode/range when its semantic target is ready.
- Montage viewport restore requires the montage plan, not tile payload completion, so placeholders and final content are presented under the same camera.
- Panel visibility is restored as managed-panel intent, then applied by the normal progressive dock controller after the window is visible.

`viewport_shape` remains a semantic observation for viewport/render planning and persistence diagnostics. It must not be used to synthesize outer window size.

## Consequences

Session re-open should not ratchet the viewport smaller or wider across repeated launches. Operation-backed restores can show placeholders or delayed tiles at the restored camera position instead of briefly auto-fitting and correcting later.

Future startup restore work should add fields to the transaction and give them one owner. It should not add independent `_pending_*` flags or after-show resize repair callbacks.
