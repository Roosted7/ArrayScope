# 0024 - Canvas preserve transaction

## Problem

The preserve-canvas logic was effective but invasive: `WindowLayoutManager` owned a procedural retry
state machine, the strong Wayland workaround was mixed into default behavior, and recent state was
printed to stdout instead of exposed through the developer diagnostics UI. Tests also depended on
private layout-controller fields.

## Decision

ArrayScope now owns preserve-canvas behavior in
`arrayscope.window.canvas_preserve.CanvasPreserveController`. The View menu exposes three resize modes:
`Off`, `Best effort`, and `Strong Wayland`. Best effort uses bounded `resize()` correction only.
Strong Wayland is explicit, gated to Wayland platforms, and may apply temporary captured
QWidget/QWindow min/max constraints plus commit pokes/nudges when ordinary correction does not settle.
The controller captures and restores the real current constraints, records recent events in a small
diagnostics buffer, and emits debug logging instead of stdout output.

Developer -> Diagnostics shows the current preserve mode, platform, last transition/result, attempts,
strong-path state, constraint state, and recent events. Phase 4g StageCache and operation-planner work
remain out of scope for this decision.

## Consequences

Panel transitions remain best effort because window managers and compositors can still constrain
top-level sizes. The strong path is opt-in and inspectable instead of being part of ordinary behavior.
`WindowLayoutManager` is smaller and focused on panel state delegation. Tests can assert behavior
through `CanvasPreserveController` diagnostics and explicit controller seams.

## Rejected Alternatives

- Remove the strong nudge path completely: rejected because it is still useful on Wayland when ordinary
  correction does not commit.
- Always use the strong nudge path: rejected because it is heavier, more platform-sensitive, and can
  interfere with detached dialog mapping.
- Keep stdout diagnostics: rejected because Developer Diagnostics is the right production-quality
  surface for this runtime state.
- Implement StageCache now: rejected because it is a separate Phase 4g feature with a larger planner
  and cache design.

## Tests Required

Settings and menu tests cover all three modes and persistence. Panel tests cover diagnostics recording,
best-effort behavior, Wayland-gated strong preserve, non-Wayland strong skips, exact constraint
restore, cancellation release, and detach avoiding strong preserve. Diagnostics tests cover the Canvas
Preserve section and visual row. Architecture guards keep preserve internals out of
`WindowLayoutManager` and prevent stdout preserve diagnostics.

## Manual Checks Required

Run the Wayland panel checklist with Developer Diagnostics open. Verify Best effort, Off, and Strong
Wayland modes; confirm no top-left jump, no stdout preserve lines, detached panels map cleanly, and
manual resizing works after any strong preserve transition.
