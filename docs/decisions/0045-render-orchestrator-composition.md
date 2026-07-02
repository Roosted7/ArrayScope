# 0045 — Render orchestration is a composed object, not window mixins

**Status:** Implemented (v32, 2026-07).

## Context

`ArrayScopeWindow` was assembled from 14 mixins sharing one flat `self`
namespace. `FrameRenderMixin` alone was ~3,900 lines and wrote 71 distinct
attributes onto the shared window. The Qt-free control-plane models extracted
in N6/X1/X2 (`FramePlanner`, `WorkGraph`, `TileAdmissionQueue`,
`PresentationGenerationTracker`, `LevelConvergenceStrategy`, `StageFanInState`)
were correct, but the orchestration *around* them still lived as mixin methods
with unpartitioned mutable state.

The v32 audit measured the consequences: eight parallel revision-counter
schemes, three session-key constructions, five staleness-check patterns, four
tile-priority systems, three work-admission paths, and eleven render-path
timers. Each existed because the invariants of one mixin were not visible or
trustworthy from another, so every fix added another token/guard/timer — which
became the habitat of the next bug. Concrete crashes traced to this structure:
unparented timer chains firing into destroyed windows (canvas preserve), a
resize retarget reading live `view_state` against a stale montage session
(line-plot switch segfault), and `FramePlanner` re-deriving montage columns
that the viewport planner had already overridden.

## Decision

Rendering orchestration is owned by one composed object:

- `RenderOrchestrator` (`window/render.py`) composes the former render mixins
  (`DisplayPresentationMixin`, `FrameRenderMixin`, `RenderPrefetchMixin`,
  `RenderResourceMixin`) and owns **all** frame-planning, montage-session,
  presentation-commit, prefetch, and resource-budget state.
- The window composes exactly one orchestrator: `window.renderer`. The
  orchestrator is a `QObject` child of the window, so orchestrator timers are
  dropped when the window's C++ object is destroyed.
- Orchestration code reaches window services (`view_state`, `document`,
  widgets, evaluation controllers, layout manager) via `self.win.<name>`.
  Window-level semantic state is never duplicated on the orchestrator.
- The window exposes a thin rendering API: public entry points (`render`,
  `request_render`, `update_image_view`, `auto_window_levels`,
  `fit_image_to_view`, `one_to_one_image`, …), signal-handler delegates, and
  read-only properties (`_montage_session`, `display_geometry`,
  `_current_montage_plan`, `_current_montage_geometry`). Everything else on
  the orchestrator is internal; sibling controllers and tests target
  `window.renderer` directly for internals.
- Helpers that receive the orchestration owner (`montage_prefetch`,
  `viewport_bridge`, module-level `_montage_*` helpers) take the orchestrator
  and use `.win` for window services.

The orchestrator being internally mixin-composed is an implementation detail
scheduled for further splitting (montage orchestration vs. presentation vs.
prefetch), but the ownership boundary — render state off the window — is the
durable decision.

## Consequences

- Local rendering fixes are local again: render state has one owner and one
  namespace; the window keeps only semantic state and services.
- Timer lifetime is structural, not per-site discipline: a `QTimer` parented
  to the orchestrator (or scheduled with a receiver context) cannot outlive
  the window.
- Test fakes must model the composition: a fake window used with unbound
  orchestrator methods sets `fake.win = fake`; unit tests construct
  `RenderOrchestrator.__new__(RenderOrchestrator)` with a stub `win`.
- The remaining unification work (one generation contract instead of parallel
  revision counters; admission exclusively through `WorkGraph`) now has a
  single home and is tracked in the roadmap (Y1).

## Related

- Supersedes the "window mixin" aspect of earlier orchestration notes in
  ADR 0038/0039; the semantic contracts in those ADRs are unchanged.
- The v32 audit (`docs/reviews/v32-composition-audit.md`) records the
  measurements and the removals that accompanied this change (dead
  pre-tiles render-decision layer, stage-warmup module, sys.modules-replacing
  test imports).
