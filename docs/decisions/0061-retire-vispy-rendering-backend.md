# 0061 — Retire the VisPy rendering backend

**Status:** Accepted and implemented (2026-07-27).

## Context

VisPy was ArrayScope's first GPU rendering path and proved the value of shader
windowing, persistent tiled residency, backend-neutral presentation semantics,
and output-driven pixel gates. Those durable contracts now live above the
renderer. WGPU implements the GPU command, page-residency, compute, overlay,
and presentation paths; PyQtGraph remains the CPU/headless/remote path.

Keeping VisPy after that migration would retain a third dependency stack, a
third settings/menu choice, and a third family of implementation tests without
owning a distinct product role. The historical benchmark and journey-matrix
evidence supports WGPU as the maintained GPU path. A backend may be retired
only if its user-visible contracts remain tested on the surviving backends.

## Decision

1. WGPU and PyQtGraph are the only maintained rendering backends. AUTO selects
   WGPU when its device gate succeeds and otherwise selects PyQtGraph.
2. Delete the VisPy runtime, dependency/extra, factory and enum surface,
   settings/menu choice, backend-specific diagnostics, and implementation-only
   tests. Do not leave import shims, aliases, or a dormant compatibility
   backend.
3. Treat a persisted legacy `vispy` setting as AUTO input at the settings
   boundary. This is data migration, not a backend compatibility layer.
4. Preserve backend-neutral contracts and their coverage. Tests that used
   VisPy only as a convenient implementation host move to WGPU and/or
   PyQtGraph as appropriate: backend selection, presentation settlement,
   progressive coverage, interaction/viewport semantics, physical-pixel
   truth, lifecycle acknowledgement, and tool/backend matrices.
5. `all` in maintained tools means WGPU plus PyQtGraph. The real-Wayland
   journey gate is six journeys across those two backends (12 cells).
6. WGPU is the current performance baseline. PyQtGraph remains first-class for
   correctness and receives the standing 2× performance allowance for its
   CPU/headless/remote role.
7. Preserve released changelog entries, old ADR rationale, archived/dated
   reviews, graveyard entries, and historical benchmark rows as evidence.
   They must be labelled historical where a live document could otherwise
   mistake them for a current gate.

## Consequences

- One GPU implementation owns current device behavior, while semantic state,
  scheduling, lifecycle, interaction, and pixel truth remain shared.
- CI and normal development no longer install VisPy or initialize its OpenGL
  context.
- A user with an old VisPy preference receives normal capability-probed AUTO
  selection rather than a startup failure.
- Removing backend-specific tests lowers the raw test count. Acceptance is
  based on surviving contract coverage, not preserving the old count.
- Historical VisPy timings remain useful decision evidence but cannot pass or
  fail a current performance or release gate.

## Alternatives rejected

- **Keep VisPy as an unmaintained fallback.** A selectable backend is a product
  promise and would keep its dependency, test, and migration burden.
- **Leave compatibility modules that forward to WGPU.** This preserves a false
  identity and lets stale imports/settings outlive their owner.
- **Delete every test that mentions VisPy.** Several such tests own
  backend-neutral correctness; those assertions must be ported before the
  implementation-specific fixture is removed.

## Validation

- Default and focused display/window/UI suites collect with no VisPy
  dependency or module.
- Backend factory, settings migration, menus, tools, stress rows, and journey
  matrices enumerate only WGPU and PyQtGraph.
- Physical-pixel/reference coverage remains for WGPU executor output and
  PyQtGraph Qt raster output.
- Reduced phase-vector pages retain both circular hue and resultant coherence
  in WGPU physical pixels; zero-resultant cancellation remains black.
- Backend-labelled diagnostics and profiler artifacts fail rather than
  accepting a PyQtGraph fallback as WGPU evidence.
- Live docs, CI, packaging, and diagnostics agree on the two-backend set.

## Related decisions

- [0038 — Render backend composition](0038-render-backend-composition.md)
- [0047 — Automatic backend selection](0047-auto-image-backend-selection.md)
- [0050 — Async multi-resolution tile residency](0050-async-multi-resolution-tile-residency.md)
- [0057 — Renderer command protocol](0057-renderer-command-protocol.md)
- [0058 — Canonical tile orientation](0058-canonical-tile-orientation-and-display-transpose.md)
