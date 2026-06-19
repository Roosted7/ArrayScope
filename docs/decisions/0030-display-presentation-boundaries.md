# 0030: Display Presentation Boundaries

## Problem

Progressive montage display state was split implicitly between the render
orchestrator, `ImageView2D`, pyqtgraph histogram state, and viewport refreshes.
Panning could appear to fix montage scaling because it started a new viewport
session that reused cached tiles and re-entered window/level code through a
different path than tile completion.

## Decision

Introduce explicit display presentation boundaries:

- `arrayscope.display.planning` decides levels and histogram ranges before Qt
  is called.
- `arrayscope.display.commit.DisplayCommitter` is the gateway that writes
  pixels, levels, histogram ranges, and viewport policy to `ImageView2D`.
- `arrayscope.display.model.frame.CommittedDisplayFrame` records the committed
  value source for hover/status.
- `arrayscope.window.montage_levels.MontageLevelTracker` tracks semantic montage
  histogram coverage independently from viewport canvas pixels.
- `ImageView2D` exposes presentation APIs that keep LUT levels separate from
  semantic histogram/data bounds.

Viewport changes may schedule new visible tile work, but they do not own
semantic window/level state. Partial visible montage canvas histograms are not
automatic window/level sources.

## Consequences

Panning cannot repair or alter image scaling unless new semantic tile coverage is
actually committed. Degenerate bounds are normalized before presentation, so a
single constant tile cannot create a zero-width display window. Future render
refactors should move orchestration into the controller modules without allowing
direct display mutation back into `render.py`.

## Tests Required

Tests cover explicit ImageView presentation state, panning without new tile data,
degenerate previous levels, committed hover coordinates, and architecture guards
that route display commits through `DisplayCommitter`.

