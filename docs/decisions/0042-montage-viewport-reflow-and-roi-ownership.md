# ADR 0042: Montage viewport reflow and ROI ownership

- **Status:** Accepted and implemented for the tiled montage path
- **Date:** 2026-06-26

## Context

Montage resize and column reflow exposed an ownership leak. `frame_renderer.py` was deciding too
much about viewport meaning while also coordinating sessions, timers, tile work, committed frames,
and backend commits. That let layout-only changes behave like semantic montage growth: manual views
could refit, visible zoom could change on resize, and ROI graphics needed ad hoc correction after the
plan moved tiles.

The desired behavior is the PyQtGraph-like mental model:

- untouched automatic and Fit views hug the relevant side of the viewport and recompute their fitted
  range as the viewport changes;
- a truly manual pan/zoom preserves screen zoom on resize and does not refit on column reflow;
- when the same montage source indices move to different tile positions, source-attached geometry
  follows the same source-local coordinates;
- when the tiled dimension scrolls to a different source set, the world position remains stable and
  samples the new content underneath it.

## Decision

Montage viewport reflow policy belongs in the Qt-free viewport layer, not in the renderer or backend.

`arrayscope.display.viewport` owns manual screen-zoom preservation across plain widget resize.

`arrayscope.window.montage_viewport` owns:

- the applied montage viewport plan;
- exact full-plan and square-pixel fit ranges;
- manual layout remapping by `source_index` plus tile-local coordinate;
- the pure `MontageViewportReflow` decision returned by `retarget_montage_viewport_plan`;
- canonical ROI remapping through old/new source layout maps.

`arrayscope.window.frame_renderer` owns:

- reading current UI/controller facts, such as Fit lock and `ViewportController.is_near_auto`;
- applying a returned view range to the ViewBox;
- updating `last_auto_view_range` only when the pure decision produced a new auto range;
- mirroring canonical `RoiSelection` geometry into graphics items and stores.

Backends own only item, texture, visual, camera, and overlay mechanics. They do not decide whether a
view is manual, automatic, or near automatic, and they do not own ROI semantic geometry.

During a resize/layout reflow:

- Fit uses the exact full applied plan bounds.
- Near-auto uses the square-pixel fitted range for the new plan and viewport size only when all four
  current edges are near the fitted range that would be applied.
- Manual resize preserves screen zoom in `ViewportController`, so shrinking the widget shows less
  content at the same scale and growing it shows more content at the same scale.
- Manual montage layout reflow uses source-local translation only when the source set is unchanged
  and the layout moved; it does not apply another resize-derived zoom change.
- Manual paths never use visible-tile count, containment of full bounds, or zero-visible-tile escape
  hatches to enter auto-fit.

ROI reflow builds source layout maps once per transition, then remaps existing selections by source
index and tile-local coordinate. This is O(tile count + ROI count) for a reflow and does not add tile
scans to pointer motion.

## Consequences

Manual resize and manual layout reflow cannot silently shrink the content by zooming out or jump to
auto-fit. The only re-entry into automatic behavior is an all-four-edges near-auto check against the
next auto range, or an explicit Fit lock.

ROI graphics remain mirrors of `RoiSelection`; there are no backend-specific ROI compatibility
wrappers. Committed tiled value semantics continue to answer ROI statistics after reflow.

The renderer still coordinates a large montage lifecycle, but it no longer owns viewport reflow
semantics. Future renderer splits should preserve this boundary rather than moving the policy back
into session or backend code.

## Alternatives considered

- **Treat any full-visible or zero-visible reflow as auto-like.** Rejected because manual zoomed-out
  views can legitimately contain all tiles or temporarily intersect no tile after a layout change.
- **Scale manual ranges inside montage layout reflow.** Rejected because resize is already handled by
  `ViewportController`; applying another scale during layout reflow made content appear smaller.
- **Let ROI graphics compensate locally.** Rejected because graphics items are views; source-local ROI
  geometry must be represented in the canonical selection model.
- **Defer ROI reflow until committed stats refresh.** Rejected because the visible geometry would no
  longer match the data it claims to inspect.

## Validation

Pure viewport tests cover manual resize, manual layout reflow, near-auto reflow, and scrolled source
sets. Montage tests cover ROI remapping for same-source layout changes, stable world position for
scrolled source windows, committed tiled ROI statistics, and PyQtGraph/VisPy geometry/value parity.

Manual regression should include resizing a tiled montage while zoomed in, far zoomed out, near auto,
and Fit-locked, plus changing the column layout with ROIs on and across tiles.
