# ADR 0044: Viewport-scoped tiled residency and acknowledged commit state

Status: Accepted; first acknowledgement repair implemented, full normal-image viewport retarget remains
roadmap work.

## Context

ADR 0039 and X1 made normal images, internally tiled large planes, and montages share one semantic frame
planner and tiled presentation model. That is the correct direction, but it creates a stricter residency
contract.

A planned tile, a requested tile, and a resident tile are different things:

- **planned** means the semantic frame contains the region;
- **active** means the current viewport/frame wants the region drawable now;
- **near** means the region is useful speculative residency;
- **resident** means the backend accepted the payload and can draw or inspect from it.

Before this ADR, some code paths could build committed frame semantics from the requested tiled
presentation before backend acknowledgement had been folded into the committed tile state. That makes
scene diagnostics and semantic value availability too optimistic when a backend defers, rejects, or
budget-limits tile upserts.

The other related issue is internally tiled normal images. The planner can split a huge single plane into
regions, but normal-image viewport changes do not yet retarget tiled active/near sets the way montage
viewport changes do. If visible-only active regions were enabled before that retarget path exists, panning
could reveal non-resident tiles without scheduling new visible work.

## Decision

Committed tiled-frame semantics must be built from backend-acknowledged tile state.

- `DisplayCommitter.commit_tile_layer()` commits through the backend first.
- The acknowledged `TilePresentationState` is the source of committed `DisplayScene` residency and
  `TiledValueSource` payload availability.
- Code must not treat `TilePresentationDelta.upserts` as resident until the backend report accepts them.

Viewport-scoped tiled rendering must be storage based, not montage-mode based.

- Montage is only one tiled layout. A huge single plane may also be tiled.
- A viewport retarget should be scheduled when the current committed scene uses tiled storage and the
  viewport changes, even when `view_state.montage_axis is None`.
- The project must not enable visible-only active-region commits for internally tiled normal images until
  that retarget path exists.

Large-image materialization should become region-first.

- A tiled frame plan should allow each `FrameRegion` to be materialized independently.
- The first implementation may wrap the existing eager display image, but the control-plane contract
  should move toward region reads with deadlines and supersession.
- Out-of-core, chunked, remote, and server-backed sources should plug in below the same region contract.

Backend and LOD defaults remain gated by hardware evidence.

- Backend capabilities should include queried limits and proven allocation outcomes.
- Allocation failure and context loss must downgrade strategy without corrupting semantic frame state.
- Async/source-provided LOD must use compatible pages, arrays, or virtual-texture/page-table mechanics;
  arbitrary reduced tile shapes must not be mixed into fixed native atlas pages.

## Consequences

Positive consequences:

- Scene residency, diagnostics, hover values, profiles, and ROI availability describe what the backend
  actually accepted.
- Normal images and montages can continue converging on one semantic model without pretending they have
  identical physical update paths.
- Future lazy/out-of-core sources have a clear contract: provide regions, not necessarily full frames.
- Backend fallback and context-loss handling can become policy decisions instead of ad hoc error paths.

Costs and tradeoffs:

- A tiled single-plane viewport-retarget path must be implemented before the active set can safely become
  visible-only.
- Region-first materialization introduces a new source boundary that existing eager sources must adapt to.
- More conformance tests are needed around deferred backend upserts and backend failure modes.

## Alternatives considered

### Keep all normal images raster-only

This avoids normal-image viewport retargeting, but it fails the performance target for very large planes
and out-of-core sources. It also preserves a semantic split between one-tile montage and one large normal
image.

### Make every image an atlas tile immediately

This simplifies one backend path, but it is not optimal for small images and creates unnecessary atlas,
quad, and residency overhead. Raster and tiled storage should remain interchangeable physical strategies
behind one semantic presentation model.

### Build committed scenes from requested deltas

This is simpler, but wrong. It makes diagnostics and value availability optimistic under backend budget,
deferral, rejection, allocation failure, or context loss.

## Migration

1. Keep the acknowledgement repair in `DisplayCommitter`.
2. Add conformance tests where a fake backend accepts only part of a tiled delta and verify committed
   scene residency follows accepted payloads.
3. Change viewport retarget scheduling from montage-mode based to tiled-scene based.
4. Add a region materialization source protocol and adapt eager display images through it.
5. Enable visible-only active regions for internally tiled normal images only after steps 2 to 4 pass.
6. Use X5 hardware traces to decide backend defaults, allocation fallback, and LOD enablement.

## Related records

- ADR 0038: backend composition.
- ADR 0039: unified image surface and deadline scheduler.
- ADR 0040: backend-aware presentation convergence.
- ADR 0041: LOD selection, materialization, and residency.
- ADR 0042: montage viewport reflow and ROI ownership.
