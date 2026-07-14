# R8 current status

Updated: 2026-07-14

R8 remains limited to viewer truth and convergence. Throughput, scheduling,
admission-policy changes, and performance cleanup remain blocked. The complete
chronological investigation is preserved in
[`archive/r8-investigation-2026-07-14.md`](archive/r8-investigation-2026-07-14.md).

## Gate status

- **R8A pixel truth:** complete.
- **R8B complex truth:** complete.
- **R8C.1 semantic transitions:** complete.
- **R8C.2 viewport/convergence boundaries:** complete.
- **R8C.3 semantic level-evidence ownership:** active.
- **R8C.4 committed manual-camera policy:** blocked on R8C.3.
- **R8C final transition certification:** pending.
- **Marathon salvage audit:** deferred until all R8C gates pass.
- **R8D performance work:** blocked.

The earlier orange-background complex-rendering failure is resolved. Both
backends now enforce typed target/acknowledgement compatibility, distinguish
scalar, complex, and RGB texture storage, and use placeholders when a current
semantic target lacks compatible physical pixels. The synthetic adversarial
complex fixture and the scalar/complex back-to-back transition coverage are
green. The current blocker is no longer complex rendering.

## Active R8C.3 invariant

> Semantic level and histogram evidence has one explicit, bounded owner
> independent of tile visibility and texture residency. Offscreen sources
> required for semantic statistics are evidence work, never visible tile work
> or generic prefetch.

Montage levels and histograms describe the complete semantic source population,
not merely current canvas pixels. The viewer may keep the predecessor committed
while successor evidence is incomplete, but the successor must eventually
appear without another user action.

## Exact current reproduction

The blocking reproduction is a PyQtGraph manual-camera source-population
transition:

- the current onscreen target is correctly acknowledged;
- plan geometry names sources `0..19`;
- committed geometry remains the predecessor `0..4`;
- the last presentation outcome is `level-evidence-wait`;
- the evidence tracker is exhausted at `MONTAGE_VISIBLE_SUBSET` with sources
  `{0,1,2,5,6,10}`;
- no pending evidence item, scan, or evidence worker remains;
- raw montages are excluded from generic prefetch;
- level evidence currently scans only rendered payloads;
- the attempted CPU-readiness and prefetch-busy changes were insufficient and
  reverted.

This is an ownership gap, not permission to classify offscreen sources as
visible or to publish partial semantic levels. A dedicated bounded evidence
path must produce reusable per-source statistics without texture upload or
visible-tile admission.

## Completed boundary work

- Visible completion is gated by the physical onscreen set rather than the
  broader coverage ring (`359f618`).
- Tiles that only touch a viewport edge with zero area are not visible
  (`4a05cdc`).
- First-commit resize intent is preserved across PyQtGraph and VisPy
  (`305aa69`).
- A settled restored viewport is released when a later top-level resize
  establishes new user intent (`8aeafec`).

These fixes prevent hidden or boundary-only tiles from masquerading as visible
obligations. They do not supply full-population semantic evidence.

## Uncommitted follow-up held outside this handoff

The working tree retains incomplete follow-up from the investigation:

- a committed-manual-camera auto-fit guard and its stronger two-backend
  transition test; PyQtGraph remains blocked on R8C.3;
- remaining canonical `_frame_session` ownership/test migration;
- ROI interaction test corrections that expose predecessor-value and duplicate
  layout-generation failures.

Those changes are intentionally not part of this documentation commit. The
broader affected UI check produced three non-green cases: cached PyQtGraph
tile-layer presentation did not settle, hidden ROI inspection read predecessor
semantics, and ROI layout refresh produced an extra generation. Do not bundle
those failures into the R8C.3 evidence-owner implementation.

## Next bounded slice

Add tests for a dedicated semantic-evidence owner before production code. The
tests must distinguish evidence evaluation from visible tile evaluation and
prove bounded work, generation-guarded reuse, full expected-source coverage,
correct levels generation, eventual PyQtGraph commit, and unchanged VisPy
truth. Do not modify generic prefetch, visible admission, LOD policy, or R8D
performance code in this slice.
