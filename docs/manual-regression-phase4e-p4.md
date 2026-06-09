# Manual Regression - Phase 4e P4

Run these checks before release-level changes to scheduler, rendering, prefetch, or operation cost
behavior.

## Visible Work

- Open a 3D or 4D array with an FFT operation.
- Rapidly scroll a non-image slice dimension.
- Confirm the previous image remains visible while work runs.
- Confirm only the newest result commits and stale results do not clear the newer overlay.

## Budget Decisions

- Set Performance > Render Memory Budget low enough to force a degraded preview.
- Confirm the preview is marked with an overlay and is not reused as an exact cached image.
- Lower the budget further or choose a view that cannot be previewed safely.
- Confirm rendering refuses with a useful status message and keeps the previous image.

## Chunking

- Trigger a large exact render that uses chunked evaluation.
- Change slices before it finishes.
- Confirm old work stops after a chunk boundary and does not commit over the newer request.

## Prefetch

- Enable nearby-slice prefetch.
- Confirm no prefetch starts while visible rendering is running.
- Stop interacting and confirm prefetch starts after the idle delay.
- Confirm cheap operation-backed prefetch can store a nearby slice.
- Confirm expensive FFT-backed prefetch is skipped.

## Montage

- Enable a broad montage range.
- Confirm bounded visible-tile behavior still works and no degraded preview path is used for montage.
