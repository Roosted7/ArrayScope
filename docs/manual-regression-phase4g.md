# Manual Regression: Phase 4g

## Runtime Region Planner And StageCache

1. Open a small 3D array.
2. Add `CenteredFFT(axis=2)`.
3. Open `Developer -> Diagnostics`.
4. Select the `Operations` details tab.
5. Confirm the text shows final region, required input region, expanded axis, transitions, stage-cache candidates, and peak estimate.
6. Scroll slices and confirm Stage cache entries/stores increase after the first slice.
7. Confirm later slices increase Stage cache hits.
8. Add `CenteredFFT(axis=2) + CenteredIFFT(axis=2)`.
9. Confirm the final expanded stage is cached and reused.
10. Open montage over the same sliced axis.
11. Confirm montage tile rendering increases Stage cache hits rather than recomputing every tile.
12. Check profile and scalar hover over nearby slices and confirm reuse.
13. Confirm image rendering, hover scalar readout, profile updates, export frame rendering, and montage tiles still match the expected transformed data.
14. Confirm diagnostics transitions update when the operation stack changes.
15. Lower memory profile or StageCache budget and confirm entries evict without crashing.
16. Edit the operation stack and confirm StageCache entries clear.
17. Export frames from the live window and confirm StageCache hits increase when frames share an expanded transform stage.
