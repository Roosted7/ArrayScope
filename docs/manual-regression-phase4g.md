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
9. Render and confirm Diagnostics -> Operations shows an optimization summary and no FFT/IFFT transform transitions for the simplified pair.
10. Confirm StageCache does not fill with unnecessary FFT/IFFT stages for the simplified pair.
11. Confirm the exact view is not refused as over budget when the requested visible slab fits, including non-contiguous axis ranges.
12. Replace the stack with a single `CenteredFFT(axis=2)`.
13. Scroll slices and confirm StageCache entries and hits increase for the expanded FFT stage.
14. Add `ReverseAxis(axis=a) + ReverseAxis(axis=a)` and confirm output matches the original data and diagnostics shows simplification.
15. Add two adjacent same-axis crops and confirm output shape/content are correct and diagnostics shows crop composition.
16. Open montage over the same sliced axis.
17. Confirm montage tile rendering increases StageCache hits rather than recomputing every tile.
18. Check profile and scalar hover over nearby slices and confirm reuse.
19. Confirm image rendering, hover scalar readout, profile updates, export frame rendering, and montage tiles still match the expected transformed data.
20. Confirm diagnostics transitions update when the operation stack changes.
21. Use `Performance -> Use Less Memory`, `Use More Memory`, `Decrease Render Budget`, and `Increase Render Budget`; confirm Diagnostics memory/cache budgets update and entries evict without crashing.
22. Edit the operation stack and confirm StageCache entries clear.
23. Export frames from the live window and confirm StageCache hits increase when frames share an expanded transform stage.
24. Save and load a recipe and confirm user operation rows remain literal, not rewritten by the optimizer.
