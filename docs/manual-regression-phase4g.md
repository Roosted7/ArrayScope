# Manual Regression: Phase 4g

## Runtime Region Planner

1. Open a small 3D array.
2. Add `CenteredFFT(axis=2)`.
3. Open `Developer -> Diagnostics`.
4. Select the `Operations` details tab.
5. Confirm the text shows final region, required input region, expanded axis, transitions, stage-cache candidates, and peak estimate.
6. Scroll slices and confirm the image updates correctly.
7. Add `Crop + Reverse + FFT`.
8. Confirm image rendering, hover scalar readout, profile updates, export frame rendering, and montage tiles still match the expected transformed data.
9. Confirm diagnostics transitions update when the operation stack changes.
10. Confirm no StageCache bar or StageCache text section is present yet.

## Stage Cache Future Checks

Do not expect runtime stage reuse in this step. The planner may report stage-cache candidates, but no
stage-cache object is allocated and repeated FFT slice scrolling may still recompute expanded stages.
