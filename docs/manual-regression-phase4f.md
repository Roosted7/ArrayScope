# Manual Regression - Phase 4f

## Progressive Montage

1. Open a 3D or 4D array with enough slices for a multi-row montage.
2. Enable montage on a non-image dimension.
3. Pan and zoom so only part of the montage grid is visible.
4. Confirm the canvas does not jump, shrink, or change origin while tiles load.
5. Confirm cached tiles appear immediately.
6. Confirm missing tiles clear old content immediately.
7. For fast loads, confirm loading overlays do not flash.
8. For slower loads, confirm loading overlays appear after a short delay.
9. Lower the render memory budget enough that an individual tile cannot fit.
10. Confirm skipped tiles show a detailed warning explaining the tile estimate, budget, shape, and recovery options.
11. Hover a loaded tile and confirm a numeric value is shown.
12. Hover a loading tile and confirm loading status is shown immediately instead of a numeric NaN.
13. Hover a gap and confirm no value is shown.
14. Enable live profile and place the marker on a loaded tile; confirm the profile updates.
15. Move the live-profile marker to a loading tile; confirm loading status appears and no stale profile is scheduled.
16. Rapidly change montage slice/range while tiles are loading; confirm stale tile results do not clear the current overlay or replace the current canvas.
17. Confirm the final montage overlay clears only for the current session.

## Memory Policy and Diagnostics

1. Open a small 3D array.
2. Open Developer -> Diagnostics.
3. Confirm memory profile, system total/available, process RSS, and policy budgets are visible.
4. Confirm the filling bars show system, RSS, image cache, tile cache, profile cache, render, canvas, and prefetch usage with readable values in the bars.
5. Confirm scheduler bars show completed, planned, cancelled, stale, and failed work for visible/pixel/profile/ROI/prefetch schedulers.
6. Switch memory profile between Conservative, Balanced, Aggressive, and Custom.
7. Confirm Diagnostics budget values and bars update without disturbing the image view.
8. Lower render cap to 128 or 256 MiB.
9. Confirm visible render/montage warnings use the lowered cap.
10. Confirm cache budgets shrink/evict without crashing.
11. Open a montage with many tiles.
12. Confirm tiles load progressively and are not skipped due to aggregate tile count.
13. Force a single tile over budget and confirm the detailed skipped warning appears.
14. Enable prefetch and confirm Diagnostics shows prefetch scheduling/blocking counters.
15. Close Diagnostics and confirm its timer stops and the main window remains usable.
16. Confirm Diagnostics does not resize or disturb the main window canvas.
