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
