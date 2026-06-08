# Manual Regression - Phase 4d

Run these checks before release-level changes to rendering, montage, panels, viewport behavior, or
background evaluation.

For Wayland-specific managed-panel checks, also run `docs/manual-regression-wayland-panels.md`.

## Rapid Visible Rendering

- Open a 3D or 4D array with at least one operation in the stack.
- Rapidly scroll a non-image slice dimension.
- Confirm the previous image remains visible during slow updates.
- Confirm only the newest requested slice commits.
- Confirm status/overlay clears after completion.

## Memory Bounds

- Open or synthesize a large stack where full montage would exceed 1 GiB.
- Enable montage over a broad range.
- Confirm ArrayScope shows a memory warning when relevant and does not allocate a full collage.
- Confirm the UI remains responsive.
- Pan through several later montage rows and confirm process memory does not visibly climb after each
  render.

## Montage

- Enable montage on a small stack and confirm tile labels/hover context use the montage source index.
- Pan to later montage tiles and confirm hover and live-profile labels show the real source indices,
  for example tile 10 reports `d<axis>=10` rather than local tile 0.
- Draw an ROI across a montage tile gap and confirm gap pixels are ignored.
- Draw or move an ROI across unloaded canvas regions after panning and confirm unloaded pixels are
  ignored as `NaN`.
- Change montage range text and confirm stale montage results do not overwrite the new selection.

## Panels

- Show, hide, detach with the title-bar button, detach by dragging the title bar, redock with the
  detached-window Dock button, and reset Operations, Profile, and Inspection panels.
- For each managed panel, run this exact lifecycle sequence: open from View, detach, close the
  detached dialog, reopen from View, detach again, redock with the detached-window Dock button, hide
  from View, and reopen from View.
- Confirm each detached panel window contains the original panel content, not only the redock controls.
- Confirm the panel content is present after every reopen/redock and the View menu check state tracks
  hidden versus visible/detached state throughout the sequence.
- On Wayland, move each detached panel with the custom title/move handle.
- Confirm detached panels use tool windows rather than native floating docks.
- Confirm opening a panel grows the main window, and hiding/detaching it shrinks the main window while
  preserving the canvas size.
- Confirm hidden Profile and Inspection panels do not trigger expensive profile/ROI computations.

## Viewport

- Toggle Fit on and confirm pan, drag zoom, and wheel zoom are disabled.
- Resize the window and change slices while Fit is checked; the image should remain fully visible.
- Press 1:1 and confirm Fit unchecks, square-pixel interaction returns, and no render is scheduled.

## Suggested Commands

```bash
PATH=~/miniconda3/bin:$PATH direnv exec . pytest tests/ui -q
PATH=~/miniconda3/bin:$PATH direnv exec . env ARRAYSCOPE_STRICT_UI=1 pytest tests/ui -q
```
