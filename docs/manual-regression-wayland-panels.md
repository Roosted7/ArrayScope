# Manual Regression - Wayland Panels

Run these checks in a Wayland session when available. They are also useful on X11 and the offscreen
test platform, but Wayland is the target environment for this checklist.

## Preserve-Canvas Panel Transitions

- Open a 2D array.
- Resize the main window so the central image area has visible margins.
- Toggle Inspection, Profile, and Operations from the View menu.
- Detach and redock each panel.
- Hide each panel from the managed title-bar Hide button and from the View menu.
- Confirm the main window top-left position does not jump during panel transitions.
- Confirm the central viewer returns to approximately the same pixel size after each transition.
- Confirm detached panels still move via the custom title/move handle.

## Preserve Setting

- In the View menu, turn off `Preserve Canvas Size on Panel Changes`.
- Open and hide Inspection, Profile, and Operations.
- Confirm ArrayScope no longer grows or shrinks the main window for panel transitions, aside from any
  unavoidable compositor or Qt minimum-size constraints.
- Turn `Preserve Canvas Size on Panel Changes` back on.
- Confirm panel transitions again use best-effort central-viewer size preservation.

## Expected Limitation

Preserve-canvas behavior is best effort. ArrayScope requests size corrections with `resize()` and does
not programmatically move the window position, but the window manager or compositor may constrain the
final top-level size.
