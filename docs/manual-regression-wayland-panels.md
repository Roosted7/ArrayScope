# Manual Regression - Wayland Panels

Run these checks in a Wayland session when available. They are also useful on X11 and the offscreen
test platform, but Wayland is the target environment for this checklist.

## Preserve-Canvas Panel Transitions

- Open a 2D array.
- Open Developer -> Diagnostics and keep the Canvas Preserve details visible.
- Set View -> Panel Resize Behavior -> Best effort.
- Resize the main window so the central image area has visible margins.
- Toggle Inspection, Profile, and Operations from the View menu.
- Detach and redock each panel.
- Hide each panel from the managed title-bar Hide button and from the View menu.
- Confirm the main window top-left position does not jump during panel transitions.
- Confirm the central viewer returns to approximately the same pixel size after each transition.
- Confirm Developer Diagnostics updates the preserve mode, platform, last transition, result, attempts,
  and recent events after each transition.
- Confirm detached panels still move via the custom title/move handle.

## Preserve Modes

- In the View menu, set Panel Resize Behavior -> Off.
- Open and hide Inspection, Profile, and Operations.
- Confirm ArrayScope no longer grows or shrinks the main window for panel transitions, aside from any
  unavoidable compositor or Qt minimum-size constraints.
- Set Panel Resize Behavior -> Strong Wayland.
- On Wayland, confirm diagnostics reports whether the strong path was used, constraints are released
  after transitions, and the main window can still be manually resized afterward.
- On non-Wayland platforms, confirm Strong Wayland behaves like best effort and diagnostics reports
  that the strong path was skipped for the platform.
- Set Panel Resize Behavior -> Best effort before finishing the checklist.

## Expected Limitation

Preserve-canvas behavior is still bounded by the window manager or compositor. ArrayScope requests size
corrections with `resize()` and does not programmatically move the window position, but the final
top-level size may still be constrained.
