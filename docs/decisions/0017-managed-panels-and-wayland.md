# 0017 - Managed panels and Wayland

## Problem

Native floating `QDockWidget` behavior is hard to reason about across window managers, especially on
Wayland where applications cannot freely move top-level windows.

## Decision

ArrayScope now has a `PanelManager` with explicit `HIDDEN`, `DOCKED`, and `DETACHED` states. Docked
panels use `QDockWidget` with a managed title bar that provides hide and detach buttons; dragging that
title bar also detaches the panel. Detached panels reparent the same panel body into a `QDialog` tool
window with a move handle that calls `QWindow.startSystemMove()` and a Dock button for redocking.
Hidden panels store their body back in the hidden dock, not in a hidden dialog. `panel.dialog` is
non-`None` only while the panel state is `DETACHED`; hiding a detached panel first takes the body back
from the dialog and destroys the dialog.

The supported close/hide paths are the managed title-bar Hide button, View menu actions, and
`WindowLayoutManager` programmatic methods. `StandardDockWidget` intentionally has no custom
`closeEvent` lifecycle override, so native dock close behavior is not a second managed-panel state
machine.

`WindowLayoutManager` owns the outer-window size transition. Opening, hiding, detaching, and redocking
managed panels preserve the central viewer with a post-layout transaction: record the central widget
size, apply the panel change, let the `QMainWindow` layout settle, then correct the top-level size with
`resize()`. A short generation-guarded `QTimer` retry loop verifies the result because Qt and Wayland
compositors may apply configure/layout changes asynchronously. If ordinary `resize()` retries do not
settle, the final retry temporarily sets the top-level minimum and maximum size to the requested window
size on both the `QWidget` and its `QWindow`, calls `resize()`, then restores the original constraints.
If Qt reports the target central layout without a remaining delta, ArrayScope still briefly fixes the
current top-level size and repeats QWidget/QWindow resize/update requests as commit pokes for Wayland
compositors. Detach transitions use only the normal correction loop, not the strong fixed-size/nudge
escalation, so the new detached tool window can map cleanly. ArrayScope does not call
`setGeometry()` or intentionally move the top-left window position for preserve-canvas behavior. Users
can disable this best-effort main-window resizing from the View menu. Temporary stdout diagnostics with
the `[ArrayScope preserve-canvas]` prefix stay in place while Wayland behavior is being debugged.

Hidden and detached panels are removed from `QMainWindow`'s dock layout so their minimum sizes cannot
affect the canvas.

## Consequences

Panel state is owned by ArrayScope instead of inferred from native dock floating state. The implementation
does not use native `QDockWidget.setFloating()`, view-size snapshots, or top-level position-setting APIs
for managed panel transitions. Perfect preservation is not guaranteed when the compositor constrains the
requested size; the contract is best effort.

## Rejected alternatives

More `visibilityChanged` and event-filter repair logic was rejected because Phase 4c already showed
that it makes dock lifecycle behavior harder to reason about.

## Tests required

Qt tests cover title-bar detach, drag-to-detach, hide, redock, re-show, and reset layout using
`PanelManager.location()` instead of `QDockWidget.isFloating()`. Regression tests cover operation-dock
open/close with no prior manual resize, operation close while Inspection remains open, and detached
dialogs containing the original panel body. Lifecycle tests also cover detach, detached-dialog close,
View menu hide while detached, reopen, redock, hide, reopen, and reset layout without stale dialogs.
Preserve-canvas tests cover resize-only correction, no intentional position movement, the off setting,
and settings persistence.

## Manual checks required

On Wayland and X11, detach each managed panel, move it with the custom title handle, hide it from the
View menu, show it again, redock it, and reset layout.
