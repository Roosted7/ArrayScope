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

`WindowLayoutManager` owns the outer-window geometry transition. Opening or redocking a panel grows the
main window by a stored panel extent plus Qt's dock separator extent; hiding or detaching consumes the
same active extent and shrinks the main window after the panel has left the dock layout. Hidden and
detached panels are removed from `QMainWindow`'s dock layout so their minimum sizes cannot affect the
canvas.

## Consequences

Panel state is owned by ArrayScope instead of inferred from native dock floating state. The implementation
does not use native `QDockWidget.setFloating()`, view-size snapshots, or retry timers for managed panel
transitions.

## Rejected alternatives

More `visibilityChanged` and event-filter repair logic was rejected because Phase 4c already showed
that it makes dock lifecycle behavior harder to reason about.

## Tests required

Qt tests cover title-bar detach, drag-to-detach, hide, redock, re-show, and reset layout using
`PanelManager.location()` instead of `QDockWidget.isFloating()`. Regression tests cover operation-dock
open/close with no prior manual resize, operation close while Inspection remains open, and detached
dialogs containing the original panel body. Lifecycle tests also cover detach, detached-dialog close,
View menu hide while detached, reopen, redock, hide, reopen, and reset layout without stale dialogs.

## Manual checks required

On Wayland and X11, detach each managed panel, move it with the custom title handle, hide it from the
View menu, show it again, redock it, and reset layout.
