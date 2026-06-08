# 0017 - Managed panels and Wayland

## Problem

Native floating `QDockWidget` behavior is hard to reason about across window managers, especially on
Wayland where applications cannot freely move top-level windows.

## Decision

ArrayScope now has a `PanelManager` with explicit `HIDDEN`, `DOCKED`, and `DETACHED` states. Docked
panels use `QDockWidget` with a managed title bar that provides hide and detach buttons; dragging that
title bar also detaches the panel. Detached panels reparent the same panel body into a `QDialog` tool
window with a move handle that calls `QWindow.startSystemMove()` and a Dock button for redocking.

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
dialogs containing the original panel body.

## Manual checks required

On Wayland and X11, detach each managed panel, move it with the custom title handle, hide it from the
View menu, show it again, redock it, and reset layout.
