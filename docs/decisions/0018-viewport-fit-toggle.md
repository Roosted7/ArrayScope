# 0018 - Viewport Fit toggle

## Problem

Fit behaved like a command, but users experienced it as a mode. The toolbar did not visibly show
whether future renders/resizes would keep fitting.

## Decision

Fit is a checkable locked viewport mode. When enabled, pan and zoom are disabled, wheel zoom is ignored,
aspect lock is released, resizes refit, and new image commits refit. 1:1 remains a momentary command
that unchecks Fit and restores square-pixel interaction.

Outside Fit mode, pan and zoom stay bounded enough that the current image or montage remains recoverable:
zoom-out is capped so content occupies at least 5% of each viewport axis, and panning must leave at
least 5% per-axis overlap. Montage constraints use full montage world bounds, not the currently
materialized viewport canvas.

## Consequences

Viewport commands remain data-free: Fit and 1:1 do not trigger array evaluation or render scheduling.

## Rejected alternatives

A non-checkable Fit action was rejected because it hides persistent viewport intent.

## Tests required

Qt tests verify Fit/1:1 do not call render, Fit is checkable, 1:1 unchecks Fit, and locked Fit ignores
wheel zoom.

## Manual checks required

Toggle Fit, resize the window, change slices, and confirm the image remains fully visible. Press 1:1
and confirm Fit unlocks without data recomputation.
