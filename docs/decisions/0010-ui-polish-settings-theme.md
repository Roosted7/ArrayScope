# 0010 - UI polish, settings, and theme abstraction

## Status

Accepted.

## Context

ArrayScope needs to feel more like a desktop tool without becoming tied to one
theme package or a large custom styling system. Layout state, dock visibility,
theme choice, and operation-stack affordances should persist across sessions,
but the application must still launch with only the required Qt/pyqtgraph stack.

The operation pipeline also needs lightweight user feedback about whether the
current view came from cache or required evaluation. This should not turn into a
background scheduler or computation graph.

## Decision

Add a small theme/settings layer instead of hard-coding a third-party theme:

- `System/Native` remains the default first-run behavior.
- `Native`, `Dark`, and `Light` are exposed as user choices.
- `Dark` and `Light` use a minimal built-in Qt palette so they visibly work
  without optional packages.
- Optional theme backends may be detected and tried in later work, but they are
  not required and are not the default path for this repair pass.

Persist the following through `QSettings`:

- main window geometry;
- dock placement and visibility;
- selected theme;

Add a reset-layout action that restores the intended first-run layout: image
view gets most of the window, the Operations dock remains visible on the side,
and the Profile dock remains hidden unless the user explicitly shows or needs
it.

The Operations dock is allowed to edit the operation list by deleting selected
operations and reordering rows by drag/drop. Reorders are validated against the
current base shape before replacing the document, so invalid reorders are
rejected without corrupting the stack. A row context menu provides delete and
move actions as a fallback. Inline operation parameter editing is deferred.

Expose cache/evaluation status as small metadata from the evaluator and show it
in the Operations dock. The status is intentionally coarse: cold, computing,
ready, cached, stale, or error. Computing is transient and is replaced by ready
or cached once a synchronous evaluation returns.

## Consequences

Theme support remains small and optional-package-free. ArrayScope uses built-in
Qt palettes for dark and light themes; optional external theme backends are not
part of the current theme contract.

The persistence boundary is the main window and app settings, not the operation
recipe format. Recipes still save operation stacks only, not UI state.

Nearby-slice prefetching is not exposed in the UI until it actually works. It
remains a separate task because it needs careful cache-key invalidation and must
not touch Qt widgets from worker threads.
