# 0011 - Progressive interaction polish

## Status

Accepted.

## Context

ArrayScope should feel as lightweight as `plt.imshow()` for quick inspection while still supporting
operation workflows and profile inspection. The previous control layout exposed many row-based
dimension buttons and kept the Operations dock prominent even when no operations existed, which made
simple 2D viewing feel like a prototype control panel.

## Decision

Use progressive disclosure:

- Quick glance: central image plus histogram, compact dimension strip, no operation dock while the
  stack is empty, no profile dock unless 1D data needs it.
- Inspect: live profile shows the Profile dock and marker; image hover updates a small pixel HUD and
  status text.
- Pipeline: adding the first operation reveals the Operations dock. Rows are draggable operation
  cards with enable/disable, delete, edit where supported, output shape, dtype, size estimate, and
  cache state.

Operation discovery is available from three paths:

- dimension chip operation menus, which use the clicked axis;
- dimension chip menus also expose profile-axis and live-profile actions so profile inspection remains
  discoverable from the same per-dimension affordance;
- Operations dock Add/Search, which uses a sensible default axis and still asks when an operation
  requires an axis;
- `Ctrl+K` command palette, which searches operations and viewer commands.

Dimension chips wrap in a grid: the narrow/default layout targets three chips per row, and wider
windows can use more columns. Active image-axis buttons display direction symbols (`^`, `v`, `<`,
`>`) and clicking the active direction flips the corresponding image axis.

Full view recipes are separate from operation recipes. Operation recipes describe pipeline steps;
view recipes describe steps plus `ViewState` and display settings. Dock geometry remains QSettings
layout state rather than recipe state.

## Consequences

The old stacked dimension control columns are no longer the visible interaction model. Some private
widget objects remain as temporary plumbing for existing mixins, but visible behavior is owned by the
compact strip and toolbar.

The first HUD intentionally stays pixel-focused. Profile-specific HUD content, ROI tools, montage,
multi-axis profiles, and session restore remain future work.

Native platform file dialogs are not used for ArrayScope save/load workflows because they can freeze
under some Qt/PySide environments. The UI uses non-native `QFileDialog` instances so navigation and
filename editing stay responsive.
