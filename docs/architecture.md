# Architecture

Main rule: Qt collects intent and displays results; Qt widgets are not the
source of array-view state.

## Ownership

- `arrayscope.core.ViewState`: authoritative current view of the derived array: shape, image
  axes, profile axis, slice indices, channel, scale, and per-axis flags.
- `arrayscope.display.slice_engine`: converts `data + ViewState` into display-ready images and
  lines.
- `arrayscope.operations.pipeline`: immutable NumPy operations plus shape prediction.
- `arrayscope.operations.pipeline.OperationStep`: ordered operation rows with enable/disable
  state. `ArrayDocument.operations` is the active enabled operation sequence; `steps` is the
  pipeline UI/document sequence.
- `arrayscope.operations.coordinator`: owns the operation document, evaluator, stack edits,
  and materialization.
- `arrayscope.profiles.model` / `arrayscope.profiles.coordinator`: maps image-space marker positions to
  profile view states and line results.
- `arrayscope.core.window_levels`: decides image window/level reuse or auto-level behavior.
- `arrayscope.display.ImageView2D`, `arrayscope.ui`, and `arrayscope.ui.docks`: Qt display and controls only.
- `arrayscope.ui.dimension_strip`, `arrayscope.ui.display_toolbar`, `arrayscope.ui.command_palette`,
  and `arrayscope.ui.hud`: compact viewer controls, operation discovery, and on-canvas pixel
  feedback. They emit user intent and do not own view state.
- `arrayscope.core.view_recipe`: serializes operations, `ViewState`, and display settings for
  full-view restore. It is pure and does not contain dock geometry.
- `arrayscope.window.main.ArrayScopeWindow`: wires Qt signals to state changes, then calls `render()`.
- `arrayscope.app.launch`: QApplication creation, multiprocessing launch, and IPython Qt event-loop handling.
- `arrayscope.io`: file loading, dataset selectors, and save workflows.
- `arrayscope.export`: video/frame export workers and UI workflow.

The public package surface is intentionally small: users should prefer
`import arrayscope as asc` followed by `asc(data)`. Internal code should import
concrete submodules rather than relying on package-root re-exports.

## Render Flow

User actions update `ViewState` or the operation coordinator. `render()` then:

1. migrates `ViewState` to the current derived shape;
2. syncs controls from `ViewState`;
3. renders image/profile data through the evaluator;
4. applies view-only axis flips;
5. updates docks, labels, HUD text, compact controls, and cache status.

Do not read widget values to reconstruct `ViewState`. Widget state is an output
of render, except transient UI-only state such as dock visibility and histogram
interaction.

## Interdependency Map

When changing `ViewState`, check:

- `arrayscope.display.slice_engine`
- `arrayscope.profiles`
- `arrayscope.operations.evaluator`
- `arrayscope.operations.coordinator`
- `arrayscope.window.render.RenderMixin.render()`
- `arrayscope.export`
- Qt smoke/artifact tests

When changing `slice_engine`, check:

- image display
- profile extraction
- video export
- window-level behavior

When changing `operation_pipeline`, check:

- operation registry and recipes
- operation dock
- evaluator cache keys
- `ViewState.for_shape()`

## Placement Guide

- Axis validation: `arrayscope.core.axis_utils`.
- Pure array transforms: `arrayscope.operations.dim_ops` or `arrayscope.operations.pipeline`.
- Display conversion: `arrayscope.display.slice_engine`.
- Colormap creation: `arrayscope.display.colormaps`.
- Window/level decisions: `arrayscope.core.window_levels`.
- User-action orchestration: `arrayscope.window` mixins or a focused coordinator.
- Qt UI controls: `arrayscope.ui` and `arrayscope.ui.docks`.
- Full-view serialization: `arrayscope.core.view_recipe`.

## Progressive Disclosure

The quick-glance layout keeps the central image and histogram first. The Operations dock is hidden
while the operation step list is empty, and it appears automatically after the first operation. That
automatic reveal does not mark the dock as user-pinned; clearing the stack hides it again unless the
user explicitly showed the dock from the View menu. The Profile dock is hidden unless the data is 1D,
live profile is enabled, or the user explicitly shows it.

Operation creation is intentionally available from three places:

- dimension chip operation buttons / context menus use the clicked axis;
- the Operations dock Add/Search controls use the last valid operation axis, then the first
  non-display non-singleton axis, then profile, X, Y, then the first valid axis;
- `Ctrl+K` uses the same defaulting, with a focused dimension chip taking precedence.
