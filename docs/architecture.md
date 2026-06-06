# Architecture

Main rule: Qt collects intent and displays results; Qt widgets are not the
source of array-view state.

## Ownership

- `ViewState`: authoritative current view of the derived array: shape, image
  axes, profile axis, slice indices, channel, scale, and per-axis flags.
- `slice_engine`: converts `data + ViewState` into display-ready images and
  lines.
- `operation_pipeline`: immutable NumPy operations plus shape prediction.
- `operation_coordinator`: owns the operation document, evaluator, stack edits,
  and materialization.
- `profile` / `profile_coordinator`: maps image-space marker positions to
  profile view states and line results.
- `window_levels`: decides image window/level reuse or auto-level behavior.
- `ImageView2D`, docks, dialogs, and widgets: Qt display and controls only.
- `ArrayScopeWindow`: wires Qt signals to state changes, then calls `render()`.

## Render Flow

User actions update `ViewState` or the operation coordinator. `render()` then:

1. migrates `ViewState` to the current derived shape;
2. syncs controls from `ViewState`;
3. renders image/profile data through the evaluator;
4. applies view-only axis flips;
5. updates docks, labels, and cache status.

Do not read widget values to reconstruct `ViewState`. Widget state is an output
of render, except transient UI-only state such as dock visibility and histogram
interaction.

## Interdependency Map

When changing `ViewState`, check:

- `slice_engine`
- `profile`
- `operation_evaluator`
- `operation_coordinator`
- `ArrayScopeWindow.render()`
- `video_export`
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

- Axis validation: `axis_utils`.
- Pure array transforms: `dim_ops` or `operation_pipeline`.
- Display conversion: `slice_engine`.
- Colormap creation: `colormaps`.
- Window/level decisions: `window_levels`.
- User-action orchestration: `ArrayScopeWindow` or a focused coordinator.
- Qt UI controls: dock/widget/dialog modules.
