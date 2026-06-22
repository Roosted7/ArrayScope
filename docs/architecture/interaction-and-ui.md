# Interaction and UI

The UI should expose array intent directly without making widgets the source of semantic truth.

## Event flow

```text
Qt event / command
  -> normalize intent
  -> update ViewState, document, ROI/profile model, or layout state
  -> request focused render/analysis work
  -> present accepted result
  -> update lightweight status/HUD
```

Callbacks should not perform operation planning, large NumPy work, cache traversal, or unbounded scene updates inline.

## Dimension interaction

The dimension strip is the core product surface. It supports:

- choosing image axes;
- choosing line/profile and montage roles;
- scalar slicing and explicit range/index selection;
- axis flip and FFT-shift state;
- dimension-local operation actions.

Role changes are normalized through `ViewState`/state synchronization. A dimension control should not maintain a private interpretation of which axes are displayed.

## Viewport

`ViewportController` owns four useful modes:

- untouched automatic behavior;
- user range;
- persistent fit;
- one-time 1:1 action returning to user mode.

The pure range constraint keeps a minimum recoverable overlap with content while allowing zoomed-in edge inspection. When max zoom-out span is clamped using an old center, overlap must still be enforced; the v28 audit fixed an early-return bug in this path.

Backend camera mechanics may differ, but fit/preserve/reset/1:1 meaning is shared.

## Committed-frame pointer semantics

Pointer coordinates are interpreted against the frame currently shown. The mapping uses committed geometry and value source, so a queued state change cannot make hover report values from a different slice than the visible pixels.

The shared interaction controller increasingly owns:

- active tool;
- hovered/selected ROI or profile element;
- hit priority and semantic target;
- cursor intent;
- handle selection.

Backends draw that state. Complete pointer capture, drag lifecycle, and event ownership still need migration from concrete widgets.

## ROI and profiles

Qt graphics items are views of Qt-free ROI/profile models. Sampling/statistics live in `core.roi`, `core.histograms`, geometry, and profile coordination.

Recommended interaction sequence:

1. update hover/selection from the committed frame immediately;
2. show cheap committed/coarse information;
3. schedule exact analysis at a lower lane priority;
4. publish only if its semantic target is still current.

Hidden panels do no continuous work. Selected/hovered entities can receive higher priority than unrelated analysis.

## Histogram and levels

The histogram widget is both a plot and an interaction surface. Its controller owns adaptive plotting, level previews/final edits, and manual value entry. Heavy or high-resolution refinement should move off the GUI thread if traces exceed budget.

Queued zero-delay refreshes are cancellable during widget shutdown. This prevents callbacks from accessing deleted graphics objects.

## Managed panels and layout

The layout controller owns panel visibility, dock/detached behavior, persisted geometry, and canvas-preservation transactions. Wayland/native-window behavior is treated as a platform constraint rather than repaired by arbitrary geometry loops.

Panel actions should be idempotent and route through one owner. Detached windows must refresh inspection content from semantic state rather than relying on a stale dock event.

## Commands and progressive disclosure

Frequently used actions remain near the array: dimension controls, display mode, levels, ROI/profile tools, fit/1:1, and operation stack. Diagnostics, performance settings, rare export options, and developer controls stay behind menus/panels.

A command palette is useful when it calls the same semantic command handlers as menus/shortcuts. Avoid a second behavior implementation for each invocation route.

## UI quality rules

- Preserve the canvas when panels open/close where platform behavior permits.
- Keep the previous valid frame during work.
- Show concise progress, degraded, stale, or refusal state without blocking dialogs.
- Do not let hover flood computation or reprioritize a full queue per mouse event.
- Coalesce high-frequency intent before expensive planning, but do not use debounce to starve exact progress indefinitely.
- Keep keyboard, menu, and direct-control routes behaviorally consistent.
- Any new visible mode must define how it composes with existing image/line/montage, ROI, channel, and viewport state.

## Current UI debt

- `ImageView2D` remains large and owns shell plus PyQtGraph mechanics.
- `VisPyImageView2D` inherits it and bridges two input/camera systems.
- Some callbacks still combine state transition, scheduling, rendering, and status updates.
- Recent slicing grammar is powerful but needs clearer inline preview/error feedback.
- Tile hover priority is sampled when a plan is built; active queue retargeting needs a coalesced design.
