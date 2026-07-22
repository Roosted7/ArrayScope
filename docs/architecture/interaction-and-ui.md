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

Montage resize and layout reflow use the same shared meaning. Fit and untouched near-auto views may
recompute a square-pixel fitted range; manual resize keeps screen zoom stable, so shrinking the
viewport shows less content at the same scale and growing it shows more. Manual column reflow does
not refit. If the same montage sources move to new tile positions, the viewport can translate by
source-local focus; if the source set changes because the tiled dimension scrolled, the world range
stays stable and samples the new content. See
[ADR 0042](../decisions/0042-montage-viewport-reflow-and-roi-ownership.md).

## Committed-frame pointer semantics

Pointer coordinates are interpreted against the frame currently shown. The mapping uses committed geometry and value source, so a queued state change cannot make hover report values from a different slice than the visible pixels.

The shared interaction controller owns:

- active tool;
- hovered/selected ROI or profile element;
- hit priority and semantic target;
- pointer capture and drag lifecycle;
- cursor intent;
- handle selection.

Backends draw that state. Qt pointer events for semantic overlays are normalized by the shared pointer
driver; PyQtGraph items and VisPy visuals mirror ROI/profile state rather than owning drag behavior.
Pointer hover first queries an indexed display-space ROI candidate set and then applies exact
backend-independent hit testing. Hit testing uses real display coordinates; only active drag updates
are clamped to the committed image bounds. If a montage range shrink leaves an ROI outside the
current tiles, the ROI remains there so expanding the range can recover it; when the user grabs its
body, the shared interaction controller translates the ROI back to the nearest allowed committed
content bounds without changing the ROI kind or size.

Background viewport navigation is separate from overlay drag semantics. `display.view_navigation`
owns backend-neutral pan/zoom range math, and `display.view_navigation_driver` translates backend
events into that model for surfaces that provide native navigation. That shared math consumes the
canonical `ViewBox` range and inversion/orientation state so flipped X/Y axes behave like the
PyQtGraph baseline. A shared touchpad path handles native pinch zoom and two-finger pan for every
backend, applying Qt's platform-provided incremental acceleration and momentum deltas without a
second animation owner. A manually calibrated angle-delta mapping preserves accelerated touchpad
motion when available, bounded against simultaneous native pixel deltas so alternate Qt encodings do
not create a speed discontinuity; angle-only devices retain the full compatibility calibration.
Mouse-wheel events remain with the existing backend wheel path. VisPy uses the native path so plain
pan/zoom updates the canonical range and camera immediately without routing through PyQtGraph scene
drag machinery. ROI/profile hits still take priority and use the shared semantic interaction
controller.

Middle-button drag is direction-locked after a small movement threshold. Vertical drag performs a
smooth focus-anchored zoom; horizontal drag steps the last manually used scrollable dimension, or
the first non-display, non-singleton dimension when no prior choice remains valid. `DimensionStrip`
owns that target, its canonical slice/range stepping, and the existing accent-border indication;
the navigation driver owns only pointer trajectory and emits bounded intent through callbacks.
Sub-threshold vertical motion is shown immediately as a provisional zoom. If horizontal motion wins
the direction lock, the driver restores the exact press-time range before emitting index intent, so
the tentative feedback cannot alter final camera or viewport-mode state.

## ROI and profiles

Qt graphics items are views of Qt-free ROI/profile models. Sampling/statistics live in `core.roi`, `core.histograms`, geometry, and profile coordination.
For montage layout reflow, canonical ROI selections remap by source index and tile-local coordinate
when the source set is unchanged. Graphics items mirror that selection state; backend-specific ROI
items do not own alternate reflow rules.

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
- Do not scan every ROI or rebuild overlay snapshots on every mouse move; maintain bounded candidate
  structures and perform exact geometry checks only on nearby candidates.
- Do not route backend-native background pan/zoom through a second scene graph when the backend can
  update the canonical viewport range directly.
- Coalesce high-frequency intent before expensive planning, but do not use debounce to starve exact progress indefinitely.
- Keep keyboard, menu, and direct-control routes behaviorally consistent.
- Any new visible mode must define how it composes with existing image/line/montage, ROI, channel, and viewport state.

## Current UI debt

- `ImageViewShell` remains large and still contains shared shell, PyQtGraph mechanics, and backend hooks.
- VisPy still uses a transparent Qt/PyQtGraph event layer for shared overlay events, but background
  viewport navigation now bypasses PyQtGraph scene drag.
- Some callbacks still combine state transition, scheduling, rendering, and status updates.
- Recent slicing grammar is powerful but needs clearer inline preview/error feedback.
- Tile hover priority is sampled when a plan is built; active queue retargeting needs a coalesced design.
