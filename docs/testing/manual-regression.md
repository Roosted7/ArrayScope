# Manual regression

Use this checklist for release candidates and rendering/UI changes. Historical phase-specific checklists remain in [`../archive/manual-regression/`](../archive/manual-regression/README.md).

## Record the environment

- commit and clean/dirty state;
- OS and X11/Wayland/macOS/Windows session;
- Python, PySide6, PyQtGraph, VisPy, NumPy/SciPy versions;
- GPU and driver;
- backend and performance settings;
- data generator/file, shape, dtype, and operation stack;
- diagnostics JSONL/benchmark artifact path.

## Basic workflow

1. Launch from Python and CLI; verify blocking/non-blocking behavior.
2. Open real, complex, 1D, 2D, 3D, and >3D arrays.
3. Change image axes, scalar slices, explicit ranges, flips, FFT shift, channel, and scale.
4. Confirm line mode, montage mode, fit lock, preserve, reset, and 1:1.
5. Add/disable/reorder/undo operations and verify values/shape.
6. Save NumPy output and export a short frame/video sequence.
7. Close/reopen windows repeatedly and watch for lingering processes/errors.

## Interaction and semantic consistency

- Hover values match the pixels currently visible during rapid slice changes.
- ROI handles/body/profile targets have stable priority and cursor feedback.
- Drag interruption by mode change/window close does not leave a stuck tool.
- User-locked levels survive progressive histogram/tile refinement.
- Double-click auto-window and revert/manual editing behave consistently.
- Changing levels/LUT does not trigger tile re-materialization/re-upload counters.
- Pan/zoom preserves semantic session and does not rerun operations.
- Cropped image-axis ranges map coordinates and profiles correctly.

## Responsiveness stress

Use a large plane, many montage tiles, complex shader mode, and at least one expensive operation.

- Continuously pan/zoom/slice/drag levels for 20–30 seconds.
- Confirm the last valid frame remains visible.
- Confirm exact or progressively improving frames still arrive; interaction must not cancel all useful work forever.
- Watch diagnostics for callbacks over 16 ms, queue growth, repeated cold uploads, and cancellation churn.
- Hover across montage center/edges and verify useful tile priority changes without mouse lag.
- Open ROI/profile panels during rendering and verify visible work remains dominant.

## Memory and recovery

- Request work just below and above configured render budget; verify clear degraded/refusal status rather than crash.
- Move from far-away/old viewport ranges to newly cropped content; verify content remains recoverable onscreen.
- Exercise stage cache fill/eviction and montage residency under pressure.
- Simulate/trigger backend replacement or context loss where practical; verify explicit recovery and no stale commit.
- Observe RSS after opening/closing repeated windows and benchmark runs.

## Backend parity

Run the same scenarios in PyQtGraph and VisPy:

- scalar real/log/symlog;
- complex real/imag/magnitude/phase;
- LUT and levels;
- one dirty tile, level-only update, pan/zoom with warm residency;
- ROI/profile/hover values and geometry;
- fit/preserve/1:1 and axis flips;
- close/reopen.

Record differences as contract failures, intentional capability gaps, or visual-library differences.

## Platform/layout

- Open/close/detach/reattach every managed panel.
- Verify canvas preservation and restored window size with/without docks.
- On Wayland, verify no geometry repair loop, flashing, or misplaced detached panel.
- Check standard and HiDPI displays, light/dark theme, keyboard traversal, menu/shortcut parity.

## Pass criteria

A manual run passes only when there are no semantic mismatches, crashes, stuck interaction states, unbounded memory growth, repeated cold uploads for presentation-only changes, or unexplained UI stalls. Visual/latency concerns should include a trace and exact reproduction steps.
