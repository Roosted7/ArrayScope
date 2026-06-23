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

## Real workflow profiling

Use this when callback-budget, scheduler, or backend changes affect pacing. Run
on a real display, not `QT_QPA_PLATFORM=offscreen`, so VisPy/OpenGL frame pacing
and swap behavior are actually visible.

```bash
mkdir -p tests/artifacts
PATH=~/miniconda3/bin:$PATH direnv exec . python -m arrayscope.tools.profile_montage_workflow \
  --backend all \
  --jsonl tests/artifacts/montage-workflow-profile.jsonl
```

The window should visibly load the bundled NIfTI dataset, draw dims 0/1 as the
image axes, render the full dim-2 tiled montage, apply FFT on dim 2, render the
FFT montage, then close. Confirm the pacing by eye and keep the JSONL.

For timing evidence, prefer the plain JSONL run above. It records
request-to-first-content, total phase time, event-loop max gap, callback records,
tile upload bytes, and montage compute counters. In particular, FFT montage runs
should not show hundreds of direct tile computations when a reusable stage is
available; the JSONL counters should show stage-backed tiles for full-axis FFT
montages.

For Python stack attribution, wrap the same visible workflow with a low-rate,
nonblocking `py-spy` sample. Treat the JSONL from the same run as timing
evidence and the raw stack file as attribution evidence:

```bash
PATH=~/miniconda3/bin:$PATH direnv exec . py-spy record \
  --format raw \
  --rate 50 \
  --nonblocking \
  --gil \
  -o tests/artifacts/montage-workflow-profile.raw -- \
  python -m arrayscope.tools.profile_montage_workflow \
    --backend all \
    --jsonl tests/artifacts/montage-workflow-profile.jsonl
```

Use `perf record -F 99 -g` when native attribution matters, such as SciPy FFT,
Qt painting, or GL driver calls. `py-spy --native` can be useful as a last
resort, but it can significantly perturb Qt/FFT timing; if used, compare it
against a plain JSONL run and do not use the native py-spy timings as pacing
evidence.

If `py-spy` is not installed on the test machine, record the JSONL run and the
missing sampler explicitly rather than substituting offscreen timing for a GPU
claim.

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
