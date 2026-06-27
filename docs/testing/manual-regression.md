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
- In montage mode, resize the viewport while zoomed in and while far zoomed out; manual content
  should keep the same screen scale, showing less content when smaller and more when larger, unless
  the view was truly near the remembered auto range.
- In montage mode, resize while Fit is locked and while untouched near-auto; the fitted view should
  hug/recompute consistently without a manual revert surprise.
- Change the applied montage column layout with the same source indices; ROI geometry and statistics
  should follow the same source-local data. Scroll the tiled dimension to a different source set; ROI
  world positions should remain stable and sample the new content under them.

## When automated interaction tests disagree with the real app

Qt, PyQtGraph, and platform plugins can transform input before ArrayScope sees
it. When a `QTest` or controller-level test passes but real interaction still
fails, troubleshoot from the real event stream before adding another special
case:

1. Launch the real file and backend that reproduces the issue, on the real
   display/session type. Prefer a bundled or copied fixture path so the run can
   be repeated, for example:

   ```bash
   PATH=~/miniconda3/bin:$PATH direnv exec . python -m arrayscope data/example.nii
   ```

2. Add a temporary, narrowly scoped trace at the ownership boundary that should
   receive the intent. For histogram interactions that boundary is the
   histogram controller, not the renderer. Log event type, target widget,
   button, local/global position, accepted state, and the semantic signal or
   slot reached.
3. Ask the tester to perform the exact real gesture while the process remains
   attached to the terminal. Compare center/edge/inside/outside regions when
   the bug is spatial.
4. Turn the observed event sequence into the regression test. Do not only test
   the ideal Qt event if the real app emits press/release pairs, swallowed
   double-clicks, propagated events, or accepted events from a child item.
5. Remove the trace before committing unless it is part of a documented
   diagnostics feature.

The histogram double-click repair is the model case: real PyQtGraph interaction
showed that a click inside the active level region opened span editing before
the second click could become auto-window. The fix moved that single-click edit
behind Qt's double-click interval, and the regression test covers the observed
release-pair behavior rather than only `QTest.mouseDClick`.

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
FFT montage, perform the `fft_level_refinement_preview` level edit, then close.
Confirm the pacing by eye and keep the JSONL.

For timing evidence, prefer the plain JSONL run above. It records
request-to-first-content, exact-visible/full-completion timing, event-loop
p95/p99/max gaps, callback records, tile upload bytes, warm/cold work counters,
histogram/level timings, and montage compute counters. In particular, FFT montage runs should not show
hundreds of direct tile computations when a reusable stage is available; the
JSONL counters should show stage-backed tiles for full-axis FFT montages. Level-only transitions should
also include `presentation_revision`, `presentation_stale_count`, `presentation_pending_count`, and
`presentation_settled`; use those fields, not retained visibility alone, to confirm convergence.

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
