# Phase 4h Manual Regression

## P0 - Stabilize and measure

1. Start ArrayScope with a small 3D array and open Developer -> Diagnostics.
2. Render a normal image view and confirm the Render tab shows timing lines. Unset timings should read `n/a`; populated timings should be non-negative millisecond values.
3. Switch to montage mode and confirm the Montage tab shows tile cache counts plus canvas/tile/overlay timing lines.
4. Enable nearby-slice prefetch, change a non-display slice, wait for idle prefetch, and confirm scheduler/cache diagnostics update without leaving pending work stuck.
5. Trigger a live profile and move the marker. Confirm exact profile updates still complete while profile prefetch is accounted under the prefetch scheduler.
6. Run the base FFT backend tests in an environment without `pyfftw`; the optional backend test should skip instead of failing.

Useful commands:

```bash
PATH=~/miniconda3/bin:$PATH direnv exec . pytest tests/operations/test_fft_backend.py tests/operations/test_chunked_evaluator.py -q
PATH=~/miniconda3/bin:$PATH direnv exec . pytest tests/ui/test_evaluation_controller.py tests/ui/test_prefetch.py -q
PATH=~/miniconda3/bin:$PATH direnv exec . pytest tests/core/test_runtime_diagnostics.py tests/ui/test_diagnostics_dialog.py -q
```

## P1 - Render coalescer and fast interactive slice path

1. Open a 3D array.
2. Use the mouse wheel over a non-display dimension slice field and confirm the slice number changes immediately.
3. Confirm the image catches up after a short delay and lands on the latest slice, not intermediate slices.
4. Hold bracket shortcuts to step slices rapidly.
5. Confirm the Operations, Profile, ROI, and Inspection panels do not visibly churn during the burst.
6. Stop interacting and confirm side panels refresh to the latest state.
7. Enable live profile and confirm profile updates resume after interaction quiet.
8. Enable nearby-slice prefetch and confirm prefetch is cancelled during visible interaction, then resumes after idle.
9. Try montage range text and confirm existing montage behavior remains correct.

Useful commands:

```bash
PATH=~/miniconda3/bin:$PATH direnv exec . pytest tests/ui/test_interactive_render_coalescing.py -q
PATH=~/miniconda3/bin:$PATH direnv exec . pytest tests/ui/test_render_scheduler.py tests/ui/test_dimension_control_interactions.py tests/ui/test_viewport_interactions.py tests/ui/test_roi_inspection_interactions.py -q
PATH=~/miniconda3/bin:$PATH direnv exec . env ARRAYSCOPE_STRICT_UI=1 pytest tests/ui -q
```
