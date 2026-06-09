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
