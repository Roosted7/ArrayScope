# 0021 - Scheduler v2 cost-aware rendering

## Problem

Latest-only evaluation prevented stale results from committing, but a stale NumPy/SciPy operation could
still burn CPU and memory until the underlying call returned. Prefetch could also compete with visible
work, and the Phase 4e P3 cost estimates were not yet used to decide whether visible rendering should
run exact, preview, or refuse.

## Decision

Visible image rendering now goes through a pure render decision model. The decision can use cache, run an
exact async render, run exact evaluation in cooperative chunks, show a marked degraded preview, or refuse
while keeping the previous image visible. Degraded previews are not written to the exact image cache.

Chunked visible evaluation splits work across independent output/display axes, including operation-backed
pipelines with reductions or transforms when a non-blocking image axis is available. It checks
cancellation before and after each chunk and around major evaluation steps. It does not split one FFT
axis internally; a single SciPy/pyFFTW FFT call remains non-interruptible.

Prefetch is idle-only, cancelled when visible work starts, capped to one or two nearby slices, skipped
while visible work is busy, skipped for montage, and allowed for operation-backed views only when P3 cost
estimates are under conservative thresholds.

Cache and scheduler diagnostics now report hit rate, degraded/refused/chunked/cancelled render counts,
and pending/running/cancelled/stale scheduler counts.

## Consequences

Visible work wins more consistently: old queued work is dropped, old chunked work can stop between chunks,
and expensive views either produce a clearly marked preview or refuse without blanking the existing
image. Exact results remain exact when they commit. Some large transform views still refuse because
there is no safe chunk axis or preview that fits the selected budget.

## Rejected alternatives

Increasing Qt worker counts was rejected because throughput would compete with interaction latency. Using
all CPU threads was rejected because FFT libraries can already use internal workers. Dask/Zarr/GPU
backends were rejected for this phase. Splitting a single FFT axis was rejected because NumPy/SciPy/FFTW
calls are not safely cancellable mid-call in this architecture.

## Tests required

Tests cover render decision selection, degraded view-state striding, chunk-axis selection, chunked raw,
FFT, reduction, and complex image equivalence, cancellation between chunks, token-aware controller
execution, render refusal/degraded/chunked UI paths, idle and cost-aware prefetch, diagnostics, and
architecture guards for Qt-free pure modules.

## Manual checks required

Rapidly scroll an FFT-backed stack and confirm only the newest result commits. Lower render budget to
trigger degraded preview and refusal while the previous image remains visible. Enable prefetch and
confirm it starts only after idle, never during visible work, and only for cheap operation-backed stacks.
Interrupt a chunked render by changing slices and confirm stale chunks do not clear the newer overlay.
