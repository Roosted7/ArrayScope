# 0020 - Operation cost model and FFT backend

## Problem

Large operations were hard to predict. NumPy/SciPy FFT calls cannot be cancelled mid-call, Qt workers can
oversubscribe CPU threads when FFT libraries also use workers, and materialize/export warnings used a
derived-array size check plus a special large-FFT axis heuristic rather than operation-level estimates.

## Decision

ArrayScope now has a conservative Qt-free operation cost model covering operation kind, shape, dtype,
full-axis requirements, output bytes, and peak-memory estimates. The model estimates reductions, RSS,
complex conversion, FFT, and view-like operations and is used for warnings, settings-driven guardrails,
and tests. It is advisory only in this phase.

Runtime FFT work goes through `arrayscope.operations.fft_backend`. The default `auto` backend resolves to
SciPy when available and falls back to NumPy if needed. FFT worker count is configurable and defaults to
`min(4, max(1, os.cpu_count() // 2))`; `all_minus_one` resolves to all but one CPU. pyFFTW is
import-guarded and only used when explicitly selected and importable. The project conda environment
includes pyFFTW so the backend can be tested, but pyFFTW is not a hard package dependency in
`pyproject.toml`.

ArrayScope keeps its existing MRI/k-space centered transform convention: `CenteredFFT` uses an inverse
FFT internally and `CenteredIFFT` uses a forward FFT internally.

App settings persist FFT backend, FFT workers, and render memory budget. Visible rendering and
interactive montage tile/canvas checks use the app render budget.

## Consequences

Materialize/export warnings can describe both output size and estimated peak memory, including the
operation responsible for the largest estimate. FFT workers can be constrained to avoid oversubscription
with Qt worker pools. The operation stack recipe remains machine-independent because backend and worker
settings are runtime app settings, not operation fields.

This does not make FFT calls cancellable or chunked. The scheduler still uses latest-only replacement,
not cost-aware refuse/degraded-preview decisions.

## Rejected alternatives

Using all CPU threads by default was rejected because it risks oversubscription during background UI
work. Adding mandatory pyFFTW as a package dependency was rejected to keep ordinary installs lighter.
Rewriting the slab evaluator or adding cost-aware scheduling in the same step was rejected to keep this
phase scoped to measurement, configuration, and warnings.

## Tests required

Tests cover SciPy/NumPy/pyFFTW backend behavior, worker resolution, centered FFT round trips, operation
dtype and peak estimates, coordinator cost delegation, settings serialization and menu persistence,
render budget plumbing, materialize/export warnings, architecture guards, and lightweight benchmark
scenarios.

## Manual checks required

Change FFT backend and workers, render FFT-backed slices, and confirm the view updates without recipe
changes or stale overlays. Change render memory budget low/high and verify visible render and montage
guards. Materialize/export FFT and RSS pipelines and verify the confirmation dialog includes output size
and peak-memory estimates.
