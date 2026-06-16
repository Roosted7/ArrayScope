# 0034 Compute Policy and Stage-Aware Warmup

## Status

Accepted.

## Context

Hot cached montage display must not be delayed by unrelated speculative work or by native FFT worker
oversubscription. The viewer already has separate Qt scheduler lanes, but FFT worker counts were
effectively global, so multiple montage tile jobs could each start multi-worker FFTs.

Stage-cache candidates are often much larger than the generic prefetch budget. Treating reusable FFT
stages as ordinary prefetch work prevents the one precompute that actually makes later interaction
cheap.

## Decision

ArrayScope uses a `ComputePolicy` to derive per-lane Qt worker counts and FFT worker counts. Visible
and stage lanes remain single-job lanes and may use the capped runtime FFT worker count. Montage tile,
prefetch, profile, ROI, and pixel lanes default to one FFT worker per job. The default montage tile
policy keeps `montage_tile_workers * fft_workers_tile` within a conservative CPU budget unless the
user explicitly selects an aggressive FFT worker setting.

Operation evaluation accepts an `EvaluationContext` carrying the compute lane, cancellation token,
FFT worker count, and memory policy. FFT operations honor the context-provided worker count instead
of reading only the global runtime default.

Stage warmup is not generic prefetch. It uses the stage-cache budget, runs on the low-priority stage
lane only while visible work is idle, and attaches to the existing stage-materialization singleflight
instead of creating duplicate expanded FFT jobs.

Rendered montage tile and next-slice prefetch remain idle-only. Expensive FFT-backed prefetch is
allowed only when the required reusable stage is cached or in-flight. Otherwise ArrayScope records a
skip decision and avoids recomputing the same FFT per predicted tile.

Large reusable stage materialization may chunk over non-blocking axes only. Blocking axes such as FFT
axes remain complete in every chunk, and cancellation is checked between chunks before the final
stage value is stored.

## Consequences

Cold stage work is less likely to starve the UI or multiply FFT workers across tile jobs.

Useful predictive work now targets the right granularity: warm one reusable stage first, then prefetch
cheap rendered tiles or slices from that stage.

Prefetch diagnostics can explain why no speculative work happened: idle blocked, budget blocked,
stage missing, stage in-flight, cache hit, scheduled, or deduped.
