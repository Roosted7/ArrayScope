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
and stage lanes remain single-job lanes and may use the capped runtime FFT worker count; automatic
FFT workers resolve to half the machine, capped at eight workers. Montage tile, prefetch, profile,
ROI, and pixel lanes default to one FFT worker per job. The default montage tile policy uses roughly
half the CPU for concurrent tile jobs, capped, while keeping `montage_tile_workers *
fft_workers_tile` within that CPU budget unless the user explicitly selects an aggressive FFT worker
setting.

Operation evaluation accepts an `EvaluationContext` carrying the compute lane, cancellation token,
FFT worker count, and memory policy. FFT operations honor the context-provided worker count instead
of reading only the global runtime default.

UI-thread result fan-in is controlled separately from compute throughput. `LatencyFeedbackController`
tracks recent elapsed time and per-item cost for named channels, then returns adaptive batch limits,
work budgets, and commit intervals. Montage tile completion uses this to patch multiple cheap tiles
in one UI tick, but falls back toward smaller batches when patching/windowing/upload cost rises or
interaction is active.

Stage warmup is not generic prefetch. It uses the stage-cache budget, runs on the low-priority stage
lane only while visible work is idle, and attaches to the existing stage-materialization singleflight
instead of creating duplicate expanded FFT jobs.

Rendered montage tile and next-slice prefetch remain idle-only. Expensive FFT-backed prefetch is
allowed only when the required reusable stage is cached or in-flight. Otherwise ArrayScope records a
skip decision and avoids recomputing the same FFT per predicted tile.

Reusable stage materialization chooses chunk axes from the operation contract, not from display roles.
The planner starts with all candidate axes, removes semantic blocking axes such as FFT axes, and then
chunks only over remaining non-blocking axes. Image axes are valid chunk axes because FFT-over-montage
cases need full montage/FFT axes preserved.

Chunking is a memory-pressure tool, not the default for every large transform. If the full retained
stage fits the stage-cache budget, ArrayScope materializes it unchunked by default because chunking
can add overhead while the final full stage still has to be stored. Explicit low targets in tests or
future memory-stressed paths may still force chunking. Blocking axes remain complete in every chunk,
and cancellation is checked between chunks before the final stage value is stored.

For visible montage, a cold fitting reusable stage can be warmed by one lead tile instead of by a
duplicate explicit stage job. Once that tile stores the retained stage, waiting tiles immediately
activate from the stage cache. If an attached stage is no longer cached or in-flight, waiting tiles
fall back to direct tile evaluation instead of remaining stuck in a loading state.

## Consequences

Cold stage work is less likely to starve the UI or multiply FFT workers across tile jobs. Fitting 3D
FFT-over-montage stages avoid unnecessary chunking and avoid duplicating a direct lead tile's full
stage computation with a separate stage job.

Useful predictive work now targets the right granularity: warm one reusable stage first, then prefetch
cheap rendered tiles or slices from that stage.

Prefetch diagnostics can explain why no speculative work happened: idle blocked, budget blocked,
stage missing, stage in-flight, cache hit, scheduled, or deduped.
