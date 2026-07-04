# Changelog

This file records user-visible release changes. Detailed development history and architecture decisions live under `docs/` and in Git.

## Unreleased

### Added

- Opt-in multi-resolution montage textures (ADR 0050, first slice): the
  `montage_lod_policy` setting (`native-only` default, `resident` opt-in,
  VisPy tiled scenes only) presents zoomed-out montages from box-mean-reduced
  pyramid levels that are materialized asynchronously in the background and
  stream in per tile, while hover/histogram/profile/ROI values stay exact.
  New diagnostics report the applied level, per-level resident tile counts,
  pyramid cache bytes/hits, and pending materializations.
- Julia and MATLAB invocation wrappers (`wrappers/julia`, `wrappers/matlab`) that open the viewer from those languages on Linux, macOS, and Windows with no in-process Python bridge. Arrays are handed off through a raw, uncompressed `.npy` file (single buffer write, native column-major layout) and loaded memory-mapped copy-on-write, so large arrays avoid compression, transposes, and eager second copies. See `docs/invocation.md`.
- CLI flags supporting that route and general use: `--mmap` (copy-on-write memory-mapped `.npy` loading), `--consume` (delete a temporary handoff file once loaded), and `--title` (window title override).
- Linked-window sync: per-facet toggle buttons synchronize window/level (display toolbar), dimension indexing (dimension strip), operation recipes (operations dock), and ROIs (inspection dock) across ArrayScope windows — including windows started as separate processes — on Linux, macOS, and Windows via per-user Qt local sockets. Mismatched shapes clamp per dimension; incompatible recipes are skipped with a notice.
- Out-of-core and lazy sources: large `.npy` and BART `.cfl` files now open as
  memory-mapped lazy sources (automatic above a memory-based size threshold),
  with region reads planned, budgeted, and cancellable above the source
  adapter. Lazy windows are labeled `[…, lazy]` in the title.
- Momentum-aware slice prefetch: sustained same-direction scrubbing warms up
  to four slices ahead of the motion (with a single reversal guard) instead
  of a fixed one-around neighborhood; a pause or direction change resets the
  depth immediately. All existing memory/cost/busy admission gates still
  apply per candidate.

### Changed

- Reusable operation stages (e.g. a full-dimension FFT chain) are computed
  once even when several evaluations need them concurrently: the stage cache
  now deduplicates in-flight computations, so a montage start, a visible
  re-render, or scrub steps that race the same uncached stage wait for the
  first computation instead of each running the full transform. New
  `compute_claims` / `compute_wait_reuses` stage-cache diagnostics make the
  deduplication observable.
- A montage whose tiles share a reusable stage now always computes that
  stage as a stage-lane job with the multi-worker FFT context, instead of
  inline in a "lead" montage tile whose lane runs FFTs single-threaded.
  The whole montage used to idle behind that one single-threaded transform
  — the "slow start" where tiles only pour in near the end; the stage now
  starts immediately and uses the configured FFT workers.
- Montage presentation no longer starts as a trickle: per-item latency
  feedback misattributed the fixed per-commit overhead (level sync,
  histogram, presentation build) to the tiles themselves, so small commits
  looked expensive per tile and pinned upload batches at 1-2 tiles until
  late in the fill. Presentation channels now split cost into per-commit
  overhead and per-item marginal rate (an exponentially-weighted
  regression over batch size vs elapsed) and size batches from the
  marginal rate. New `overhead_ewma_ms` / `marginal_per_item_ms` feedback
  diagnostics, plus `presented_order_sample` and a priority retarget
  counter in montage diagnostics so fill-order violations are visible in
  diagnostics logs.
- Montage presentation no longer sags in the middle of a fill and bursts
  at the end: the upload byte cap was derived from a per-byte rate that
  folded the fixed per-commit overhead into it, capping commits at a
  fraction of what the budget sustains while tiles were still streaming
  in. The byte cap now uses the same overhead/marginal cost split as the
  item batch, and idle fills may amortize up to two overheads of upload
  work per commit. Back-to-back reloads of a 272-tile complex montage now
  present tiles at the compute rate throughout the fill.
- Montage fill order is now correct by construction instead of by
  carefully-timed retargets. Three structural fixes, each verified against
  the actual GPU upload order on a live display: (1) queued tile objects
  are rebound to the current plan when a layout reflow changes the column
  count — previously tiles captured under a transitional geometry were
  scheduled by stale coordinates but drawn at new positions, so reload
  fills visibly ignored the priority order; (2) tile priority queues read
  the montage session's single scheduling context through a provider and
  re-key on change, instead of each queue holding a stale copy that was
  re-keyed only in bounded batches; (3) the "fairness aging" pop was
  removed — it degraded any bulk drain (stage fan-in activation, upsert
  admission) to insertion order after the first few items. Queues are
  always drained completely, so priority order cannot starve a tile.
- Reloading a file no longer fills the montage from a corner or wherever
  the pointer last crossed the image: the stale hover focus is cleared on
  reload (the transitional reload viewport let it pass the in-viewport
  validation), and priority retargeting prefers the viewport-continuity
  target range over the mid-restore camera range. Presentation batch sizing
  also no longer flaps between one tile and the maximum when the estimated
  per-commit overhead hovers around the budget — the batch now grows
  continuously to amortize the overhead.
- Montage tiles now genuinely fill center/mouse-first. Three paths ignored
  the priority order: stage fan-in released waiting tiles in the plan's
  row-major order under budget caps (filling from a corner regardless of the
  pending queue's priority), tile priorities were computed against the
  stale pre-montage viewport range until a mouse hover retargeted them, and
  near-viewport prefetch picked candidates in plan order. Fan-in waiting
  lists are priority queues now, queue priorities are rebuilt from the live
  viewport at the first commit and at stage activation, and prefetch
  candidates are ordered by focus proximity.
- Scheduling responsiveness during interaction: resource-governor decisions
  (worker counts, drain batch limits, callback budgets) are now reapplied on
  interaction start/stop edges instead of waiting for the next 250 ms–1 s
  sampling tick, so interactive budgets apply from the first drag event.
- Result-drain feedback now measures and controls the same channel
  (`<lane>_queue_drain`), closing a loop where most controllers' batch
  limits were decided from channels that never received observations.
- Latency feedback filters isolated outliers (GC pauses, one-off relayouts)
  on all channels and regrows drain batches from the measured under-budget
  rate, preventing a single slow callback from pinning throughput at
  one-item batches during interaction.
- Linked-window sync publishes on the leading edge after a quiet period, so
  a discrete change (slice step, level nudge, ROI drop) reaches peers
  immediately; bursts still coalesce through the existing 120 ms trailing
  timer, which now also flushes periodically during continuous drags.
- Background-work fallback polling backs off (10 → 100 ms) while the
  queue is empty, cutting idle wakeups and GIL churn during long
  computations; it snaps back to 10 ms when a poll finds pending queue work.
  New `fallback_event_polls`/`fallback_idle_polls` diagnostics counters make
  this observable without claiming every event-bearing poll was a missed
  signal.
- Worker pools are sized from the CPUs actually available to the process
  (affinity mask / container limits via `os.process_cpu_count` or
  `os.sched_getaffinity`) rather than the machine's total CPU count.
- Live-profile mouse tracking starts at a 16 ms coalesce interval instead of
  40 ms until governor feedback takes over.

## 0.8.0 — ArrayScope v28 release candidate

This is the first ArrayScope release-candidate baseline after the rebrand from
the historical ndslice line. It establishes package/runtime identity,
reproducible release diagnostics, and the current v28 correctness baseline.

### Added

- Reversible dimension-operation stack with recipes, runtime optimization, cost estimates, and reusable operation-stage caching.
- Live profiles, ROI inspection, ROI histograms, comparison helpers, and managed inspection panels.
- Progressive, viewport-bounded montage rendering with typed tile payloads and explicit requested/materialized/resident/presented state.
- Runtime memory policy, lane-aware compute policy, latency feedback, resource governance, diagnostics snapshots, trace logging, and rendering benchmarks.
- Experimental VisPy raster/tiled backend with shader-based scalar and complex display mapping.
- Explicit viewport modes, fit lock, 1:1 view, cropped image-axis ranges, and adaptive/manual histogram controls.
- A deterministic release diagnostics command and benchmark JSONL output for RC evidence.

### Changed

- PySide6 is the default Qt binding through PyQtGraph’s abstraction.
- Display semantics, backend mechanics, operation planning, caching, and UI orchestration have been split into focused packages.
- Documentation now separates live guidance from archived phase notes and provides a progressive architecture/roadmap path.
- Package metadata and runtime version now share the canonical ArrayScope `0.8.0` version.

### Fixed in the v28 audit

- The normal-image over-budget fallback now imports and uses the byte formatter instead of raising `NameError`.
- Viewport zoom-out constraints still enforce recoverable content overlap when an old viewport center is far outside new content.
- Display widgets cancel queued histogram refreshes and close VisPy resources during shutdown.
- Rendering benchmarks release parentless Qt/VisPy object graphs and share one module result set, avoiding very slow test-process teardown.
- Two stale evaluator tests now vary actual sliced axes, and test package bootstrapping is collection-order independent.

## Legacy ndslice releases

The entries below predate the ArrayScope rebrand. Their version numbers do not describe the current package maturity.

### 0.7.0

- Added DICOM directory conversion through `dcm2niix`.
- Added Ctrl+S NumPy export with range selection and optional singleton squeezing.

### 0.6.1

- Real arrays default to the real component; complex arrays default to magnitude.
- Fixed macOS emoji rendering.

### 0.6.0

- Added file monitoring/live reload and cross-platform CI.
- Adopted the Fusion Qt style.
- Closed HDF5/NPZ containers promptly to avoid file-lock issues.

### 0.5.1 — 2026-04-09

- Fixed switching back to the gray colormap without matplotlib.

### 0.5.0 — 2026-02-18

- Added PyQt6/HiDPI support in the legacy line, more colormaps, video export, and MATLAB v7.3 fallback loading.
- Fixed window/level reset when reselecting linear or symlog scale.
