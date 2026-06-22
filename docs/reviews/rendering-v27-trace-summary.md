# ArrayScope v27 rendering trace summary

> **Historical review.** This document records a dated evidence set. Use [`../current-state.md`](../current-state.md), [`../architecture.md`](../architecture.md), and [`../roadmap.md`](../roadmap.md) for current guidance.

## Scope

This note analyzes the two diagnostics captures shipped with the v27 source archive:

- `vispy-arrayscope-diagnostics-20260620-123159.jsonl`
- `PQtGraph-arrayscope-diagnostics-20260620-123345.jsonl`

Both captures used a derived `float64` array with shape `(336, 336, 272)` and an aggressive memory
profile. The actions were intentionally similar. A diagnostics snapshot was requested every 500 ms,
so a long gap between snapshots is direct evidence that the Qt event loop did not service the
logging timer.

The summaries can be reproduced without Qt:

```bash
python -m arrayscope.core.diagnostics_trace \
  vispy-arrayscope-diagnostics-20260620-123159.jsonl \
  PQtGraph-arrayscope-diagnostics-20260620-123345.jsonl
```

## Headline result

The longest freeze in each capture was not dominated by evaluation, texture upload, or the final
display commit. Both backends recorded approximately nine seconds in synchronous render
orchestration while all 272 montage tiles were already cached.

| Backend | Trace duration | Maximum sampling gap | Stalls > 1.5 s | Largest synchronous render |
|---|---:|---:|---:|---:|
| VisPy | 53.894 s | 9,487.5 ms | 1 | 9,180.070 ms |
| PyQtGraph | 86.201 s | 18,255.3 ms | 5 | 9,210.305 ms |

This is strong evidence of a backend-independent UI-thread defect above both concrete renderers.

## VisPy capture

The single large stall occurred at sequence 67, montage session 37:

- sample gap: 9,487.5 ms;
- loaded/pending/visible tiles: 272 / 0 / 272;
- synchronous render orchestration: 9,180.070 ms;
- display commit: 1.915 ms;
- canvas/tile-layer commit: 41.884 ms;
- tile payload build: 5.531 ms;
- tile cache lookup: 5.507 ms;
- texture upload was not the dominant cost.

The resource governor reported approximately 96% CPU headroom and low memory pressure, while UI
pressure was elevated. That combination is important: the application was not compute-saturated or
memory-thrashing. One GUI-thread call was monopolizing the event loop.

At this point the feedback channel for `montage_commit` recorded `last_count=1` for the whole 41.884
ms commit and therefore estimated one item as costing the complete callback. It reduced the batch
limit to one and spaced commits to 40 ms. The feedback mechanism was reacting to a bad observation,
not to the true per-tile cost.

Scheduler totals at the stall also show substantial discarded work:

- visible lane: 18 completed, 16 stale, 5 cancelled;
- montage lane: 316 completed, 63 stale, 3 cancelled;
- prefetch lane: 10 scheduled and 169 limited.

Those counters do not prove each cancellation was avoidable, but they establish that replacement and
speculation policy is a material part of total throughput.

## PyQtGraph capture

The capture contained five event-loop stalls above 1.5 seconds:

| Sequence | Sampling gap | Loaded tiles | Important changed timings |
|---:|---:|---:|---|
| 86 | 18,255.3 ms | 272 | 9,072.813 ms synchronous render; 48.329 ms commit |
| 85 | 9,798.1 ms | 272 | 294.442 ms commit; 205.458 ms display commit; 166.247 ms RGB windowing |
| 91 | 9,550.6 ms | 272 | 9,210.305 ms synchronous render; 44.839 ms commit |
| 83 | 3,541.7 ms | 99 | 72.841 ms commit; 21.238 ms RGB windowing |
| 82 | 2,671.1 ms | 72 | 102.994 ms commit; 44.340 ms RGB windowing |

The 9.1-second events have the same shape as the VisPy event: all tiles are cached, tile lookup and
payload construction are single-digit milliseconds, and the final display commit is small. They are
the same orchestration defect.

Sequence 85 exposes a separate PyQtGraph-specific fan-in problem. One GUI callback prepared or
updated 173 tile items:

- 166.247 ms CPU RGB windowing;
- 202.385 ms inside `setImage`/display mutation;
- 205.458 ms display commit;
- 294.442 ms complete montage commit;
- approximately 58.6 MB of visible upload/source data;
- 173 updated items and 99 skipped/unchanged items.

The existing controller limited the number of *commit callbacks*, not the number of tile upserts
inside one callback. Therefore a nominally progressive path could still execute hundreds of
CPU-windowing and item-update operations in one event-loop turn.

The governor again learned from `last_count=1`. It interpreted the complete 294 ms callback as one
item, selected a batch limit of one, reduced the budget to 2 ms, and spaced commits to 232 ms. This
creates a bad feedback loop: an unbounded callback causes excessive backoff, and excessive backoff
then makes tiles visibly pop in one at a time.

Scheduler totals at this event were:

- visible lane: 31 completed, 31 stale, 20 cancelled;
- montage lane: 348 completed, 51 stale, 1 cancelled.

The visible lane discarded at least as many results as it committed. This matches the code-level
finding that rapid single-image interaction repeatedly invalidates latest-only work.

## Root causes

### Cached montage restart performed semantic work before first presentation

Before the review fixes, a fully cached montage restart synchronously fed every cached tile through
level/histogram accumulation before committing pixels. The aggregate tracker repeatedly rebuilt
combined samples while tiles were added. The result was effectively quadratic growth in repeated
array work for a cache-hit path.

The corrected order is:

1. materialize or resolve cached tile payloads;
2. merge level summaries for the tiles that will become active;
3. choose a level source whose coverage includes the active payloads;
4. commit those pixels;
5. acknowledge the backend-presented tiles;
6. refine the detailed histogram curve in small timer slices without invalidating texture residency.

### Progressive commits bounded scheduling but not GUI work

The tile-delta model already represented upserts and removals, but the complete pending delta was
passed into one backend call. PyQtGraph then performed all CPU display preparation and item updates in
that callback. VisPy could similarly receive a large upload burst.

The first fixed path sliced actual upserts, but the 14:16/14:18 rerun showed this was still the wrong
unit of control: resident VisPy tiles and existing PyQtGraph items were paced like cold uploads. The
repair path keeps resident rebinds, existing-item shows, and geometry moves cheap and immediate, while
backends report cold uploads, new items, and CPU windowing as separate feedback channels.

### Continuous normal-image interaction can starve exact rendering

The normal image path is independent of the montage freeze. Every interactive request advances the
visible generation and clears render-dependent groups. Each coalesced `render()` clears them again,
and `EvaluationController.start_latest()` cancels the previous `visible-image` job before submitting
the replacement. With one visible worker, a stream of requests faster than evaluation completion can
cancel every job. No frame is committed until the input rate becomes low enough for one evaluation to
finish.

The eventual fix must use an active-plus-latest target model rather than unconditionally restarting
work:

```text
presented frame  <- remains visible
active target    <- allowed to make bounded progress
latest target    <- replaces only the queued target
```

A completed active target may be cached but rejected for presentation when it no longer matches the
latest view. The important point is that continuous input must not reset all progress every 16 ms.

## Fixes made in this review branch

- cache-hit montage pixels commit before complete detailed histogram reconstruction;
- level coverage for presented tiles is required before backend presentation;
- semantic level identity is independent of requested coverage and LUT/presentation state;
- materialized tiles remain loading until backend presentation is acknowledged;
- resident/reused tiled work is not throttled as cold upload work;
- retargeting preserves compatible source-keyed payloads and avoids destructive clears;
- stale tiled deltas are rejected; backend clears carry explicit reasons;
- progressive queues use constant-time front removal;
- VisPy no longer advertises or exposes the broken montage canvas fallback;
- phase-level synchronous montage timings were added;
- a Qt-free JSONL trace summarizer and tiled work counters were added.

## Follow-up 14:16/14:18 rerun

The follow-up captures confirmed that the long event-loop freezes were gone, but exposed regressions in
tiled presentation semantics: VisPy level/histogram flicker, one-by-one tile pop-in in both backends,
scroll retargets that cleared retained content, placeholders that disappeared before visible content,
and occasional queued full refreshes. The repair commits are therefore aimed at preserving the freeze
fix while restoring presentation invariants. The important diagnostic distinction is cold work versus
resident/reused work: a large commit with zero uploads is not the same risk as a large cold-upload
commit and must not be throttled the same way.

## Measurements required for the next capture

The next capture should include, per session and per callback:

- viewport planning time;
- cached tile resolution time;
- stage planning time;
- session construction time;
- first pixel commit time;
- number and bytes of tile upserts/removals;
- cold tile uploads, new PyQtGraph items, CPU-windowed tiles, resident rebinds, existing-item shows,
  and geometry-only moves separately;
- CPU preparation milliseconds per cold item;
- GPU upload milliseconds and bytes;
- draw/vertex submissions;
- time from request to first usable frame;
- time from request to exact complete frame;
- event-loop gap p50/p95/p99/max;
- cancelled work milliseconds, not only request counts;
- presented-frame age and target-frame distance.

A backend comparison is meaningful only after both paths obey the same per-callback work budgets and
report presented-frame latency rather than setter submission time.
