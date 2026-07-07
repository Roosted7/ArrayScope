# 0053 — Execution kernel and the modular rendering pipeline

**Status:** Accepted (2026-07-07). Redesign branch `redesign`.
**Drives:** the plan set in [`docs/redesign/`](../redesign/README.md).

## Problem

Regressions ping-ponged between subsystems because scheduling was split
across five cooperating-but-independent systems: `WorkGraph` (bookkeeping
that never ran anything), eight `EvaluationController` QThreadPools (where
`EvalPriority` never reached execution order — pools ran FIFO), the resource
governor's timers, montage dispatch derivation, and ~28 pacing/watchdog
`QTimer`s. Most fan-in work (level stats scans, tile result application,
priority rebuilds) ran on the GUI thread in budgeted timer slices; the
entire pacing/budget apparatus existed to compensate. On a 16-core machine,
one pool did evaluation and everything else fought for the GUI thread.
`FrameRenderMixin` grew to 6,100 lines / ~150 methods because every new
concern (LOD, previews, floors, pacing) had nowhere else to live.

## Decisions (confirmed with Thomas, 2026-07-07)

1. **Own scheduler kernel, Dask-inspired, not Dask-hosted.** GUI deadlines,
   supersession, and lane quotas do not map onto Dask's scheduler; we keep
   its ideas (keyed task graphs, dependencies, futures) with zero new
   dependencies. `arrayscope/kernel/`.
2. **Test bar during the redesign:** Qt-free core/operations/lifecycle/
   kernel/render tests stay green at every commit; window/display tests may
   break only if listed in [known-red.md](../redesign/known-red.md) with the
   plan step that fixes or deletes them.
3. **Both backends stay first-class.** VisPy is the GPU path (residency,
   atlas, uniform level changes); PyQtGraph the no-GPU/server path. Feature
   decisions branch on *declared capabilities*, never backend names, and
   each backend exploits its native strengths (no lowest-common-denominator
   rendering).
4. **Threads sized to cores are the primary parallelism**, behind a
   swappable `WorkerBackend` seam so free-threaded 3.14t (or processes) can
   be adopted without kernel changes. NumPy/FFT release the GIL, so this is
   real multi-core use today.

## Architecture

### Kernel (`arrayscope/kernel/`, Qt-free, implemented + tested)

- `TaskSpec`: key, fn, lane, priority, scope, deps, supersession family,
  deadline, cost estimates, `reusable`, cooperative token.
- Priorities order actual worker pulls (heap over lane-rank, priority,
  deadline). Dependencies gate actual execution; a failed/dropped dependency
  drops dependents with `dependency_failed` — never silently.
- **Staleness has one arbiter** with three sources: supersession family
  advanced, scope cleared (hierarchical: clearing `"a"` covers `"a:b"`), key
  resubmitted. Re-checked at GUI dispatch time so a completion racing a
  target change can never commit stale pixels. `reusable` tasks finish and
  deliver through `on_reuse` (cache fill) instead of being cancelled.
- Optional lanes (prefetch/speculative) park under a worker-fraction quota
  and whenever visible backlog exists; visible work is never starved.
- One `CompletionQueue`; `QtKernelBridge` is the single GUI drain, bounded
  by `GuiCallbackBudget`, with one adaptive anti-hang fallback timer
  (observable via `fallback_event_polls` — a busy fallback is a bug report,
  ADR 0051 discipline retained).

### Modular pipeline (`arrayscope/render/`, nucleus implemented)

Stage boundaries are types (`render/stages.py`): `RenderIntent → TileWork →
CommitBatch → AckExpectation`. One owner per state:

| state | owner |
|---|---|
| tile lifecycle (3 axes + claims) | `presentation/tile_lifecycle.py` (unchanged) |
| which quality rung next | `render/ladder.py` (pure planner) |
| task execution + staleness | `arrayscope/kernel` |
| commit batching queue | `render/pipeline.py` |
| GPU/CPU application | backend adapter, GUI thread only |

### Unified LOD ladder (`render/ladder.py`)

FLOOR → PREVIEW → DESIRED → EXACT, one pure planner replacing the four
scattered answers (montage_lod planning, frame_renderer preview/floor
methods, ingest-reduction admission, native-only checks). Invariants:
coarse before fine within and across tiles (floor-first fill generalized),
nothing committable is recomputed, resident levels come only from
acknowledged lifecycle claims, exact inspection always native, operations
run once per rung (commuting ops on reduced input), native-only collapses
the ladder to one EXACT step.

## The GUI thread is a gateway

Allowed on the main thread: intent snapshots, kernel submits, bounded
drains, bounded commit application, widget updates. Everything else —
evaluation, reduction, statistics, histogram scans, planning heavier than a
few microseconds per tile — is a kernel task. Timers may only exist as
anti-hang fallbacks or UI cosmetics (toast durations); every scheduling
timer deleted in R4 must be replaced by a completion event or a capacity
waiter.

## Deletion targets (end state)

`core/work_graph.py`, `core/scheduler.py` (compat aliases already point at
the kernel), `window/evaluation_controller.py` (after R1),
`window/frame_renderer.py` and `window/montage_lod.py` dissolved per the
[method map](../redesign/frame-renderer-map.md) (R2/R3), pacing-governor
batch machinery reduced to governor-as-telemetry (R4). Tests pinning
deleted machinery are deleted with it (R5) — a test asserting timer pacing
of work the kernel now schedules is a wrong-path signpost, not coverage.

## Rejected alternatives

- **Dask as the runtime**: wrong latency class for 16 ms interaction
  budgets; supersession/deadline semantics would live outside the scheduler
  anyway; heavy dependency.
- **Free-threaded-only design**: ecosystem risk (Qt bindings, VisPy) still
  real in mid-2026; kept as a backend swap instead.
- **VisPy-only rendering**: PyQtGraph remains required for servers and
  no-hardware-GL environments and stays first-class per decision 3.
