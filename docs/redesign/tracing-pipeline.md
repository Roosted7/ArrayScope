# Tracing pipeline — design and build plan

**Goal:** one structured, timestamped event stream capturing every user
action, render request, kernel task, lifecycle transition, commit, backend
acknowledgement, camera change, and level publication — plus analysis tools
that (a) verify correctness invariants from a captured trace of a real
on-screen session and (b) drive the performance program with causal
latency data instead of folklore.

This replaces "certification by counter" with **verification by trace**: we
run realistic scripted interactions on a real display, capture everything,
and let tools prove correctness and attribute latency afterwards.

## What already exists (build on it, do not duplicate)

- **JSONL snapshot layer:** `core/diagnostics_jsonl.py` (schema-versioned
  records), `ui/diagnostics_logging.py` (file lifecycle),
  `core/diagnostics_trace.py` (analyzer CLI that computes inter-snapshot
  gaps and attributes stalls). Limitation: 500 ms *state snapshots*, only
  while the diagnostics dialog is open — cause cannot be linked to effect.
- **Deterministic counters:** kernel `LaneCounters`/`KernelDiagnostics`
  (`kernel/task.py:195`, `scheduler.py:370`) — always-on, per-lane
  queued/started/completed/superseded/stale.
- **Per-drain telemetry:** `GuiCallbackBudget.observation()` recorded at
  every bridge drain (`kernel/qt_bridge.py:119`).
- **Per-commit records:** `_last_montage_level_decision`
  (`frame_effects.py:1652`), `TileCommitReport` (`display/commit.py`),
  `TileLayerUpdateStats` (backend truth incl. uploads/bytes), tile truth
  overlay rows (`display/tile_truth_overlay.py` — ready-made event dicts).
- **Scripted drivers:** `tools/profile_montage_workflow.py` (phase-tagged
  JSONL, settlement-aware, event-loop probe) and `tests/gpu_interaction/`
  `Harness` (real display, screenshot/pixel assertions, `heartbeat_gaps`).
- **Marathon instrumentation to port first** (salvage Tier 1): worst-event
  drain timing, UI-observation epochs, `render_sync` sub-phases, paint
  observations, `callback_cpu_ms`.

## Gaps this pipeline closes

1. **Events, not snapshots** — discrete timestamped events with identities,
   so a scroll can be causally followed to its pixels.
2. **Input→pixel latency** — no input timestamps exist today.
3. **Production event-loop gap monitor** — the 1 ms probe lives only in
   test drivers.
4. **Paint/draw order** — only a sample is recorded, never a stream.
5. **Trace-based invariant checking** — GPU-harness assertions currently
   run only against live objects.

## Design

### The bus

`arrayscope/core/trace.py`: `TraceEvent(ts_ns, kind, **ids)` + a
process-wide `TraceBus` writing into a bounded ring buffer. Qt-free,
allocation-light; when disabled, emission is a single branch. Sinks:

- **Ring buffer** (always available): last N seconds retrievable from the
  diagnostics dialog ("dump trace") and automatically on a stall assertion.
- **JSONL file** (opt-in): `arrayscope --trace FILE` (new CLI flag — today
  JSONL capture requires clicking a dialog button) and env var for tests.
  Reuses the `diagnostics_jsonl` serialization + schema-version pattern.

### Event kinds (schema v1 — freeze small, extend by version bump)

`input`, `render_request`, `coalesce_flush`, `kernel_submit`,
`kernel_start`, `kernel_finish{outcome}`, `bridge_drain`,
`lifecycle{tile, from→to}`, `commit_batch`, `backend_ack{slot, identity,
uploads}`, `camera`, `levels_publish`, `heartbeat_gap`, `stall{owner
chain}`. High-rate kinds (`input`, `camera`) may be coalesced; there are
never per-pixel events.

### Emission points (the choke points — identities already in scope)

| Event | Where | Identities available |
|---|---|---|
| render_request | `window/render.py:971` `request_render` | reason, interactive, target_key, render generation |
| coalesce_flush | `window/render_coordinator.py:202` | RenderRequest, requested/coalesced counts |
| kernel_submit | `kernel/scheduler.py:150` | key, lane, priority, scope, deps, supersession, seq |
| kernel_finish | `kernel/scheduler.py:458` | seq, outcome (completed/stale/superseded/…) |
| bridge_drain | `kernel/qt_bridge.py:96` | events count, budget observation, worst event |
| lifecycle | `presentation/tile_lifecycle.py` transition methods (`:425–:567`) | tile, TileIdentity, payload ref (source, lod, quality), phase edge |
| commit_batch | `window/display_presenter.py:78/:205` | render generation, CommitKind, frame_key, commit report |
| backend_ack | backend `tiles.py` layer updates (`vispy/tiles.py:1439` etc.) | slot, acknowledged identity, uploads/bytes, lod |
| camera | `window/montage_viewport.py:94/:211` | view_range, viewport plan, frame_session_key |
| levels_publish | `render/level_stats.py:136` | level_key, source, evidence quality, refined flag |

### Analysis tools (extend `core/diagnostics_trace.py` / new `tools/trace_*`)

- **`trace_verify`** — replays invariants over a captured trace: at
  settlement no visible tile lacks an acknowledged payload (black-tile
  class); every visible target reaches ack; stale work never commits;
  placeholders only where no compatible source; quality monotonic per
  tile; paint order center-out within a rung; repeated identical
  `commit_bail` state is bounded so a barrier without a complement producer
  fails mechanically. These are the GPU-harness assertions generalized to
  any recorded session.
- **`trace_latency`** — input→first-pixel waterfalls per interaction;
  per-stage spans (submit→start→finish→drain→commit→ack); heartbeat-gap
  attribution (which events occupied the gap). This is what the
  performance program optimizes against.
- **`trace_work`** — wasted-work accounting: superseded/stale per lane,
  uploads by cause, replans per gesture. Detects convoy/storm patterns
  (e.g. the R2 per-completion commit storm would be one query).

## Build phases

- **T1 — spine** (with salvage Tier 1): the bus, the JSONL sink + `--trace`
  flag, kernel_submit/finish + bridge_drain + lifecycle + commit_batch +
  backend_ack events, `trace_latency` MVP. The profile tool and GPU
  harness always run with `--trace`; their artifacts become trace files.
- **T2 — verification**: `trace_verify` with the invariant set above; V3's
  loud non-convergence emits the `stall` event with the owner-chain
  snapshot and dumps the ring buffer; harness scenarios assert
  "trace-clean" in addition to pixels.
- **T3 — production always-on**: ring buffer on by default, production
  heartbeat-gap monitor (port the probe pattern out of the drivers),
  diagnostics-dialog "dump last 30 s".

## Bounds (so this stays fun)

One bus, one schema, no plugin architecture. An event is one flat dict.
If an analysis needs an event that doesn't exist, add the event — never a
second stream. The ring buffer has a fixed byte budget. Tools are plain
scripts over JSONL; no database, no dashboard until the P-steps prove we
need one.
