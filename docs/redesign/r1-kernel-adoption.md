# R1 — All background execution on the kernel

**Goal:** one `Kernel` + one `QtKernelBridge` replace the eight
`EvaluationController` QThreadPools and the bookkeeping-only `WorkGraph`.
Priorities and lanes become real for every submission in the app.

**Deliverable shape:** a `KernelEvaluationController` adapter in
`arrayscope/kernel/eval_adapter.py` that implements the *public* surface of
`window/evaluation_controller.EvaluationController` over a shared kernel, so
the ~40 call sites keep working while frame_renderer still exists; R2 then
removes the adapter's callers cluster by cluster, and the adapter dies with
the last one.

## Steps

1. **Composition first.** In `window/main.py`, construct one
   `Kernel(ThreadWorkerBackend())` + one `QtKernelBridge(kernel, self)`,
   with `handler_error_hook=handle_ui_exception`. Keep the eight controller
   *attributes* but assign them adapters:
   `KernelEvaluationController(kernel, bridge, name="visible", lane_default=Lane.VISIBLE_MATERIALIZATION, priority_default=…)`.
   Name→lane map: visible→VISIBLE_MATERIALIZATION, montage_tile→
   VISIBLE_MATERIALIZATION, stage→STAGE_MATERIALIZATION, histogram→
   HISTOGRAM_REFINEMENT, pixel/profile/roi→PROFILE_ROI_HOVER, prefetch→
   SPECULATIVE_RESIDENCY.
2. **Adapter semantics** (map, don't reinvent — kernel already owns these):
   - `replace_group` (hierarchical `a:b:c`) → kernel scope, prefixed by the
     adapter name: scope=`f"{name}:{replace_group}"`. `clear_group` →
     `kernel.clear_scope`; `advance_group`/`group_generation` keep a local
     int mirror for the few callers that read it.
   - `supersession_key/value` → `Supersession((name, group, key), value)`.
   - `start_latest` = clear-family-then-submit; `start_active_plus_latest`
     = submit with `reusable=True` + `on_reuse` (the kernel preserves
     running reusable work by design).
   - `start_prefetch` keeps its local gates (idle, cost, dedupe by key,
     max_prefetch) and then submits at `Priority.PREFETCH`; the kernel
     quota replaces the WorkGraph visible-backlog gate.
   - `on_slow`: submit-time `QTimer.singleShot(slow_ms, bridge, cb)` that
     re-checks via handle state — acceptable UI-cosmetic timer.
   - `notify_when_capacity` → `bridge.notify_when_capacity`.
   - `set_max_workers` → per-lane quota hint on the kernel (add
     `kernel.set_lane_quota(lane, n)`; TODO marker exists in scheduler).
   - `diagnostics()` → build `SchedulerDiagnostics` from
     `kernel.diagnostics()` lane counters (names already match).
3. **Conformance gate.** Adapt `tests/ui/test_evaluation_controller.py`
   (32 tests) + `tests/window/test_evaluation_capacity_waiters.py` to
   construct the adapter. Behavioral deltas are expected in exactly two
   places — update the tests, citing this plan:
   - drain-fallback counters live on the bridge, once, not per controller;
   - pool-clear semantics (`clear_queued` calling `pool.clear()`) become
     scope clears; late results are stale by scope, not by pool luck.
4. **Delete** `window/evaluation_controller.py` internals (keep the file as
   a re-export of the adapter until R2 retires the last import), delete
   `core/work_graph.py` `WorkGraph` (counters now come from the kernel;
   `WorkItem` call sites in frame_renderer pass their lane/priority to the
   adapter instead — mechanical, ~30 sites), and delete
   `tests/ui/test_render_scheduler.py` assertions that pin WorkGraph
   admission bookkeeping the kernel now enforces structurally.
5. **Governor rewire (minimal).** `resource_governor` stops adjusting
   per-controller worker counts; it sets kernel lane quotas and the bridge
   drain budget. Everything else it touched is R4's problem — leave TODOs.

## Exit gate

- Full suite green (adapted tests included); GPU harness green.
- `grep -rn "QThreadPool" arrayscope/` → only the adapter file (or zero).
- Scrub benchmark (`profile_montage_workflow`, vispy/resident) within ±10%
  of the pre-R1 numbers recorded in the commit message.
- `kernel.diagnostics()` visible in the diagnostics dock (replace the
  scheduler panel's data source).

## Risks / notes

- The visible controller was created with `max_workers=1` for ordering
  reasons in some tests (`test_visible_pool_max_thread_count_is_one`);
  ordering now comes from priorities + supersession. Delete that test with
  a pointer to kernel ordering tests.
- `memory_budget_bytes` on requests was advisory; keep passing it through
  to `TaskSpec.estimated_bytes`.
