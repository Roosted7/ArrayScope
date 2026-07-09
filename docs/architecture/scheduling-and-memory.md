# Scheduling and memory

> **Redesign banner (2026-07-07):** execution now belongs to
> `arrayscope/kernel` — real priorities, dependencies, lane quotas, one
> staleness arbiter, one GUI fan-in ([ADR 0053](../decisions/0053-execution-kernel-and-modular-pipeline.md)).
> WorkGraph and per-controller drains are gone after R1. R4 deleted the
> remaining scheduling timers and shrank the governor to telemetry plus bridge
> and commit-budget knobs; do not add new scheduling systems beside the kernel.

ArrayScope must remain responsive when the requested work is larger than one event-loop turn or one safe allocation.

## Work classes

The system already distinguishes several practical lanes:

- visible image evaluation;
- visible montage tile materialization;
- display preparation and Qt/GPU commit;
- histogram/level refinement;
- ROI/profile/scalar inspection;
- reusable stage materialization;
- nearby-slice/tile prefetch and GPU warm residency.

These lanes do not have equal urgency or cost. Visible first-pixel work has a deadline; selected/hovered analysis has interactive value; speculation runs only with measured spare capacity.

## Current coordination

### Normal image

A coalescer/generation model chooses the latest requested state, performs cache/cost checks, and starts immediate, asynchronous, chunked, degraded, or refused work. Latest-only cancellation avoids stale commits but can discard repeated progress during continuous interaction.
The normal visible path now uses active-plus-latest submission with a stable supersession key and a separate target value.
Queued obsolete visible work is collapsed while already-running reusable work can finish and be stored without committing stale pixels.

### Montage

A persistent `MontageSession` tracks target, requested/materialized/presented sets, payloads, levels, and commit acknowledgement. Pan/zoom retarget visible/near priorities and residency instead of rebuilding the semantic session. Ready results are progressively committed through bounded batches.

### Stage and prefetch

Reusable stages use singleflight. Nearby slice/tile work and warm residency are lower priority, gated by memory, scheduler busy state, feedback, and kernel lane quotas.

## Kernel execution

ArrayScope owns a single Qt-free kernel scheduler. Tasks carry a lane,
priority, scope, dependencies, supersession family/value, deadline, cost
estimates, expected value, reusable-output policy, and cooperative
cancellation token. Priorities order real worker pulls; dependencies gate
execution; scope clears, supersession, and key resubmission are the only
staleness arbiters.

The former controller attributes are temporary public surfaces over this
kernel while `frame_renderer` still exists. They do not own thread pools or
GUI drains. Production worker limits are kernel lane quotas set once per
canonical lane; the adapters only keep compatibility diagnostics for legacy
callers.

A task needs at least:

- semantic/viewport/presentation target keys;
- lane and supersession key;
- hard/soft deadline;
- estimated CPU time and GPU bytes;
- dependencies;
- expected quality/latency gain;
- cancellation/reuse policy.

After hard visible deadlines, optional admission should be value-based rather
than timer-based:

```text
expected value = probability of use × latency saved × quality gain / estimated cost
```

Local gates run before kernel submission. Idle state, memory cost, dedupe, and in-flight caps reject
prefetch and retained warmup before a `WorkItem` becomes active, which keeps diagnostics from showing
work that never actually ran. `STAGE_MATERIALIZATION` can represent exact visible tile dependencies or
retained stage warmup; retained quality is optional and must yield to visible backlog.

## GUI-thread contract

All paths that mutate Qt or OpenGL state follow these limits:

- interactive callbacks target **≤ 4 ms**;
- idle presentation callbacks target **≤ 8 ms**;
- **16 ms** is a warning threshold, not a normal batch allowance;
- no callback loops over an unbounded data/user-sized collection;
- every batch has item, byte, and elapsed-time limits;
- partial progress is published and remaining work is rescheduled;
- queueing many individual Qt events is not equivalent to one bounded callback.
- result fan-in is budgeted before visible admission; ready tile bursts must be admitted in bounded
  item/byte/time batches rather than drained unconditionally.

`QtKernelBridge`, montage tile result fan-in, stage-wait release, and backend
commit paths publish bounded callback observations and kernel counters.
Priority rebuilds, histogram refresh, and some presentation updates still
need broader large-tile-count traces before release-level performance claims.

## Cancellation and supersession

Cancellation protects correctness and scarce resources; it is not a substitute for scheduling.

- Stale results are rejected by semantic key/revision even if cancellation arrives late.
- Stale visible work is dropped before visible admission.
- Exact cache entries are written only by complete accepted results.
- Work that is cheap to finish or reusable may be allowed to complete.
- Presentation-only changes supersede presentation work, not materialization.
- Camera changes retarget visible regions/residency, not the operation pipeline.
- Side-analysis results are guarded by the committed semantic target.

## Memory policy

`core.memory_policy` derives budgets from configured profile, system total/available memory, process RSS, and hard per-render caps. Separate budgets cover:

- visible render output/peak;
- visible tile materialization and presentation residency;
- display cache;
- profile/scalar cache;
- reusable stage cache;
- speculative/prefetch allowance.

`memory_budget` contains estimation/formatting helpers; it is not the source of runtime policy.

Estimates are conservative admission inputs, not proof that allocation will succeed. Diagnostics should record estimated versus observed bytes and refusal/degradation reasons.

## GPU residency

GPU residency has its own budget and lifecycle. It must consider queried device limits, actual allocation outcomes, texture format/shape, context identity, and pressure. CPU cache presence does not imply GPU residency; GPU eviction does not invalidate semantic CPU data.

Visible residency has priority over warm/speculative residency. Warm tile promotion, prefetch, and
retained-source residency use a separate lower-priority budget and must shrink, pause, or cancel when
visible materialization, visible GPU residency, or admitted visible commit fan-in needs capacity.

## Feedback and resource governance

Latency feedback records callback duration and work count with EWMA plus
one-sample outlier suppression. Resource telemetry samples CPU/memory without
blocking the UI. The resource governor combines those signals with policy and
kernel state to set only:

- `QtKernelBridge` drain budget (`set_budget_ms` and `set_max_items_per_drain`);
- presentation commit batch bounds (`CommitBatch.max_items` / `max_bytes`).

Kernel lane quotas are the worker/admission throttle. Compute policy supplies
the initial lane quota values; the kernel owns stale-work pruning and
speculative admission. There is no governor sampling timer, per-controller
worker clamp, per-channel `ui_work_decision`, or interaction-edge reapply loop.

Feedback loops must be closed on the channel that is actually measured. The
single kernel completion drain observes and is governed as
`kernel_bridge_drain`; presentation commit observations govern commit batch
bounds. Non-kernel GUI callbacks such as `histogram_refresh`, `roi_refresh`,
and `profile_update` may use the shared callback-budget vocabulary, but they
do not create separate scheduling controllers.

The bridge fallback timer is a safety net for missed cross-thread signals,
not a scheduling mechanism: it backs off exponentially (10 -> 100 ms) while
polls come up empty and snaps back to 10 ms whenever it finds pending queue
work. The `fallback_event_polls` / `fallback_idle_polls` counters make
fallback activity observable once per bridge without treating an
event-bearing poll as proof of signal failure.

Idle slice prefetch is momentum-aware (`core.prefetch_policy`): sustained
same-direction scrubbing deepens speculation ahead of the motion (bounded,
with a single reversal guard), while a pause or direction change collapses
depth immediately. Planning is separate from admission; every candidate
still passes the memory, cost, busy-state, and kernel quota gates.

## Required metrics

At minimum capture:

- input event to first usable frame;
- input event to exact-visible frame;
- event-loop max/percentile gap;
- queue delay and presented-frame age;
- CPU preparation time per item/byte;
- upload/commit time and bytes;
- cancellation time and reusable output retained;
- cold upload versus warm rebind/visibility counts;
- cache/stage hit rates and evictions;
- process RSS and estimated GPU residency;
- backend, dtype/component, tile layout/commit path, and interaction state.

## Testing limits

Deterministic counters can prove “no upload on pan” or “one dirty tile only.” Headless wall-clock numbers cannot prove GPU execution or frame pacing. Release-level performance claims require the hardware matrix in [manual regression](../testing/manual-regression.md).
