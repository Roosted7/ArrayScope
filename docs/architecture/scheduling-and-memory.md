# Scheduling and memory

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

Reusable stages use singleflight. Nearby slice/tile work and warm residency are lower priority, gated by memory, scheduler busy state, feedback, and resource-governor decisions.

## Work graph admission

ArrayScope now owns a Qt-free `WorkGraph` above the Qt worker controllers. It records visible
planning/cache lookup, visible materialization, display preparation, backend commit, GUI fan-in,
histogram refinement, ROI/profile/hover work, stage materialization, and speculative residency as
lane-specific `WorkItem`s. A work item carries a frame target, quality, supersession key/value,
deadline, estimated CPU/byte cost, dependencies, expected value, and reusable-output policy.

The graph admits exact visible work before optional work, drops stale queued work before admission,
preserves already-running reusable visible work when a newer target arrives, and records deterministic
counters for queued, admitted, dropped, superseded, completed, failed, rescheduled,
reusable-finished, deadline-missed, and budget-blocked work by lane. Supersession is indexed by
supersession key, so replacing a target touches only the affected queued family instead of scanning
unrelated profile, ROI, stage, or prefetch queues. Re-admitting queued work never advances the current
supersession value; if the queued value is stale it is dropped before it can become visible. Repeated
budget polling reports the same queued blocked item once until its state changes.

## Scheduler behavior

Visible controllers hold explicit presented, active, and latest-queued targets. A new interaction
replaces queued obsolete work but does not automatically kill an active item that is nearly complete
or produces reusable cache data.

A work item needs at least:

- semantic/viewport/presentation target keys;
- lane and supersession key;
- hard/soft deadline;
- estimated CPU time and GPU bytes;
- dependencies;
- expected quality/latency gain;
- cancellation/reuse policy.

After hard visible deadlines, optional admission should be value-based rather than timer-based:

```text
expected value = probability of use × latency saved × quality gain / estimated cost
```

Local gates run before graph admission. Idle state, memory cost, dedupe, and in-flight caps reject
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

`EvaluationController`, montage tile result fan-in, stage-wait release, and backend commit paths now
publish bounded callback observations and work-graph counters. Priority rebuilds, histogram refresh,
and some presentation updates still need broader large-tile-count traces before release-level
performance claims.

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

Latency feedback records callback duration and work count. Resource telemetry samples CPU/memory without blocking the UI. The resource governor combines those signals with policy and scheduler state to adjust:

- lane worker counts;
- callback/result fan-in;
- upload byte/item batches;
- commit interval;
- prefetch/speculation admission.

Overload backoff should be immediate; recovery gradual. Metrics must be path/backend/payload aware, or a cheap warm rebind can incorrectly justify a larger cold-upload batch.

Feedback loops must be closed on the channel that is actually measured: each
evaluation controller's drain observes and is governed by its own
`<lane>_queue_drain` channel. Isolated latency outliers (GC pauses, one-off
relayouts, event-loop stalls) are suppressed for a single sample on every
channel — a repeat is accepted as a real cost change — and drains that finish
under budget while hitting their batch cap regrow the cap from the measured
rate rather than waiting for the EWMA to decay.

Governor decisions are applied on two paths: a periodic sampling timer
(250 ms active, 1 s idle) that also refreshes telemetry, and an immediate
lightweight reapplication on interaction start/stop edges so interactive
budgets and worker clamps take effect with the first drag event rather than
up to a sampling period late.

The per-controller drain-fallback timer is a safety net for missed
cross-thread signals, not a scheduling mechanism: it backs off exponentially
(10 → 100 ms) while polls come up empty and snaps back to 10 ms whenever it
recovers an event the signal path should have delivered. The
`fallback_recovered_events` / `fallback_idle_polls` counters make signal
health observable per controller.

Idle slice prefetch is momentum-aware (`core.prefetch_policy`): sustained
same-direction scrubbing deepens speculation ahead of the motion (bounded,
with a single reversal guard), while a pause or direction change collapses
depth immediately. Planning is separate from admission; every candidate
still passes the memory, cost, busy-state, and work-graph gates.

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
