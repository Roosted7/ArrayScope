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

### Montage

A persistent `MontageSession` tracks target, requested/materialized/presented sets, payloads, levels, and commit acknowledgement. Pan/zoom retarget visible/near priorities and residency instead of rebuilding the semantic session. Ready results are progressively committed through bounded batches.

### Stage and prefetch

Reusable stages use singleflight. Nearby slice/tile work and warm residency are lower priority, gated by memory, scheduler busy state, feedback, and resource-governor decisions.

## Target scheduler behavior

The next scheduler should hold explicit presented, active, and latest-queued targets. A new interaction replaces queued obsolete work but does not automatically kill an active item that is nearly complete or produces reusable cache data.

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

## GUI-thread contract

All paths that mutate Qt or OpenGL state follow these limits:

- interactive callbacks target **≤ 4 ms**;
- idle presentation callbacks target **≤ 8 ms**;
- **16 ms** is a warning threshold, not a normal batch allowance;
- no callback loops over an unbounded data/user-sized collection;
- every batch has item, byte, and elapsed-time limits;
- partial progress is published and remaining work is rescheduled;
- queueing many individual Qt events is not equivalent to one bounded callback.

Current code does not yet enforce this everywhere. In particular, stage-wait release, priority rebuilds, histogram refresh, and some presentation updates need traces at large tile counts.

## Cancellation and supersession

Cancellation protects correctness and scarce resources; it is not a substitute for scheduling.

- Stale results are rejected by semantic key/revision even if cancellation arrives late.
- Exact cache entries are written only by complete accepted results.
- Work that is cheap to finish or reusable may be allowed to complete.
- Presentation-only changes supersede presentation work, not materialization.
- Camera changes retarget visible regions/residency, not the operation pipeline.
- Side-analysis results are guarded by the committed semantic target.

## Memory policy

`core.memory_policy` derives budgets from configured profile, system total/available memory, process RSS, and hard per-render caps. Separate budgets cover:

- visible render output/peak;
- montage canvas and individual tiles;
- display image cache;
- montage payload cache;
- profile/scalar cache;
- reusable stage cache;
- speculative/prefetch allowance.

`memory_budget` contains estimation/formatting helpers; it is not the source of runtime policy.

Estimates are conservative admission inputs, not proof that allocation will succeed. Diagnostics should record estimated versus observed bytes and refusal/degradation reasons.

## GPU residency

GPU residency has its own budget and lifecycle. It must consider queried device limits, actual allocation outcomes, texture format/shape, context identity, and pressure. CPU cache presence does not imply GPU residency; GPU eviction does not invalidate semantic CPU data.

## Feedback and resource governance

Latency feedback records callback duration and work count. Resource telemetry samples CPU/memory without blocking the UI. The resource governor combines those signals with policy and scheduler state to adjust:

- lane worker counts;
- callback/result fan-in;
- upload byte/item batches;
- commit interval;
- prefetch/speculation admission.

Overload backoff should be immediate; recovery gradual. Metrics must be path/backend/payload aware, or a cheap warm rebind can incorrectly justify a larger cold-upload batch.

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
- backend, dtype/component, storage strategy, and interaction state.

## Testing limits

Deterministic counters can prove “no upload on pan” or “one dirty tile only.” Headless wall-clock numbers cannot prove GPU execution or frame pacing. Release-level performance claims require the hardware matrix in [manual regression](../testing/manual-regression.md).
