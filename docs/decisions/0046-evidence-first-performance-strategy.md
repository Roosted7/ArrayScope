# 0046 — Evidence-first performance strategy after ownership convergence

**Status:** Accepted (2026-07).

## Context

Y1–Y3 repaired important ownership problems: one render staleness contract, shared
backend surface semantics, declarative UI binding, and one bounded-cache core. That
work was necessary, but it does not by itself prove that ArrayScope is fast across
low-power integrated GPUs, mobile-class hardware, desktops, and render servers.

The recent rendering line is directionally correct at the semantic level: a normal
plane, a huge plane, a one-tile montage, and a many-tile montage should share the
same presentation meaning and inspection contracts. The risk is mistaking that
semantic convergence for a requirement that every case use identical physical GPU
mechanics. A tiny frame should not pay atlas, quad, residency, or scheduling overhead
when a direct singleton surface is measurably faster. A huge plane should not require
a full eager display image before the first visible regions can appear. A montage
should not report requested tiles as resident before the backend accepts them.

Headless or software-GL runs are useful for API and contract coverage, but they are
not evidence for backend defaults, texture limits, frame latency, memory behavior,
or interaction feel. X5 is therefore an evidence gate, not another broad rewrite.

## Decision

Keep one semantic image-surface contract and one frame-planning vocabulary. Do not
restore the old separate normal-image semantic path, degraded/refuse render branch,
or synchronous CPU LOD pyramid.

Allow multiple physical strategies underneath the same semantic surface:

- **singleton/direct** for small or one-region frames when measured faster and
  within proven texture limits;
- **resident tiled** for montage and large-frame cases that need bounded region
  upload and reuse;
- **virtual/region-first tiled** for huge, out-of-core, chunked, or remote sources
  where first pixels must not wait for full-frame materialization;
- **multi-resolution tiled** only after compatible residency pages, arrays, or a
  virtual-texture/page-table design are proven.

X5 is split into ordered gates:

1. Capture real-device telemetry: texture and format limits, allocation outcomes,
   upload timings, accepted/rejected tile counts, RSS, event-loop gaps, and context
   loss/fallback behavior.
2. Strengthen acknowledged-residency conformance: partially accepted, deferred,
   rejected, evicted, and context-lost tiled commits must leave `DisplayScene`
   residency and value availability truthful.
3. Move viewport retargeting from montage-mode checks to tiled-scene/storage checks,
   then introduce region-first materialization for internally tiled normal images.
4. Decide backend defaults and singleton fast paths from measured traces, not from
   theoretical throughput or code-path elegance.
5. Enable asynchronous/source-provided LOD only after the acknowledgement,
   viewport-retarget, region-first, and compatible-residency contracts pass.

Performance-sensitive queues and caches must be bounded before work is admitted.
Speculative/warm residency must be an ordered queue that yields to visible work and
supersession. It must not copy the remaining payload map on every timer tick, and it
must not rely on post-insert trimming to recover from large batches.

## Consequences

- The current semantic route survives; the project does not need a throwaway rewrite.
- The next work is less glamorous but more decisive: telemetry, conformance, and
  strategy policy.
- The PyQtGraph backend remains the production default until hardware traces justify
  a different default or a capability-based split.
- VisPy remains a serious candidate for large tiled scenes, but real device stability,
  context handling, and latency decide where it is used.
- Small-image performance is treated as a first-class product requirement, not as
  collateral damage from tiled unification.
- Backend mechanics may diverge physically while sharing the same semantic frame,
  value-source, interaction, and commit contracts.

## Alternatives considered

### Continue broad renderer refactoring before X5

Rejected. Further refactoring without hardware and residency evidence risks producing
cleaner folders around the wrong physical policy. Split files only when the split
follows a proven workflow boundary and leaves tests or traces behind.

### Make every image use the same atlas/tile physical path

Rejected. This makes one backend easier to reason about, but it can pessimize small
images and one-region scenes. Semantic unification is the durable contract; physical
storage is a strategy decision.

### Restore the old direct normal-image semantic path

Rejected. That would bring back duplicated level, hover/value, cache identity,
interaction, and presentation behavior. A direct physical strategy is allowed; a
separate semantic world is not.

### Enable CPU LOD immediately for responsiveness

Rejected. Mixed-size LOD tiles in fixed native atlas slots are structurally unsafe.
LOD must use compatible pages, texture arrays, or virtual texture/page-table
mechanics and must preserve exact semantic values independently of display LOD.

## Migration

1. Keep the Y1–Y3 ownership contracts and architecture guards.
2. Add the X5 telemetry and conformance matrix before changing backend defaults.
3. Move viewport-retarget scheduling to tiled-scene/storage ownership.
4. Introduce a physical strategy policy below `ImageSurface`; include a singleton
   strategy but keep semantic presentation identical.
5. Implement region-first materialization for huge normal planes.
6. Revisit VisPy defaulting and LOD only after measured traces and conformance pass.

## Related records

- ADR 0038: backend composition.
- ADR 0039: unified image surface and deadline scheduler.
- ADR 0040: backend-aware presentation convergence.
- ADR 0041: LOD selection, materialization, and residency.
- ADR 0044: viewport-scoped tiled residency.
- ADR 0045: render orchestrator composition.
