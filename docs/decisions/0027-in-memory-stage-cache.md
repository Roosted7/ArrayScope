# 0027: In-Memory StageCache

## Problem

Image, montage tile, profile, scalar, and export caches were keyed around final requests. That meant an
FFT or IFFT over a sliced axis could force a full-axis intermediate result for every nearby slice,
montage tile, profile, or export frame, but only the final display/profile result was cached. Phase 4g
P1 exposed planner transition boundaries and candidate stage-cache points, but no runtime cache used
them yet.

## Decision

Add an in-memory `StageCache` owned by `OperationEvaluator`. Cache keys include document
identity/revision, operation prefix, region, dtype, and shape. `operations.slabs` uses planner
candidates as lookup/store boundaries and can reuse broader cached regions for narrower requests. The
cache is priority-aware LRU, uses `MemoryPolicy.stage_cache_budget_bytes`, and is visible in Developer
Diagnostics through an overview bar and detailed text.

The cache is intentionally in-memory only. Disk-backed cache, memmap, and Joblib-inspired persistence
remain future work.

Amendment: the final Phase 4g runtime optimizer can eliminate some candidate stages before StageCache
sees them. Candidate planning now marks the useful retained stage in the optimized linear pipeline;
slab execution stores retained candidates rather than every cacheable transition, with an earlier
candidate used only as fallback when the preferred retained stage is oversized.

## Consequences

Repeated sliced transform requests become cheap after the first expanded compute. Simplified operation
pairs, such as same-axis FFT/IFFT, avoid both transform work and stage-cache allocation. For
non-simplified stacks, memory use is visible and policy-controlled, while invalidation remains
conservative: operation edits, base replacement, or base revision changes clear the StageCache.

No persistence or larger-than-memory behavior is provided.

## Rejected Alternatives

- Reuse image/profile caches for stages: rejected because final display/profile keys cannot represent
  reusable operation prefixes and expanded regions cleanly.
- Cache only final display images: rejected because it still recomputes expensive expanded transforms
  for every nearby request.
- Implement disk cache now: rejected to keep Phase 4g focused on the in-memory planner/cache path and
  avoid persistence/invalidation complexity before usage data exists.
- Implement algebraic simplification instead of caching: rejected because simplification is useful but
  does not solve general expanded-stage reuse across requests.
