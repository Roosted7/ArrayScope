# 0025 - Operation capabilities and region contracts

## Problem

Operation cost and request-expansion behavior was spread across central type/name checks in
`operations.cost` and operation-specific branches in `operations.slabs`. That made expensive operation
reuse hard to plan: the evaluator could estimate that FFT needs a full axis, but there was no shared
region/stage vocabulary for deciding what future StageCache entries should look like.

## Decision

Each registered operation now declares its output dtype and `OperationCapabilities`: kind, blocking
axes, chunkable axes, expanded request axes, temporary multiplier, stage-cache preference, and fusion
eligibility. `operations.cost` consumes those declarations for public cost estimates instead of
recognizing built-in operations through central type switches.

ArrayScope also has pure `operations.regions` and `operations.planner` contracts. Slab planning now
records final regions, required input regions, stage metadata, and candidate stage-cache points for
diagnostics and later runtime integration. This first increment deliberately does not allocate or use a
runtime StageCache.

## Consequences

Adding a new operation requires declaring its behavior on the operation itself. Cost estimates,
diagnostics, and future planner/cache code can then consume the same contract. The existing lazy slab
execution path remains behavior-preserving until the next Phase 4g increment moves runtime request
expansion fully behind the planner.

## Rejected Alternatives

- Keep type/name switches in `operations.cost`: rejected because every expensive operation would become
  another central special case.
- Implement StageCache immediately: rejected to keep the first increment focused and reviewable.
- Put capability metadata in the registry only: rejected because shape/dtype/capability behavior belongs
  with the operation implementation and should also work for non-UI construction.

## Tests Required

Tests cover every registered operation's dtype and capability declarations, cost estimates derived from
custom operation declarations, hashable region roundtrips, planner cache candidates for FFT/IFFT, slab
plan region metadata, and architecture guards keeping the new modules Qt-free with no runtime
StageCache allocation.

## Manual Checks Required

No new manual behavior is expected from this increment. Use Developer -> Diagnostics after rendering an
operation-backed view to confirm the Operations details show capability stages and stage-cache
candidates when relevant.

## Amendment

The follow-up runtime planner increment moved slab execution onto `RegionPlan` transitions. Registered
operations now own both backward region mapping and regional application. `operations.slabs` executes
planner transitions and no longer contains registered-operation request expansion branches. Runtime
StageCache allocation remains deferred.
