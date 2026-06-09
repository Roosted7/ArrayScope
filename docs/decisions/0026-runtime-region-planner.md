# 0026 - Runtime region planner

## Problem

Operation capabilities and region contracts existed, but runtime slab evaluation still used duplicated
operation-specific branches in `operations.slabs`. That kept the future StageCache insertion point
unclear: the evaluator could describe final and required regions, but execution did not actually move
through explicit stage transitions.

## Decision

ArrayScope now plans display/profile/scalar/export requests with `RegionPlan` and
`StageRegionTransition`. Registered operations own their own `required_input_region()` and
`apply_to_region()` methods. `operations.slabs` builds a region plan, applies the base required input
region, then executes transitions forward through the operation stack.

Developer Diagnostics exposes final region, required input region, expanded axes, transition summaries,
candidate stage-cache points, and peak estimates. StageCache allocation and cache lookup/storage remain
future Phase 4g work.

## Consequences

Runtime behavior is still intended to match materialized evaluation exactly, but request expansion now
has explicit stage boundaries. Future StageCache work can insert lookup/store operations at those
boundaries without duplicating planner logic. Adding a registered operation requires a region contract;
ArrayScope does not keep an internal materialization fallback for registered operations.

## Rejected Alternatives

- Keep duplicate `_evaluate_ops` branches in `operations.slabs`: rejected because it would keep the
  planner and executor from sharing one source of truth.
- Implement StageCache simultaneously: rejected to keep this increment focused on behavior-preserving
  runtime planning.
- Keep operation region mapping in a central planner type switch: rejected because operation behavior
  should stay with operation implementations.

## Tests Required

Tests cover final regions for image/line/scalar/export requests, registered operation backward region
mapping, transition ordering, disabled operation handling, cache candidate derivation, planner-backed
slab evaluation against materialized results, diagnostics formatting, and architecture guards that keep
registered-operation branching out of `operations.slabs`.

## Manual Checks Required

Open an operation-backed 3D array, render a sliced FFT view, inspect Developer -> Diagnostics ->
Operations, and confirm final/required regions, expanded axes, transitions, and cache candidates update
while the image remains correct. Confirm no StageCache UI appears yet.
