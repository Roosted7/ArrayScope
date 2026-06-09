# 0028: Runtime Operation Optimizer

## Problem

StageCache avoids repeated work after an expanded stage has been computed, but it does not prevent
canceling operations from being computed once. Some view and elementwise operation stacks can be
composed or removed safely, and storing every cacheable intermediate can waste memory when a later
stage is the reusable result developers actually want.

## Decision

Add a Qt-free internal optimizer for enabled operation stacks. The optimizer produces a runtime
execution plan only; it does not mutate user operation rows, recipes, step IDs, undo/redo history, or
operation counts.

The optimized plan must preserve output shape and dtype. Same-axis `CenteredFFT`/`CenteredIFFT` pairs
are removed, with an internal dtype cast inserted when needed to preserve current FFT/IFFT dtype
behavior. Reverse pairs, conjugate pairs, adjacent same-axis crops, and adjacent internal dtype casts
are simplified. General fused-operation abstraction is deferred until there are multiple nontrivial
elementwise operations.

StageCache candidate planning now distinguishes retained and skipped candidates. Slab execution stores
the retained useful candidate by default and uses an earlier fitting candidate only as fallback when
the preferred retained stage is oversized.

## Consequences

Runtime evaluation does less work before caching, especially for operation pairs that cancel out.
StageCache holds fewer unnecessary intermediate entries. Developer diagnostics show original and
optimized operation counts, optimization summaries, and candidate retain decisions. Recipes remain
literal and user-authored.

## Rejected Alternatives

- Mutate the operation stack automatically: rejected because it would surprise users and complicate
  undo/redo, recipes, row IDs, and operation diagnostics.
- Expose optimizer as a public command now: rejected because Phase 4g only needs runtime efficiency.
- Ignore dtype preservation for FFT/IFFT identity: rejected because existing stack output dtype is
  part of visible behavior.
- Store every intermediate candidate: rejected because current linear pipelines usually only need the
  latest reusable stage, with fallback for oversized retained stages.
