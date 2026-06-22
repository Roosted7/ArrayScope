# State and operations

This document describes the authoritative array-view state and the reversible operation/evaluation pipeline.

## View state

`ViewState` is an immutable description of how the *derived* array is being viewed. It contains:

- array dimensionality and current derived shape;
- two image axes and a line/profile axis;
- scalar slice positions for non-image axes;
- optional range selections, including cropped image-axis selections;
- montage axis, selected montage indices, and column policy;
- complex component/channel and scale mode;
- per-axis flip and FFT-shift flags.

UI controls create a new state and hand it to a focused window/controller method. A spinbox, button, or menu is never the only copy of a semantic choice.

### Selection parsing

`core.slice_selection` normalizes text into explicit index selections. The current parser accepts Python-like and compatibility forms; ambiguous inputs must be previewed and tested rather than interpreted differently by separate widgets. Clamping or repairing input should remain visible to the user.

### Shape changes

Operation edits can change dimensionality. State synchronization remaps roles and selections to the new derived shape while preserving stable choices where possible. This logic belongs in state/operation coordination, not renderer callbacks.

## Array document

`ArrayDocument` contains:

- the base array;
- an explicit `revision` token;
- ordered `OperationStep` rows, including enabled state and stable row identity;
- derived shape/dtype information.

The base array is treated as immutable for cache identity. In-place external mutation requires `notify_data_changed()` or a new document/source revision.

`operations` means the enabled runtime sequence. `steps` means the user-visible document sequence. Undo/redo, recipes, row IDs, disabled rows, and UI presentation use `steps` and must not be mutated by optimizer shortcuts.

## Operation declaration contract

A registered operation owns its own:

- shape and dtype transformation;
- value implementation;
- blocking axes and chunkable axes;
- output-region to input-region expansion;
- temporary-memory multiplier/cost hints;
- stage-cache usefulness;
- fusion/cancellation/optimization eligibility.

Render and slab code must not grow a parallel switch over operation types. Architecture guards intentionally enforce this.

## Planning and optimization

The runtime sequence is:

```text
ArrayDocument steps
  -> enabled operations
  -> optimizer (behavior-preserving internal plan)
  -> requested output region
  -> backward region expansion
  -> forward stage transitions
  -> evaluation
```

The optimizer can remove inverse pairs, fuse compatible transforms, or add dtype adapters, but only when derived shape, dtype, and values remain equivalent. It returns an internal plan plus diagnostics; it does not rewrite `ArrayDocument.steps`.

`RegionPlan` records the requested final region and each required intermediate region. This makes image, profile, scalar-hover, montage-tile, and export requests use the same operation semantics.

## Evaluation modes

The visible render decision may choose:

- cached result;
- immediate exact evaluation;
- asynchronous exact evaluation;
- chunked evaluation;
- degraded preview;
- refusal when the estimated output/peak exceeds policy.

Chunk axes are selected only from operation-safe non-blocking axes. A blocking axis, such as an FFT axis, remains complete in each chunk.

Background functions receive an immutable document/state snapshot and cancellation token. They return `EvaluationResult` values. The UI-thread `OperationEvaluator` records status and stores accepted results after generation/key checks.

## Caches

ArrayScope separates caches because their reuse and cost differ:

- exact display images/export frames;
- montage tile payloads;
- profile/scalar inspection results;
- reusable operation-stage arrays;
- backend GPU/item residency (owned separately by the backend).

Keys include document identity/revision and all semantic inputs that affect values. Presentation-only state is excluded from source/materialization cache keys.

### Stage cache

`StageCache` stores selected expanded intermediate arrays under a dedicated memory budget. Candidates include operation prefix, region, shape, dtype, and usefulness metadata. When the preferred candidate is oversized, the planner may use an earlier fitting stage rather than forcing an unsafe allocation.

### Stage materialization

`StageMaterializationCoordinator` provides singleflight: concurrent consumers of the same `StageKey` attach to one in-flight job. Stage warmup is optional lower-priority work and must yield to visible deadlines and memory pressure.

## Correctness requirements

- Every operation has shape, dtype, and value tests.
- Region/slab evaluation matches full evaluation for representative and property-generated requests.
- Cancellation does not store partial values in exact caches.
- Degraded previews never occupy exact-result cache entries.
- Disabled operations remain in document history but do not run.
- Cache tests vary a real sliced/ranged axis; changing a currently displayed axis’s scalar placeholder is not a semantic view change.
- Complex data keeps raw semantic values separate from histogram/display mapping, especially in shader paths.

## Extension checklist

When adding an operation:

1. Implement the immutable operation and registration metadata.
2. Define shape/dtype and region expansion.
3. Declare blocking/chunking/cost/stage behavior.
4. Add full and slab/chunked value tests.
5. Add optimizer rules only when equivalence is provable.
6. Add UI/recipe integration without putting operation semantics in the widget.
7. Update the relevant ADR only when the contract itself changes.
