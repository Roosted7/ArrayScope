# Testing strategy

ArrayScope’s tests are layered because no single environment can prove numerical correctness, Qt behavior, GPU performance, and platform integration at once.

## Pure core and operation tests

Locations: `tests/core`, `tests/operations`, much of `tests/display`.

They prove:

- immutable state transitions and selection parsing;
- ROI/histogram/window-level math;
- operation shape/dtype/value semantics;
- region-plan, slab, chunked, optimizer, and stage-cache equivalence;
- memory/compute/resource policy decisions;
- geometry, shader-equivalent mapping, viewport, montage planning, and frame models.

Use property tests where ranges/shapes have many edge cases. These tests should not import Qt unless the module contract truly requires it.

## Architecture guards

Location: `tests/app/test_architecture_guards.py`.

They prevent known regressions such as operation type switches in renderer/slab code, direct graphics-item ownership outside the layer owner, static runtime budgets, or widget-owned presentation policy. Guards are supplements to good module design, not a reason to encode every line layout as a string assertion.

## Display/backend conformance

Locations: `tests/display`, `tests/window`.

These cover semantic presentation, commit acknowledgement, stale rejection, tile residency/upload counters, shader mapping, level/value equivalence, and widget lifecycle. Prefer deterministic work counters over elapsed-time assertions.

Rendering benchmark helpers must close/delete parentless views and collect Qt/Python cycles. Re-running the full backend matrix per assertion is both slow and a lifecycle stressor; module-scoped results are appropriate for deterministic assertions.

## UI interaction tests

Location: `tests/ui`.

Use `pytest-qt` to exercise dimension controls, viewport, montage, ROI/profile, panels, settings, coalescing, and scheduler integration. Assert semantic state and committed result rather than fragile pixel coordinates whenever possible.

High-frequency tests should process events in bounded loops with explicit conditions/timeouts. A passing interaction test does not establish good feel or frame pacing.

## Stress and benchmark tests

Locations: `tests/app/test_memory_stress.py`, `tests/app/test_operation_benchmarks.py`, `tests/display/test_rendering_benchmarks.py`, diagnostics/trace tools.

Use them to prove bounded allocations, deterministic upload/rebind behavior, callback work counts, and relative algorithmic properties. Wall-clock gates are optional and environment-specific.

Record separately:

- submission time;
- event-loop drain/first presented frame;
- callback max gap;
- preparation/upload counters and bytes;
- cache/residency state;
- process RSS.

## Manual and real-hardware tests

[Manual regression](manual-regression.md) covers interaction feel, rendering artifacts, Wayland/panel behavior, HiDPI, GPU limits, and lifecycle/context loss. Record OS, session type, Qt/PySide/PyQtGraph/VisPy versions, GPU/driver, data shape/dtype, backend, and settings.

Headless `offscreen` runs cannot validate:

- actual GPU upload/execution time;
- swap/frame pacing;
- maximum usable texture allocation;
- Wayland native-window behavior;
- pointer capture feel;
- HiDPI visual correctness.

## Test hygiene

- Import the real `arrayscope` package in the shared conftest before direct-import isolation tests can install package stubs.
- Close widgets/controllers in `finally` or through `qtbot.addWidget`.
- Cancel single-shot semantic callbacks on close; a zero-delay callback can outlive its graphics object.
- Do not silently change a stale test to match output. First establish whether it varied a real semantic input.
- Keep fixture data deterministic and small; generate large patterns when possible.
- Mark optional-backend/platform skips with a concrete reason.

## Recommended change matrix

| Change | Minimum validation |
|---|---|
| Pure state/math | focused unit + property/edge cases |
| Operation | shape/dtype/value + slab/chunked + cache key |
| Presentation/frame | display model/commit + stale revision tests |
| Backend mechanics | deterministic counters + lifecycle + conformance |
| Scheduler/memory | pure policy simulation + integration + trace |
| Visible UI | focused pytest-qt interaction + manual smoke |
| GPU/Wayland claim | real-hardware matrix and trace artifact |
