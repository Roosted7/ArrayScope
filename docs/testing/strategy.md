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

Use `arrayscope.tools.profile_montage_workflow` when scheduler or backend
changes may affect perceived pacing. It drives a real window through the
bundled NIfTI dataset, full dim-2 tiled montage, and FFT-over-dim-2 montage.
With `--jsonl`, the tool writes phase records with first-content timing, total
phase timing, event-loop max gap, callback observations, tile upload bytes, and
montage compute counters. Use those JSONL fields as the timing evidence.
Full-axis FFT montage should be stage-backed once the shared stage is available;
hundreds of direct FFT tile computations are a scheduler/cache regression even
if tiles eventually appear.

Wrap the workflow with external profilers only for attribution. Prefer a
low-impact `py-spy record --format raw --rate 50 --nonblocking --gil` sample for
quick Python hot-stack hints, a duration-bounded blocking `py-spy --rate 80` run
when complete sampled Python-thread stacks matter more than pacing, and
`perf record -F 99 -g` for native SciPy/Qt/GL stacks. Use `cProfile` only as
opt-in deterministic Python call-count evidence because it substantially
perturbs the GUI workflow. Avoid treating high-rate, blocking, cProfile, or
`py-spy --native` runs as timing evidence unless they are compared against a
plain JSONL run. Run the workflow on a real display for OpenGL/VisPy claims.

The built-in `--profile-suite` runner emits plain timing JSONL, low-impact
py-spy, full sampled py-spy, and perf artifacts by default; cProfile is
available with `--include-cprofile`. It prints the focused `suite-summary.md`
interpretation to stdout and leaves raw JSONL/profiler details on disk. Treat
`suite-manifest.jsonl` as authoritative: a complete suite requires
`overall_valid: true`, every child command to return `0`, every expected
artifact to be nonempty, and full sampled py-spy to report samples without
missed-stack errors. Missing tools, profiler failures, or partial artifacts are
degraded or failed evidence that must be fixed before using the suite as
benchmark support.

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
