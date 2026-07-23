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

Every user-visible step has one repository-wide acceptance budget: 2 s target,
5 s hard failure, owned by `arrayscope.tools.interaction_budget`. The limit is
per open/render/zoom/pan/scroll/scrub/level/refinement step, not per multi-step
scenario. Local budgets may be shorter only through the shared cap. Never
widen a timeout to make slow settlement pass; a longer process-deadlock guard
may terminate a whole child but is not evidence of successful settlement.

## Stress and benchmark tests

Locations: `tests/app/test_memory_stress.py`, `tests/app/test_operation_benchmarks.py`, `tests/display/test_rendering_benchmarks.py`, diagnostics/trace tools.

Use them to prove bounded allocations, deterministic upload/rebind behavior, callback work counts, and relative algorithmic properties. Wall-clock gates are optional and environment-specific.

Record separately:

- submission time;
- first useful display;
- exact-visible frame;
- full completion;
- event-loop p95/p99/max gap;
- preparation/upload counters and bytes;
- cold upload/preparation versus warm rebind/visibility work;
- cache/residency state;
- process RSS.

Use `arrayscope.tools.profile_montage_workflow` when scheduler or backend
changes may affect perceived pacing. It drives a real window through the
bundled NIfTI dataset, full dim-2 tiled montage, and FFT-over-dim-2 montage.
Its displayed-X and displayed-Y stages each perform short crop-window scrolls,
swap the image axes, transition through current/+1/current single-slice
indices, and restore the montage. Those stages gate committed-frame currency
on every successor and, on WGPU, zero-upload reuse of already-resident source
pages across the montage/single-slice boundary. The default run executes the
same stages on WGPU and PyQtGraph.
After the FFT montage is visible it also performs a deterministic
`fft_level_refinement_preview` level edit so histogram/level presentation
latency is captured on the same onscreen tiled workflow.
With `--jsonl`, the tool writes phase records with first-content timing, total
phase timing, event-loop gap statistics, callback observations, tile upload
bytes, and montage compute counters. Use those JSONL fields as the timing
evidence. Cold and warm runs are separate evidence: a warm resident pan, clean
flush, histogram refinement, or level-only update must not be folded into the cold initial-display
number.
Full-axis FFT montage should be stage-backed once the shared stage is available;
hundreds of direct FFT tile computations are a scheduler/cache regression even
if tiles eventually appear.

Rendering benchmark coverage must include small, medium, and large tiled cases.
Small tiled cases protect one-tile/small-tile latency in the unified tiled
engine; medium cases catch item/page fan-in behavior; large cases expose
residency pressure, event-loop starvation, and backend commit scaling.

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
`suite-manifest.jsonl` as authoritative. Benchmark artifacts must record git
revision, clean/dirty state, command line, platform, backend, and tool status.
A complete suite requires `overall_valid: true`, every child command to return
`0`, every expected artifact to be nonempty, and full sampled py-spy to report
samples without missed-stack errors. Missing tools, profiler failures, partial
artifacts, or unavailable stack samples are degraded evidence, not clean
completion. Degraded evidence can help attribution, but it must not support a
performance claim until the missing or failed tool status is recorded and the
claim rests on a valid timing artifact.

## Manual and real-hardware tests

[Manual regression](manual-regression.md) covers interaction feel, rendering artifacts, Wayland/panel behavior, HiDPI, GPU limits, and lifecycle/context loss. Record OS, session type, Qt/PySide/PyQtGraph/VisPy versions, GPU/driver, data shape/dtype, backend, and settings. When pytest-qt interaction tests disagree with the real app, capture the real widget event stream at the ownership boundary and convert that observed sequence into the regression test.

Headless `offscreen` runs cannot validate:

- actual GPU upload/execution time;
- swap/frame pacing;
- maximum usable texture allocation;
- Wayland native-window behavior;
- pointer capture feel;
- HiDPI visual correctness.

## Parallel execution

The suite runs in parallel by default via [`pytest-xdist`](https://pytest-xdist.readthedocs.io/).
The configuration lives in `pyproject.toml`:

```toml
addopts = "-n auto --dist loadfile"
```

**Why xdist (processes), not threads.** xdist runs each worker as a separate OS process, so
global C-extension and Qt state is fully isolated. That is essential here: `QApplication`, all
`QObject`/widget code, and the GL surfaces must live on one main thread and are not thread-safe.
Thread-based runners (e.g. `pytest-parallel`) would share one interpreter and corrupt that state —
do not use them. `pytest-xdist` is also the actively maintained, pytest-org tool.

**`--dist loadfile`.** Every test in a file runs on one worker. Module-scoped fixtures build once,
related tests stay together, and running a single file is effectively serial — convenient for
debugging. The session-scoped `qt_app` fixture is *correct* under xdist: each worker is its own
process, so each builds and reuses exactly one `QApplication`. Do not make it function-scoped.

**Worker cap.** `-n auto` is capped at half the logical cores by
`pytest_xdist_auto_num_workers` in `tests/conftest.py`. Many tests create real GL contexts
(vispy/pyqtgraph surfaces); one worker per core saturates the CPU and has every worker building GL
contexts against the same offscreen/software-GL stack simultaneously, which intermittently
**segfaults** the driver. Leaving half the cores free for each worker's Qt/GL threads keeps workers
stable while still giving a large speedup (full suite ≈150s serial → ≈35s here). On 2-core CI
runners the cap floors at 2, so CI parallelism is unaffected.

**Per-worker filesystem isolation.** Workers share one filesystem, so `tests/conftest.py` gives each
worker (keyed on `PYTEST_XDIST_WORKER`) its own directory for the two shared on-disk resources:

- **QSettings** — the test `QApplication` uses a fixed org/app name, so all workers would otherwise
  read/write the same store and the autouse `_clear_qt_settings` fixture in one worker would wipe
  another's writes. Each worker gets its own `XDG_CONFIG_HOME`.
- **`tests/artifacts/`** — the Qt smoke test writes fixed filenames; each worker gets its own
  `ARRAYSCOPE_ARTIFACT_DIR`.

Both redirects are no-ops for serial runs (`-n 0` / no xdist), so CI steps that must publish PNGs to
the canonical `tests/artifacts/` run those commands with `-n 0`.

**No fixed-time assertions.** Parallel load makes wall-clock timing nondeterministic. A test that
launches background work and then asserts state after a *fixed* window — `QTest.qWait(220)`, or a
short `qtbot.waitUntil(..., timeout=250)` — passes only on an idle CPU and flakes under load. Wait on
the actual signal or condition, using the shared 5 s hard limit for a
user-visible step. Parallel load is not permission to raise that acceptance
limit: if the condition misses it, the path is too slow or the test must be
made smaller while preserving the behavior. This is the counterpart to the
"deterministic work counters, not elapsed time" rule elsewhere in this doc.

**Debugging serially.** Append `-n 0` to any command to disable parallelism and get clean tracebacks
and working `-s`/`pdb`: `pytest -q -n 0 tests/ui/test_foo.py::test_bar`.

## Test hygiene

- Import the real `arrayscope` package in the shared conftest before direct-import isolation tests can install package stubs.
- Close widgets/controllers in `finally` or through `qtbot.addWidget`.
- Cancel single-shot semantic callbacks on close; a zero-delay callback can outlive its graphics object.
- Do not silently change a stale test to match output. First establish whether it varied a real semantic input.
- Keep fixture data deterministic and small; generate large patterns when possible.
- Mark optional-backend/platform skips with a concrete reason.

## Coverage reporting

CI runs one dedicated full-suite coverage job on Linux/Python 3.12 and uploads
`coverage.xml` as both a GitHub artifact and a Codecov report. Keep coverage
as a trend and review signal: it can reveal unexercised code, but it does not
replace the semantic assertions, deterministic counters, visual smoke tests,
or real-hardware checks required elsewhere in this strategy.

Run the same coverage pass locally with:

```bash
pytest tests/ -q --cov=arrayscope --cov-report=term-missing:skip-covered --cov-report=xml
```

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
