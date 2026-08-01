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

High-frequency tests should process events in bounded loops with explicit conditions. A passing interaction test does not establish good feel or frame pacing.

**Wait with `tests/settle.py`, not with a stopwatch.** `settle_window(win,
predicate)` pumps the event loop and watches the counters the kernel and the
presentation path already publish, re-arming its budget on every advance and
while any task is queued, running, or waiting to be delivered. It therefore
fails on exactly one thing: an idle kernel, an empty completion queue and no
armed timer with the goal still unmet — a lost wakeup (ADR 0051), which looks
the same on a saturated machine as on an idle one. There is no parameter for
the budget, deliberately.

**Never pause for a fixed number of event-loop turns.** `process_events(qtbot,
count=N)` is `N x qtbot.wait(10)` — a sleep that asks the product nothing, so
it is simultaneously too long on an idle box and too short on a busy one. It
had 672 call sites, ~90 s of static sleep per pass over the suite, and eleven
of the fourteen tests that failed under three concurrent suites on 2026-08-01
were waits of that shape.

`tests/ui/helpers.py` has the replacements, in increasing strength:

- `settle_layout(win, predicate=None)` — the queued dock-visibility requests
  have been dispatched and the canvas-preserve retry chain has finished. Says
  nothing about the frame, for tests that hide a dock in a state where the
  frame session is deliberately never going to settle.
- `settle_idle(win, predicate=None)` — the above plus a settled frame session.
  This is the default: it is what a pause after building a window, rendering,
  applying an operation or opening a dock was covering.
- `settle_quiescent(win, predicate=None)` — the above plus nothing armed to do
  any more work, via `window_work_in_flight`. **The exception, not the
  default**: only for claims of the shape "and that scheduled nothing further"
  — a coalesced resize that must not emit a second range change, a retarget
  that must stay one transaction. Reaching for it by reflex is a hazard rather
  than merely wasteful: `settle_until` re-arms its budget whenever work is in
  flight and has no absolute cap, so on a lane that keeps producing work an
  over-strong quiescent wait cannot time out at all — the run ends in a
  `faulthandler` abort naming some arbitrary later test. It was measured at 249
  call sites and entered a state stronger than `settle_idle` in ~3% of them, so
  it is down to 25. The three families that kept it are the ones whose claim
  needs it: an interactive render held on the coordinator's quiet timer, a
  resize or retarget that must stay one transaction, and
  `test_viewport_convergence_latch.py`, an A/B that samples both arms' committed
  frames and would otherwise compare them at different points. Demoting the rest
  was not free either — two sites needed something stronger still and now wait
  for the fact itself, since a prefetch *store* is not covered by a settled
  frame: prefetch is by definition work the visible frame does not wait for.

All three run one event-loop slice when the state already looks settled, so a
wait placed immediately after an action whose continuation is still sitting in
the event queue cannot answer before that continuation has been delivered.
`tests.settle.deliver_posted_events()` is that one slice on its own, for the
few sites where delivery is the entire requirement.

**A converted wait must be shown to actually wait.** The cheap proof is to
break the wakeup it depends on and check it goes red rather than green:
monkeypatch `CanvasPreserveController._single_shot` to a no-op, or
`set_dock_visible_later` to bump its generation without scheduling, or
`FrameSession.visible_plan_complete` to `False`, and the sites that depend on
each raise `SettleTimeout` naming the line.

The thing to rule out is a predicate that **cannot be false** — `tx is None or
tx.released` before the transaction exists is true of every possible product,
so that wait can never wait, on any machine, and is worse than the pause it
replaced. That is not the same as a predicate that is *usually* already true:
the panel-location predicates in `test_panel_layout.py` hold at the call in 49
of 52 measured cases on a quiet box, and they earn their place on the other
kind of box, which is the one this whole section is about. Keep them, and note
the cost: when such a predicate does not come true, the report says "lost
wakeup (ADR 0051)" for what is really "the panel went somewhere else", so read
the label — it names the waiting line — before believing the diagnosis.

The one legitimate remaining use of a fixed pause is a claim *about a time
window*, where the behaviour under test is itself a throttle:
`tests/ui/test_interactive_render_coalescing.py` asserts that a burst of slice
changes produces one render *within the coalescing interval*, and waiting for
the coalesced render to arrive would assert the opposite thing. Its module
docstring records the reasoning; a new fixed pause should say which throttle
bounds it.

"Should" is now checked, and checked structurally: `tests/timing_guard.py`
requires the duration to be **read** from the throttle — a constant the product
or the toolkit publishes, or the live object owning the timer — rather than
typed. `qtbot.wait(2 * PUBLISH_COALESCE_MS)` states its reason in a form that
breaks when the throttle is deleted, where a comment saying the same thing
outlives the mechanism it describes. A comment marker was the obvious
alternative and was rejected for that, and because fifty-odd of them is an
excuse list with the entries scattered through the tree rather than gathered at
the top of a file — the shape `819789f8` deleted three of. The settlement
budget is deliberately not a throttle for this purpose: a pause bounded by
`INTERACTION_SETTLE_HARD_LIMIT_MS` is a five-second sleep with a respectable
import in front of it.

The alternative it replaced was `qtbot.waitUntil(pred,
timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)`, which asks two questions and
reports one answer: "did the product get there?" and "within five seconds of
*this machine's* wall clock?". Measured 2026-08-01, three concurrent suites
turned that into fourteen red tests that all pass quiet, eleven of them
reporting `waitUntil timed out` — the same sentence a real regression prints.

Every user-visible step still has one repository-wide acceptance budget: 2 s
target, 5 s hard failure, owned by `arrayscope.tools.interaction_budget`. The
limit is per open/render/zoom/pan/scroll/scrub/level/refinement step, not per
multi-step scenario. **It is an acceptance criterion, and pytest is not where
it is adjudicated** — a worker sharing sixteen cores with fifteen others cannot
measure a 5 s bar. `arrayscope.tools.profile_montage_workflow` measures it, on
a quiet machine, with per-step settlement recorded. Never widen a timeout to
make slow settlement pass; a longer process-deadlock guard may terminate a
whole child but is not evidence of successful settlement.

## Stress and benchmark tests

Locations: `tests/app/test_memory_stress.py`, `tests/app/test_operation_benchmarks.py`, `tests/display/test_rendering_benchmarks.py`, diagnostics/trace tools.

Use them to prove bounded allocations, deterministic upload/rebind behavior, callback work counts, and relative algorithmic properties. Wall-clock gates are optional and environment-specific.

**An absolute size or duration does not belong in pytest.** A threshold derived
from the machine measures the machine. The clearest case:
`test_montage_tile_residency_rss_stays_bounded` compared process RSS and cache
bytes against caps that `arrayscope.core.memory_policy` computes as *fractions
of `psutil.virtual_memory().available`*, so a second workload shrank the caps
under a working set that was already resident and the gate failed
(`assert 2948096 <= 2423808`) with nothing having changed in the product. What
stayed in pytest is the relative claim — a second walk over the same semantic
windows must not grow the accounted set — and the numbers moved to
`arrayscope.tools.montage_residency_memory_benchmark`, which records the policy
inputs and the load average beside every result so a reading taken on a busy
box is identifiable rather than silently comparable.

The same rule catches durations wearing a correctness test's clothes. Before
relocating one, check whether the product already publishes the fact the
duration stood for — it usually does, and then the answer is neither a
benchmark nor a looser constant: kernel shutdown emits `action="complete"`
versus `action="timeout"`, a bounded `join` that returns already proves
cancellation reached the child, and "it refused before spawning" is best said
by a child that never wrote its start marker.

**"Ratio-free and threshold-free" is not the same as "not a stopwatch."** The
last wall-clock comparison in the suite,
`test_empty_delta_commit_is_cheaper_than_one_that_does_work`, compared two
measurements of the same thing — `empty_ms <= working_ms` — with no constant
anywhere in it. That made the *comparison* machine-independent and left the
*measurement* exactly as fragile as any other: both numbers are 1–7 ms, so
under three concurrent suites either one can absorb a scheduling delay big
enough to invert them, and the `[pyqtgraph]` arm did while passing serially.
The test to be suspicious of is any that reads a clock, whatever it then does
with the reading.

Restating it followed the rule above and found something the duration was
hiding. The claim was structural — an empty delta must stay on the incremental
path — and both backends already publish which path a commit took, so it is now
read rather than timed (`tests/render/test_commit_cost_scaling.py`; the
durations stay in `docs/architecture/progressive-render-contract.md` as
evidence, gating nothing). Forcing the defect back three ways showed the
stopwatch caught the WGPU form comfortably (14.8 ms against 7.5), caught one
PyQtGraph form by 14% (4.54 against 3.97 — the margin load inverts), and
**missed the other entirely**: 4.29 against 6.69 ms with the delta dropped, the
whole-active-set republication, still "cheaper", still green. PyQtGraph has no
empty-delta-only fast path, so a regained fixed cost lands on both arms of a `≤`
and largely cancels — the arm that was flaking was the arm least able to fail
for the reason it named. Check that a relocated timing assertion could ever
have failed before deciding where its replacement belongs; "it went red under
load" is evidence about the box, not evidence that the test was doing work.

None of that survived contact with a hurried commit on its own, so
`tests/app/test_timing_guard.py` reads the tree for it in ring 0. **A clock
reading may not reach a verdict**: not an `assert`, and not an `if elapsed >
...: pytest.fail()` either, since an assert-only rule steps straight over that.
The tree is at nought and the gate says nought, with no allowance to widen —
recording a duration beside a result stays free, and `tests/conftest.py`,
`tests/load_report.py` and `tests/seams.py` all still do it. The fixed-pause
half cannot be at nought yet, so it is a counted backlog that is only ever
revised downwards, printed in full by path and line on either failure; it is a
number rather than a list of paths precisely so that no file is grandfathered.
Both mutations were forced back to check the gate can fail: a bare
`qtbot.wait(250)` in `test_toasts.py` and `empty_ms <= working_ms` restored in
`test_commit_cost_scaling.py` were each named with their line.

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
indices, and restore the montage. Before those transitions, each stage runs
the same eleven-state crop matrix: one cropped displayed axis; both cropped
axes; independent, diagonal, and reversing one-pixel moves; an exact
256-sample page edge; a boundary crossing and return; and odd 99x101 extents.
Every successor must settle with a current committed frame. Per-step records
include settlement time, upload delta, binding-cache work, physical tile
coverage, and minimum/maximum pages sampled per tile. WGPU additionally gates
zero-upload reuse of already-resident source pages across the entire matrix
and the montage/single-slice boundary. It also projects every live payload
through the cold binding selector: distinct displayed-axis source windows must
produce distinct crop-local physical identities, even though their canonical
source-page identity is intentionally reusable. This catches the failure mode
where page-table metadata is current but the page still contains a predecessor
crop. The default run executes the same stages on WGPU and PyQtGraph.
While both displayed axes remain cropped, each displayed-axis stage also
scrolls every dimension: a three-input coalesced burst, settled restore,
settled +1 step, and settled return. Every one of those four checkpoints per
dimension is compared against a CPU semantic framebuffer oracle on WGPU and
the PyQtGraph raster. WGPU additionally compares the committed global source
origin with the immutable payload/session anchor. Upload accounting is
role-specific: displayed-axis scrolls must rebind with zero upload, while a
montage-axis move may report genuinely cold/prefetched pages separately.
Supplying `--screenshot-dir` captures each settled checkpoint for visual
review; periodic screenshot timing remains diagnostic because capture can
perturb draw acknowledgement.

WGPU page-pool byte policy is retention headroom, not visible admission.
Capacity tests count the physical pages the command path actually submits
(including a native source plane carried by a reduced preview), grow to the
complete immutable frame-plan working set within the device limit, and assert
that a valid full montage cannot fail merely because it crosses a policy-sized
page count. The byte preference is shared across all representation pools,
and a cold factor-misaligned crop spanning multiple stored pages must take the
bounded packed-upload fallback rather than fail geometry validation.
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
`perf record -F 99 -g` for native SciPy/Qt/render-driver stacks. Use `cProfile` only as
opt-in deterministic Python call-count evidence because it substantially
perturbs the GUI workflow. Avoid treating high-rate, blocking, cProfile, or
`py-spy --native` runs as timing evidence unless they are compared against a
plain JSONL run. Run the workflow on a real display for WGPU/Vulkan claims.

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

[Manual regression](manual-regression.md) covers interaction feel, rendering artifacts, Wayland/panel behavior, HiDPI, GPU limits, and lifecycle/context loss. Record OS, session type, Qt/PySide/PyQtGraph/WGPU/rendercanvas versions, GPU/driver, data shape/dtype, backend, and settings. When pytest-qt interaction tests disagree with the real app, capture the real widget event stream at the ownership boundary and convert that observed sequence into the regression test.

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
`QObject`/widget code and rendering surfaces must live on one main thread and
are not thread-safe.
Thread-based runners (e.g. `pytest-parallel`) would share one interpreter and corrupt that state —
do not use them. `pytest-xdist` is also the actively maintained, pytest-org tool.

**`--dist loadfile`.** Every test in a file runs on one worker. Module-scoped fixtures build once,
related tests stay together, and running a single file is effectively serial — convenient for
debugging. The session-scoped `qt_app` fixture is *correct* under xdist: each worker is its own
process, so each builds and reuses exactly one `QApplication`. Do not make it function-scoped.

**Worker cap.** `-n auto` is capped at half the logical cores by
`pytest_xdist_auto_num_workers` in `tests/conftest.py`. Many tests create Qt
surfaces and some initialize GPU device state; one worker per core saturates
the CPU and makes every worker initialize rendering state simultaneously,
which has intermittently destabilized the platform stack. Leaving half the
cores free for each worker's Qt/render threads keeps workers
stable while still giving a large speedup. On 2-core CI
runners the cap floors at 2, so CI parallelism is unaffected.

**Sized for the selection, not for the machine.** The same hook caps the pool
again at the number of test files [change selection](test-selection.md) left,
because `--dist loadfile` makes files — not tests — the unit of parallelism.
One file, or under 2.5 s of recorded work, runs in-process: a worker costs
~1.3 s to boot Qt and the package, which is more than those tests do. Test
files the map has never seen count as work, never as absence. A `-k`/`-m`/
`--lf`/node-id run is not narrowed by the map, so it gets the full pool.

**Per-worker filesystem isolation.** Workers share one filesystem, so `tests/conftest.py` gives each
worker (keyed on `PYTEST_XDIST_WORKER`) its own directory for the two shared on-disk resources:

- **QSettings** — the test `QApplication` uses a fixed org/app name, so all workers would otherwise
  read/write the same store and the autouse `_clear_qt_settings` fixture in one worker would wipe
  another's writes. Each worker gets its own `XDG_CONFIG_HOME`.
- **`tests/artifacts/`** — the Qt smoke test writes fixed filenames; each worker gets its own
  `ARRAYSCOPE_ARTIFACT_DIR`.

Both redirects are no-ops for serial runs (`-n 0` / no xdist), so CI steps that must publish PNGs to
the canonical `tests/artifacts/` run those commands with `-n 0`.

**Run order is a schedule, not a contract.** `--dist loadfile` hands out whole
files and gives a worker the next one when it runs dry, so the order files
appear in *is* the schedule. `tests/conftest.py` orders them
sub-500 ms files first (immediate feedback) and then longest-first. Measured,
that is worth nothing in wall time against the shortest-first order testmon
applies by itself — a queue model said 100 s and the suite said 0
([why, with both numbers](test-selection.md#run-order)) — so treat it as a
latency and determinism choice. Declaration order holds *within* a file. A
test that depends on what an earlier test left behind must still say so with
`@pytest.mark.coupled_order("<group>")`; everything else must stand alone.

**Foreign load is reported, not guessed at.** Every run prints the system load
before the first test and, at the end, the share of it that did not come from
this run (`tests/load_report.py`). Both the flake advice below and every
performance number depend on knowing the machine was quiet; that used to be
established after an hour of bisecting a phantom.

**No fixed-time assertions.** Parallel load makes wall-clock timing nondeterministic. A test that
launches background work and then asserts state after a *fixed* window — `QTest.qWait(220)`, or a
short `qtbot.waitUntil(..., timeout=250)` — passes only on an idle CPU and flakes under load. Wait on
the actual signal or condition, using the shared 5 s hard limit for a
user-visible step. Parallel load is not permission to raise that acceptance
limit: if the condition misses it, the path is too slow or the test must be
made smaller while preserving the behavior. This is the counterpart to the
"deterministic work counters, not elapsed time" rule elsewhere in this doc.

**Debugging serially.** Append `-n 0` to a **node id** to disable parallelism for that one test and
get clean tracebacks and working `-s`/`pdb`: `pytest -q -n 0 tests/ui/test_foo.py::test_bar`. A wide
`-n 0` is refused — serial is a different regime with its own standing failures, no per-test timeout
and no worker to survive a segfault; see
[test-selection.md](test-selection.md#why-a-wide--n-0-is-refused). The serial rings above
(2, 3 and 4) are never refused.

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
