# A test that skipped left a record that read as evidence

**Status:** fixed. `tests/skip_ledger.py` is new; `tests/testmon_policy.py`,
`tests/conftest.py`, `tests/coverage_map.py` and `tools/test_selection.py`
changed. Found while widening the WGPU screen-present path to X11
(`claude/relaxed-einstein-46087f`, see
[wgpu-screen-on-x11-2026-07-31.md](wgpu-screen-on-x11-2026-07-31.md)).

## What the harness said, and why it was false

With `arrayscope/display/backends/wgpu/screen_canvas.py` heavily rewritten in
the working tree:

```
test selection: nothing to run — no line your working tree changed is executed
  by any of the 11 recorded tests.
  That is a result, not a gap.
11 deselected in 0.27s
```

and `pytest --since` reported *"branch reaches 104 tests — all 104 verified
green against this exact tree from recorded runs, nothing left to run"*.

Those 11 tests were the change's only real coverage. One of them
(`test_auto_present_method_activates_screen_on_a_native_session`) genuinely
needed updating for the change. A regression would have passed every gate the
workflow prescribes.

## Root cause: one defect, not two

testmon records the source lines a test executed, and it records them for a test
that **skipped** exactly as for one that ran. The map's schema keeps `failed`
and `duration` and nothing else (`testmon_core.process_result`), so a skip lands
as a fast green whose dependency set is whatever the module import and the gate
that turned it away happened to touch. Nothing in the map distinguishes them.

Every `tests/gpu_interaction` test is opt-in — `tests/gpu_interaction/conftest.py`
sets `ENABLED` from `ARRAYSCOPE_GPU_TESTS=1` plus a real display — so under the
default `offscreen` environment they all skip, and all get recorded. Queried
from the branch's own `.testmondata`, for the ten
`test_wgpu_screen_present.py` tests:

| environment | dependency files per test | records `screen_canvas.py`? |
|---|---|---|
| `offscreen` (id 3) | 2–3 | **no**, 0/10 |
| `xcb+gpu` (id 4) | 26–224 | yes, 10/10 |
| `wayland+gpu` (id 5) | 26–224 | yes, 10/10 |

An *empty* map is handled correctly — it runs everything. The bug is
specifically the skipped-recording case, which looks like knowledge and is not.

### The suspected second defect is not one

The report wondered whether the decision had been taken against the `offscreen`
map while recording into a freshly created `xcb+gpu` one, because the message
said "11 recorded tests". It had not. Counted per environment in that map:

| environment | recorded tests | of which in `tests/gpu_interaction` |
|---|---|---|
| `offscreen` | 4488 | 41 |
| `xcb+gpu` | **11** | 11 |
| `wayland+gpu` | 10 | 10 |

`mapped` in that sentence is `len(data.all_tests)` for the resolved
environment. 11 is `xcb+gpu`'s entire population — the correct environment, read
correctly. The 11 records were skip records because the first `xcb+gpu` run
happened before the X server was up, and they have since been overwritten with
real ones (26–224 files) by the node-id runs that followed. Environment
resolution is sound; there is one defect here, not two.

### Blast radius beyond ring 4

Estimated first from the record shape — 56 of the 4488 `offscreen` records
carried a dependency set of ≤3 files — and then **measured** by the ledger this
change adds, over a full `pytest --testmon-noselect` of this checkout: **69
tests in the `offscreen` map skipped when they were recorded**. Not only
ring 4:

| file | tests |
|---|---|
| `tests/gpu_interaction/test_onscreen_interaction.py` | 12 |
| `tests/display/test_imagesurface_contract.py` | 11 |
| `tests/gpu_interaction/test_wgpu_screen_present.py` | 10 |
| `tests/stress/test_synthetic_stress_matrix.py` | 5 |
| `tests/gpu/test_wgpu_astc_render.py` | 4 |
| `tests/gpu_interaction/test_framebuffer_cpu_reference.py` | 4 |
| `tests/gpu_interaction/test_progressive_open_reference.py` | 4 |
| `tests/gpu_interaction/test_pyqtgraph_raster_cpu_reference.py` | 4 |
| remaining files | 15 |

The gap between the estimate and the measurement is the argument for the ledger
in one line: the record shape is a *guess* at what happened, and 13 of the 69
were invisible to it.

## The fix

The outcome is a fact only a run can see, so it is written down as it happens —
`tests/skip_ledger.py`, the same shape and the same argument as the red and
flake ledgers, per environment, carried across map seeding.

**A mark means "this record is not evidence of independence", and it clears on
exactly one event: the test runs for real.** Nothing expires, nothing is tuned.

1. **`--since` reaches marked tests**, and *reaching* is the point — the gate
   derives its reached set from recorded dependencies, so a skip-recorded test
   was absent from that set no matter what the branch rewrote. Partitioning
   alone would have fixed nothing: there was nothing in the set to partition.
   The widening lives in `since_verification`, the single owner both the
   controller and the worker already go through. New run reason:
   `skipped when recorded`.
2. **The inner loop reports an unmeasured remainder** instead of folding them
   into "unaffected by construction", and drops "that is a result, not a gap".
   The loop makes no completeness claim, so it does not pay to reach them; it
   just stops claiming what it cannot support.
3. **The coverage side** (`coverage_map.new_code`) had the same hole read from
   the other end — it reported `screen_canvas.py`'s `ensure_context` and
   `_on_surface_resized` as new functions with no test. It now asks the other
   environments, so those read `executed under xcb+gpu, wayland+gpu, not here`.
   Its docstring claimed *"'no test executes it' cannot be an artefact of a
   stale map"*, which was true only within one environment; that is now stated.
4. **Existing maps cannot heal themselves.** The ledger records outcomes as they
   happen, so it can only speak for tests some run has observed since it
   existed; on an older map every entry's skip status is unknown, and unknown is
   exactly what reads as "unaffected by construction". There is no way to
   inspect an old record and tell what it was. So the sweep record carries a
   `skips` stamp beside `attributed`, `--since` refuses without it, and the
   inner loop names it every run — the same mechanism, for the same reason, as
   the per-test attribution fix earlier the same day. `pytest --testmon-noselect`
   retires both.

### Verified end to end

Recording the ring offscreen into a scratch map, then editing a body ring 4
executes (`ensure_context`), the same command that produced the false claim now
produces:

```
test selection: nothing to run — no line your working tree changed is executed
  by any of the 10 recorded tests.
  Unmeasured remainder: 10 tests skipped when the map recorded them, so their
  recorded dependencies describe a gate, not their code —
  tests/gpu_interaction/test_wgpu_screen_present.py (10).
  `pytest --since` reaches them; running them where their gate opens is what
  measures them.
```

and `pytest --since main`:

```
--since main: branch reaches 10 tests — 0 verified green against this exact tree
from recorded runs, 10 run now (10 skipped when recorded).
```

## The limitation, stated rather than hidden

Opening the gate is invisible to selection — the gate's inputs are the
environment, not the code — so the ordinary inner loop deselects a marked test
and the mark survives a run that would have cleared it. Widening the inner loop
to re-attempt every mark was the alternative and was rejected: it would drag the
whole opt-in ring into every offscreen run and cost the `nothing to run` fast
path. The escape hatch is therefore named on the remainder line, on every run
that has one — `pytest --since`, or any node-id run, which turns selection off
for that run. `test_a_mark_clears_the_moment_the_test_actually_runs` pins both
halves.

## Traps hit

* **The xfail lookalike.** xfail arrives as `skipped` with `wasxfail`, but its
  body ran to completion and its record is ordinary evidence. Marking it would
  put a permanent mark on every xfail in the suite. The guard test caught this
  within one session — because the first draft **duplicated the outcome rule**
  into a test fixture's conftest instead of calling the production one, and the
  copy drifted immediately. The rule now has one owner, `SkipLedger.observe`.
* **"Reached" is not "measured".** The first draft subtracted what the run
  reached from the remainder, so `--since` — which reaches every mark and, in
  the offscreen environment, watches every one of them skip again — reported an
  empty remainder. That is the same false claim one step along. The remainder is
  read from the *live* ledger, which is the outcome ledger: a test that ran for
  real has already had its mark cleared, one that skipped again still carries
  it.
* **The `skips` stamp is narrower than `attributed`.** A sweep also counts when
  the policy turned selection off (`--no-testmon`, `--cov`, CI), and those runs
  open no ledger — they witness every skip and write none down. Stamping them
  would claim exactly the knowledge the key exists to certify.

## Not fixed here

`tests/app/test_test_selection.py::test_every_test_that_spawns_a_child_process_declares_what_it_runs`
is red on `main`: `tests/app/test_log_noise.py` (added by `09d60ad1`) spawns a
child process without an entry in `OUT_OF_PROCESS_DEPENDENCIES`. Inherited,
unrelated to this change, chip spawned.
