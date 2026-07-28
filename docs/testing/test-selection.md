# Change-driven test selection

`pytest` runs the tests your working tree affects, and skips the ones it
provably does not. This page is the reference: what that means, what it cannot
see, and which command to reach for.

Policy and mechanism live in [`tests/testmon_policy.py`](../../tests/testmon_policy.py)
and [`tests/conftest.py`](../../tests/conftest.py); the reporting tool is
[`tools/test_selection.py`](../../tools/test_selection.py). The rings this sits
inside are in [README.md](README.md) — selection changes *which tests a ring
runs*, never *which ring a claim needs*.

## The problem it solves

The defect that keeps recurring is not a red suite that got shipped. It is:

> the change was in `render/`, so `tests/render` was the thing that got run,
> and the regression surfaced in `tests/ui`.

Directory intuition is what fails, and it fails silently. The full suite is
~230 s, so "just run everything, every time" loses to the clock during an
edit loop, and the loop is exactly where the guess gets made.

[pytest-testmon](https://testmon.org) records, for each test, the source lines
it actually executed. That turns the question into evidence. Reverting
`987d6bdc` — a 22-line change to `arrayscope/window/frame_effects.py`, shipped
with one new `tests/window` file — selects **20 tests across 16 files: 13 of
them in `tests/ui`, one in `tests/display`.** None of those directories would
have been guessed from the diff.

So selection is on by default. It is not primarily a speed feature; it is what
makes the *honest* set of tests the cheap one to run.

## Everyday use

```bash
pytest                    # everything this working tree affects
```

That is the whole inner loop. No flags, no path to guess. The run announces
what it did in two places:

```
test selection: on (default) — 20 of 3540 mapped tests affected
...
test selection: 3520 of 3540 mapped tests were unaffected by this working
tree and did not run. The whole suite is `pytest --no-testmon`.
```

Read the second line before writing "suite green" anywhere. A selected run that
passes says the affected tests pass — a strictly smaller claim.

| Goal | Command |
|---|---|
| Inner loop | `pytest` |
| The whole suite (pre-merge, or after anything below) | `pytest --no-testmon` |
| Blast radius without running anything | `python tools/test_selection.py` |
| What the map does *not* cover, and which tests are known red | `python tools/test_selection.py status` |
| Re-run the known-red tests too | `ARRAYSCOPE_TESTMON_RERUN_FAILING=1 pytest` |
| Rebuild the map from scratch | `rm .testmondata && pytest` |
| Re-record without deselecting | `pytest --testmon-noselect` |
| Coverage, artifacts, CI | already exhaustive — see below |

Scoping still works and still narrows further: `pytest tests/ui` runs the
affected tests *in* `tests/ui`. Manual selectors (`-k`, `-m`, `--lf`, a
`file.py::test` argument) turn deselection off entirely and run exactly what you
asked for, while still recording.

## Reading the blast radius

`tools/test_selection.py` answers "what does this change reach?" in about a
quarter of a second, without collecting or running anything:

```bash
python tools/test_selection.py            # grouped summary
python tools/test_selection.py --tests    # every affected node id
python tools/test_selection.py --json     # for scripts
```

This is the number worth putting in a handoff or a commit message. "Touches 84
tests across `ui`, `window` and `display`" is a review signal that a diff alone
does not give you.

## What it cannot see

testmon traces Python lines executed **in the pytest process**. Three things are
therefore invisible, and none of them are fixed by running `pytest` again:

1. **Child processes.** Coverage stops at the process boundary. The tests that
   genuinely spawn one declare what that child runs, in
   `OUT_OF_PROCESS_DEPENDENCIES` — the entry point is then re-attached to the
   test's recorded dependencies, so those tests are selected exactly when it
   changes. What the entry point *imports* is still out of reach.
   `tests/app/test_test_selection.py` fails if a new spawning test appears
   undeclared, or if a declared path moves.
2. **Non-Python inputs.** Icons, fixture arrays and JSONL traces have no
   fingerprint.
3. **Real rendering.** Rings 3–4 prove things about pixels a compositor drew.
   Selection has nothing to say there, and [the ring rules](README.md) are
   unchanged: whoever touches a display/render/kernel/window lane still runs
   those rings themselves.

The escape hatch for all three is one flag:

```bash
pytest --no-testmon       # everything, untraced, exactly as before
```

Run it before merging. Selection is for the loop, not for the gate.

## Where selection turns itself off

The policy (`testmon_policy.decide`) refuses in the runs that must be
exhaustive, so no stale map can quietly truncate them:

| Situation | Why |
|---|---|
| `--no-testmon` | explicit |
| `--cov` / `--cov-report` | a coverage number over a subset of the suite is a false number |
| `CI` is set | every CI job is a gate; see below for the one exception |
| `ARRAYSCOPE_TESTMON=0` | manual override, e.g. while bisecting |
| pytest-testmon not installed | nothing to select with |

An explicit `--testmon`, `--testmon-noselect`, `--testmon-nocollect` or
`--testmon-forceselect` always wins and is passed through untouched.

### In CI

Selection is off by default in CI because Actions sets `CI`, so no map — stale,
absent, or restored from another branch — can narrow a gate.

One job opts back in with `ARRAYSCOPE_TESTMON=1`: **Affected GUI tests (fast
signal)**, which caches `.testmondata` between runs and runs only what the diff
affects. That is safe there and nowhere else, because everything it runs is also
run exhaustively by the `coverage` job in the same workflow — an under-selecting
map costs a late signal, never a missed regression. It also sets
`ARRAYSCOPE_TESTMON_RERUN_FAILING=1`, since a cached map recorded from a failing
run would otherwise keep reporting that job green while the test stayed broken.

Note what this does *not* do: it does not shorten the workflow. The `coverage`
job runs the whole suite in parallel and is on the critical path, so selection
buys runner minutes and an earlier red, not wall-clock. Making the exhaustive
jobs selective would buy wall-clock and cost the gate; that trade is not taken.

The cache key is the interpreter plus a hash of `pyproject.toml`, with a prefix
`restore-keys` fallback to the newest matching map. Restoring a map from an
unrelated commit is harmless by construction — content-addressed fingerprints
mean whatever differs is re-run — and a genuinely different package set is part
of testmon's environment key, so it discards the map by itself.

Regenerating the canonical `tests/artifacts/` PNGs is a `--no-testmon` job. A
selected run redirects `ARRAYSCOPE_ARTIFACT_DIR` to a private directory the way
an xdist worker already does, so it cannot overwrite them from a partial run.

## The map

`.testmondata`, at the repo root, gitignored, one per checkout — a worktree
gets its own. It is a SQLite file of about 2.5 MB for this suite.

* **It maintains itself.** Every selected run re-records the tests it ran, and
  the tests it skipped keep fingerprints that are still valid by construction.
  There is no refresh chore. A first run on an empty map runs everything, in
  order to record everything.
* **It is keyed by regime.** `environment_expression` in `pyproject.toml` keys
  entries by `QT_QPA_PLATFORM` plus the `ARRAYSCOPE_STRICT_UI`, `_GPU_TESTS` and
  `_STRESS` ring variables. The same test executes different code offscreen and
  on a real compositor, so merging their fingerprints would let a ring-1 run
  mark a ring-4 test "unaffected". A ring you have never run simply has no
  entries, and everything in it runs.
* **It is checksum-based, not timestamp-based, and works on the AST.** Pulling
  50 commits invalidates exactly what changed. Reformatting, re-indenting or
  moving a comment invalidates nothing.
* **It reorders the run**, and `tests/conftest.py` replaces that order — see
  [Run order](#run-order) below.
* **Tracing costs ~3.5%.** A full traced run measured 236 s against a 228 s
  untraced baseline, with byte-identical outcomes (3459 passed, 13 failed, same
  13). It is not a source of timing flakes at this scale.

If the map is missing or unreadable, every code path degrades to running
everything. There is no state in which selection silently skips more than it
should because the map broke.

### A new worktree seeds itself

A fresh worktree has no map, so its first run would be a full traced suite —
exactly when selection would help most, since an agent branch usually touches a
handful of files. So the first run **copies** a map in, and says so:

```
test selection: seeded this checkout's map from /home/you/projects/ArrayScope/.testmondata
```

It prefers the main checkout (a worktree is normally branched from it) and
otherwise takes the largest sibling worktree's map. This is safe rather than
merely convenient: every fingerprint is content-addressed and path-relative, so
the donor's records are compared against *this* tree's files and whatever
differs is invalidated. A seeded map can only cause more tests to run, never
fewer. A map recorded under a different package set or Python version reads as
empty, because both are part of the environment key.

**Copy, never link.** Two checkouts sharing one map would fight: each run
records its own tree's fingerprints over the other's and vacuums what the other
still references, so both would end up seeing everything as changed — with
SQLite serializing their runs on top of that.

Set `ARRAYSCOPE_TESTMON_SEED=0` to opt out and record from scratch.

## Tests that were already red

testmon's own rule is that a test which failed last time re-runs unconditionally,
whatever changed. The reasoning is sound in general: a failed test stops at the
failure, so the lines it recorded are the lines of a run that ended early, and
its dependency set may be short of the file that would fix it.

ArrayScope does not do that by default, because the cost is not theoretical.
Branches here carry documented incumbent failures, and re-running them measured
**124 s on an otherwise clean tree** — an entire inner loop spent re-confirming
what everybody already knew. So a red test whose dependencies did not change is
treated like any other unaffected test, and every run says how many:

```
test selection: 10 of those were already failing before this run and nothing
they use changed, so they were not re-run (ARRAYSCOPE_TESTMON_RERUN_FAILING=1
re-runs them; python tools/test_selection.py status names them).
```

`tools/test_selection.py status` lists them with the time they would cost.

Turn it back on while you are actually fixing one of them:

```bash
ARRAYSCOPE_TESTMON_RERUN_FAILING=1 pytest
```

A red test whose dependencies *did* change is affected like anything else and
runs either way — skipping known reds never reaches a test your change touches.

## When the machine is busy

A competing workload invalidates two things this repository leans on: the
timing-fragile tests flake under CPU saturation, and any duration measured while
something else owned the cores is not evidence. Both are normally discovered
after an hour of bisecting a phantom, so every run reports it — once in the
header, once in the terminal summary:

```
WARNING: ~4.2 of the load on 16 cores during this run came from elsewhere.
Re-run before attributing a failure or a duration to your change.
```

The closing figure subtracts this run's own parallelism (its children's CPU time
over wall time), so a 16-way suite does not warn about itself. It under-reports
rather than over-reports: the one-minute load average has already partly
forgotten a burst that ended mid-run, so a quiet line is weaker evidence than a
loud one. See [`tests/load_report.py`](../../tests/load_report.py).

## Run order

`--dist loadfile` hands out whole files and gives a worker the next one when it
runs dry, so the item order *is* the schedule. testmon orders what it selected
shortest-first; `tests/conftest.py` replaces that with every sub-500 ms file
first, then longest-first, declaration order within each file.

**This is for determinism and first-feedback latency, not for wall time**, and
the distinction is worth stating because the modelling said otherwise. A queue
simulation over this suite's recorded durations (256 files, 8 workers, floor
196 s) predicted 296.5 s for shortest-first against 196.4 s for longest-first —
a 100 s win, with the run ending on one busy worker and seven idle. Measured on
the real suite, three runs each:

| order | modelled makespan | **measured** |
|---|---|---|
| shortest first (testmon's own) | 296.5 s | **226.7 / 227.4 s** |
| longest first, sub-500 ms leading | 196.7 s | **225.7 / 226.1 / 226.2 s** |

Indistinguishable. The model is wrong because xdist prefetches the next work
unit as a worker's current file drains, so the tail it predicts never forms, and
because durations recorded under 8-way parallel load do not compose the way a
serial model assumes. Do not quote the modelled column as a result.

What the order *is* chosen for: pure longest-first starts every worker on
something enormous and reports nothing for 49 s, so every sub-500 ms file leads
and the first results land in about a millisecond; longest-first for the
remainder is free and is the right shape if the balance ever does matter; and
files the map has never timed lead, since unknown cost is more safely assumed
large.

**Within a file, declaration order is restored.** Recorded durations change every
run, so testmon's sort reshuffles each file differently each time; that turned
three latent order dependencies into failures that moved between files from run
to run (`tests/gpu/test_chunk_codec.py`, `tests/ui/test_window_sync.py`,
`tests/operations/test_operation_library.py` — all real bugs, all still there,
none caused by selection). A run order that changes underneath them makes every
red ambiguous; surfacing them deserves a deliberate shuffle, not a side effect.

Tests that only mean anything in sequence say so with
`@pytest.mark.coupled_order("<group>")`, which keeps the group adjacent and in
declaration order. `tests/core/test_test_state_isolation.py` is the only user —
one test installs a module, the next asserts its identity survived teardown.
Use it only for a test that observes what a previous one left behind.

## Worker sizing

`--dist loadfile` keeps a file on one worker, so *test files* — not tests —
bound useful parallelism. `pytest_xdist_auto_num_workers` now sizes the pool for
the selection before collection starts:

* still capped at half the logical cores (the GL-context stability cap);
* capped again at the number of affected test files;
* **zero workers for one file or under 2.5 s of recorded work** — booting a Qt
  worker costs ~1.3 s, more than those tests do. The run happens in-process.

Test files the map has never seen count as work, never as absence: unknown means
"assume it runs".

The effect on a small change: `pytest tests/kernel` after a one-function edit is
**0.9 s**, against 2.4 s for the same command with eight workers and no
selection.

## Debugging selection itself

* **"Why did this test run?"** — `python tools/test_selection.py` lists the
  files that differ from the map above the tests they reach.
* **"Why did this test *not* run?"** — `tools/test_selection.py status` lists
  test files the map has never recorded and declarations that have gone stale.
  If the test is mapped and unaffected, the map is saying your change does not
  execute any line it covers; `--no-testmon` settles it either way.
* **"Nothing is mapped"** — the environment key of your shell differs from the
  one the suite writes. Check `QT_QPA_PLATFORM` and the `ARRAYSCOPE_*` ring
  variables, or pass `--environment`.
* **Suspect selection itself** — `ARRAYSCOPE_TESTMON=0 pytest ...` removes it
  from the picture without editing anything.
