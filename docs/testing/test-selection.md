# Change-driven test selection

`pytest` runs the tests your working tree affects, and skips the ones it
provably does not. This page is the reference: what that means, what it cannot
see, and which command to reach for.

Policy and mechanism live in [`tests/testmon_policy.py`](../../tests/testmon_policy.py)
and [`tests/conftest.py`](../../tests/conftest.py); the reporting tool is
[`tools/test_selection.py`](../../tools/test_selection.py). The rings this sits
inside are in [README.md](README.md) — selection changes *which tests a ring
runs*, never *which ring a claim needs*.

## The rules

1. **`pytest` is the loop.** No flags, no path to guess.
2. **`pytest --since` before merging** — and after a rebase, or in a checkout
   somebody else was iterating in. Still selected, still fast. It reports the
   whole set your branch reaches and runs only the part the map cannot already
   vouch for; `--fresh` re-runs the lot. See
   [`--since`](#since-a-verification-report-for-the-whole-branch).
3. **`pytest --rerun-reds` when the red you are fixing is one you inherited.**
   Selection does not re-run those, so even a node id reports it deselected.
   `--since` implies it: a merge gate that declined to run a red is not a gate,
   and the moment the branch lands an inherited red is main's red.

   A bare `--since` picks its baseline from `ARRAYSCOPE_BASELINE_REF`, then the
   upstream, then `main`, and **prints which one it found and how many commits
   back it is**. That matters on `main` itself, where the upstream resolves and
   may be far behind if nothing has been pushed — the run is then honest but
   enormous, and the printed distance is what says so.
4. **`--no-testmon` runs only a short list of node ids.** Everything else —
   a bare sweep, a directory, a file, or node ids costing over 30 s — is
   **refused** (exit 4), and the answer is `--testmon-noselect`, which runs
   the same tests *and updates the map*. That one is refused only when it
   would repair nothing. See [Why `--no-testmon` is
   refused](#why---no-testmon-is-refused).
5. **No `-p no:...` here.** Three spellings arrive from other repositories'
   habits and all three abort:
   - `-p no:randomly` — pytest-randomly is not installed, so the flag does
     nothing. It is worse than inert: it reads as "the ordering is pinned", so a
     flake it never had a chance of preventing gets blamed on something else.
   - `-p no:cacheprovider` — **crashes the run**, every time. testmon reads
     `config.option['lf']`, which only exists while the cache plugin is
     registered, so blocking it exits 3 with an INTERNALERROR that looks like
     this repository is broken. Here the cached state is the *feature*:
     `.testmondata` is what makes `pytest` narrow.
   - `-p no:pytest-testmon` — turns selection and recording off with nothing to
     say so.
6. **`-n 0` is for one test, not for the suite.** A wide serial run is refused;
   see [Why a wide `-n 0` is refused](#why-a-wide--n-0-is-refused).
7. **Suspect the map? Report it.** Do not route around it silently.

Rules 4, 5 and 6 are enforced by the run rather than by anyone having read this —
all abort, and `pytest --help` carries the per-flag detail.
Reasoning: [Why not just sweep?](#why-not-just-sweep) and [Why `--no-testmon` is
refused](#why---no-testmon-is-refused).

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
test selection: ran the 20 tests your tree reaches. The other 3520 of 3540 are
  unaffected by construction — their recorded fingerprints still hold, so running
  them could only re-confirm them (~222 s). This is the complete affected set,
  not a sample.
  Branch-sized question, before merging:  pytest --since
```

and when nothing is affected at all:

```
test selection: nothing to run — no line your working tree changed is executed
  by any of the 3540 recorded tests.
  That is a result, not a gap.
```

Read the closing line before writing "suite green" anywhere. A selected run that
passes says the affected tests pass — a strictly smaller claim, and a different
one from "the suite is green". What it is *not* is an unfinished job: the tests
that did not run could only have re-confirmed fingerprints that still hold.

| Goal | Command |
|---|---|
| Inner loop | `pytest` |
| Everything this whole branch changed — pre-merge, after a rebase, or in a borrowed checkout | `pytest --since` (it prints which baseline it picked, how far back, and what it verified without running) |
| The last five commits, or one named sha | `pytest --since HEAD~5`, `pytest --since <sha>` |
| Distrust the recorded greens — a flake hunt | `pytest --since --fresh` (re-runs everything the branch reaches) |
| Landing a worktree branch: give the main checkout the map that recorded it | automatic on merge (`.githooks/post-merge`), or `python tools/test_selection.py adopt-map` |
| Blast radius without running anything | `python tools/test_selection.py` |
| What the map does *not* cover, which reds are yours | `python tools/test_selection.py status` |
| The current reds are not yours (stale worktree, stacked branch) | `python tools/test_selection.py accept-reds` |
| Re-run the inherited reds too | `pytest --rerun-reds` (implied by `--since`) |
| Rebuild the map from scratch | `rm .testmondata && pytest --no-seed-map` (without the flag the next run **re-seeds from a sibling checkout**, handing you back the map you distrusted) |
| Re-record without deselecting | `pytest --testmon-noselect` |
| Run a whole file or directory regardless of selection | `pytest --testmon-noselect <paths>` |
| Run specific tests regardless of selection | `pytest <node ids>` — a `file.py::test` argument already defeats deselection, so adding `--testmon-noselect` is a no-op and the run says so |
| Which functions the suite executes | `python tools/test_selection.py coverage` |
| New code on this branch that no test runs | `python tools/test_selection.py coverage --since` |
| The coverage baseline is not yours | `python tools/test_selection.py accept-coverage` |
| Regenerate artifacts, or settle a map you suspect | `pytest --no-testmon-force-please-i-really-consider-it-needed` (a plain `--no-testmon` sweep is refused — see below; the flag has no short form, pytest disables abbreviation) |
| Artifacts, CI | already exhaustive — see below |

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

testmon traces Python lines executed **in the pytest process**. Four things are
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
4. **Python read as *input*.** The architecture guards —
   `tests/app/test_dead_code.py`, `test_architecture_guards.py`,
   `test_g5_architecture_guards.py`, `test_import_health.py` — `ast.parse` the
   tree rather than executing it, so testmon records no dependency on what they
   inspect. Measured, the whole recorded dependency set of the dead-code guard
   is `['tests/app/test_dead_code.py', 'tests/dead_code.py']`: editing anything
   in `arrayscope/` never selects it. This is under-selection, and it is a
   variant of item 2 the phrase "non-Python inputs" does not name.

   It has one useful consequence. The residual risk in module-block forgiveness
   is a test that *introspects* a module — `dir()`, `getattr()`, a `__all__`
   scan — and so notices a newly added name without executing it. In this tree
   the tests that would do that are exactly the ones selection already never
   runs for an `arrayscope/` edit, so the hole is bounded by a measurement
   rather than by an argument. The commit hook runs the dead-code guard on every
   commit regardless.

Item 3 has no local escape hatch at all: even a forced untraced sweep runs the
whole *offscreen* suite, and rings 3–4 are not in it either way, so nothing on
this page substitutes for running them — see [the ring rules](README.md).

## Where selection turns itself off

The policy (`testmon_policy.decide`) refuses in the runs that must be
exhaustive, so no stale map can quietly truncate them:

| Situation | Why |
|---|---|
| `--no-testmon` | explicit |
| `--cov` / `--cov-report` | a coverage number over a subset of the suite is a false number |
| `CI` is set | every CI job is a gate; see below for the one exception |
| pytest-testmon not installed | nothing to select with |
| pytest-testmon installed but not registered (`-p no:pytest-testmon`) | the plugin, not the package, is what would have done the work |

That last row is the invariant behind all of them: **a run may only print
`test selection: on` if it can prove it** — the plugin registered, it built its
configuration, and the map could be read. Everything else is a bypass and gets
priced like one. Every degradation path in this layer fails *open*, so the
absence of proof is reported rather than assumed away.

An explicit `--testmon`, `--testmon-noselect`, `--testmon-nocollect` or
`--testmon-forceselect` always wins and is passed through untouched.

### In CI

**Read this first: CI is not a gate here.** It has run seven times in the life
of this repository, most recently 2026-07-25 against a `main` that was already
247 commits ahead, and no branch protection or required check exists. The rest
of this section describes how the workflow behaves *when somebody runs it* — it
is not a backstop anything else on this page may lean on. The tracked
`.githooks/pre-commit` and the [cadence nudge](#the-cadence-nudge) are the ones
that actually fire.

Selection is off by default in CI because Actions sets `CI`, so no map — stale,
absent, or restored from another branch — can narrow a gate.

One job opts back in with an explicit `--testmon`: **Affected GUI tests (fast
signal)**, which caches `.testmondata` between runs and runs only what the diff
affects. That is safe there and nowhere else, because everything it runs is also
run exhaustively by the `coverage` job in the same workflow — an under-selecting
map costs a late signal, never a missed regression. It also sets
`--rerun-reds`, since a cached map recorded from a failing
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

## Why not just sweep?

It is the flag people reach for reflexively, and as a routine wide sweep it is
close to pure cost: ~222 s against a few seconds, and it records nothing, so the
next selected run still plans from the same map it started with.

**`pytest --since` is the pre-merge command.** It asks the question that actually
matters before merging — everything *this branch* changed against its baseline,
not merely what moved since the last run — and it still selects through the map,
so it stays fast. It is equally the right call after a rebase, or in a checkout
somebody else was iterating in, where "since the last run" is simply the wrong
window.

But a routine sweep is not the same thing as *never* sweeping:

- The map-erosion hole that once made this the only trustworthy run is closed
  (`protect_map_outside_the_scope`, below).
- To repair a map, `pytest --testmon-noselect` runs the same tests *and*
  re-records them, which `--no-testmon` does not.
- The [blind spots](#what-it-cannot-see) are real and only an exhaustive run
  covers them. That is what the cadence nudge below is for — occasionally, not
  every loop.

And if you do suspect the map, **report it**: a wrong map is worth fixing once
for everybody, and quietly sweeping past it is how it stays wrong.

### The cadence nudge

This page used to say CI swept every push, and hand the blind spots to it.
Measured on 2026-07-30: **CI has run seven times in the life of this
repository**, most recently 2026-07-25 with `main` 247 commits ahead of it.
There is no branch protection, no ruleset and no required check. So the blind
spots had no owner at all, and nothing said so.

They have a local one now. Every run that really did execute the whole suite —
`--testmon-noselect`, or any run where the policy turned selection off — records
its HEAD sha and the date beside the map. When that record is more than **21
days or 40 commits** old, a bare `pytest` prints one quiet line:

```
blind spots last swept exhaustively 27 days ago, 63 commits — pytest --testmon-noselect (~4 min) when convenient.
```

That is the whole mechanism: no gate, no refusal, no colour. A sweep is four
minutes and the reader is mid-loop, so the line has to be droppable or it
becomes noise and then becomes invisible. A checkout with no record yet starts
the clock on its first run rather than opening with a nag, and says "never been
swept exhaustively here" if it later ages out — a different claim from a stale
sweep, and worded as one.

CI, when it runs, is a bonus on top of this rather than the thing being relied
on.

### Why `--no-testmon` is refused

The runner exits 4 with a `UsageError` unless the run is **a list of specific
node ids costing under 30 s**:

```
pytest --no-testmon                                   # refused: sweeps the whole suite
pytest --no-testmon tests/ui                          # refused: names a directory
pytest --no-testmon tests/ui/test_thing.py            # refused: names a file
pytest --no-testmon tests/ui/test_thing.py::test_one  # runs (and still warns)
```

The shape is checked before the price, and that ordering is the fix for a real
hole: until 2026-07-30 the rule was price-only, so a *cheap* file ran — the
third line above was a lie for as long as this page has carried it.

The permission is written as a narrow allow rather than a list of bans because
there is exactly one question `--no-testmon` answers that nothing else can —
*is the tracer itself causing this?* — and that question is asked of named
tests and answered in seconds. Every wider shape is someone reaching past
selection, and pays twice: once in wall clock, and again because the run
records nothing, so the map still does not know how those tests did.
`--testmon-noselect` runs any of them, is equally immune to selection, and
**updates the map**. Measured on one file: 0.32 s against 1.83 s.

Refused rather than warned about because **the wrong shapes are fastest exactly
when they are wrong.** One narrowed sweep costs seconds and reads as a clean
pass, so nothing pushes back, and the habit it forms is later aimed at the
whole suite. It is also what a deselected test invites — the moment someone is
least inclined to read a yellow line. Added 2026-07-30 after that sequence
played out twice in two sessions, the second time burning ~6 minutes of sweeps
that recorded nothing, despite the docs and an after-the-fact runtime hint.

Every run that actually bypassed selection — a permitted one, a CI one, a
coverage one, and one forced through — closes with what it cost and a defect
ask:

```
Bypassed selection (explicitly forced) and ran 214 s.
!!! IF THIS WAS NEEDED, REPORT AND FIX THE TEST HARNESS !!!
```

The override is **`--no-testmon-force-please-i-really-consider-it-needed`**,
in full: pytest disables argparse's prefix abbreviation, so any shorter
spelling exits 4 with `unrecognized arguments`. Force is what regenerating
`tests/artifacts/` or settling a distrusted map uses.

### Why keep `--no-testmon` at all, then?

Because they are not the same tool, and the difference cuts both ways.

`--testmon-noselect` runs every test **and re-records**, so it leaves the map
repaired — which is why it, not `--no-testmon`, is the answer when the map is the
thing you are fixing. It is the better command whenever "run it all" and "leave
the map right" are both wanted.

What it cannot do is take testmon out of the picture. Two cases need that:

- **Suspecting the tracer itself.** `--testmon-noselect` still traces every line.
  In a suite this timing-sensitive — Qt event loops, real GPU contexts, 2 s/5 s
  interaction budgets — "is the instrumentation causing this?" is a legitimate
  question, and `--no-testmon` is the only way to ask it.
- **Recording nothing on purpose.** CI throws its map away, so tracing there
  would be ~3.5% spent on a file nobody reads. CI gets this via the `CI`
  variable rather than the flag, but it is the same choice.

There is also a cost on the other side, which is why `--testmon-noselect` is not
simply promoted to "the sweep": it reorders, running the deselected group last.
Declared `coupled_order` groups are reassembled afterwards
(`testmon_policy.py`), but *undeclared* order coupling is not, and this suite has
enough of it that a reshuffle has measured about one spurious failure per run. A
sweep whose job is to be trusted is the wrong place to spend that.

So: `--since` before merging, `--testmon-noselect` to repair the map,
`--no-testmon` to get the tracer out of the way. If the reordering flakiness ever
gets driven to zero — the latent couplings declared with `coupled_order`, or
fixed — then `--testmon-noselect` really would dominate `--no-testmon` locally,
and this section should be revisited.

### When `--testmon-noselect` is refused

It is priced on whether it has a **product**, not on what it costs. That is the
difference between the two flags in one sentence: `--no-testmon` leaves nothing
behind, so its cost is the whole story; `--testmon-noselect` leaves a repaired
map, so its cost is an investment — and an investment is refusable exactly when
there is nothing to repair.

| The map | Verdict |
|---|---|
| holds nothing for this environment | **runs** — nothing to compare against, so let it record |
| has unmapped test files, or the tree differs from it | **runs** — this is what the command is for (a big rebase, a >400-file `--since`, a refactored directory) |
| complete and current | **refused** — ~222 s to write a byte-identical map, plus about one spurious failure from the reorder |

That last refusal is unusual in being *itself the answer*: if nothing in the
tree differs from what was recorded, there is nothing a sweep could find that
the map does not already know. If you distrust the map rather than its coverage,
`rm .testmondata && pytest --no-seed-map` is the command that rebuilds it.

Until 2026-07-30 this was priced like `--no-testmon`, so a bare
`--testmon-noselect` was refused at 1320 s — while three separate places (this
page, `tools/test_selection.py`'s epilog, and the `--since` >400-file error)
named it as the sanctioned repair. The documented remedy exited 4.

## The map

`.testmondata`, at the repo root, gitignored, one per checkout — a worktree
gets its own. It is a SQLite file of about 2.5 MB for this suite.

* **It maintains itself.** Every selected run re-records the tests it ran, and
  the tests it skipped keep fingerprints that are still valid by construction.
  There is no refresh chore. A first run on an empty map runs everything, in
  order to record everything.
* **A scoped run no longer shrinks it** — see
  [What a scoped run used to delete](#what-a-scoped-run-used-to-delete). That was
  a real hole, and a bad one.
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

### Adding a function does not re-run the file's world

testmon puts a module's **whole top-level body in one AST block spanning line 1
to EOF**, so that block is in 100% of the file's fingerprints, and its
comparison is a subset test against the file's current checksum list. One
top-level statement moving therefore retired one 4-byte checksum and with it
every test that had ever touched the file. Measured before the fix: appending
one inert, uncalled function to `arrayscope/window/frame_session.py` selected
**258 of 3933 tests and cost 123.16 s of wall clock**, and the number of tests
whose executed code had changed was zero.

Selection now *forgives* a retired checksum when it can prove the code is the
same. Two rules, both applied per file against the revisions the sidecar
retains:

* **The module block**, when the current revision is the previous one with
  statements *inserted* — nothing removed, modified or reordered — and every
  inserted statement is inert (below).
* **Any block whose index-free body digest still exists.** `Block.code` is
  `f"{index}:{ast_repr}"`, so moving a function down a file changes the
  checksums of everything below it while the code is byte-identical. Identical
  digest, identical AST; only the ordinal moved, and nothing can observe an
  ordinal.

Measured on a settled map of 3987 tests, working tree clean at 0 affected:

| Edit | Before | After |
|---|---:|---:|
| append a function to `arrayscope/render/lod.py` | 164 tests / 324 s recorded | **0** |
| append a function to `arrayscope/window/frame_session.py` | 266 / 372 s | **0** |
| insert a function above the first `def` in `lod.py` | 164 / 324 s | **0** |
| add a method to `ViewState` (`core/view_state.py`) | 144 / 146 s | **0** |
| append one test to `tests/display/test_odd_origin_crop_geometry.py` | 297 | **0** |
| change a constant in `lod.py` | 164 | 164 |
| add an unused import to `lod.py` | 164 | 164 |
| append a *decorated* function to `lod.py` | 164 | 164 |

**What is inert**, and nothing else is:

* an **undecorated** `def` or `async def` whose name was not already bound in
  that scope, and whose default arguments are literals, plain names or
  attributes, or containers of those. Decorators run at import and mutate shared
  registries — this tree has them, in `arrayscope/operations/library.py`.
  Defaults evaluate at definition time; annotations do not (PEP 649).
* a **class** with no decorators, no bases and no keywords, whose body is only
  methods, docstrings, `pass` and literal assignments. A base class gets
  `__init_subclass__` called on it, and `metaclass=` is a call.
* a method added to an **existing** class only if that class is undecorated or
  `@dataclass` — a class decorator reads the finished body. (Census: 415
  decorated classes here, 411 `@dataclass`, 4 `@runtime_checkable`; adding a
  method to a runtime-checkable Protocol changes what `isinstance` answers.)

**What is never forgiven**, beyond the above: a name starting with `__`
(a `def __repr__` replaces something with no block to compare, and a module-level
`def __getattr__` changes every attribute lookup on the module); any file whose
source mentions `__all__`; any file declared in `OUT_OF_PROCESS_DEPENDENCIES`,
where the whole file is the honest dependency because nothing about the child
process is traced; and any file with no retained previous revision, which is a
fresh sidecar or more than five revisions of drift.

The direction of the risk is fixed by construction. Forgiveness only appends to
the right-hand side of testmon's subset test, so a mistake can turn "affected"
into "unaffected" and never the reverse — it can never make more tests run, and
it changes nothing else testmon does. `tests/app/test_selection_forgiveness.py`
drives testmon's own comparison against testmon's own fingerprints, so every
case carries the control arm that says what selection did without it.

The previous revisions live in `.testmondata-coverage.json`, the sidecar the
coverage figure already writes, extended to track every file the map depends on
(572 here, against the 295 under `arrayscope/`) and to keep the last five
revisions of each. It costs 0.79 MB per generation of drift on a 1.6 MB file,
about 1 s of parsing on a cold sidecar, and ~25 ms per changed file when
selection reads it.

### What a scoped run used to delete

testmon garbage-collects the map on every run: `TestmonData.sync_db_fs_tests`
deletes each recorded test the run neither collected nor called unaffected. That
is correct for a whole-suite run, where "not collected" does mean "gone". For a
scoped one it was a silent, permanent hole, and it took out exactly the test you
most needed:

```
edit arrayscope/core/roi.py     ->  affects 14 tests, none of them in tests/kernel
pytest tests/kernel             ->  all 14 are affected, none were collected,
                                    all 14 are deleted from the map
pytest                          ->  "affected: nothing". 1 test runs. The other
                                    13 never run again.
```

The last step is the damage, and it compounds: with no map entry nothing marks
the test affected, and testmon's file-level `pytest_ignore_collect` skips its
whole file because every *remaining* test in it is unaffected — so collection
never rediscovers it and nothing re-adds it. The edit was then untested and
reported as fully tested by `pytest`, by `pytest --since`, and by
`tools/test_selection.py status` (which counts unmapped *files*, and the file was
still mapped). Only `pytest --no-testmon` caught it. Repairing the map needed a
full `--testmon-noselect` run.

**A node id was the worst case, and it is the everyday debugging move.** `pytest
tests/kernel/test_kernel.py::test_x` collects one file, so a single such run
deleted **49** entries — the edit's entire affected set — after which `pytest`
ran nothing at all on a modified tree.

`testmon_policy.protect_map_outside_the_scope` narrows the deletion to the scope
the run actually looked at: every mapped test whose file lies outside the run's
arguments is retained regardless. Cleanup *inside* the scope still happens, which
is the part with a legitimate job (a renamed or deleted test), and it is still
exact there, because that scope was collected. A run that covers every
`testpaths` root is left alone entirely — note a bare `pytest` carries `tests` as
an argument, so "has path arguments" is not the same question as "was scoped"
(`collected_the_whole_suite`).

Measured, matched A/B from an identical map, same 14-test affected set:

| | map entries deleted | tests the next `pytest` runs |
|---|---|---|
| before | 14 of 14 | 1 |
| after | 0 | 24, including all 13 that had been lost |

This is also why the coverage figure below refuses to report on a scoped run, and
why it publishes the number of tests the map holds: **this is what had eroded
`main`'s map to 2680 of 3941 tests**, and a coverage number over two thirds of the
suite would have read as a 15-point regression that did not exist.

### The erosion canary

That defect was fixed, and the fix is one `except AttributeError: return 0` away
from silently coming back — a testmon upgrade that renames `sync_db_fs_tests` or
`stable_files` would degrade `protect_map_outside_the_scope` to a no-op with
nothing to show for it. Every degradation path in this layer fails *open*, and
the map does not complain when it shrinks.

So the sidecar keeps a **high-water mark** of the map's recorded test count, per
environment, and a run that comes in more than 5% under it says so:

```
test selection: the map lost 1261 of 3941 recorded tests since this checkout's
high-water mark. That is the erosion signature. Repair: pytest --testmon-noselect
```

The mark only rises, with one exception: a whole-suite `--testmon-noselect` run
re-records everything with no scope to have missed anything, so its count *is*
the truth and is allowed to lower it. That is what keeps an honest deletion from
warning forever.

Two guards sit under it. `tests/app/test_test_selection.py` asserts that the
testmon attributes this policy rewrites still exist — including
`TestmonSelect.deselected_files`, `TestmonData.sync_db_fs_tests` and
`stable_files` — and that the set of option keys testmon *subscripts* is exactly
what we expect. That second one is the tripwire that would have documented the
`-p no:cacheprovider` crash before anybody found it the hard way: `lf` is
registered by pytest's cache plugin, and testmon indexes it rather than
`.get`-ing it.

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

Pass `--no-seed-map` to opt out and record from scratch.

## Tests that were already red, and tests you just broke

testmon's own rule is that a test which failed last time re-runs unconditionally,
whatever changed. Doing that here measured **124 s on an otherwise clean tree** —
an entire inner loop spent re-confirming what everybody already knew. But
skipping every unchanged red has a hole big enough to drive the feature into:

```
edit foo.py  ->  test T runs, fails, and the map records T as failed
edit bar.py  ->  T's dependencies are unchanged *since the map was written*,
                 so T is skipped — two minutes after you broke it
```

The map cannot close that hole, because the map is rewritten by every run,
including the run that introduced the red. So the two cases are told apart by
observation rather than inference. Every time a test *transitions* into failing —
from passing, or from not existing — [`tests/red_ledger.py`](../../tests/red_ledger.py)
writes it down. A test in that ledger is never skipped and is named at the end of
every run until it passes again:

```
BROKEN HERE (1): these were passing in this checkout and are not; they run on
every selected run until they pass again.
    tests/render/test_effects.py::test_pyqtgraph_shared_fft_preview_retains_...
```

That makes the baseline exactly right: whatever was already failing when this
checkout's map arrived (usually seeded from `main`) is inherited and stays quiet;
everything that broke afterwards is yours and stays loud. Nothing to bootstrap —
an absent ledger means "nothing has broken here yet", which is the correct
reading of a checkout that has not run anything.

Measured end to end: break a function in `render/lod.py`, then edit
`kernel/scheduler.py` — an unrelated file. The run is **2 tests in 1.65 s**: the
one the unrelated edit affected, plus the one you broke, with the 14 inherited
reds still skipped. Selection stays optimal; it just stops lying.

### When the reds are not yours

The ledger infers ownership from transitions observed *here*, which is right
while you work and wrong after the ground moves underneath it — a worktree that
jumps fifty commits, a branch stacked on another branch's reds, a feature branch
split in two. The failures are real but not yours, and left alone they re-run
forever. One command says so:

```bash
python tools/test_selection.py accept-reds
```

They become ordinary inherited reds. Anything that breaks from then on is yours
again.

To go the other way and re-run the inherited ones too — the right choice while
you are actually fixing one, since the fix can land outside the truncated
dependency set of the run that failed:

```bash
pytest --rerun-reds
```

## `--since`: a verification report for the whole branch

The map answers *what changed since the last run*. That is the right question in
an edit loop and the wrong one before merging: by then every file has been
recorded, so the map reports nothing affected while the branch has changed twenty
files. Measured on this branch — plain `pytest` reaches **1 test**; `--since`
reaches **65**.

```bash
pytest --since                  # resolve the baseline (below)
pytest --since origin/release   # or name it
pytest --since HEAD~5           # any ref works: a tag, a sha, a range endpoint
pytest --since 7d08135a
```

Merge-base semantics (`ref...HEAD`), plus the working tree and untracked files,
so it answers "what has this branch done" rather than "how does it differ from
wherever `ref` has got to since". Any ref git can resolve is a valid argument,
including a raw sha or `HEAD~N` — the merge base of `HEAD~5` and `HEAD` is
`HEAD~5`, so that spelling means "the last five commits plus my working tree".

### What it runs, and what it reports instead of running

The map is an **evidence ledger**. A test's fingerprint is content-addressed:
if it is stable, the last recorded run of that test executed byte-identical code
to what is on disk now. So a stable test whose last outcome was green is
*verified for this tree* — running it again cannot produce information, only a
copy of the record.

`--since` therefore partitions the reached set rather than re-running it:

```
--since main: branch reaches 65 tests — 60 verified green against this exact tree
from recorded runs, 5 run now (1 broken here, 3 without evidence, 1 known flake).
```

| Bucket | Meaning |
|---|---|
| **verified** | stable, recorded green, not in the flake ledger — reported, not run |
| **broken here** | the red ledger owns it; the map cannot vouch for a red it recorded |
| **without evidence** | unstable (a bare `pytest` would run it anyway), or never recorded — every new test lands here |
| **inherited red** | stable, recorded, last outcome red. A merge gate owns those: the moment the branch lands, an inherited red is `main`'s red |
| **known flake** | stable and green, but observed to flip outcome with no code change (below) |

This is the part worth being precise about: **a verified green is evidence
acquired earlier, not a test that just passed.** It is exactly as good as the
claim that the fingerprint describes everything the outcome depends on — which
is true for ordinary code and false for a flake, which is why the flake ledger
exists. It was measured at ~95% of the reached set on a faithfully-iterated
branch, i.e. `--since` used to spend almost all of its time re-confirming
records it already held.

Any failure to compute the partition — an unreadable ledger, a testmon upgrade,
a map that will not answer — falls back to running everything reached. Never to
skipping more.

### `--fresh`: re-acquire all evidence

```bash
pytest --since --fresh
```

Runs every test the branch reaches, verified or not — the behaviour `--since`
had before the partition. It is the right call for a flake hunt, or for a map
you distrust for a reason you can name (in which case also
[report it](#the-map)). On its own, without `--since`, it is an error: there is
no reached set for it to be fresh about, and a flag that silently does nothing
is how a phantom sweep gets believed.

### The flake ledger

One case defeats the evidence model: a test whose outcome does not depend only
on the code its fingerprint describes. Those are observable, and every run used
to throw the observation away.

A red that re-ran **only because it was red** — its dependencies unchanged, so
it was in the forced set rather than the affected set — and then *passed* has
flipped its outcome with no code change. That is a flake, witnessed rather than
guessed, and noticing it is free: the run had already paid to execute the test.
It is written to a ledger beside the map
([`tests/red_ledger.py`](../../tests/red_ledger.py)) and listed by
`python tools/test_selection.py status`.

A marked test is never verified-skipped; `--since` always runs it. The mark
clears on exactly one event — its dependencies actually changed, so it
re-recorded. New code is new evidence, and the old flakiness was never about it.
No score, no decay, no threshold: one observation is enough to know the
fingerprint does not determine the outcome.

**The baseline is resolved, not assumed**, and the run prints which ref it used.
In order: `ARRAYSCOPE_BASELINE_REF`, then the branch's upstream, then `main`,
then `origin/main`. The upstream comes first so a branch stacked on another
branch measures against its parent rather than inheriting the parent's whole
diff; a branch with no inferable parent — split in two, cut from somewhere
unusual — should be told, either per run or once via the environment variable. An
unresolvable baseline is an error, never a silent fallback to the wrong branch.

Comparison is **method-level**, using testmon's own fingerprints against the
branch point, not "any test that recorded a changed file". On a four-file diff
including `tests/conftest.py` that is 41 tests against 55, and the gap grows with
how widely the touched code is executed. Non-Python changes — a fixture array, an
icon — have no method structure, so they fall back to reaching every test that
recorded them.

That figure used to be the *best* case rather than the typical one: the same
module block that made an added function re-run a whole file made `--since`
degrade to file-level for every file the branch added one to. The same two rules
apply here, mirrored — the fingerprints are current and the checksum list is the
branch point's, so it is the current checksums that get appended — and no
sidecar is needed, because `git show` has already produced the baseline
revision.

Measured by translating each recorded fingerprint forward through the edit the
way the re-record would write it (a disk edit alone cannot show it: `--since`
compares *recorded* fingerprints, and the recording only moves when the tests
re-run): appending an inert function to `render/lod.py` goes **165 reached →
0**, adding a method to `ViewState` **144 → 0**, appending one test to a
297-test file **297 → 0**. A changed constant in `lod.py` stays at 165, and so
do a changed body and a new import.

`--since` is the pre-merge command precisely because it is still a *selected*
run: the branch-sized answer at map speed. It does not pretend to be a sweep —
it still cannot see [the blind spots above](#what-it-cannot-see), which is what
the occasional `--testmon-noselect` is for (see [the cadence
nudge](#the-cadence-nudge)), and rings 3–4 for real pixels.

## Coverage, for free

The map records the AST blocks each test executed. Unioned across every recorded
test, that answers a question nobody was asking it — **which functions in
`arrayscope/` does the suite execute?** — for the price of one SQL query. No
coverage pass, no wall clock.

```bash
python tools/test_selection.py coverage
```

```
functions entered by a test   5306 of 6204  (85.5%)
files a test reaches at all   281 of 295
```

The same figure appears at the end of a run, **but only when it moved**:

```
coverage: 5311 of 6205 functions executed (85.6%), +3 since this checkout's
baseline (3 newly covered, 0 no longer).
```

It is legitimate on a *selected* run, which is the point: a deselected test keeps
its recorded fingerprint, and that fingerprint is still valid by construction —
the same invariant selection itself rests on. It self-heals for the same reason a
selected run is honest: when a file changes, every test that recorded it is
affected, so it re-runs and re-records in the same breath, and stale checksums
stop matching and drop out of the numerator.

### It is not coverage.py's number

This measures **functions entered**, and it must never be quoted against CI's
Codecov line percentage. Measured on identical execution data (`tests/core` plus
`tests/kernel`, coverage.py's own `executed_lines` folded through testmon's own
`create_fingerprint`):

| | |
|---|---|
| coverage.py lines | 14.8% |
| testmon blocks | 13.1% |
| per-file delta | median +0.0, **spread −36.1 to +17.4 points** |

The aggregate agreement is two large biases cancelling, not accuracy. Per file
they diverge violently, in both directions, for one structural reason each:

* **Upward** — a block counts as covered when *any* line in it ran, so a function
  abandoned at its first guard reads exactly like one run to completion.
* **Downward** — a module's top-level statements (imports, class bodies, every
  `def`, every constant) are a *single* block spanning the whole file, so merely
  importing a module makes coverage.py call most of its statements covered.

The second is severe enough that the module block is excluded from the metric
entirely: it spans line one to EOF, so "covered" there only means "some test
touched this file". That is reported separately, as reach.

### How close is it to the truth?

Validated against a full `pytest --cov` run on the same tree, with a completely
recorded map:

| | functions |
|---|---|
| ground truth (full `--cov` run) | 5271 of 6204 — 84.96% |
| the map, for free | 5312 of 6204 — **85.62%** |

**0.66 points apart**, from 99 the map knew about and the coverage run skipped
(tests conditionally skipped that day, recorded earlier) and 58 the coverage run
saw and the map cannot (mostly child processes, blind spot 1 above).

Cost: 2.1 s cold, **47 ms warm**. Block structure is cached beside the map keyed
by each file's git blob sha, so a warm run parses nothing and the cost after the
first is proportional to the diff.

### It is a floor, not an estimate

A test the map has never recorded contributes nothing, so the figure is a floor
whose tightness is exactly the share of the suite the map holds — which is why
`coverage` prints that count. With `main`'s eroded 2680-of-3941 map, the same
tree measured 70.2% against the true 85.0%: **a 15-point phantom regression**.
That erosion is [fixed](#what-a-scoped-run-used-to-delete); the reporting still
refuses on a scoped run rather than trusting it.

### What it compares against

Three bases, in preference order, and never an invented one:

1. **`--since`, before merging.** Not a percentage — for a file this branch
   edited, the map holds only the current revision's fingerprints, so the
   baseline revision's coverage is *not recoverable* and a "percent since main"
   would read every edited file as newly uncovered. What is exact is the useful
   half:

   ```
   coverage: of 1 functions this branch adds or changes since main, 1 are
   executed by no test:
       arrayscope/core/roi.py:462  _never_called_by_any_test
   ```

   Exact because a function this branch touched necessarily re-recorded in the
   run that just happened. Comparison is on an index-free body digest, so
   inserting a function at the top of a file does not report every function below
   it as changed — which testmon's own checksum would, and which is why it
   over-selects there.

2. **The inner loop: what this checkout inherited.** Observed rather than
   inferred, exactly like [the red ledger](#tests-that-were-already-red-and-tests-you-just-broke)
   and for the same reason. A worktree seeds its coverage baseline from the same
   donor as its map, so "since this checkout's baseline" means "since `main`"
   with nothing to bootstrap. `accept-coverage` re-baselines when the ground has
   moved under it.

3. **Nothing.** A fresh clone gets the absolute figure and no delta.

State lives in `.testmondata-coverage.json` beside the map — gitignored, one per
checkout, holding the parse cache and the baseline. Losing it costs one re-parse
and one baseline, never a wrong number.

`--cov` still turns selection off, and that rule is unchanged: coverage.py over a
subset of the suite *is* a false number. This is a different measurement, over
the union of everything the map has recorded.

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

An **explicit** `-n N` used to walk straight past the half-cores cap, because
that cap lives in `pytest_xdist_auto_num_workers` and only `-n auto` consults
it. It is now clamped, and the run says so:

```
worker sizing: -n 16 capped to 8 (half the logical cores). Concurrent GL context
creation past that intermittently segfaults the driver and starves the
timing-sensitive UI tests.
```

### Why a wide `-n 0` is refused

`-n 0` is right for one test and wrong for the suite, and nothing used to say
so. What it costs depends entirely on what it is paired with: bare on a clean
tree it is 0.37 s; after an edit it is the selected set run honestly, ~8×
slower; paired with any bypass it is the whole suite in one process.

Three things make that last shape worse than merely slow:

1. **There is no per-test timeout.** `pytest-timeout` is not installed; the
   settlement watchdogs are deadlock guards inside individual helpers. A test
   that never settles hangs the process indefinitely at 100% CPU. Under `-n auto`
   that costs one worker and the other seven keep printing, so it reads as
   "slow"; serially the terminal simply stops. `faulthandler_timeout = 300` now
   turns that into a dumped traceback naming the test — a floor, not a fix.
2. **A segfault takes the run, not a worker.** xdist reports a crashed node and
   carries on; `-n 0` has nothing to carry on with.
3. **Serial is a different regime with its own reds** — ~25 of them standing. A
   wide `-n 0` run is not a de-noised version of the parallel run; it is a
   second, less-tested one, and most of what it produces is not yours.

So an explicitly typed `-n 0` is priced, against **what the run will execute**:
node ids sum to their own recorded seconds, a still-selecting run is priced at
its affected set within scope, and only a run that has stopped selecting is
priced at the whole scope. `pytest -n 0 tests/ui` after a one-line edit runs
three tests, and refusing it because `tests/ui` *could* cost 1164 s would be the
"unknown is large" mistake pointed the other way.

The threshold is **60 s**, and it is calibrated rather than chosen — every
documented serial command sits well below it and every harmful shape well above,
with nothing in the 3.8× gap between:

| Serial by design | recorded | | Refused shapes | recorded |
|---|---|---|---|---|
| ring 0 fast loop (`tests/kernel tests/render tests/presentation`) | 5.6 s | | `tests/app` | 105.9 s |
| ring 1 presentation contract (3 files) | 16.4 s | | `tests/ui` | 1164.8 s |
| ring 2 artifacts (`tests/ui/test_qt_smoke_artifacts.py`) | 27.5 s | | whole suite | 1320.0 s |
| CI's serial step | 29.7 s | | | |

Never refused, at any price:

* `CI` is set — its serial steps are the artifact gate;
* the run is **not** the offscreen regime (`ARRAYSCOPE_GPU_TESTS`,
  `ARRAYSCOPE_STRESS`, `ARRAYSCOPE_STRICT_UI`, or a non-offscreen
  `QT_QPA_PLATFORM`). Rings 3–4 *require* serial and their price is unknowable
  from a map that has never held their environment key;
* every path argument is under `tests/gpu_interaction`, `tests/stress`, or
  `tests/ui/test_qt_smoke_artifacts.py`;
* `--pdb` or `--trace` — pdb genuinely cannot work under xdist. Deliberately
  **not** `-s`: "I want to see prints" is a reflex, not a reason to spend ten
  minutes;
* `--collect-only`, which executes nothing;
* `--serial-force-please-i-really-consider-it-needed`.

Note the direction on unknown cost is the **opposite** of the bypass gate's.
Rings 3 and 4 have zero recorded seconds by construction, so "unknown is large"
would refuse the two rings [ground rule 1](README.md) depends on. Scope is the
backstop instead: an unpriceable serial run is refused only when it names the
whole suite, which is the fresh-clone `pytest -n 0`.

The override is separate from the bypass one on purpose. "I need the tracer out
of the way" and "I need this in one process" are unrelated claims, and a single
flag granting both would hand each exemption to somebody who only meant the
other.

One stale scare is worth retiring here: `-n 0` no longer reads the developer's
real `~/.config`. `tests/conftest.py` points `XDG_CONFIG_HOME` at a private
directory for serial runs too, so the QSettings half of the old `-n 0` trap is
closed.

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
* **Reverting an edit does not restore the map** once a run happened in
  between. The map is re-recorded by the run, so the revert is just another
  change: mutate → `pytest` (123.16 s, 296 tests) → revert → `pytest` (73.66 s,
  262 tests), measured. That doubles the price of exactly the experiment a
  suspicious developer runs, so do the revert *before* the confirming run, not
  after it. Without a run in between the fingerprints match again and nothing
  re-runs.
* **Suspect selection itself** — report it (rule 6); there is deliberately no
  environment variable that turns it off
  from the picture without editing anything.
