# Redesign — course and queue

**Date:** 2026-07-14. **Branch:** `redesign`.
**This file is the single source of truth for what happens next.**
`docs/roadmap.md` defers to it. Everything that used to live in this
directory is history in [archive/](archive/) — read it for evidence, never
for direction.

## Where we are

The architecture rewrite (R1–R7) landed: one kernel, one pipeline, one
lifecycle machine, `frame_renderer.py` deleted. That part worked and stays.

The certification program that followed (R8A–R8D) is **closed**. Not because
its gates passed — because the gates stopped measuring the product. It
produced real fixes, but its later fixes narrowed internal predicates
(coverage ring → onscreen → strict intersection → physical targets) until
the counters said "converged" while the screen showed black tiles and
wrong-order rendering. The post-mortem is in
[retro-2026-07.md](retro-2026-07.md); the R8 logs are in the archive.

## What is broken right now (user-visible)

1. **Persistent black tiles** — tiles that stay black indefinitely while the
   system reports completion. Root-cause dossier:
   [black-tiles-and-priority.md](black-tiles-and-priority.md) §B.
2. **Priority rendering order** — tiles do not fill center-out / visible-first.
   Dossier §A. (The "fix" in e6665315 changed one of three drifted rankers
   and was verified only by an isolated sort-order unit test.)
3. **Silently dead code paths** — `window/montage_prefetch.py` still imports
   the deleted `frame_renderer` module inside `except Exception`, so
   interaction-awareness and the retained-preview admission path have been
   off since R7, with no test failure. Dossier §C.

## Performance bars (restored from R2/R4/R8D — commitments, not history)

Closing R8 closed its *bureaucracy*, not its goals. These bars are the
product promise and stay binding until met, verified by the harness on
real hardware:

- GUI callbacks < **50 ms**, always; event-loop heartbeat gap ≤ **16 ms**
  during fills and scrub; warm scrub input ≤ **15 ms**.
- Settled-idle CPU **0%**.
- **#1 throughput target:** fast montage index scroll on FFT data is
  ~**4 fps** today vs ~17 fps scalar (2026-07-09 measurement, realistic
  human scroll rate). Bring FFT scroll toward the scalar rate.
- Once a baseline is frozen (T1), benchmark deltas stay within ±10% unless
  a P-step improves them.

## The queue (execute in order)

| Step | What | Done when |
|---|---|---|
| V0 | Repair the dead imports in `montage_prefetch.py`; add an import-health guard (import every `arrayscope` module; forbid `except`-swallowed imports) | Guard test exists and fails on a re-broken import |
| T1 | **Measurement foundation.** Port the marathon benchmark harness ([marathon-salvage.md](marathon-salvage.md) Tier 1: profile-tool certification phases + session fixture + PanelSession + `gui_gc` + instrumentation) and land the trace-event spine ([tracing-pipeline.md](tracing-pipeline.md) T1) | Certification report + trace file produced on this branch, both backends; frozen baseline recorded |
| V1 | **Black tiles.** One owner for "which tiles must render": admission, completion, and evidence scoping all read the same set (fix dossier B1+B2, including the level-evidence deadlock). Delete the tests that pin the narrowed predicates | Harness scenario: one-index scroll with a boundary-landing tile settles fully on real Wayland, both backends; no black tile, no parked evidence pass; `trace_verify` clean |
| V2 | **Priority order.** One ranker. Per-tile viewport distance becomes part of kernel ordering (or per-tile rung interleaving); delete the other two rankers (fix dossier A1–A4) | Harness scenario: cold montage load + fast scroll paints center-out, proven from the recorded commit/ack trace, not a unit sort |
| V3 | **Loud non-convergence.** Any tile unsettled with no work in flight for >2 s emits the `stall` trace event with the owner-chain snapshot and a visible diagnostic (tracing-pipeline T2; this failure class has recurred ~5×) | Injecting a stranded tile produces the diagnostic + ring-buffer dump |
| V4 | Merge `redesign` → `main`. The branch is 116 commits adrift; every week unmerged is risk | `main` runs the fixed viewer; roadmap (X5…) resumes |
| P1… | **Performance program to the bars above**, one measured cause at a time against the frozen T1 baseline, drawing from [marathon-salvage.md](marathon-salvage.md) Tiers 2–3 (order given there: prefetch-busy, level_source, viewport intent, histogram aggregation, coalesced drain, cadence throttle, stage-cache snapshot, governor policy, admission batching, gate pacing, slot relocation) | Each P-step: one cause, before/after trace + benchmark numbers in the commit; bars trend green |

A step is done only when its harness scenario passes **on a real display**.
Counters, lifecycle diagnostics, and unit tests are debugging aids, not
acceptance.

## The visible-truth harness (the only gate)

One scripted scenario runner on real Wayland, assembled from pieces that
exist: the **marathon profile-tool certification harness** (T1 port — ~30
named gates, event-loop probe, presentation-continuity probe, portable
session fixture), `tests/gpu_interaction/` (real-display pixel
assertions), `tools/probes/`, the per-tile truth overlay
(`display/tile_truth_overlay.py`), and the **trace pipeline**
([tracing-pipeline.md](tracing-pipeline.md)): every harness run records a
trace; `trace_verify` proves invariants, `trace_latency` attributes every
millisecond. Each scenario drives the real app, waits on settlement, and
asserts **pixels / per-tile content / trace invariants**, not internal
counters. V1 and V2 each add their scenario before their fix.

## Ground rules (all seven — the old eleven collapsed into these)

1. **Pixels are the gate.** A claim of "fixed" requires the harness scenario
   green on real hardware. An offscreen or unit pass is never acceptance.
2. **Fix by deleting a duplicate, not by adding a gate.** If a fix adds a
   predicate, a generation, or a new set, it is probably wrong. One owner per
   decision: one ranker, one visible set, one completion predicate.
3. **GUI thread never hangs.** Synchronous GUI-thread step >50 ms is a bug;
   pan/scrub heartbeat target ~16 ms.
4. **Tests pin user-visible behavior.** A test that pins an implementation
   detail (upload counts, internal predicate scoping) may be deleted when it
   blocks a user-visible fix — say so in the commit message.
5. **No silent fallbacks.** `except Exception` around an import or a lookup
   that turns a missing symbol into a default is forbidden; broken must be
   loud.
6. **Bounded sessions.** End every working session with the app visibly
   better or the change reverted. Update this README's queue and nothing
   else; new process documents require Thomas's explicit ask.
7. **Backends: VisPy is the certification bar on Linux.** PyQtGraph keeps
   truth tests (right pixels) but no longer blocks on performance/upload
   gates. (Proposed course change — Thomas to confirm; revert to
   both-first-class by deleting this rule. [Thomas 2026-07-14: both are first-class, but PyQtGraph is ment for GPU headless/remote work. Lets give it 2x the perf allowance of VisPy!]).

## Environment & commands

- Python: `~/miniconda3/envs/arrayscope/bin/python` (host conda env; the
  Cowork sandbox cannot load PyQt6 — GUI/GPU work runs on the host).
- **Full suite** (~35 s):
  `~/miniconda3/envs/arrayscope/bin/python -m pytest tests -q -n 16 --ignore=tests/gpu_interaction`
- **Fast Qt-free loop:** `… -m pytest tests/kernel tests/render tests/presentation -q -n 0`
- **GPU harness:** `ARRAYSCOPE_GPU_TESTS=1 XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-0 QT_QPA_PLATFORM=wayland … -m pytest tests/gpu_interaction -n 0`
- **Workflow benchmark:** `… -m arrayscope.tools.profile_montage_workflow --backend {vispy|pyqtgraph} --montage-quality-policy resident`
  (`native-only` does NOT exercise the LOD ladder).
- Known parallel-only flakes (pass alone): `test_selecting_fft_workers_updates_settings`,
  `test_compute_policy_configures_stage_and_montage_lanes`, teardown of
  `test_montage_ready_display_payloads_commit_immediately`.

## Debugging gotchas (carried over, still true)

- `print()` in app code is swallowed under pytest/Qt — append to a /tmp file.
  py-spy can't keep up with 3.14 — use cProfile or JSONL diagnostics.
- JSONL wedge evidence lives in the STATIC TAIL of the file.
- `pkill -f "pytest tests"` in a DC shell kills the shell itself.
- VisPy offscreen `canvas.render()` needs int-rounded `physical_size`;
  `QTimer.singleShot` needs the 3-arg receiver form.
- Real rendering/visual/Wayland claims must never use
  `QT_QPA_PLATFORM=offscreen`.
- `with_montage_axis(axis, text=...)` does NOT set the index window — pass
  `indices=range(...)`.
- Committing from a Cowork sandbox: delete stale `*.lock` under `.git` first.
