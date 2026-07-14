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

### Codex execution record (added 2026-07-14; not part of the original plan)

> **[Codex 2026-07-14 — V0 complete]** Repaired both dead
> `montage_prefetch.py` imports: interaction state now comes from the live
> frame-effects owner and retained-preview conversion from `render.effects`.
> Removed the broad exception fallbacks that had made both paths silently
> inert. Added `tests/app/test_import_health.py`, which imports all 214
> `arrayscope` modules and rejects internal imports hidden behind unreported
> broad exception handlers. The guard also exposed and removed stale silent
> internal-import fallbacks in the kernel, evaluator, colormap, backend-probe,
> and benchmark paths. Focused result: 14 tests passed (import health,
> prefetch, live side-panel owner, deleted-module guard).
>
> **[Codex 2026-07-14 — V0 rejected approach / recurrence note]** Merely
> changing `frame_renderer` to another private window-module import would
> keep the same deletion hazard. V0 instead imports the existing canonical
> functions. No V0-scoped issue is carried forward; T1 is the next queue item.
>
> **[Codex 2026-07-14 — open broad-suite debt discovered during V0]** The
> documented full-suite command completed with **47 failed, 1844 passed,
> 3 skipped, 2 teardown errors**. A pristine `eeee204a` worktree reproduced
> representative failures in the stale timer allowlist and tests that still
> reach the deleted flat `window._montage_session` owner, proving those two
> classes predate V0; the remaining failures have not yet been individually
> baseline-certified. The failures also include render-coalescer timing,
> viewport/session ownership, montage settlement, and VisPy upload assertions.
> Do not repeat the R8 mistake of calling the branch broadly green: T1 must
> freeze an honest baseline, and V1/V2 must migrate or delete tests that pin
> superseded owners while preserving their user-visible assertions.

> **[Codex 2026-07-14 — T1 rebase log after two failed real-display
> starts]** The first ported-harness start rejected a fully presented,
> fully refined 60-tile frame because the marathon harness compared backend
> `TileIdentity` acknowledgements with semantic `source_id` values. Rebased
> that check to the live `tile_ack_identity()` contract; no runtime predicate
> was added. The second start then proved the marathon fixture itself was
> branch-specific: at its saved `1400×940` Wayland window size the current UI
> produces a `739×1247` image viewport, not the checkpoint's `753×1245`.
> Re-froze the checked-in fixture to the current production-restored geometry
> instead of porting the marathon's later viewport-continuity machinery. If
> this geometry drifts again without an intentional UI change, treat it as a
> production restore regression; do not widen the harness tolerance.

> **[Codex 2026-07-14 — T1 complete; frozen real-Wayland baseline]** Landed
> the production-session workflow harness, checked-in 59 MB NIfTI session
> fixture, `PanelSession`, latency-oriented GUI GC policy, divisible mean
> fast path, read-only callback/bridge instrumentation, and the schema-v1
> trace bus plus `--trace` / `trace_latency`. The bus is one flat stream with
> an 8 MiB bounded in-memory tail; optional JSONL is intentionally complete.
> Visible Wayland runs used the production restore path at the exact
> `1400×940` window / `739×1247` image viewport, and screenshots were
> inspected with all 272 requested tiles visibly populated on both backends.
> Local ignored artifacts are under
> `tests/artifacts/redesign-t1-2026-07-14/`.
>
> | Frozen cold/raw baseline | PyQtGraph | VisPy |
> |---|---:|---:|
> | phase elapsed | 2840 ms | 3469 ms |
> | event-loop max gap | 618 ms | 657 ms |
> | first input → backend ack | 2692 ms | 808 ms max across captured phases |
> | kernel queue p95 | 0.83 ms | 1.14 ms |
> | kernel run p95 | 12.37 ms | 103.74 ms |
> | largest observed GUI callback | 153.87 ms | 133.63 ms direct / 56.14 ms observed |
>
> These are failure baselines, not acceptance claims. PyQtGraph raw recorded
> a presentation blackout. VisPy raw recorded missing rough level evidence;
> VisPy FFT recorded both missing first-pixel evidence and a blackout. The
> VisPy refinement phase passed its current gates, then FFT index scroll
> timed out with 60/60 exact tiles presented, no materialization work in
> flight, and draw acknowledgement stuck at `2457/2458`. The complete VisPy
> trace contains 230,305 backend acknowledgements for this run; do not hide
> that churn by sampling the trace. V1 owns the black transition and evidence
> scope; V3 owns the stranded final-draw/non-convergence report.
>
> **[Codex 2026-07-14 — T1 rejected geometry approaches / recurrence
> note]** A backend-specific fixture or wider viewport tolerance would have
> certified two different sessions. The first production fix—merely refusing
> to settle in the same callback that resized the outer window—was necessary
> but insufficient: VisPy's child stack changed geometry after the transaction
> had looked settled. The final fix keeps the saved viewport authoritative for
> child-layout resize events while top-level user resize still releases it.
> Regression coverage forbids same-turn settlement and requires a late child
> layout change to reopen restoration. Both backends now reach the same exact
> frozen geometry; do not reintroduce per-backend geometry in the harness.
>
> **[Codex 2026-07-14 — T1 validation boundary]** The post-T1 full parallel
> non-GPU suite reported **47 failed, 1866 passed, 3 skipped, 2 teardown
> errors**: the same failure count/classes as the V0 baseline, with 22 added
> T1 tests passing. That run also exposed the already-stale timer allowlist;
> it has since been updated for the bounded harness/restore timers and its
> focused architecture guard passes. The focused T1 slice is **211 passed,
> 2 skipped**, apart from the previously baseline-reproduced preview-first
> viewport test. Compileall, F821/E9 lint, and `git diff --check` pass. Do not
> call the branch broadly green until the remaining live-owner/timing tests
> are migrated or fixed by the queue steps.

> **[Codex 2026-07-14 — V1 complete; required-tile owner and real-pixel
> gate]** Replaced the drifted onscreen/admission/completion/evidence scopes
> with `FrameSession.required_tile_numbers()`: ladder admission, first-pass
> evidence, semantic side-work, presentation batching, and
> `visible_plan_complete()` now consume that one set. Pixel-center viewport
> intersection includes a tile whose edge lands exactly on the view boundary.
> The new GPU-harness scenario renders 36 constant-value tiles, lands the
> sixth column exactly on the boundary, shifts `0:36` to `1:37`, and then
> reveals that column. On real Wayland, both PyQtGraph and VisPy presented the
> strictly increasing analytic gray ramp (within 12 gray levels), with all
> 36 required targets exact, no required tile parked, and no black tile.
> Schema-v1 lifecycle retarget/release edges let `trace_verify` replay the
> final scope: PyQtGraph was clean at 2,226 events / 36 acknowledgements and
> VisPy at 2,994 / 36. Focused model/UI coverage is 214 passed.
>
> **[Codex 2026-07-14 — V1 rejected predicates / recurrence note]** The
> first real-harness run never reached its own `settled()` even though the
> live frame had converged: the harness still read deleted
> `renderer._montage_session` and independently reconstructed completion from
> six queues plus total plan length. Both were deleted in favor of the live
> `_frame_session.visible_plan_complete()` owner. Do not restore that second
> completion model. The old one-index VisPy test also required at most one
> texture upload; VisPy correctly used one 4-byte fallback followed by one
> 192-byte exact upload while maintaining compatible pixels throughout. That
> implementation-count assertion was removed, while its physical-identity,
> semantic value, ROI, hover, cache-reuse, and no-black assertions remain.
> A one-off PyQtGraph workflow-profile timeout with all 272 identities already
> current did not reproduce at a 20-second timeout and caused no runtime
> change. V3 remains responsible for making any real stranded draw loud.

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
