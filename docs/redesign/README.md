# Redesign — course and queue

**Date:** 2026-07-14. **Status:** V0–V4 and review hardening are linear on `main`.
**[Codex 2026-07-14 — linear-history correction]** The temporary V4 and
review integration merges were removed by rebase. The redesign, V4 closure,
and adversarial-review commits now form one merge-free sequence on `main`.
Older execution-record wording below that said “merged” is corrected where it
affected the live status; the recorded pre-integration validation numbers are
unchanged.
**[Codex 2026-07-14 — post-V4 ownership update]** This file remains the
execution/evidence record for the redesign performance program. With V4
complete, `docs/roadmap.md` again owns the product queue. Everything that
used to live in this directory is history in [archive/](archive/) — read it
for evidence, never for direction.

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
| V4 | **Done 2026-07-14.** Rebased the completed redesign linearly onto `main`; no merge commit remains | `main` runs the fixed viewer; roadmap (X5…) resumes |
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
> the production-session workflow harness, local git-ignored 59 MB NIfTI data
> plus the checked-in JSON session fixture,
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

> **[Codex 2026-07-14 — V2 complete; one ranker reaches the kernel and
> pixels]** Deleted the independent resident-ladder and viewport-seeding
> distance formulas. `display.model.tile_priority.tile_priority_key()` is now
> the single ranker used by the native queue, one-shot presentation admission,
> viewport seeding, prefetch, and resident ladder. Its ordinal rank rides each
> rung into `TaskSpec.scheduling_rank`, ahead of rung priority in the kernel
> ready heap, so a center refinement no longer queues behind an edge floor.
> Fully cold source successors now present in bounded priority bands; an
> already-compatible successor remains atomic so V1's no-black one-index
> handoff is preserved. The real-Wayland 36-tile `0:36` → `36:72` scenario
> passed on both backends with the correct analytic pixel ramp and 36 exact
> final acknowledgements. The recorded first exact cohort contained at least
> 6/8 canonical nearest tiles on PyQtGraph and 14/16 on VisPy (bounded worker
> completion may permute two tiles); trace replay was clean at 2,762 and 9,031
> events respectively.
>
> **[Codex 2026-07-14 — V2 rejected atomicity approach / recurrence note]**
> Simply deleting CPU atomic successors made the cold trace progressive, but
> regressed the established one-index contract: PyQtGraph discarded 59
> compatible predecessor slots and rebuilt acknowledgements 12 → 25 → 37 →
> 49 → 59 → 60. The accepted rule is semantic: atomicity preserves compatible
> pixels; it is not a backend-wide or every-source-window rule. A disjoint
> successor has zero compatible pixels and therefore streams center-out. Do
> not restore first-display/full-successor atomic waits, and do not remove the
> compatibility test to chase a prettier cold trace. Tests that asserted the
> deleted CPU atomic readiness helper or forbade a presentable preview floor
> were removed/migrated; pixel identity, one-index continuity, and trace order
> are now the gates.

> **[Codex 2026-07-14 — V3 complete; non-convergence is loud]** The
> receiver-owned watchdog now observes every live frame session, not only an
> open diagnostics dialog. A required tile whose owner signature is unchanged
> for at least two seconds, with the kernel/completion queue and presentation
> continuation idle, emits one schema-v1 `stall` event containing the required
> tile IDs, lifecycle rows, queue/evidence counts, stage ownership, and commit
> flags. It dumps the bounded trace tail to
> `arrayscope-stall-<session>-<count>.trace.jsonl` under
> `ARRAYSCOPE_STALL_DUMP_DIR`, the configured test artifact directory, or the
> platform temp directory (in that order), stores that path in
> diagnostics state, and shows a persistent status-bar error with the path.
> When no JSONL sink was requested, the watchdog activates only the bounded
> 8 MiB ring so production stalls still have evidence. `trace_verify` now
> rejects any captured stall event. The injected stranded-tile regression
> proves the event, owner chain, ≥2 s threshold, dump contents, and visible
> diagnostic.
>
> **[Codex 2026-07-14 — V3 rejected repair / recurrence note]** The old
> diagnostics-only probe silently called `release_idle_evaluation_claims()`
> when the kernel was idle. That changed live lifecycle state, erased the
> owner chain being investigated, and then retried the same architecture.
> V3 deletes that repair and the dialog-visible gate. The watchdog is evidence
> only: it never schedules, releases, retries, or marks a tile complete.

> **[Codex 2026-07-14 — V4 merge-readiness boundary; broad suite remains
> red]** All required V-step acceptance evidence is now present: the V1 and
> V2 pixel/trace scenarios pass on real Wayland on both backends, and the V3
> stranded-tile injection produces the visible diagnostic plus trace dump.
> The final pre-merge parallel non-GPU suite is **42 failed, 1874 passed,
> 3 skipped, 2 teardown errors** in 98.56 s. A serial replay of that failed
> set is **41 failed, 1 passed, 430 deselected, 2 teardown errors**, so this
> is not being dismissed as xdist noise. The set includes tests that still
> read the deleted flat `window._montage_session`, the intentionally
> superseded exclusive boundary-intersection expectation, and unresolved
> coalescer, levels, viewport/ROI, cache-rebind, and transition behavior.
> Those failures remain explicit post-integration product/test debt; V4
> certifies the named visible-truth fixes, not a broadly green test suite.
> Do not later cite V4 integration as evidence that the full suite passed.

> **[Codex 2026-07-14 — V4 complete; linear-history correction]** The
> completed redesign and V4 closure were rebased directly onto `main`; the
> temporary merge commit is intentionally absent. V0–V4 are closed. The next redesign-derived work is the measured
> P-program, while the live product ordering is again maintained in
> `docs/roadmap.md`.

> **[Codex 2026-07-14 — P1 rejected; narrowed prefetch-busy was not a
> measured improvement]** Replaced the broad session-collection busy test
> experimentally with V1's canonical `required_target_settled()` plus live
> kernel/controller work, then ran the same real-Wayland scroll phases before
> and after. VisPy FFT elapsed **12,855.7 → 12,899.7 ms** (+0.3%) and heartbeat
> max **223.2 → 229.7 ms**; scalar elapsed **7,448.7 → 9,383.5 ms** (+26.0%)
> while heartbeat max changed **180.6 → 165.5 ms**. Both variants remained
> gate-red. PyQtGraph froze identically before and after at **50/60 presented,
> 10 dirty upserts, idle kernel, 4.0 s stall**. The runtime/test change was
> reverted, leaving no P1 code delta. Do not reapply the marathon's narrowed
> busy predicate on architectural plausibility alone; require a trace showing
> prefetch suppression on the critical path. P2 (committed-frame
> `level_source`) is next.

> **[Codex 2026-07-14 — P2 rejected; committed `level_source` crossed an
> unproven maturity boundary]** Experimentally carried the selected
> `LevelSource` through `DisplayTiledPresentation` into
> `CommittedDisplayFrame`, including viewport re-commits and concrete level
> changes. The VisPy workflow stopped failing
> `first_visible_level_evidence_quality`, but did not improve the bars: FFT
> elapsed **12,855.7 → 13,139.3 ms** (+2.2%) and heartbeat max
> **223.2 → 230.2 ms**. More importantly, the real V2 VisPy center-out gate
> regressed from the accepted **14/16** nearest tiles in the first cohort to
> **4/16**. The focused relative-level scenario also mapped `5:15` over a
> rough committed `9:10` source and produced **24:214** instead of
> **105:115**. All P2 runtime and test edits were reverted. The missing
> prerequisite is an explicit maturity/semantic rule for when level evidence
> may become committed-frame truth; do not blindly port the marathon field.
> P3 (viewport intent replay) is next.

> **[Codex 2026-07-14 — P3 complete as a viewport-truth fix; no performance
> credit claimed]** `setViewportContentExtent()` now reports a real semantic
> extent transition, and acknowledgement replays AUTO/FIT against that
> successor extent while preserving an explicit USER camera. The canonical
> controller recognizes the accepted square-pixel content fit even when its
> previous auto baseline belongs to the predecessor. VisPy publishes hidden
> bounds inside the same programmatic transaction and restores the committed
> camera until acknowledgement, so backend scene geometry cannot masquerade
> as user input. The focused display/window/harness slices are **328 passed,
> 2 skipped**; the V1 boundary and V2 center-out real-Wayland gates remain
> green on both backends (**4 passed**).
>
> **[Codex 2026-07-14 — P3 negative result / remaining stall]** The exact
> real-Wayland FFT/scalar workflow still stops in the FFT phase before it can
> write a phase metrics record. Before and after both ended with **7/60 active
> tiles, 53 dirty tiles, no work in flight, `report_committed=0`, `ack_new=0`,
> and `ViewportMode.USER`**. The trace changed only within run noise:
> **61,712 → 59,500 events**, **136 → 137 backend acks**, and input-to-first-
> ack **6819.9 → 6811.6 ms**. This is expected for the fixture's explicit USER
> camera: P3 must not move it. Do not cite P3 as a fix for the idle
> presentation-gate stall or broaden AUTO promotion to capture USER state.
> The failed workflow evidence is retained under the ignored
> `tests/artifacts/redesign-p3-2026-07-14/{before,after}` directories. P4
> (background histogram aggregation) is next.

> **[Codex 2026-07-14 — P4 complete; histogram aggregation leaves the GUI
> transaction]** The Qt-owned `MontageLevelTracker` now exposes a guarded
> `(revision, expected sources, covered sources, samples)` snapshot; bounded
> sample aggregation runs as explicit kernel work and installs only if every
> identity still matches. The presentation path consumes current cached data
> or schedules it—it never derives the aggregate inside a tile commit. The
> conceptual-stride selector was also changed from one full-vector filter per
> tile to arithmetic intersections. On the deterministic 60×8192-source
> workload, identical output measured **2.433 → 0.248 ms median** (9.8×) and
> **3.719 → 0.280 ms max**. The real-Wayland trace records **24** completed
> aggregate jobs, with **36.64 ms max / 2.46 ms median** worker runtime; that
> work no longer occupies a GUI callback.
>
> **[Codex 2026-07-14 — P4 regression record / accepted fix]** The first
> completion design delegated to the generic evidence wake and stranded the
> VisPy first-pixel phasing scenario after six preview acknowledgements. An
> unconditional wake then exposed a second invariant violation: an atomic
> delta prepared at rough level revision 1 was reused with refined uniforms,
> crossing payload levels **254.54:1860.67** with backend levels
> **250.00:1860.67**. The accepted path makes prepared atomic transactions
> level-revision-aware before issuing the histogram wake. The exact phasing
> regression now passes; focused coverage is **222 passed, 2 skipped** and
> broad `tests/display tests/window` is **840 passed**. That broad run also
> migrated the remaining test that still expected V1's deleted exclusive
> boundary predicate; the boundary tile is again pinned as required.
>
> **[Codex 2026-07-14 — P4 overall-bar boundary / open issue]** P4 does not
> receive credit for the separate presentation deadlock. The canonical
> workflow remains at **7/60 active, 53 dirty, no work in flight**. Relative
> to the unmodified P3 trace, input-to-first-ack changed **6811.6 → 6717.5
> ms**, bridge-drain max **54.38 → 53.02 ms**, and total events
> **59,500 → 60,939**; the phase still aborts before a report is written.
> Physical V1/V2 gates remain green on both backends (**4 passed**). Do not
> move histogram work back to Qt to chase this unrelated stall. P5 (coalesced
> completion drain) is next.

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

- Python: `~/miniconda3/envs/arrayscope/bin/python` (host conda env with
  **PySide6**; GUI/GPU work runs on the host).
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
