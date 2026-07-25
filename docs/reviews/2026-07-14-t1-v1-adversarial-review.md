# Adversarial review of V0/T1/V1 (2026-07-14)

Scope: commits b482a999 (V0 import health), 0f11a22c (T1 measurement
foundation), 609b8410 (V1 required-tile owner), reviewed and break-tested in
worktree `.worktrees/review-t1v1` (branch `review/t1-v1-hardening`, offscreen
PyQtGraph; no real-display claims). Hardening fixes for everything in §1 are
committed on that branch.

## 1. Breaks found (fixed on the review branch)

1. **The V0 guard missed three swallowing forms.** Planted probes proved
   `except ImportError:`, `except ModuleNotFoundError:`, and
   `except (RuntimeError, Exception):` around an import of the deleted
   `frame_renderer` module all passed the guard — the exact hazard class V0
   exists for, since a deleted internal module raises ImportError. Fixed by
   broadening the swallowing-handler set and covering tuple forms; the scanner
   is now a testable function with a synthetic-tree regression. The production
   tree is clean under the stricter rule.
2. **The benchmark fixture crashed on any non-canonical dataset.** The
   checked-in session fixture pins montage window `106:166`; `--data` with a
   smaller axis-2 dataset died with a raw `ValueError` five frames deep in
   session parsing. Now raises one actionable error naming the shapes and the
   `--session-fixture ''` escape hatch.
3. **The stall guard had a post-visible blind spot.** On synthetic data
   (64×64×12, 6 tiles) the FFT phase burned its entire 60 s timeout: the
   frame was fully visible, so the fail-fast path was skipped, while the
   completion gate could never close. New `_post_visible_gate_blockers`
   guard: fully visible + no work in flight + no dirty payloads + a blocked
   completion gate for 4 s → bail, naming the stuck gates. The same scenario
   now fails in ~7 s with `['physical_drawn']`.
4. **`trace_verify` could pass vacuously.** An empty trace, or one whose
   `lifecycle` edges were never emitted (renamed kind, broken emitter, wrong
   file), replayed as an empty scope and reported `ok: true`, exit 0. Now:
   empty traces and lifecycle-free traces are violations, and
   `--expect-targets N` pins the final scope size (the V1 GPU scenario
   already asserted 36 externally; the CLI had no such guard).
5. **`trace_latency` scanned all acks per input** (O(N·M); real traces carry
   200k+ acks) — now bisect — and dropped `kernel_finish` outcomes; it now
   reports the completed/superseded/stale breakdown (wasted-work signal).

## 2. Production bug surfaced (not fixed here — belongs to the queue)

**Idle presentation re-commit/re-ack loop.** The synthetic FFT run's trace
shows, for a 6-tile montage sitting *idle* in a wait loop (35 kernel tasks
total, nothing in flight): ~33,100 `backend_ack` events (~550/s, i.e. each
tile re-acknowledged with the *identical* exact identity ~92×/s), ~33,300
`lifecycle presented` edges, and ~5,550 `commit_batch` events in under a
minute. `presentationDrawPending()` never clears, so the physical-draw gate
never closes — this is what the stall guard now names. Codex's T1 baseline
saw the same signature at scale (230,305 acks; VisPy FFT scroll stuck at
draw ack 2457/2458).

Repro (seconds, no NIfTI needed):
`python -c "import numpy as np; np.save('/tmp/s.npy', np.random.default_rng(7).normal(size=(64,64,12)).astype('float32')*100)"` then
`QT_QPA_PLATFORM=offscreen python -m arrayscope.tools.profile_montage_workflow
--data /tmp/s.npy --load-mode native --backend pyqtgraph --max-tiles 6
--timeout-s 60 --session-fixture "" --trace /tmp/trace.jsonl` → FFT phase
stalls; the trace quantifies the churn. This likely also burns idle CPU
against the settled-idle-0% bar and competes with real work during scroll
(the 4 fps FFT target). Candidate owner: V2/V3 or an early P-step; the
dossier's B2/B3 seams (level settlement ↔ presentation re-commit) are the
place to look.

## 3. Observations (no action taken)

- The V1 GPU scenario (36-tile boundary shift) is well-constructed: analytic
  ramp, both backends, trace-verified scope. It asserts
  `required_targets == 36` externally, which is the right guard; consider
  passing `expect_targets=36` now that `verify_trace` supports it.
- README's T1 note says "checked-in 59 MB NIfTI session fixture" — the NIfTI
  is git-ignored local data (`data/*.nii`); only the 98-line JSON session is
  checked in. Fresh clones skip the two realistic-data tests and cannot
  reproduce the frozen baseline without obtaining the dataset. Worth a
  correction and, longer term, a synthetic baseline dataset.
- The environment actually ships **PySide6** (`prefer_pyside6()` at the tool
  entry); AGENTS.md and several docs still say PyQt6. Worth one sweep.
- A pre-existing failure (`test_montage_tile_residency_rss_stays_bounded`)
  reproduces on clean 609b8410 — part of the documented 47-failure baseline
  debt, not introduced by these commits.

## 4. Validation on the review branch

- Focused: `tests/app/test_import_health.py` (3),
  `tests/app/test_profile_montage_workflow.py` (39+2 skips),
  `tests/core/test_trace.py` (7) — all green.
- Broad offscreen `tests/app tests/core`: 291 passed, 3 skipped, 1
  pre-existing failure (above).
- `compileall`, Ruff `F821,E9` clean.
- End-to-end: synthetic runs confirm the friendly fixture error and the
  named post-visible stall; the raw phase still settles and records
  milestones as before.

---

# Addendum: adversarial review of V2/V3 (same day)

Scope: 6d073742 ("Carry canonical tile priority into execution") and
b2025289 ("Make stranded tile ownership loud"), reviewed on this branch
after rebasing onto them.

## What holds up

- V2 is the dossier fix executed properly: `tile_priority_key()` is the one
  ranker; the two duplicate formulas are deleted; the ladder path now reads
  the live `tile_priority_context` owner (closing the stale-`view_range`
  hazard, dossier A3); the ordinal rank rides `RungStep` →
  `TaskSpec.scheduling_rank` → the kernel heap. A synthetic offscreen
  cold-load first-ack order contained both grid-center tiles in the first
  four acks (weak positive; real-display evidence is the committed gate).
- V3 rightly deletes the silent `release_idle_evaluation_claims()` repair,
  runs always-on (armed from three frame_controller sites, not the dialog),
  and the injected-stall regression covers event, dump, and diagnostic.
- The V2/V3 `frame_effects` rework cut the idle churn ~14× on the synthetic
  repro (5,521 → 398 identical acks per tile). Real improvement.

## New findings

1. **Priority inversion surface: `scheduling_rank` sorts before `priority`,
   and every non-pipeline task defaults to rank 0.** Level-evidence work
   (`render/level_stats.py:1081,1179`, VISIBLE_MATERIALIZATION lane) and
   stage materialization (`window/frame_effects.py:3000`,
   STAGE_MATERIALIZATION lane) now execute ahead of every tile with
   rank ≥ 1 at any priority — before V2, priority ordered them. Evidence
   admission is gated on visible settlement so its window is small, but
   per-tile stage work carrying rank 0 competes directly with ranked rungs
   during scroll — exactly where the 4 fps FFT target lives. Recommend:
   non-tile visible-lane submissions get an explicit rank floor (or stages
   inherit their consuming tile's rank), pinned by a kernel test.
2. **The livelock is still live, and V3 cannot see it.** On the synthetic
   FFT repro the presentation layer still re-acknowledges identical exact
   payloads (~398 per tile) and `physical_drawn` never closes; the run
   contains **zero `stall` events** because the watchdog early-returns
   while `_montage_presentation_gate_armed` is set and
   `required_target_unsettled_tiles()` is empty. V3 detects deadlock
   (nothing happening), not livelock (the same thing happening forever).
   Complement added on this branch: `trace_verify` now flags
   `no_acknowledgement_churn` when any identity is re-acked more than
   `--max-identical-acks` (default 25) times — it fires on both the old
   (5,521) and new (398) churn traces and passes healthy scenario traces.
3. **Real watchdog firings pollute `/tmp` during test runs.** The injected
   V3 test cleans up its dump, but incidental ≥2 s stalls inside other
   offscreen UI tests each wrote an up-to-8 MB
   `/tmp/arrayscope-stall-<session>-<n>.trace.jsonl`. Two per suite run
   observed. Recommend a configurable dump directory (or suite-level
   TMPDIR) and possibly a dump-size cap; also note those incidental stalls
   are themselves evidence worth triaging.
4. Nit: `TaskSpec.__post_init__` clamps `scheduling_rank` with
   `max(0, int(rank))`, so an accidental negative rank silently becomes the
   *best* rank instead of an error.

## Validation

Rebased cleanly onto b2025289 (conflicts in `trace_verify.py` /
`test_trace.py` resolved keeping both sides' invariants; V3's stall-only
fixture gained a lifecycle event since lifecycle-free traces are now
violations). Touched suites: 209 passed, 2 skipped; compileall and Ruff
clean. The churn detector validated against three real traces: pre-V2
pathological (6 violations, worst 5,521), post-V2 (still fires at 398),
and the healthy raw-phase trace (clean).

## Resolution after review integration — Codex 2026-07-14

The five review commits were rebased into `main` without a merge commit. The
review then directly changed the next production slice:

- Stage materialization now inherits the best canonical rank of its consuming
  tiles; visible level-evidence tasks use an explicit non-tile rank floor.
  Negative ranks raise instead of silently becoming rank zero.
- Repeated confirmation of an unchanged presented identity is idempotent and
  emits no fictitious backend-ack edge. Level evidence is queued only for
  backend-accepted payload transitions, not the full active set on every
  report. A settled revision is no longer mistaken for presentation backlog,
  and no-op tiled commits no longer arm a physical-draw obligation.
- Stall dumps honor `ARRAYSCOPE_STALL_DUMP_DIR`, then the test worker's
  `ARRAYSCOPE_ARTIFACT_DIR`, before falling back to the platform temp
  directory. Parallel tests therefore keep evidence without polluting
  `/tmp`.

The exact six-tile synthetic repro no longer stalls in its FFT phase. Its
worst identical acknowledgement count fell **411 → 10 per identity**, and
backend-complete commits for the formerly wedged session fell **443 → 1**.
The complete non-canonical workflow now reaches all later scroll/zoom phases
with **zero stall events**. It still exits nonzero because the tiny dataset
does not meet the canonical workflow's presentation-continuity and
full-grid-cap gates; those gate failures are recorded separately and are not
being relabeled as idle-loop regressions.

**Rejected approach / recurrence note:** making *every* unranked `TaskSpec`
default to the rank floor stranded VisPy startup after fallback pixels:
control-plane frame-admission work is itself the producer of ranked tile work.
The accepted boundary is explicit: tile-producing control tasks retain rank
zero, stages inherit consumers, and genuinely non-tile evidence receives the
floor. Do not restore a global default floor.

---

# Addendum 2: review of the review-response, P1–P8 (2026-07-15)

Scope: 028cd444 (review-response) through ee28ff9d (P8 evidence), reviewed
on branch `review/p8-stress`.

## Going well

- **The feedback loop closed properly.** 028cd444 fixed the rank inversion
  at the root (`UNRANKED_SCHEDULING_RANK` floor, stages inherit consuming
  tile rank, negative ranks raise), and P8 then resolved the design tension
  the right way: priority-before-rank restored in the kernel with the
  plan-wide preview barrier carrying cross-rung ordering — pinned by two
  kernel tests. The synthetic livelock went 5,521 → 398 → 1 identical acks.
- **Honest rejection discipline held.** P1, P2, and P5 were measured,
  failed their gates, and were reverted with the numbers recorded. The
  churn detector and trace replay were used as acceptance instruments in
  P8's own record.
- **The full suite is green again** (1,955 passed / 8 skipped) without
  weakening user-visible assertions.

## Needs attention

1. **Trace vocabulary gap (P9-blocking for replay):** idempotent
   acknowledgement means a target satisfied by a retained compatible
   payload emits *nothing*; whole-workflow replay reports 0/12 final acks
   on a run the harness calls settled. Emit `target_satisfied_retained`
   at the point `required_target_unsettled_tiles()` closes a tile without
   a fresh backend ack; `verify_trace` on this branch already accepts it.
2. **Mid-scroll transient wedge + identity aliasing:** the V3 watchdog
   fired once during a synthetic scroll (recovered later); its probe row
   shows tile 4 acknowledged with tile 7's identity and the *same physical
   plane pointer*. Stall dump preserved. Retarget-under-churn identity
   rebinding deserves a focused regression.
3. **Synthetic convergence is nondeterministic** (stress-matrix runs flip
   between pass and fail for identical inputs), **raw complex64 input
   deadlocks PyQtGraph deterministically** (6/10 tiles stuck at
   dirty/pending_upsert, `report_committed=0`), and tiny-montage level
   settlement is racy. The canonical fixture converging is necessary, not
   sufficient.
4. **R8 gate calibration:** `full_grid_not_capped` and
   `presentation_continuity` fail on capped/synthetic offscreen runs
   regardless of convergence — fine for the canonical fixture, but the
   harness should mark them n/a (not FAIL) when their preconditions
   (uncapped run, fixture geometry) don't hold.

## Shipped on this branch

- `verify_trace`: retains acks across compatible retargets, accepts the
  (future) `target_satisfied_retained` edge with sequence guards; tests for
  both directions.
- `tests/stress/test_synthetic_stress_matrix.py`: opt-in workflow × input-
  class matrix with trace-replay oracles (found the complex64 deadlock and
  the convergence nondeterminism on its first runs).
- `docs/testing/stress-and-trace-strategy.md`: the drivers × oracles model,
  the four cost-ordered rings, and the rules for new features (new input
  class → matrix row; new behavior → trace event + oracle rule) and for
  performance work (wasted-work counters as the early-warning channel).
