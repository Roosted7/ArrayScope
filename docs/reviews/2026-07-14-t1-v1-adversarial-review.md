# Adversarial review of V0/T1/V1 (2026-07-14)

Scope: commits 79c8ff11 (V0 import health), 3cdbd49e (T1 measurement
foundation), 38811162 (V1 required-tile owner), reviewed and break-tested in
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
  reproduces on clean 38811162 — part of the documented 47-failure baseline
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
