# Execution plans — LOD / tile-lifecycle work

**Date:** 2026-07-06.
**Source of truth for current priority:** `docs/roadmap.md`.
**Source of truth for LOD/lifecycle rationale and history:** ADR 0050, ADR 0051, and commit
messages. These plans are execution recipes on top of them; if they disagree, update the plan or
the ADR rather than carrying two queues.

## What these plans are

Self-contained, step-by-step plans a less experienced developer (or model) can execute:

| Plan | Item | Status |
|---|---|---|
| [01-delta-commit-walk.md](01-delta-commit-walk.md) | Delta-commit walk cost (~20–30 ms warm scrub) | Done — 2026-07-06 |
| [02-pyqtgraph-lod-ab.md](02-pyqtgraph-lod-ab.md) | Re-measure PyQtGraph resident-LOD A/B → default decision | Done — 2026-07-06; PyQtGraph resident LOD remains opt-in |
| [03-p3-residency-axis.md](03-p3-residency-axis.md) | P3: residency axis authoritative | Done — 2026-07-06 |
| [04-preview-reduce-before-display.md](04-preview-reduce-before-display.md) | Preview-quality reduced display/evaluation, then exact refinement | Active next item in the roadmap |

The roadmap owns the queue beyond these recipes. Add another numbered plan only when an item
reaches the head of the roadmap and needs command-level execution detail.

## Ground rules (read before ANY plan)

1. **Thomas's bar:** the GUI event loop never hangs noticeably. Interaction outranks all speculation, everywhere. Any synchronous GUI-thread step >50 ms is a bug. Pan/scrub heartbeat max gap target ~16 ms.
2. **Defect class to check FIRST when anything wedges:** optimistic bookkeeping / repairs that only run on event X. Triage order (ADR 0051 "Wedge triage"): `lifecycle_identity_rejections>0` → false-ack door; `backend_stale_identities>0` at idle with 0 dirty → missing convergence trigger; churning → backend can't converge; `stall_repairs>0` → dispatch-construction violation. Every watchdog rescue is a bug report, not a fix.
3. **No new parallel state.** Tile state changes go through `TileLifecycle` events/effects (`presentation/tile_lifecycle.py`); dispatch decisions go through `derive_montage_dispatch` (`presentation/dispatch.py`). Never add a collection the machine doesn't own or a pump the derivation doesn't imply.
4. **Montage fast paths must gate on a real montage axis** — normal slices flow with `montage_axis=None`.
5. **One change per commit, measure after each.** Commit messages follow the existing style (see `git log`): what + why + counters/numbers.
6. **Never run the test suite concurrently with profile-workflow runs** on the host — CPU contention pushes the 35 s suite past 4 min (looks hung).
7. **No fixed sleeps for waiting.** Run commands foreground via Desktop Commander `start_process` (timeout_ms ≈ 40000), then block on `read_process_output(pid, timeout_ms≈40000)` — returns the moment the process finishes. For jobs >~2 min: nohup + sentinel line, waited on with one bounded early-exit grep loop. Treat >3× historical runtime as hung.

## Environment & commands

- Python: `~/miniconda3/envs/arrayscope/bin/python` (host conda env; the Cowork sandbox cannot load PyQt6 — GUI/GPU work must run on the host, e.g. via Desktop Commander).
- **Full suite** (~35 s, KEEP PARALLEL): `cd ~/projects/ArrayScope && ~/miniconda3/envs/arrayscope/bin/python -m pytest tests -q -n 16 --ignore=tests/gpu_interaction`
  - Known parallel-only flakes (pass alone; pre-existing, ignore unless newly consistent): `test_selecting_fft_workers_updates_settings`, `test_resource_governor_applies_worker_and_callback_limits`, `test_compute_policy_configures_stage_and_montage_lanes`, teardown of `test_montage_ready_display_payloads_commit_immediately`.
- **GPU harness** (~14 s, real hardware; asserts `stall_repairs==0`):
  `ARRAYSCOPE_GPU_TESTS=1 XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-0 QT_QPA_PLATFORM=wayland ~/miniconda3/envs/arrayscope/bin/python -m pytest tests/gpu_interaction -n 0` (a bare DC shell lacks the display env — always set these).
- **Workflow benchmark** (~30–40 s/run; `--data` optional, default = synthetic 272-tile scene):
  `~/miniconda3/envs/arrayscope/bin/python -m arrayscope.tools.profile_montage_workflow --backend {vispy|pyqtgraph} --montage-lod-policy {resident|native-only} [--json-out FILE]`
  (check exact flag names with `--help` before first use). Count `STALL WATCHDOG` on stderr — must be 0.
- **Probes** (in `tmp_probes/` and `/tmp/lod-baseline/`): `verify_scrub_fastpath.py`, `profile_cached_rebuild.py` (prints retargets/rejects; ONSCREEN — tell Thomas hands-off before running), `verify_stale.py` (phase machine + DISPATCH counters), `/tmp/lod-baseline/verify_scrub.py` (10 ms heartbeat gap probe), `profile_scrub.py`.
- **PROBE TRAP:** `with_montage_axis(axis, text=...)` does NOT set the index window — you MUST pass `indices=range(...)`. `tmp_probes/profile_cached_rebuild.py` does it right; copy from there.
- **Kill switches:** `ARRAYSCOPE_DISABLE_SCRUB_FASTPATH`, `ARRAYSCOPE_DISABLE_SESSION_RETARGET`, `ARRAYSCOPE_PYQTGRAPH_RESIDENT_LOD=1` (opt-in), `ARRAYSCOPE_ATLAS_MIPMAPS=1` (opt-in).

## Debugging gotchas

- `print()` in app code is swallowed under pytest/Qt — append to a /tmp file instead. The watchdog assertion prints to stderr deliberately.
- py-spy cannot keep up with Python 3.14 — use cProfile or the JSONL diagnostics.
- App JSONL rows are per-phase dicts (`montage_lod_*` keys); recorder snapshots use `event=snapshot` + nested `diagnostics.montage`. Wedge evidence lives in the STATIC TAIL of the file.
- `pkill -f "pytest tests"` inside a DC shell kills the shell itself (pattern self-match) — use a more specific pattern or the PID.
- VisPy offscreen `canvas.render()` needs int-rounded `physical_size`. `QTimer.singleShot` needs the 3-arg receiver form.
- Committing from a Cowork sandbox: delete stale `*.lock` under the main repo `.git` first (needs the cowork file-delete permission).

## Definition of done (every plan)

1. Full suite green (`-n 16`), GPU harness green including `stall_repairs==0`.
2. Numbers recorded (before/after) in the commit message and, when they change a decision, in the ADR.
3. Docs updated the same day: the ADR's phase/status section, `docs/roadmap.md` when priority
   changes, and `docs/current-state.md` only for high-level state changes.
4. Claude memory appended (`arrayscope-lod-residency` note: queue position, new tip, new
   recipes/gotchas discovered). Do not delete historical memory notes.

## Backlog

No independent backlog lives here anymore. Current next steps, including small follow-up tests and
probe polish, are listed in `docs/roadmap.md` under X5. Historical detail remains in ADR 0050,
ADR 0051, and the completed plans above.
