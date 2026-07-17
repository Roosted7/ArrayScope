# Testing — rings, enforcement, and the laws

Front page for test policy. Deep dives:
[strategy.md](strategy.md) (what each layer proves),
[stress-and-trace-strategy.md](stress-and-trace-strategy.md)
(drivers × oracles), [manual-regression.md](manual-regression.md),
[release-candidate.md](release-candidate.md).

## The laws (why recurring defects recur, and what stops them)

1. **A defect's regression test goes in the ring that can actually see the
   failure.** Black/stale/orange tiles, stalls, and livelocks were invisible
   to the offscreen ring for weeks — an offscreen test for a live-Wayland
   failure class is documentation, not protection. Name the ring in the test
   docstring along with the dossier/ADR it pins (36 files already do this —
   keep the convention).
2. **Every silent failure mode found gets a loud channel in the same fix**:
   a `trace_verify` invariant, a stats counter asserted somewhere, or a
   visible diagnostic. Examples that now exist because of this rule:
   `no_identity_rejected_commits`, `no_acknowledgement_churn`
   (`--max-identical-acks`), `no_stall_events`, the import-health guard.
3. **Implementation-detail tests are deletable** when they block a
   user-visible fix (say so in the commit) — but user-visible assertions are
   never weakened to green a suite ([ground rules](../ground-rules.md) #4).
4. **No fixed-wait timing assertions and no permissive settle timeouts.** The
   suite is parallel by default; `qWait(220)`-style windows flake under load.
   Wait on the actual signal/condition. For every user-visible step the target
   is 2 s and the hard failure is 5 s, per
   `arrayscope.tools.interaction_budget`. A step-specific timeout may be
   shorter and must remain capped by that owner; it may never be widened to
   green a slow path. Longer whole-process deadlock guards are not settlement
   success budgets.
5. **Oracles must be proven able to fail.** A gate that passes on an empty
   trace or a clamped framebuffer rectangle is vacuous — pair new oracles
   with fault injection (`trace_verify` grew `trace_not_empty` /
   `--expect-targets` for exactly this reason).

## The rings

| Ring | What | Trigger | Command |
|---|---|---|---|
| 0 — fast Qt-free loop | kernel/render/presentation semantics (~0.5 s) | while editing | `pytest tests/kernel tests/render tests/presentation -q -n 0` |
| 1 — default offscreen suite | everything except stress/gpu_interaction; ~2081 tests, xdist (workers capped at half cores — GL segfault guard) | every `pytest`; **CI on every push/PR** (`.github/workflows/ci.yml`, incl. 3.10–3.14 compat, coverage, strict-UI, 3-OS wheel validation) | `QT_QPA_PLATFORM=offscreen pytest tests -q` |
| 2 — serial artifact ring | canonical screenshot/JSONL artifacts | CI (`-n 0` steps); before UI-visual claims | `pytest tests/ui/test_qt_smoke_artifacts.py -n 0` |
| 3 — stress ring (opt-in, serial) | synthetic stress matrix + live churn convergence; the livelock/stall reproducers | **manually, before merging scheduling/lifecycle/presentation changes** | `ARRAYSCOPE_STRESS=1 pytest tests/stress -n 0` (live half needs Wayland + local NIfTI under `data/`) |
| 4 — real-GL/Wayland acceptance | `tests/gpu_interaction` pixel/heartbeat harness + live gate tests; the only ring that satisfies ground rule #1 | **manually, before any rendering/scheduling "fixed" claim or perf claim** | `ARRAYSCOPE_GPU_TESTS=1 XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-0 QT_QPA_PLATFORM=wayland pytest tests/gpu_interaction -n 0` |
| benchmarks/harness | `profile_montage_workflow`, `rendering_benchmarks`, `profile_scroll_input` + trace tools | per queue-step evidence | `python -m arrayscope.tools.profile_montage_workflow --backend {vispy,pyqtgraph}` (cwd = repo root for `data/` paths) |

The 5 s interaction limit applies to each step in every ring and harness, not
to the cumulative duration of a scenario with several steps. Profile CLI
values above the limit are clamped, and the architecture guard rejects local
settlement-timeout owners.

**Enforcement gap, stated honestly:** rings 3–4 are machine-bound (real
Wayland, real GPU, local data) and cannot run in CI — CI is entirely
offscreen software-GL. The rule is therefore personal, not scheduled:
**whoever (human or agent) changes a display/render/kernel/window lane
runs rings 3–4 themselves before claiming the change works**, and records
the run in the commit or PR description. No background runner will catch
it for you; an unrecorded ring-3/4 run means the claim is "compiles",
not "fixed".

## Known suite state (2026-07-16)

- ~2081 passed / 24 skipped (~124 s parallel). The skips are the opt-in
  rings.
- Open xfails that are *tracked work, not noise*: churn-convergence
  (queue step 1, strict=False), complex64 PyQtGraph deadlock (standing
  lane, strict=True), tiny-3-slices raciness (strict=False).
- 4 pre-existing `tests/gpu_interaction` baseline failures (P9-era) —
  re-triage queued.
- Shared fakes: `tests/display/vispy_test_utils.py`; live-window harness:
  `tests/ui/helpers.py` — use these, don't re-roll.

## Environment facts (hard-won; trust these)

- Python: `~/miniconda3/envs/arrayscope/bin/python` (conda env, PySide6,
  editable install). **The editable-install pointer decides which checkout
  the live app AND every test run import — check it before interpreting any
  field report or gate result:**
  `python -c "import arrayscope; print(arrayscope.__file__)"`.
  Re-pointing it (`pip install -e <checkout>`) is a deliberate, announced
  act: it silently redirects the *other* party's runs too (a branch agent's
  live gates would import `main`, or the user's field session imports WIP).
  Field reports must name the checkout + commit they ran. As of 2026-07-17
  the pointer is at `.worktrees/g5-sparse-pyramid` (the G5 program's live
  verification channel) — the user's field sessions currently test that
  branch, not `main`.
- Real display works from the harness (`QT_QPA_PLATFORM=wayland`). Real
  rendering/Wayland claims must never use `offscreen`. Hybrid GPU: Intel
  default; NVIDIA offload is *slower* to first frame — don't default to it.
- **py-spy is unusable here** (unprivileged sampling can't keep up); use
  cProfile in-process, `perf`, or the JSONL diagnostics.
- Headless hang diagnosis: `timeout -s ABRT <s> pytest …` → faulthandler
  stack dump. A swallowed exception in `request_operation` surfaces as a
  modal QMessageBox that hangs offscreen runs.
- `print()` in app code is swallowed under pytest/Qt — append to a /tmp
  file. JSONL wedge evidence lives in the static tail. VisPy offscreen
  `canvas.render()` needs int-rounded `physical_size`.
  `with_montage_axis(axis, text=...)` does NOT set the index window — pass
  `indices=range(...)`.
- Artifacts convention: `tests/artifacts/<gate>-<date>/` (gitignored);
  diagnostics land as `arrayscope-diagnostics-<stamp>.jsonl` in the repo
  root (gitignored).
