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
   (`--max-identical-acks`), `no_identical_commit_bail_loop`
   (`--max-identical-commit-bails`), `no_stall_events`, the import-health
   guard.
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
| 1 — default offscreen suite | everything except stress/gpu_interaction; ~2300 tests, xdist (workers capped at half cores — GL segfault guard) | every `pytest`; **CI on every push/PR** (`.github/workflows/ci.yml`, incl. 3.10–3.14 compat, coverage, strict-UI, 3-OS wheel validation) | `QT_QPA_PLATFORM=offscreen pytest tests -q` |
| 2 — serial artifact ring | canonical screenshot/JSONL artifacts | CI (`-n 0` steps); before UI-visual claims | `pytest tests/ui/test_qt_smoke_artifacts.py -n 0` |
| 3 — stress ring (opt-in, serial) | synthetic stress matrix + live churn convergence; the livelock/stall reproducers | **manually, before merging scheduling/lifecycle/presentation changes** | `ARRAYSCOPE_STRESS=1 pytest tests/stress -n 0` (live half needs Wayland + local NIfTI under `data/`) |
| 4 — real-GL/Wayland acceptance | `tests/gpu_interaction` pixel/heartbeat harness + live gate tests; the only ring that satisfies ground rule #1 | **manually, before any rendering/scheduling "fixed" claim or perf claim** | `ARRAYSCOPE_GPU_TESTS=1 XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-0 QT_QPA_PLATFORM=wayland pytest tests/gpu_interaction -n 0` |
| journey matrix — real Wayland, serial | `{cold fill, zoom-in, zoom-out, scroll shuffle, index scroll} × {VisPy, PyQtGraph, Wgpu}`; JSONL phase ordering/priority/LOD plus screenshot-output latency | **pre-merge for every `display/`, `render/`, `kernel/`, or `window/` change** | `XDG_RUNTIME_DIR=/run/user/$(id -u) WAYLAND_DISPLAY=wayland-0 QT_QPA_PLATFORM=wayland python -m arrayscope.tools.journey_matrix run --artifact-dir tests/artifacts/journey-matrix-$(date +%F)` (add `--wgpu-present-method screen` to run the wgpu rows on the native swapchain; the driver fails loudly if screen cannot activate) |
| 4 — real-display/GL Wayland acceptance | `tests/gpu_interaction` physical-pixel/heartbeat harness + live gate tests (real GL for VisPy, real Qt raster for PyQtGraph); the only ring that satisfies ground rule #1 | **manually, before any rendering/scheduling "fixed" claim or perf claim** | `ARRAYSCOPE_GPU_TESTS=1 XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-0 QT_QPA_PLATFORM=wayland pytest tests/gpu_interaction -n 0` |
| benchmarks/harness | `profile_montage_workflow`, `rendering_benchmarks`, `profile_scroll_input` + trace tools | per queue-step evidence | `python -m arrayscope.tools.profile_montage_workflow --backend {vispy,pyqtgraph,wgpu}` (cwd = repo root for `data/` paths) |

The 5 s interaction limit applies to each step in every ring and harness, not
to the cumulative duration of a scenario with several steps. Profile CLI
values above the limit are clamped, and the architecture guard rejects local
settlement-timeout owners.

**Enforcement gap, stated honestly:** rings 3–4 are machine-bound (real
Wayland, real GPU where applicable, local data) and cannot run in CI — CI is
entirely offscreen software-GL. The rule is therefore personal, not scheduled:
**whoever (human or agent) changes a display/render/kernel/window lane
runs rings 3–4 themselves before claiming the change works**, and records
the run in the commit or PR description. No background runner will catch
it for you; an unrecorded ring-3/4 run means the claim is "compiles",
not "fixed".

### Journey-matrix trajectory gate

The journey matrix is output-driven: `profile_montage_workflow` emits the
gesture boundaries and structured work/commit facts, while its periodic
visual timeline supplies the screenshot sequence. The verifier requires, for
every applicable journey/backend cell:

1. zero phase-2 `kernel_submit` events while lifecycle-owned phase-1 coverage
   is open (`trace_verify` independently enforces
   `no_phase2_submit_during_coverage`);
2. at least the journey/backend's declared `N` payload commits, every commit
   within its emitted cap. Shader-backend atomic successors are unbounded;
   GPU shader-backend commits may atomically include already-resident rebinds
   beyond the item cap when their explicitly reported cold-upsert tile subset
   remains within it (the legacy all-zero texture/upload-byte/vertex-upload
   proof is also accepted). Uploaded tiles never bypass the cold cap. Rank
   correlation compares each commit's local presentation ordinal
   with the immutable current-camera ranks captured at its final backend
   boundary when two or more ranked payloads make ordering observable. Each
   batch is normalized to its first rank so batching offsets and later-ready
   work do not masquerade as an ordering decision (minimum correlation
   `0.50`);
3. session LOD demand matches demand recomputed from the live camera within
   5 s — the strict xfail in `tests/ui/test_lod_demand_freshness.py` remains
   the red pin for the open 2026-07-18 defect and must not be weakened;
4. the screenshot pixels change within the 2 s interaction target after each
   gesture (metadata-only progress is not accepted as new output); and
5. applied LOD is no coarser than desired within 5 s after phase 1 closes.
   A journey which never opens a coverage pass (for example a resident
   zoom-out) uses gesture start as the close edge; once an open pass is
   observed, its explicit close transition is mandatory.

The `N` values live beside the oracle in
`arrayscope.tools.journey_matrix.MIN_COMMITS`: cold VisPy and scroll shuffles
must visibly progress through at least two bounded commits; PyQtGraph's cold
CPU-windowed fill must do the same rather than appearing in one complete pop;
zoom-out may legitimately reuse finer resident pixels without a payload
commit. Every oracle has a fault-injection test in
`tests/app/test_journey_matrix.py`.

For a quick software-GL diagnostic, append `--offscreen-smoke`. It exercises
the trace/replay plumbing and PyQtGraph output trajectory, but it is not a
rendering, scheduling, timing, GPU, or Wayland acceptance result and never
replaces the command above.

The driver also reports the older `profile_montage_workflow` R8 verdicts as
`driver_failures`. Those nonzero exits are diagnostic only when the owned
journey artifacts are complete: the matrix verdict comes exclusively from
the output oracles above. A 180 s whole-process watchdog remains a blocking
failure, but it cannot turn a step that exceeded 5 s into a pass.

## Known suite state (2026-07-17)

- 2026-07-17 branch run: 2277 passed / 34 skipped / 1 xfailed (~116 s
  parallel), plus one pre-existing architecture-guard failure also reproduced
  on untouched `main` (`test_lod_demand_freshness.py` owns two uncapped
  `waitUntil` timeouts). The skips are the opt-in rings.
- Open xfails that are *tracked work, not noise*: churn-convergence
  (queue step 1, strict=False), tiny-3-slices raciness (strict=False), and
  live-camera LOD-demand freshness after zoom (strict=True). The native
  complex64 PyQtGraph stress row has been a hard pass since `14f0fbc5` and
  was re-verified serially on 2026-07-17.
- `tests/gpu_interaction`: 16/16 green on real Wayland (2026-07-17 full
  lane, strict=True), tiny-3-slices raciness (strict=False).
- `tests/gpu_interaction`: 20/20 green on real Wayland (2026-07-17 full
  serial run) — the 4 P9-era baseline failures no longer reproduce post-G5.
  The ring now includes physical-pixels-to-CPU reference gates for both
  first-class backends: VisPy framebuffer
  (`test_framebuffer_cpu_reference.py`) and PyQtGraph Qt raster
  (`test_pyqtgraph_raster_cpu_reference.py`). Their shared oracle is
  `tests/oracles/framebuffer_reference.py`; default-ring smokes are
  `tests/ui/test_framebuffer_cpu_reference.py` and
  `tests/ui/test_pyqtgraph_raster_cpu_reference.py`.
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

## wgpu environment traps (2026-07-18/19, all field-proven)

1. Import rendercanvas ONLY through
   `arrayscope.display.wgpu_imageview2d.import_qrenderwidget()` — a bare
   import force-sets `QT_QPA_PLATFORM=xcb` on Wayland hosts at import
   time, silently flipping the AUTO backend probe for every later window
   in the process (cross-file test pollution, 2026-07-18).
2. Every wgpu adapter probe pins
   `set_instance_extras(backends=["Vulkan"])` BEFORE its first
   `request_adapter_sync`: an all-backends instance re-inits EGL during
   GL adapter enumeration and SIGABRTs workers holding vispy GL state
   (wgpu-hal panic `gles/egl.rs:305`).
3. `winId() == wl_surface*` is undocumented Qt behavior, pinned per Qt
   minor by `tests/gpu_interaction/test_wgpu_native_wayland_pin.py`
   (ring 4) — run it after any Qt upgrade.

## Load-variance measurement protocol (perf claims)

- Attribution profiles (cProfile) may run under load; TIMING claims may
  not: quiet machine, no concurrent suites/agents (xdist worker storms
  turned real-Wayland journey rows red spuriously more than once).
- p95 claims need repeat runs; discard the first run immediately after a
  matrix/fill (cold caches: a 13.8 s cold fill poisoned a 157 ms sample).
- Benchmark completion must require equal visual maturity (level
  convergence), or faster useful work reads as regression — pinned by
  the 2026-07-19 perf program.
- py-spy is unusable on this machine (ptrace_scope); use cProfile
  in-process, GPU timestamp queries, and the trace timeline.
