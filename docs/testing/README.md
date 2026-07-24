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
| 3 — stress ring (opt-in, serial) | synthetic stress matrix + live churn convergence on WGPU and PyQtGraph by default; the livelock/stall reproducers | **manually, before merging scheduling/lifecycle/presentation changes** | `ARRAYSCOPE_STRESS=1 pytest tests/stress -n 0` (live half needs Wayland + local NIfTI under `data/`; override with `ARRAYSCOPE_STRESS_BACKENDS`) |
| 4 — real-GL/Wayland acceptance | `tests/gpu_interaction` pixel/heartbeat harness + live gate tests; the only ring that satisfies ground rule #1 | **manually, before any rendering/scheduling "fixed" claim or perf claim** | `ARRAYSCOPE_GPU_TESTS=1 XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-0 QT_QPA_PLATFORM=wayland pytest tests/gpu_interaction -n 0` |
| journey matrix — real Wayland, serial | `{cold fill, zoom-in, zoom-out, scroll shuffle, index scroll} × {VisPy, PyQtGraph, Wgpu}`; JSONL phase ordering/priority/LOD plus screenshot-output latency | **pre-merge for every `display/`, `render/`, `kernel/`, or `window/` change** | `XDG_RUNTIME_DIR=/run/user/$(id -u) WAYLAND_DISPLAY=wayland-0 QT_QPA_PLATFORM=wayland python -m arrayscope.tools.journey_matrix run --artifact-dir tests/artifacts/journey-matrix-$(date +%F)` (add `--wgpu-present-method screen` to run the wgpu rows on the native swapchain; the driver fails loudly if screen cannot activate) |
| 4 — real-display/GL Wayland acceptance | `tests/gpu_interaction` physical-pixel/heartbeat harness + live gate tests (real GL for VisPy, real Qt raster for PyQtGraph); the only ring that satisfies ground rule #1 | **manually, before any rendering/scheduling "fixed" claim or perf claim** | `ARRAYSCOPE_GPU_TESTS=1 XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-0 QT_QPA_PLATFORM=wayland pytest tests/gpu_interaction -n 0` |
| benchmarks/harness | `profile_montage_workflow`, `histogram_pipeline_benchmark`, `rendering_benchmarks`, `profile_scroll_input` + trace tools | per queue-step evidence | `python -m arrayscope.tools.profile_montage_workflow` runs every stage on WGPU and PyQtGraph; `python -m arrayscope.tools.histogram_pipeline_benchmark --output /tmp/histogram.json` covers dtype/storage/population plus real low/high-power WGPU evidence; pass `--backend {vispy,pyqtgraph,wgpu}` for an explicit profile backend (cwd = repo root for `data/` paths) |

The 5 s interaction limit applies to each step in every ring and harness, not
to the cumulative duration of a scenario with several steps. Profile CLI
values above the limit are clamped, and the architecture guard rejects local
settlement-timeout owners.

The profiler's displayed-X and displayed-Y stages keep both view axes cropped
while they apply fast and slow scrolls to every dimension. Their settled
checkpoints are physical-pixel gates: WGPU's executor target and PyQtGraph's
raster must match the CPU semantic reference, and WGPU's source origin must
match the immutable crop anchor. `--screenshot-dir` preserves those
checkpoints for human comparison. The WGPU upload verdict distinguishes
display-axis rebinds (zero upload required) from montage-axis motion (cold
demand reported independently).

**Enforcement gap, stated honestly:** rings 3–4 need a real compositor and
real GL, and CI is entirely offscreen software-GL. The rule is therefore
personal, not scheduled: **whoever (human or agent) changes a
display/render/kernel/window lane runs rings 3–4 themselves before claiming
the change works**, and records the run in the commit or PR description. No
background runner will catch it for you; an unrecorded ring-3/4 run means
the claim is "compiles", not "fixed".

What *has* changed (2026-07-21): those rings no longer need an **attached
display**. `arrayscope.tools.headless_display` owns a headless Weston with
the real GL renderer, so ring 3–4 commands run on a machine with no logged-in
session — see [Headless real rendering](#headless-real-rendering-rings-34-without-an-attached-display).
They still need a real GPU and local data, so this widens *where* the rings
can run; it does not make them automatic, and it does not make an offscreen
run into evidence.

### Headless real rendering (rings 3–4 without an attached display)

Prefix any ring-3/4 or harness command with the launcher; it exports
`WAYLAND_DISPLAY` and `QT_QPA_PLATFORM=wayland` for the child:

```
python -m arrayscope.tools.headless_display -- \
    env ARRAYSCOPE_GPU_TESTS=1 python -m pytest tests/gpu_interaction -n 0 -q
```

The journey matrix and the profiler run the same way, and the matrix labels
its report `"ring": "headless-weston"` so a verdict is never filed as if it
came from the developer's own session:

```
python -m arrayscope.tools.headless_display -- \
    python -m arrayscope.tools.journey_matrix run --artifact-dir tests/artifacts/...
```

**One compositor per batch, not per test.** A whole batch — including
parallel xdist workers, which inherit the socket — shares one Weston.
Separate *activities* get separate compositors, so a profiler's full-output
screenshots never photograph another activity's windows.

**Screen evidence owns its compositor.** `profile_montage_workflow`'s wgpu
screen path starts its own compositor in `exact_window` mode and never joins
a batch — a batch has its own output size and keeps its panel, which would
offset the window and quietly break the capture == window identity. This
replaced the kiosk shell: the profiler reads the **session fixture first**,
sizes the sole output to the `panels.window_size` that session restores, and
runs with no panel and no window decoration. The window then fills the
output at (0, 0), so one capture is the window, byte for byte — the same
identity kiosk provided, without kiosk's effect on viewport aspect and
montage layout. A capture whose size is not the window's is still a hard
failure; it is never saved as window evidence.

**Measured parity on the reference laptop (2026-07-21, quiet machine):**

| Ring / suite | Real session | Headless Weston |
|---|---|---|
| `tests/gpu_interaction` (ring 4, `-n 0`) | 28 passed, 72.9 s | 28 passed, 72.9 s |
| window / viewport geometry | 600x800 / 447x553 | 600x800 / 447x553 |
| full suite failure set | 11 failures | same 11 (+1 xdist load flake) |

The full-suite failures are the same tests in both environments because they
assert *offscreen* behaviour (`test_..._falls_back_to_bitmap_off_wayland`,
the diagnostics/prefetch/window-sync dialogs). **They belong in the offscreen
ring and must not be "converted"** — a test that pins the offscreen fallback
path is only meaningful offscreen. Default `pytest` still runs offscreen.

Traps the launcher pins (all field-proven here, all silent if unhandled):

1. **EGL vendor order.** `10_nvidia.json` sorts before `50_mesa.json`, so a
   headless compositor with no parent session resolves to the NVIDIA GPU
   while the real session uses Intel — the documented *slower* path, which
   would move every performance bar for a reason nobody would look for. The
   launcher pins the Mesa vendor file.
2. **Capture orientation.** On that NVIDIA path `weston-screenshooter` returns
   a **y-flipped** image, which would silently invert every pixel oracle
   while leaving it green. Pinning trap 1 also fixes this; prove orientation
   with a known asymmetric scene rather than assuming it.
3. **Never the kiosk shell.** Kiosk force-fullscreens every window to the
   output size, changing viewport aspect and therefore montage layout: at
   1600x1000 it turned
   `test_one_index_boundary_scroll_has_pixels_and_trace_clean[pyqtgraph]`
   red through geometry alone. The desktop shell gives natural window sizes,
   matching the real session exactly.
4. **`--debug` is required** for `weston-screenshooter` to be authorized;
   without it capture returns "unauthorized" and a black image.
5. **Missing Weston fails loudly.** The launcher never degrades to the
   offscreen platform, because an offscreen run labelled as compositor
   evidence is exactly the vacuous oracle law #5 forbids.

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
`driver_failures`. Screenshot capture perturbs its timing bars, so
performance-only nonzero exits remain diagnostic inside this ring. Correctness
bars, unsettled/stale final state, stall-tile probes, tracebacks, completion
timeouts, and the 180 s whole-process watchdog are blocking. The matrix keeps
its output oracles as the primary verdict, but it may not report green after a
driver has already observed stale pixels or incomplete product state.

Visual-timeline records name their screenshot source. `qt-window-grab` is a
full Qt-window capture. Native-Wayland WGPU cannot be captured by Qt, so its
portable fallback is explicitly named `wgpu-offscreen-replay`; that replay is
useful command/pixel evidence but is **not** compositor or full-window truth.
WGPU draw acknowledgements may be much more frequent than a compositor grab;
the probe therefore applies `--screenshot-interval-s` to acknowledgement
captures too. Periodic WGPU trajectory samples use the explicitly labelled
`wgpu-offscreen-replay`; they remain synchronous so each image and its trace
metadata describe the same scene state. For a WGPU screen-path run with
`--screenshot-dir`, the profiler owns one private headless Weston compositor:
it sizes the sole output to the session's window, launches the child with no
panel and no window decoration, captures that output once with
`weston-screenshooter` for each phase-end exact-window image, then closes
Weston and removes its private socket, config, and capture temporaries.
Callers do not provide a screenshot helper or pre-start a compositor. A
replay fences the GPU and nested composition can perturb settlement, so
timing from any
screenshot-enabled run is diagnostic: use a trace-only repeat to decide
whether a photographed coverage delay belongs to the renderer or to capture.
The managed run fails loudly when Weston, `weston-screenshooter`, the private
Wayland output, or its exact window geometry is unavailable; it never labels a
Qt grab or renderer replay as compositor evidence.
WGPU timeline tile counts and geometry come from the committed executor
instances whose page-table spans are currently resident; an empty physical-row
set is a failing diagnostic, not evidence that a visibly populated frame is
empty.

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
  `arrayscope/tools/framebuffer_reference.py`; default-ring smokes are
  `tests/ui/test_framebuffer_cpu_reference.py` and
  `tests/ui/test_pyqtgraph_raster_cpu_reference.py`.
- The shared framebuffer oracle also reads WGPU's physical executor target.
  `test_cropped_display_axis_scroll_keeps_complete_montage` applies it after
  rapid displayed-axis crop churn and X/Y swaps, so current lifecycle labels
  cannot hide stale page texels. The profile crop matrix separately gates the
  cold crop-local identity that becomes relevant under page pressure.
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
