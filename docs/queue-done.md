# Queue — Done ledger

Completed steps from [queue.md](queue.md), most recent first, one line each with
evidence links. Moved rows land at the top of the Done section below. Narrative
blocks graduated during cleanups are preserved after it.

## Done (most recent first — one line each, evidence linked)

- 2026-07-22 — **Exact semantic-evidence sparse-read hotspot FIXED:** point/slice
  selectors now collapse axes before sparse image gathers; an isolated
  production-shaped 272-source sweep fell to 0.16 s. See the
  [compression follow-up](reviews/2026-07-22-compression-live-benefit-review.md).

- 2026-07-22 — **G7 live compression benefit CLOSED — measured NO:** the
  [matched live and architecture review](reviews/2026-07-22-compression-live-benefit-review.md)
  found 8.5–138× slower **16-page synchronous** submission, 8.4–42.5× slower compressed-source
  LOD, 9–18% larger configured raw+codec pools, synchronous host-cache work at
  the wrong miss owner, and no established capacity/transfer bottleneck.
  Correctness, alias, one-budget, physical-truth, and byte-accounting defects are
  repaired. Cross-session retained payloads are now byte-bounded, while the
  evaluator display cache, exact ROI-demand cache, and selected backend's
  physical storage remain correctly distinct. Production defaults are RAW/OFF.
  Follow-up added optional GIL-free Numba encoding, removed the full raw+codec
  allocation mirror, and timed physical tile presentation separately from
  evidence settlement. AUTO is still slower and a framebuffer counterexample
  keeps the safe gate at 40 dB. Codec paths remain explicit experiments until
  one physical byte cap and retention telemetry show prevented eviction/re-upload,
  or another real transfer bottleneck is established.

- 2026-07-22 — **Montage-relevel stall (the "reds") FIXED — pyqtgraph level-only
  fast-path (`cca02e74`):** the profiler-gated fix corrected the diagnosis (the
  ~1.2 s/commit cost was the backend re-resolving ALL 272 resident payloads each
  level commit, not the O(N) prioritization). `build_tile_presentation` now flags
  a `level_only_drain` delta when every upsert is a rewindow of an already-
  resident/presented tile; the pyqtgraph backend then resolves ONLY the committed
  slice and skips the cold-deadline bail → **~1.2 s → ~73 ms per commit (17×),
  12/12 committed**, and `fft_level_refinement_preview` now SETTLES
  (`level_stale`→0, 272/272, zero TimeoutError, well under the 5 s budget;
  verified trace-only, 3 runs). Correctness unchanged; vispy/wgpu untouched.
  Residual: the `gui_callbacks_below_50ms` bar on that phase stays red — but that
  is pre-existing/cross-backend (wgpu fails it too; ~4 s full-frame composite
  windowing), a parked "merely-slow" item (#1), NOT the stall. Owner chose this
  fast-path over accept-slow / GPU-only. Red-first tests in
  `tests/ui/test_frame_session.py` + `tests/window/test_montage_backend.py`; full
  suite 2837 passed. Supersedes the diagnosis in
  [`redesign/pyqtgraph-level-convergence-2026-07-22.md`](redesign/pyqtgraph-level-convergence-2026-07-22.md).

- 2026-07-22 audit note — the following G7 entries are **component history, not a
  completed live-benefit win**. The matched review found hidden cold/LOD costs,
  duplicate physical pools, and host-budget/alias defects. The product gate
  completed honestly with the NO above.

- 2026-07-22 — **G7 Phase B — native BC/ASTC compressed texture components
  transfer + VRAM win (`27ac87e9`):** the GPU's texture sampler decompresses
  BC/ASTC in hardware for free at sample time — no decode pass. Our own BC4
  (scalar) + BC5 (complex, stored as **(real, imag)** to match `rg32float` and
  avoid ±π phase-wrap smear) CPU encoders + a **WGSL on-GPU BC4 encoder** (for
  compressing GPU-derived tiles on-device), plus ASTC via `astc_encoder` for the
  Intel block-size quality knob. Topology policy (`decide_texture_codec`):
  discrete→BC, integrated→ASTC-or-BC; **default OFF**, wgpu render path
  byte-identical. **Measured on real hardware** (`tools/g7_gpu_texture_benchmark.py`):
  NVIDIA A2000 BC4 scalar **8.0× VRAM/transfer @ 40.7 dB**, BC5 complex **8.0× @
  41.5 dB mag / 0.027 rad phase**, GPU-encoder == CPU quality, hardware-sampler
  == CPU-decode oracle (57.5 dB); Intel ASTC-6×6 **8.9× @ 42.0 dB** (4×4 → 53 dB).
  Quality measured in the DISPLAY domain (post window/level mapping). Full suite
  2857 passed; real-GPU BC tests pass on the A2000. Later commits wired the live
  pools, but the audit shows this did not yet create an end-to-end win.

- 2026-07-22 — **G7 Phase A — compressed host-cache component, topology-aware
  (`6e767001`):** two-level cache (large compressed backing tier under the raw
  cache via a default-None `BoundedCache.on_evict` hook — byte-identical for
  existing callers); a raw-cache miss is served by a decode (µs) instead of a
  recompute/re-read (ms–s). `device_topology.py` detects integrated/discrete +
  unified memory from the wgpu adapter (confirmed: Intel UHD = integrated/unified);
  `cache_policy.py` engages the tier only under RAM pressure, best lossless codec
  per dtype, off for small data, with a `discrete_transfer_candidate` seam for
  Phase B. Measured (real 336³): 2.1–2.2× compression, **40→91 chunks per RAM
  budget** in the original model. The audit later found raw/tier overlap was
  double-counted and the synthetic FFT did not belong to the live display-cache
  miss; treat the 2.0–2.7× number as superseded, not product evidence.

- 2026-07-22 — **G7 compressed transport — codec + benchmark, default OFF
  (queue step 4):** `arrayscope/gpu/chunk_codec.py` (raw/zfp/blosc2, lossless by
  default, dtype-driven `resolve_codec` with safe fallback so a lossy request or
  unsupported dtype can never silently lose pixels); `chunk_transport_codec`
  setting defaults to `RAW` and the wgpu upload path is untouched (production is
  byte-identical). Benchmark `arrayscope/tools/g7_transport_benchmark.py` on real
  data: compression is 1.7–2.6× but the inequality (compress+transfer+decompress
  < raw) **does NOT hold for CPU decode** (break-even ~0.01–0.04 GB/s vs ~12 GB/s
  PCIe), so **default correctly stays off** — the gate ("prove before flipping")
  satisfied with an honest "no CPU-decode win; the transfer win needs GPU-side
  decode (nvCOMP/wgpu compute) — recorded follow-up." Evidence: `3450b75f`;
  lossless-exactness + default-off tests, 121 gpu+import-health passed.

- 2026-07-22 — **Plugin ops v3 — BART subprocess pack (queue step 10):**
  `bart:fft`/`ifft`/`cabs` run as ops via a self-contained cfl temp-file handoff
  + `bart <cmd>` subprocess (working child env: BART_TOOLBOX_PATH + MKL);
  cancellation SIGTERMs the child's process group then SIGKILLs after a 0.25 s
  grace — **measured 22 ms kill** (<1 s gate), no orphan, temp dir always cleaned;
  OPAQUE heaviest-admission cost class. `bart:pics` deferred (multi-input, doesn't
  fit unary `fn`). Evidence: `6973c134` (+ arch-guard barrier fix `4c7ad83f`);
  16 tests run against the live bart binary. bart installed at `~/projects/bart/`
  (MKL from the conda pkgs cache).

- 2026-07-22 — **Plugin ops v2 — sigpy pack (queue step 9):** first shipped as
  `sigpy:fft`/`ifft` (`2e4c7202`), then **removed the same day** (`3d38bce7`) as
  redundant — ArrayScope already has the built-in `centered_fft`/`centered_ifft`
  ops *and* a pluggable FFT backend, and `sigpy.fft` is `numpy.fft` underneath, so
  the sigpy FFT added nothing (see docs/graveyard.md). What **did** durably land
  from this step is the first-party pack-registry seam (`register_pack_operation` /
  `load_operation_packs` / `_PACK_MODULES`), reused by the BART pack; dock/palette/
  export enumerate via `all_operations()`. **Superseded 2026-07-24** by the sigpy
  *threshold + resize* pack (`0a09d83e`) — see the entry below — which ships the
  genuinely-additive sigpy ops (`sigpy:soft_thresh`/`hard_thresh` as verified
  Tier-2 windowable, `sigpy:resize` as OPAQUE k-space zero-fill) instead of the FFT.

- 2026-07-24 — **Plugin ops v2 — sigpy threshold + resize pack (supersedes the
  removed fft pack):** reinstated an optional in-process sigpy pack
  (`arrayscope/operations/packs/sigpy_pack.py`) with the ops sigpy does that the
  13 built-ins do not — deliberately **no FFT**. `sigpy:soft_thresh` /
  `sigpy:hard_thresh` are strictly pointwise complex thresholds (MRI
  sparsity/denoise views), declared **Tier-2 `region_capable=True`** and *honored*
  by the conformance harness — the first pack ops to exercise the honored Tier-2
  fast path (BART is all OPAQUE). `sigpy:resize` is centered zero-pad/center-crop
  of one axis (k-space zero-fill; additive over the shrink-only built-in `crop`),
  shape-changing → Tier-1 OPAQUE. Numeric-precision: sigpy's thresholds always
  return complex128, so the ops narrow back to the input dtype (float32/complex64
  discipline); the narrowing is pointwise so it preserves the windowable property.
  Enabling change: added a `"float"` parameter kind (the `lamda` threshold) beside
  `"int"` in both create paths. Deferred (unchanged): `nufft`/`espirit`
  (coordinate/calibration args + dim changes), and now `fwt`/`iwt` (the wavelet
  inverse needs the forward's oshape + coeff_slices, which the scalar-param unary
  contract cannot carry). Evidence: `0a09d83e`; targeted operations suite green
  (56 passed incl. the 20 pack tests). sigpy pip-installed + declared as the
  optional `sigpy` extra (pyproject) and mirrored in environment.yml.

- 2026-07-22 — **Montage-relevel "red" DIAGNOSED — pyqtgraph throughput fork,
  not a bug (reds investigation):** the `level_stale` timeout is real and
  reproducible (trace-only, both harness backends) but is NOT a convergence
  defect — pyqtgraph bakes levels into pixels, so relevelling 272 complex FFT
  tiles is ~180 ms/commit of fixed O(N) overhead ≈ 45 s vs the 5 s budget. vispy
  passes the level phase (~447 ms, GPU-uniform levels); wgpu is 6/6 in the matrix.
  So the red is confined to the CPU-windowing (headless/remote) backend and does
  not affect the GPU backends. It is an ADR-level fork (level-only fast-path vs
  accept-slow vs VisPy/wgpu-only) — full diagnosis + fix locations in
  [`redesign/pyqtgraph-level-convergence-2026-07-22.md`](redesign/pyqtgraph-level-convergence-2026-07-22.md).
  No engine change committed (awaiting the fork decision).

- 2026-07-22 — **Compare v1c — open A−B as a third linked window (queue step 7
  complete):** `open_difference_window()` builds `CompositeArraySource(A.base_data,
  B.base_data, op="subtract", own_inputs=False)` (wrapped in `LazySourceArray`
  per the file-source precedent), opens it as a third window linked into the same
  dims+camera+levels group, rendering progressively through the unchanged pipeline
  (region-only `read_region`, no whole-array materialization). Lifecycle: the
  `own_inputs=False` guard means closing A−B never tears down the still-live A/B
  sources; the linked cursor now reads via `read_region` (also fixes lazy/memmap
  compare windows). Menu: "Open difference (A − B)…". Evidence: `d4ca8cac`;
  real-Wayland `tests/ui/test_compare_difference.py` 4/4; progressive spy proves
  region-only reads + exact vs `A_np − B_np`; mismatched-shape refusal + lifecycle
  tests; full parallel suite 2752 passed. **Compare v1 (steps 5+6+7) is complete.**

- 2026-07-22 — **Test isolation: QSettings hermetic under serial `-n 0`
  (standing lane):** `tests/ui/test_diagnostics_dialog.py` had 4 tests failing
  only under `-n 0` because pytest-qt's empty `organizationName` made `QSettings`
  merge the developer's real `~/.config/Unknown Organization.conf`
  (`image_rendering_backend=wgpu`), which `.clear()` cannot reach, so serial runs
  built a WgpuSurface instead of pyqtgraph. Fix: `tests/conftest.py` now redirects
  `XDG_CONFIG_HOME` to a private empty dir for EVERY run (per-process for serial),
  not only xdist workers. Not a production bug (the app sets a real org name).
  Evidence: `aa597939`; serial+parallel 14/14, fail-then-pass proven, full suite
  2752 passed.

- 2026-07-22 — **Compare v1b — "Compare with…" launcher + linked complex cursor
  (queue step 6):** `CompareLauncherMixin.open_compare_window` opens an
  in-process sibling pre-linked on dims+camera+levels (reusing the sync
  controllers, no new transport); an in-process `CompareCursorGroup` shares the
  source array index so every window's HUD shows A and B (magnitude+phase for
  complex) read exactly from each window's own `base_data` — rides the existing
  hover-refresh path, no new scheduler. Evidence: `cca17a4c` (+ test-hygiene fix
  `7fbcfc2d`); real-Wayland `tests/ui/test_compare_launcher.py` 2/2 headless-weston,
  values exact vs NumPy oracle (float + complex64); full parallel suite 2746
  passed. **Integration bug caught + fixed:** the compare test leaked its sibling
  window (app-global retention list, no `WA_DeleteOnClose`) so a later test's
  nested event loop segfaulted in `_release_reference`; fixed test-side by fully
  disposing windows (close→drop retention→deleteLater→wait on `destroyed`).
  Semantics note: the cursor reports raw `base_data`, not post-operation values
  (v1). The camera facet is coupled to the existing "Sync window/level" link.

- 2026-07-22 — **Plugin ops v2 — Tier-2 conformance harness (queue step 9,
  dependency-free half):** `PluginOperationSpec.region_capable` opt-in; a region
  (windowable) claim is honored only after `plugin_conformance.verify_region_conformance`
  property-tests `fn(whole)[region] == fn(whole[region])` across seeded probes,
  else it is downgraded to OPAQUE whole-array with a loud WARNING +
  `region_conformance_stats()` tally. Evidence: `800efe4c`;
  `tests/operations/test_plugin_conformance.py` red-first (mis-declared roll /
  global-mean rejected, honest `x*2+1` honored) + non-vacuity proof; 303
  operations+import-health passed. **Remaining (dep-blocked):** the sigpy
  operation pack — sigpy is not installed in this environment.

- 2026-07-22 — **Compare v1c core — CompositeArraySource (queue step 7, core):**
  `CompositeArraySource(A, B, op="subtract")` is a pure `ArraySource`
  (shape/dtype/ndim/read_region), reads the same `index_spec` from both inputs,
  applies the op region-only, and propagates the cancellation token — so A−B
  flows through the unchanged unary pipeline/tile engine. Evidence: `39d313d8`;
  `tests/core/test_composite_array_source.py` 25 tests (NumPy-oracle exactness,
  region-only spy sources, token propagation, progressive-input streaming);
  203 core tests passed. **Remaining:** opening A−B as a third linked window,
  building on the step-6 launcher (lifecycle: must NOT close shared A/B sources).

- 2026-07-22 — **Compare v1a — camera/viewport sync facet (queue step 5):**
  new `FACET_CAMERA` in `sync/messages.py` (world/data-space view range,
  JSON-plain) with symmetric `camera_state_payload`/`merged_camera_state`;
  `WindowSyncController` publishes on the ViewBox `sigRangeChanged` through the
  existing leading-edge+coalesce `schedule_publish` (no new scheduler/timer) and
  applies peer ranges via `setRange` guarded by `view_ranges_near`, with echo
  suppression by the existing `_applying` window; the "Sync" toolbar link now
  couples window/level **and** pan/zoom. Evidence: `30b2d25e`; real-Wayland
  `tests/ui/test_window_sync.py::test_camera_pan_zoom_syncs_between_windows` +
  `::test_camera_apply_does_not_echo_a_republish` (2/2 headless-weston);
  `tests/sync/test_messages.py` round-trip/keep-current/non-finite pins; full
  offscreen suite 2708 passed.

- 2026-07-22 — **Plugin ops v1 — Tier-1 registry (queue step 8):** entry-point
  group `arrayscope.operations`, entry-point name = namespaced stable id
  (must contain `:`), lazy `entry_point.load()` on first use + cache;
  un-namespaced/built-in-collision/duplicate ids rejected loudly. `PluginOperation`
  wraps a pure `fn(ndarray)->ndarray` (or `build` factory) as an `OPAQUE`
  whole-array, cache-stage-able step reusing the existing region engine; recipe
  round-trip via the namespaced id, uninstalled ids raise a clear caught error.
  Evidence: `327aa80f`; `arrayscope/operations/plugins.py`,
  `tests/operations/test_plugin_operations.py` (12 tests, in-test fake entry
  point with real `EntryPoint.load()`), lazy proof in
  `tests/app/test_import_health.py`; `docs/plugin-operations.md` is the author
  contract; operations+import-health 293 passed, ruff clean.

- 2026-07-22 — **Progressive-load publication correctness — residual closed
  (standing lane):** the real-Wayland gate the offscreen atomic-read test
  (`447cbe42`) could not see now exists —
  `tests/gpu_interaction/test_progressive_open_reference.py` drives the
  production window over a `ProgressiveArraySource` on VisPy (real GL) and
  PyQtGraph (real Qt raster), proving zero unread/zero-fill regions at
  completion with a truth-anchored full-coverage gate, plus a red-then-green
  fault-injection (a torn/partial fill fails the gate) that also exposed that
  the payload pixel-oracle passes *vacuously* on an unread tile. Evidence:
  `db1c5393`; ring-4 4/4 under headless-weston.

- 2026-07-22 — **Render crash fixed: windowability anchored on base shape
  (found while integrating step 5):** `source_anchoring_for_view` fed the
  post-operation display shape into `pipeline_windowable_display_axes`, so a
  reduction on a non-display axis (e.g. Mean over the slider axis) raised
  `axis 2 out of bounds for 2D data` and crashed the render. Now passes the
  document base-data shape. Evidence: `14dc44a7`; two UI sync tests red on
  `main` → green, new
  `tests/display/test_source_anchoring.py::...::test_reduction_on_a_non_display_axis_does_not_raise`.

- 2026-07-19 — **Kernel completed-key memory bounded (redesign-R1 TODO
  close-out):** `Kernel._completed_keys` now maps key → completion scope;
  `clear_scope` purges the cleared scope's entries (a cleared result no
  longer satisfies a later dependency) and `forget_results(prefix)` releases
  completion memory for long sessions without touching staleness. Evidence:
  `tests/kernel/test_kernel.py` pins non-monotonic growth across scope
  clears, post-clear dependency parking, and scoped/inline forget;
  kernel/scheduler/pipeline/montage-residency rings green.

- 2026-07-19 — **wgpu SCREEN presentation landed (ceiling program step 2 —
  the tax flip):** `wgpu_present_method` setting (bitmap default, screen
  opt-in) drives a paint-less native child with its own Mailbox swapchain
  (`arrayscope/display/backends/wgpu/screen_canvas.py`, gate-B recipe,
  rendercanvas bypassed; 30 fps draw pacing load-bearing); acks key on the
  real present edge; harness screenshots read the physical framebuffer
  (widget grabs are blind to swapchain pixels). Evidence: screen-enabled
  journey matrix all five wgpu rows green
  (`tests/artifacts/journey-matrix-wgpu-screen-2026-07-19/`), contract
  suite + `wgpu-screen` twin 44/44 on Wayland, ring-4 screen gate 5/5,
  paired perf fast-scroll p95 118.0 → ~108 ms, zoom/pan parity
  (`tests/artifacts/wgpu-screen-perf-2026-07-19/`; endpoint entry 4,
  ADR 0057).
- 2026-07-19 — **codex post-merge review of the perf stack: two real-risk
  defects fixed red-first, shutdown contract pinned.** (1) Coverage-evidence
  bypass: a COLD wgpu histogram obligation deferred during interaction could
  close COVERAGE evidence-empty; the quiet-edge forced commit then ran in
  REFINE, where dispatch is gated off, and the rough histogram never ran.
  Both deferral sites (frame_effects evidence configuration, level_stats
  interaction bail) now arm the phase owner's coverage-evidence barrier so
  the phase holds until the quiet-edge commit re-dispatches; end-to-end pin
  (real policy + real deferral + real quiet-edge replan):
  `tests/window/test_montage_backend.py::test_deferred_cold_histogram_obligation_holds_coverage_and_dispatches_on_quiet_edge`.
  (2) Lost wakeup #7 (reproduced by the review): cancel/drop of the LAST
  visible item never ran the parked-quota release only completions perform,
  stranding optional work (`wait_idle` false) until an unrelated quota
  transition; the kernel now shares the completion wake edge on the
  cancellation/drop paths with a loud `kernel_wake_edge` trace
  (`tests/kernel/test_kernel.py::test_cancelling_last_visible_item_releases_parked_optional_work`
  + scope-clear variant). (3) Shutdown contract pinned: queued records now
  DELIVER `CANCELLED` completions at shutdown — every admitted task owes
  exactly one terminal completion for its `on_stale` cleanup owner; the Qt
  bridge closes before kernel shutdown by design, so delivery is the
  kernel's contract and draining is the consumer's
  (`tests/kernel/test_kernel.py::test_shutdown_delivers_cancelled_completions_for_queued_cleanup_owners`).
  Perf behaviors survive: post-fix full suite 2491/0 (rebased over the
  exit-gate landing), two real-Wayland journey matrices each all-five-wgpu
  green with only the standing incumbent cold_fill demand-freshness reds
  (`tests/artifacts/journey-matrix-2026-07-19-codexfix{,-v2}/`), wgpu
  fast-scroll p95 83.2/88.3 ms on warm repeats — inside the accepted
  77–100 ms band (`tests/artifacts/codex-review-fixes-2026-07-19/`).
- 2026-07-19 — **Last standing journey-matrix red closed: pyqtgraph
  cold_fill demand freshness ADJUDICATED = real product latency at the
  viewport bridge, fixed** (`6fd0c262`,
  [dossier](redesign/demand-freshness-cold-fill-2026-07-19.md)): the
  montage-entry auto-fit range change (~170 ms) was dropped by
  `ViewportBridge` (no committed tiled frame yet — the dead-gesture-edge
  live path), freezing LOD demand at the entry fit until the profile
  driver's fit pulse rescued it at 4.9–5.8 s under matrix load (straddling
  the 5 s budget; in the field nothing rescues it). Camera intent now
  defers and replays at the first commit teardown after the coverage pass
  closes (two-phase contract preserved; replay-at-first-commit was tried
  and rejected — it perturbed entry choreography). Companion fixes: the
  interaction-deferred wgpu histogram-evidence queue is pumped directly at
  the settle edge (deterministic offscreen wgpu cold stall — the barrier
  latched with an idle kernel because the pump lived only in the
  commit-ack path), and `selected_lod_factor` emits a permanent
  `lod_demand` transition trace. Follow-ups landed during acceptance:
  the intermittent zoom_in red proved to be a systematic ungoverned
  floor-progress commit — `tile_layer_upsert_limits` now governs commits
  while unsettled required targets exist (`b30d9940`, red-first pin) —
  and the freshness oracle reads the `lod_demand` transition timestamp,
  honored only when a later sample confirms the state stuck
  (`f2dbd556`, dual fault-injection pins: unconfirmed transition stays
  red, late transition stays red; sampler starvation during the gen-2
  replan burst was over-reporting by 350–900 ms). Evidence: suite
  **2488/0**; seven real-Wayland matrices
  `journey-matrix-2026-07-19-v1…v7` re-verified with the final oracle —
  **v5/v6/v7 = three consecutive 15/15** with the full stack (v1 also
  15/15; v2–v4's single red is the zoom_in commit pre-dating its fix);
  pyqtgraph cold ground-truth demand freshness 2.6–4.3 s (was
  5.25–6.3 s), first pixels ~350 ms; vispy cold margin widened to
  1.8–3.0 s (was 4.7). Residual lane = gen-1 coverage-close time itself
  (pyqtgraph fill speed, perf-bars territory).
- 2026-07-19 — **wgpu native overlay TEXT landed — last overlay gap closed,
  screen-present-mode experiment unblocked (ceiling program step 1):**
  CPU-baked glyph atlas (`arrayscope/display/glyph_atlas.py`; QPainter bake
  off the frame path, (font, pixel-size, glyph) cache with DPR in the key,
  bounded growth + loud `wgpu_glyph_atlas_evicted` eviction) sampled by new
  `glyph_quad`/`screen_rect` instances in the same flat instanced overlay
  pass (`UpdateGlyphAtlas` command, one `r8unorm` binding, nearest
  `textureLoad`); tile-truth labels are executor pixels in the wgpu view
  (QLabels replaced), world-anchored + constant screen size, camera-only
  pans move them with the image at zero atlas uploads/buffer rewrites;
  red-first oracles in `tests/display/test_glyph_atlas.py`,
  `tests/gpu/test_wgpu_command_protocol.py`, and
  `tests/display/test_wgpu_imageview2d.py` (ADR 0057 status updated).
- 2026-07-19 — **Wgpu interaction-path stalls and queued shutdown drain closed:**
  gesture histogram resolves are deferred to settle (`1fa2e0f2`), floor lookup
  is residency-epoch memoized with a stale-store guard (`f6a9e329`), and queued
  kernel work cancels under one global shutdown deadline (`112343f8`). Real-
  Wayland fast-scroll p95 is 100.8 ms (was 194–214 ms), all five Wgpu journey
  rows pass, and the close callback completes in 56.8 ms; whole-process exit
  on a current long worker item remains explicitly open above, as do the
  shared callback/heartbeat bars. [Promotion evidence](proposals/tensor-engine-endpoint.md#promotion-evidence-entry-2-2026-07-19--interaction-stalls-removed-at-their-owners).
- 2026-07-19 — **Wgpu/system interaction follow-through:** content-family
  plane indexing (`a8e0ee06`), quality-converged perf phases (`18986810`),
  wake-free unchanged governor quotas (`17dfb948`), and single-consumption
  viewport LOD decisions (`ca2b1846`). Every measured row finished 60/60
  exact with zero pending/stale levels; representative Wgpu fast-scroll p95
  reached 77.3 ms, while accepted repeat zoom/pan controls were 141.5–145.2
  ms. Final real-Wayland matrix: 14/15 overall, Wgpu 5/5 and VisPy 5/5; only
  the standing PyQtGraph cold-fill freshness row is red. Rejected variants
  are recorded in the graveyard. [Promotion evidence](proposals/tensor-engine-endpoint.md#promotion-evidence-entry-3-2026-07-19--quality-equivalent-system-pacing).
- 2026-07-19 — **PyQtGraph montage-entry blackout fixed (~7.5 s black → 2 s
  to first pixels, R8 continuity gate green):** three mechanisms — CPU first
  pixels accept a provisional refined first evidence batch instead of the
  full population (`tile_layer_first_pixels_wait_for_level_source`,
  `_publish_first_cpu_histogram`; contract point 6 amended in
  [rendering.md](architecture/rendering.md)); visible-dependency evidence
  producers run at INTERACTIVE/rank-0 instead of queueing behind the very
  fill they gate (`FIRST_PIXEL_EVIDENCE_SCHEDULING_RANK`); and
  `plan_presentation_transition` keeps a settled predecessor visible as an
  honest **montage-axis bridge** (never atomic, survives pixel-less rebirths
  via `presentation_bridge_pending`). One refined levels/histogram update at
  sweep completion — no per-batch re-bake. Suite green (one pre-existing
  wgpu-test guard red, chip out); real-Wayland matrix 14/15 with NO new
  reds and pyqtgraph cold_fill first pixels 353 ms (was ~1 s) —
  the surviving cold_fill red is the incumbent demand-freshness lane at
  incumbent magnitude (blackout-as-its-identity refuted). Details in the
  [fill-throughput dossier](redesign/fill-throughput-2026-07-18.md)
  follow-ups.
- 2026-07-18 — **G6(d) live GPU LOD generation on wgpu:**
  [`GenerateLodPages`](../arrayscope/gpu/command_protocol.py) runs one
  disjoint-subresource 2×2 component-mean pass per parent, recursively builds a
  requested coarser page from already-resident finer pages, and binds only the
  destination identity into the executor page table. The live view substitutes
  that path for a texel upload; physically cold targets and non-`mean` families
  stay on the CPU path. Complex values reduce real and imaginary components
  (never magnitude), and bound planes isolate reducer families. Default-ring
  oracles: [`test_wgpu_command_protocol.py`](../tests/gpu/test_wgpu_command_protocol.py)
  and [`test_wgpu_imageview2d.py`](../tests/display/test_wgpu_imageview2d.py).
- 2026-07-18 — **G6(b) rough→refined collapse rejected by measurement:**
  timestamped resident-page evidence on the real T2 dataset, offscreen and in a
  real Wayland window, found the native-L0 candidate heartbeat-safe but absent
  from the montage phase-1 L2 frontier; exact GPU work was 3.7× rough at 60
  tiles and 2.3× at 272. The existing quality machinery and phase-2 refinement stay
  unchanged; benchmark artifacts:
  `tests/artifacts/g6b-histogram-collapse-2026-07-18/`.
- 2026-07-18 — **wgpu live backend COMPLETE (row 3 slices a–c):** executor
  grown to multi-plane sessions + per-representation pools (`d675d57a`);
  live viewer commits every payload shape — scalar/complex/RGB8/windowable-
  float-RGB montages, all mapping modes and scales, LOD ladder from physical
  reduction factors — with ZERO remaining `_wgpu_commit_plan` rejections and
  physical page-table acks throughout (`50ab831d`, `cf00ae7b`/`d286b135`,
  `657e4a34`, `77bb9fee`); journey/profile harness registration + real-
  Wayland v3 matrix all-five-green (`a2568b52`); settlement livelock fixed —
  retained-fallback presentation restores first-pass coverage evidence
  WITHOUT acknowledging exact targets, and the 4 s offscreen 60-tile repro
  settles 60/60 (`14633cd0`/`6670b9df`); v7 independent matrix: cold_fill
  passes outright, 4/5 rows green, zoom_out under adjudication (resolved
  harness-gap — entry above).
  Suite 2422/0. Three field defects were found ONLY by the journey matrix
  (warm-arity seam `df4f6286`, ladder-factor mismatch, settlement livelock)
  — every fix strengthened oracles, none weakened.
- 2026-07-18 — **G6(a) live GPU histogram evidence on wgpu**
  (`c875a121..04977931`): committed planes dispatch dynamic-bound histograms
  over their exact resident ADR 0056 frontier, completion-token fenced, into
  the existing level/first-pass machinery via a coverage evidence barrier
  INSIDE `ProgressiveSchedulingPolicy` (no GUI aggregation, no parallel
  scheduler); superseded task results are ABSORBED content-keyed with
  refined-first cache reuse instead of discarded (`9f8b3970`, Thomas's
  design — cancellation cancels presentation ownership, never useful
  computation); every queue bail is a loud trace event (`7567fb3a`).
  G6(c) RESOLVED: keep the PyQtGraph histogram widget (`03b46d88`).

- 2026-07-18 — **Renderer command protocol (ADR 0057) + wgpu executor seed
  landed:** `arrayscope/gpu/command_protocol.py` (the only renderer seam) +
  `WgpuPlaneExecutor` carrying the gate-B oracles as default-ring tests
  (14 green: zero-upload mode/levels/shift/scroll, pinned-ancestor fallback,
  exact histogram, completion token), plus `arrayscope.tools.wgpu_preview`
  rendering the real T2 NIfTI through protocol commands in a bitmap
  QRenderWidget with a live Qt overlay (smoke artifact:
  `tests/artifacts/wgpu-gate-b/preview-smoke.png`).
- 2026-07-18 — **Renderer runtime experiments CLOSED — verdict: wgpu GO.**
  Datoviz gate A (branch `codex/datoviz-v04-renderer-gate-a`): overlay
  composition + upload lifetime FAIL. wgpu gate B: all tiers pass —
  native-Wayland screen presentation from pure Python; bitmap = default
  composition mode (4 ms p50 @1300×650, overlays intact; 26 ms @4K is its
  boundary); virtual tensor pixel-exact with zero-upload interactions;
  exact 2-pass histogram 3.2 ms; 16-page burst 2.6 ms; completion token
  0.19 ms; mappable-primary 12.4 GB/s. Verdict tables in
  [tensor-engine-endpoint](proposals/tensor-engine-endpoint.md); plan +
  evidence in [wgpu-renderer-experiment](proposals/wgpu-renderer-experiment.md);
  artifacts `tests/artifacts/wgpu-gate-b/`.
- 2026-07-18 — **Camera re-anchor order retained end to end:** priority
  retargets atomically rebuild the kernel ready heap for unstarted session
  tasks, and the final presentation boundary reorders delta upserts against
  the same current-camera context before either backend sees them. The
  journey oracle consumes that immutable rank snapshot per commit; VisPy
  zero-texture/zero-vertex-upload rebinds are explicitly item-cap exempt,
  while pixel uploads remain capped. Real-Wayland coverage includes the saved
  60-source VisPy predecessor through a 272-tile expansion and fit without a
  physically hidden frame.
- 2026-07-18 — **One COVERAGE→REFINE scheduling-policy owner:**
  `ProgressiveSchedulingPolicy` now owns the per-required-scope phase,
  lifecycle first-pixel close predicate, and refinement replan edge. Ladder,
  admission/lane quotas, level/histogram work, atomic handoff, and commit
  batching read its verdict; duplicate first-pass derivations and the
  PyQtGraph first-commit cap bypass are deleted. Red-first contract coverage:
  [`test_progressive_scheduling.py`](../tests/render/test_progressive_scheduling.py)
  and bounded PyQtGraph first-frame assertions in
  [`test_montage_backend.py`](../tests/window/test_montage_backend.py).
  Live evidence: real-Wayland raw fills settled exact targets 60/60 on both
  backends with zero `trace_verify` violations; VisPy submitted zero phase-2
  jobs before lifecycle coverage close; ring 3 4 passed/1 skipped/1 xpassed,
  ring 4 20/20 passed.
- 2026-07-17 — **Output-driven journey matrix delivered** (standing lane):
  `{cold fill, zoom-in, zoom-out, scroll shuffle, index scroll}` × both
  backends now records gesture-scoped JSONL + screenshot timelines and gates
  phase ordering, bounded priority-ordered commits, camera-demand freshness,
  first-pixel latency, and post-coverage LOD convergence. Every oracle has a
  fault injection; `trace_verify` independently rejects phase-2 submission
  during coverage. The first real-Wayland run is intentionally red and
  mechanically exposes the open 2026-07-17/18 defects (artifact:
  `tests/artifacts/journey-matrix-2026-07-17-v3/`). Contract and pre-merge
  command: [testing/README.md](testing/README.md#journey-matrix-trajectory-gate).
- 2026-07-17 — **PyQtGraph physical readback oracle** (standing lane):
  `tests/oracles/framebuffer_reference.py` now reads the painted Qt graphics
  viewport and compares every required scalar tile against
  `cpu_display_rgba` under semantic levels/LUT, with exact-set and sample-floor
  vacuity guards plus an independently calibrated 2/255 raster tolerance.
  Default-ring smoke proves exact-set/sample-floor/wrong-level failures;
  real-Wayland audit proves wrong levels, a stale cached tile `QImage`, and
  swapped physical tile positions each fail and recover. Evidence:
  `tests/ui/test_pyqtgraph_raster_cpu_reference.py` and
  `tests/gpu_interaction/test_pyqtgraph_raster_cpu_reference.py`; full
  both-backend real-Wayland ring 20/20 green.
- 2026-07-17 — **Framebuffer-to-CPU reference oracle + fault injection**
  (standing lane): `tests/oracles/framebuffer_reference.py` compares the
  live VisPy framebuffer per required tile against the CPU shader mirror;
  real-GL audit `tests/gpu_interaction/test_framebuffer_cpu_reference.py`
  (wrong uniform / stale atlas page / swapped tile each fail the oracle,
  restore turns it green) + default-ring smoke
  `tests/ui/test_framebuffer_cpu_reference.py`. Evidence: full
  `tests/gpu_interaction` ring 16/16 green on real Wayland 2026-07-17 —
  which also closes the "4 pre-existing P9-era baseline failures" row: none
  reproduce post-G5.
- 2026-07-17 — **ImageViewShell duplication lane closed**
  (`b657bb5d..d71d4c8e`): the shell is now the single owner of ROI/
  interaction emphasis, the tiled-commit skeleton, and tiled-layer queries;
  PyQtGraph tile mechanics moved to `ImageView2D` behind declared backend
  hooks; VisPy's seven override+mirror methods and its duplicate
  hover/selection owners deleted. Behavior-preserving refactor gated on
  ring 1 (full offscreen suite green); no ring 3–4 run — no rendering
  behavior change intended.
- 2026-07-17 — **`target_satisfied_retained` emitted in production** (standing
  lane): the lifecycle emits it once per target requirement closed by a
  retained compatible payload (retarget/ack/confirm edges + the settled
  noop-commit re-affirmation in `frame_effects`); `trace_verify` re-judges
  the edge with the production settlement rule (fallback needs strictly finer
  level); `TOLERATED_INVARIANTS` is empty — the strongest invariant
  (`final_required_target_acknowledged`) is enforced in the stress matrix,
  which passed 5/5 rows serially. Gates: `tests/core/test_trace.py`
  (`*retained_satisfaction*`, red-first).

- 2026-07-17 — **G5 merged to main** (`661b6ba5`): canonical source-grid page
  route, reducer families, page cache, both-backend consumers, legacy
  whole-plane ownership deleted with a resurrection guard; the progressive
  presentation contract (docs/architecture/rendering.md) is enforced at work
  submission (coverage before refinement, plan-wide);
  live evidence: PyQtGraph raw 272/272 in ~11.4 s coverage-then-refine,
  churn ring green, zero refinement-during-pass commits on both backends
  (tests/artifacts/g5-coverage-first-*-2026-07-17). Red and owned by the
  perf-bars program: 50 ms GUI-callback, VisPy 4.5 s draw settle.

- 2026-07-16 — **Churn-convergence stall net closed** (members 4+5 of the
  deferred-stage lost-wakeup family: stale-render-generation discard/resubmit
  livelock in stage-plan/stage-value completions; exact-pass candidacy
  starvation for non-exact payloads at the target level). Commit chain and
  stage-plan callbacks now emit loud bail/decision trace events; the live
  churn scenario converges 3/3 in ~23 s and its xfail is removed. Dossier:
  [stale-empty-tiles-2026-07-16](redesign/stale-empty-tiles-2026-07-16.md).
- 2026-07-16 — **Native complex64 PyQtGraph convergence restored**
  (`14f0fbc5`): the canonical native level-zero page route preserves complete
  handoffs, and the stress-matrix complex64 row is a hard pass. Re-verified
  serially on 2026-07-17 with 10/10 required targets acknowledged, zero
  identity-rejected commits, and no `trace_verify` violations.
- 2026-07-16 — PyQtGraph identity-rejected upserts made loud (`6f95ce70`).
- 2026-07-16 — Session-148 identity-aliasing follow-ups: canonical full
  ranges, per-tile ack-vs-target coverage, re-commit backoff, trace_verify
  invariant (`37979222`; dossier
  [stale-empty-tiles-2026-07-16](redesign/stale-empty-tiles-2026-07-16.md)).
- 2026-07-16 — Orange floor tiles (per-session preview metadata vs
  persistent pyramid cache) + three deferred-stage lost-wakeups (`18a207fb`).
- 2026-07-16 — Identity-aliasing starvation stall root-caused and fixed
  (`dff723b4`).
- 2026-07-16 — Montage scroll-direction GPU warming; retained-slice
  staleness fixed at the rung-label owner (dossier
  [slice-retention-staleness-2026-07-16](redesign/slice-retention-staleness-2026-07-16.md)).
- 2026-07-15 — G1–G5 slice 1 of the GPU engine landed and real-GL verified;
  physical presentation truth standing invariant; see
  [proposals/gpu-engine-plan.md](proposals/gpu-engine-plan.md) status and
  the [continuation brief](proposals/gpu-port-continuation.md).
- 2026-07-14 — V0–V4 visible-truth program closed and merged; execution
  record in
  [redesign/archive/v-program-execution-record-2026-07.md](redesign/archive/v-program-execution-record-2026-07.md).
- 2026-07-14/15 — P1–P9 measured performance program; log in
  [redesign/archive/p-program-log-2026-07.md](redesign/archive/p-program-log-2026-07.md);
  its open cause (per-step 60-slot rebind) is solved structurally by the
  GPU engine's chunked residency.


### Row-3 status narrative (moved from the Now table, 2026-07-19 cleanup)

Preserved verbatim; the condensed row in [queue.md](queue.md) points here.

| 3 | **wgpu strangler — promotion evidence** (slices (a)–(c) LANDED, see Done; ADR 0057). zoom_out full-matrix red ADJUDICATED 2026-07-18: **harness gap, closed** (`da22dad7`, see Done — the canvas repainted ~24 ms after the journey-end capture; sampler now drains presentation-draw acks before the end sample; three consecutive FULL Wayland matrices v10/v11/v12 wgpu-green). The 2026-07-18 field stalls `259-1`/`1-1` are **not** the standing 272-tile fill: both are one wgpu physical-first-pass quality drift (exact latch followed by a mixed exact+fallback snapshot), fixed in `43287f8` and recorded in [the field-stall dossier](redesign/wgpu-field-stalls-2026-07-18.md). **2026-07-19 dogfood crash (complex FFT chain, L2 view) FIXED:** a submission's own `EnsureChunkResident` commands could LRU-evict a page snapshotted for a later `DispatchHistogram` in the SAME submission; the executor's loud KeyError then aborted the whole commit mid-batch (ensures applied, present never ran). Two-layer fix: the executor pre-scans each submission and shields histogram frontier keys from its own eviction (scoped pin owner, always released per submission; pool pressure beyond the shield yields the page honestly), and a key missing at histogram time becomes `FrameReport.histogram_missing` — the view drops that evidence spec with a loud `wgpu_histogram_queue_bail reason="evicted_in_batch"` trace, the commit completes, and evidence retries via the normal re-queue machinery (design rationale in ADR 0057 status). **Dogfood overlay parity:** wgpu now draws camera-locked ROI outlines/handles, live-profile cursor geometry, and tile loading/skipped boxes + symbols natively after the tiles in the same render pass; camera-only frames update one transform uniform and never rewrite the world-anchored overlay buffer. **TEXT GAP CLOSED (2026-07-19):** overlay text is GPU-native — a CPU-baked glyph atlas (`arrayscope/display/glyph_atlas.py`, QPainter bake off the frame path, DPR-keyed cache, bounded growth with loud `wgpu_glyph_atlas_evicted` eviction) feeds `glyph_quad`/`screen_rect` instances in the SAME flat instanced overlay pass via one `UpdateGlyphAtlas` command; tile-truth labels render as executor pixels in the wgpu view (QLabels replaced), world-anchored with constant screen size, camera-only pans move them with the image at zero atlas uploads and zero buffer rewrites (`FrameReport.glyph_atlas_uploads` + border-corner pixel oracles in `tests/display/test_wgpu_imageview2d.py`, executor oracles in `tests/gpu/test_wgpu_command_protocol.py`). With Qt-widget overlays no longer required over the canvas, the screen-present-mode experiment was unblocked — and **SCREEN PRESENTATION LANDED 2026-07-19** behind the new `wgpu_present_method` setting (bitmap default; `auto` = screen exactly where the measured native-Wayland recipe applies; `screen` explicit pin — selectable from Performance → wgpu Presentation, enabled with the wgpu backend): a paint-less native child drives its own swapchain via the gate-B recipe (QNativeInterface wl_display + winId-as-wl_surface + Vulkan-only instance; rendercanvas fully bypassed on this path), re-configured for **Mailbox** where offered so the ~15 ms Fifo acquire block never reaches the GUI thread (measured steady-state acquire 0.09–0.16 ms, present 0.03–0.12 ms), draw-paced at rendercanvas's 30 fps default (unpaced glides exhausted the mailbox chain — ~1.5–2× worse zoompan event-loop p95 until the cap), with draw-acks keyed on the real `wgpuSurfacePresent` edge and loud bitmap fallback anywhere screen cannot exist. Native-child risks pinned in ring 4 (`tests/gpu_interaction/test_wgpu_screen_present.py`: input transparency, close-cancels-drag, present-edge ack drain, resize reconfigure) plus a `wgpu-screen` contract-suite twin (44/44 on real Wayland). Screen-enabled journey matrix: **all five wgpu rows green** (`tests/artifacts/journey-matrix-wgpu-screen-2026-07-19/`; the first run exposed that widget grabs are blind to swapchain pixels — harness screenshots now use the view's physical framebuffer readback). Paired same-tip perf: fast-scroll p95 118.0 → 107.1/109.5 ms (~8–10 % win, the readback tax); zoom/pan parity within the standing jank band — at this window size the 4–7 ms readback is a small slice of the shared row-1 presentation tail; the decisive screen case remains 4K (26 ms readback), still unmeasured (endpoint entry 4). Three consecutive full real-Wayland matrices v17/v18/v19 hold the text geometry on every sample and pass all five wgpu rows outright (including index 10/10); zoom-in/out remain zero-commit. **Codex post-merge review of this landing found two real-risk defects — the coverage-evidence bypass and lost wakeup #7 — both fixed red-first the same day with the kernel shutdown completion contract pinned (see Done); post-fix acceptance: full suite 2491/0 on the rebase over the exit-gate landing, TWO fresh full Wayland matrices (pre- and post-rebase) each all-five-wgpu-rows green with only the standing incumbent cold_fill demand-freshness reds (vispy+pyqtgraph), fast-scroll p95 83.2/88.3 ms warm repeats (in band; one post-matrix cold outlier 157 ms discarded per the load-variance protocol; artifacts `tests/artifacts/codex-review-fixes-2026-07-19/`, `journey-matrix-2026-07-19-codexfix{,-v2}`).** Open: **(d)** promotion by evidence — STATUS after the 2026-07-19 perf landing: wgpu LEADS on fast-scroll (p95 77.3 ms vs VisPy ~106-124) and matches on zoom/pan steady-state (141-145 ms repeat controls); both 5/5 in the final matrix. Remaining before the AUTO flip: the shared callback/heartbeat bars (row 1), Thomas's continued dogfood hours (screen mode now dogfoodable from Performance → wgpu Presentation, incl. Auto), and the FFT-scroll 4→17 fps headline measured on the new tip. VisPy retirement review only after a release cycle — never a flag-day switch. **Successor traps:** import rendercanvas ONLY via `import_qrenderwidget()`; every wgpu adapter probe pins `set_instance_extras(backends=["Vulkan"])` BEFORE its first request (an all-backends instance re-inits EGL during GL enumeration and SIGABRTs workers holding vispy GL state — `8c57a7bf`). | Promotion gate: journey matrix + perf bars on real data; written verdict in tensor-engine-endpoint.md |

### Graduated standing-lane items (2026-07-19 cleanup)

- **CLOSED 2026-07-18 — the 272-tile raw fill "stall" was a throughput
  collapse, not a lost wakeup.** Bisect verdict: no breaking commit in
  `661b6ba5..976ea275` — the timeout reproduces at the G5 merge and at
  `62904128` itself; the offscreen invocation was never validated green.
  Gate-liveness instrumentation (now permanent trace events
  `presentation_gate`/`replan_gate`/`commit_rearm`) proved every wakeup
  fires; the fill was O(tiles²)-slow against a mis-applied 5 s gesture
  budget. Fixed at the roots: page-set resolution memo per residency
  revision (pyramid.py), fixed-cost-amortizing idle commit cohort for the
  shader-windowing backends (frame_effects.py `_idle_backlog_cohort`),
  indexed floor lookups (lod.py), and a build-scale completion budget for
  the cold-fill harness stages (`COLD_FILL_BUILD_TIMEOUT_S`, churn-harness
  rule). Repro: vispy PASS 272/272 ~9.7 s, wgpu PASS ~10.9 s; pyqtgraph
  completes 272/272 (~24 s) but its R8 gate was red on a PRE-EXISTING
  montage-entry blackout (CPU windowing gated on the full level-evidence
  sweep — fixed 2026-07-19, Done entry). Journey matrix post-fix: cold_fill GREEN on all three
  backends (the standing vispy+pyqtgraph cold_fill reds were this
  mechanism); remaining reds are the demand-freshness family (pyqtgraph
  zoom_in stable, zoom_out flips backends run-to-run). Row-3(d) perf-bars
  measurement is UNBLOCKED. Dossier:
  [fill-throughput-2026-07-18](redesign/fill-throughput-2026-07-18.md).
- **PyQtGraph ungoverned early preview commit — FIXED 2026-07-19**
  (`b30d9940`): an early frontier-tile preview commit ran with
  `max_upserts=0` and no recorded `unbounded_reason` (journey zoom_in red
  in `journey-matrix-wgpu-2026-07-18-v19`/`-v11` and
  `journey-matrix-2026-07-19-v2/v3/v4`). Not a race: floor-progress
  commits carry no dirty/pending work at limits-decision time — the
  build's floor pass materializes preview upserts during assembly, so
  `tile_layer_upsert_limits`' early-`{}` gate systematically skipped
  them. The gate now also governs commits while unsettled required
  targets exist; red-first pin
  `test_pyqtgraph_floor_progress_commits_stay_governed`. zoom_in green in
  v5/v6/v7.
- **AUTO-camera demand freshness / dead gesture edge — LIVE PATH FIXED
  2026-07-19** (`6fd0c262`, [dossier](redesign/demand-freshness-cold-fill-2026-07-19.md)):
  the live-path form was `ViewportBridge.on_view_range_changed` dropping a
  montage range change whenever no committed *tiled* frame existed yet —
  exactly the montage-entry auto-fit, whose loss froze `session.view_range`
  and LOD demand at the stale entry fit and was the standing pyqtgraph
  cold_fill demand-freshness red (adjudicated REAL latency, not a sampler
  gap). Camera intent now defers to commit teardown and replays after the
  coverage pass closes. Remaining open item here: the unit gate's fixture
  (window carries no committed display frame, so the deferred obligation
  never replays) — red pin stays strict xfail with instrumented probes:
  `tests/ui/test_lod_demand_freshness.py`.
