# The queue — what to do next

**This is the only active queue.** If any other document claims to order
current work, that document is stale — fix it to point here.
[`roadmap.md`](roadmap.md) says *why* this order serves the mission;
this file says *what, in what order, and when it counts as done*.

**Rules for this file** (they exist because the last three queues drowned):

1. Update rows **in place**. When a step lands, move its row to *Done* below
   with one line of result + a link to the evidence. Never append status
   blockquotes or execution logs here — those go in the commit message, a
   dossier under [`redesign/`](redesign/), or a dated review.
2. Every step names its **exit gate in the ring that can actually see the
   failure** ([testing/README.md](testing/README.md)). "Code exists" and
   "offscreen suite green" are not completion.
3. A rejected/reverted attempt gets a [`graveyard.md`](graveyard.md) row in
   the reverting commit.
4. Re-order only with a stated reason in the commit message.

## Now (2026-07-16, in order)

| # | Step | Exit gate |
|---|---|---|
| 1 | **Performance-bars program on the engine** (parked — Thomas 2026-07-17: act only on true stalls/no-progress, never on merely-slow). The bars (below) are the product promise. One measured cause at a time, before/after real-Wayland harness evidence per commit; a step that regresses a bar is reverted and buried in the graveyard. | Bars trend green in `profile_montage_workflow` on real Wayland, both backends (PyQtGraph at 2× allowance) |
| 2 | **G6 — GPU histogram/levels** (slice 1 landed 2026-07-18: bounded source-weighted chunk summaries + ADR 0056 frontier feeding phase-1 rough evidence). Remaining slices per the endpoint scope in [gpu-engine-plan.md](proposals/gpu-engine-plan.md#g6--gpu-compute-consumers), now written ONCE against the renderer command protocol (ADR 0057) and executed on the wgpu executor — the gate-B verdict unblocked this (row 3 Done): **(a) IMPLEMENTED 2026-07-18; Thomas's real-Wayland acceptance remains:** wgpu committed scalar, complex, and windowable-RGB planes dispatch dynamic-bound histograms over their exact resident ADR 0056 frontier; the coverage worker fences the completion token and installs the bounded result through the existing level/first-pass machinery with no GUI aggregation, while incumbent backends retain CPU chunk summaries; **(b)** evidence-gated: collapse rough→refined into one exact pass IF measured inside the phase-1 budget; **(c)** RESOLVED 2026-07-18 (Thomas): KEEP the PyQtGraph histogram widget — pyqtgraph stays a dependency as the SW-render fallback backend regardless, the widget consumes ≤256 bins (~1 KB; draw cost irrelevant in any toolkit), and the efficiency that matters — levels/histogram COMPUTATION staying on the GPU — is slice (a), not widget rendering. The private-pyqtgraph-API exposure is accepted for now; a renderer-agnostic plain-Qt widget remains the recorded fallback design if that risk ever fires; **(d)** GPU LOD generation from resident chunks (protocol executor has the in-pool reduction pass). | Levels/histogram converge from chunk summaries; sampling measured ON the GPU; no GUI-thread aggregation; real-GL gate; journey matrix not regressed |
| 3 | **wgpu strangler integration** (ADR 0057; verdict + seed landed — see Done). Ordered slices, each behind tests: **(a)** DONE 2026-07-18 (`d675d57a`): executor grown to the live payload shapes — `BindContentPlanes` multi-plane sessions (GPU flat table/LOD spans rebuilt from the *bound* planes; unbound chunks stay warm in the PageTable, so plane rebind is physically zero-upload), honest pools per representation (scalar `r32float` — zero-imag waste gone; complex `rg32float`; `rgb8` `rgba8unorm` display-ready with levels/LUT bypass), per-pool budgets with same-pool LRU eviction of unpinned only (pinned-full and unbudgeted pools raise loudly), scalar planes ignore complex modes, `DispatchHistogram` over scalar pages; gate-B oracle set extended — tests/gpu, none weakened; **(b)** live viewer DONE 2026-07-18 (`50ab831d` + rejection lifts `cf00ae7b`/`d286b135` + LOD ladder `657e4a34`; final rejection lifts in this slice): montages of N scalar, complex, display-ready uint8 RGB, or windowable/float RGB tiles commit through per-tile `ContentPlane`s with dst rects from the shared `tile_layout_map`; complex magnitude/phase/real/imag, canonical cyclic phase hue × normalized magnitude, linear/log/symlog display scales, montage scroll/scroll-back, component/scale switches, and levels moves are shader/descriptor-only and physically zero-upload with independent CPU pixel mirrors. Windowable RGB deliberately matches VisPy's semantics — preserved color multiplied by one levels-normalized histogram/luminance plane, packed into one physical `rgba32float` page — rather than inventing per-channel levels. LOD-invariant per-plane keys place coarse payloads in higher-LOD pages behind native-coordinate tile requests with pinned fallback coverage through refinement (view fallback→refine oracle mirrors the executor oracle); acknowledgement is physical page-table truth per tile (one-layer scalar, complex, and windowable-RGB pool oracles prove partial residency acks only the resident subset). **No recorded `_wgpu_commit_plan` rejections remain.** Traps for successors: import rendercanvas only through `import_qrenderwidget()`; every wgpu adapter probe must call `set_instance_extras(backends=["Vulkan"])` BEFORE its first `request_adapter_sync` — an all-backends instance re-inits EGL during GL adapter enumeration and SIGABRTs workers holding vispy GL state (wgpu-hal panic `gles/egl.rs:305`, root-caused in `8c57a7bf`). *MVP: `WgpuImageView2D` + `WgpuSurface` stay behind an explicit settings pin; AUTO never resolves to wgpu; executor reports are the commit-stats oracle.* **(c)** DONE 2026-07-18: the recorded real-Wayland v3 matrix (`tests/artifacts/journey-matrix-wgpu-2026-07-18-v3/`) has all five wgpu rows green (`cold_fill` 1/1, `zoom_in` 1/1, `zoom_out` 1/1, `scroll_shuffle` 1/1, `index_scroll` 10/10), with no LOD-geometry or Vulkan validation exception. The incumbent comparison retains the standing VisPy cold coverage/level red and PyQtGraph zoom freshness red; one PyQtGraph cold freshness sample also went red under varying system/user load and is recorded as observed timing noise outside this wgpu-view slice. All nine non-zero driver exits are `diagnostic_only` and do not gate row truth. The fix derives executor LOD from physical payload reduction, bounds pool headroom by active residency/device limits, waits for physical endpoint draws, and caps only explicitly traced cold GPU tiles while resident rebinds remain atomic; **(d)** promotion decision by evidence: perf bars vs VisPy on real data, then VisPy retirement review after a release cycle — never a flag-day switch. | Each slice: default-ring tests + real-Wayland journey rows green; promotion gate: journey matrix + perf bars on real data, written verdict in tensor-engine-endpoint.md |
| 4 | **G7 — compressed transport.** Codec ladder, measured topology; ZFP-class first. After G6. | Measured end-to-end win on real data |

## Performance bars (commitments, not history — restored from R2/R4/R8D)

- GUI callbacks < **50 ms** always; event-loop heartbeat gap ≤ **16 ms**
  during fills and scrub; warm scrub input ≤ **15 ms**; settled-idle CPU 0%.
- **#1 throughput target:** fast montage FFT index scroll ~4 fps → toward
  the ~17 fps scalar rate (2026-07-09 measurement, realistic human scroll).
- Benchmark deltas stay within ±10% of the frozen baseline unless a step
  improves them. PyQtGraph gets 2× the VisPy allowance (it targets
  headless/remote use); both backends stay first-class for correctness.

## Standing lane — test hardening & debt (parallel-safe, any order)

Safe to pick up alongside the numbered queue; each is self-contained.

- **AUTO-camera demand freshness / dead gesture edge.** After montage
  entry, a programmatic `setRange` zoom invokes NOTHING in the viewport
  chain within 5 s — zero calls to `retarget_montage_viewport` /
  `apply_montage_viewport_retarget` / `_montage_viewport_plan`, identical
  offscreen and on real Wayland — so LOD demand freezes at the fit level.
  The post-policy journey matrix measures fresh demand on every live cell,
  so the unit gate's fixture (window carries no committed display frame) is
  a prime suspect; field zoom verdict pending. Red pin (strict xfail with
  instrumented probes): `tests/ui/test_lod_demand_freshness.py`.
- **Remove the `montage_key_batch_fallbacks` runtime guard** once the
  consolidated key owner is proven in the field. 2026-07-17: derivation is
  consolidated — every layout has one owner
  (`_display_tile_key_from_parts`/`_request_key_from_parts`/
  `_view_state_key_with_slices` in `evaluator.py`; the batch's slow path *is*
  `display_tile_key`) and parity + fallback are pinned in
  `tests/operations/test_cache.py`. The runtime guard and counter stay until
  a release cycle shows the counter at zero.
- **Audit `_resident_source_matches_expected(source, None) → True`**
  (controller-side expected-source coverage during session switches).
- **Upstream rendercanvas contributions** (from gate B): a native-Wayland
  screen-presentation hook (wl_display via QNativeInterface + winId-as-
  wl_surface, Vulkan-only instance) and making the import-time
  `QT_QPA_PLATFORM=xcb` override opt-out. Until merged upstream, ArrayScope's
  `qt_platform` policy owns the platform decision.
- **Screen-mode follow-ups, evidence-gated by 4K/high-res need** (bitmap is
  the default; its measured boundary is ~26 ms readback at 4K): Mailbox or
  off-thread acquire (Fifo acquire blocks the GUI thread ~15 ms/frame), and
  the GPU-overlay layer design that screen mode requires.
- **Renderer measurements not yet taken:** NVIDIA/discrete adapter cells for
  Tier 1/4 (PRIME copy changes upload and present arithmetic), real 4K
  swapchain. (The `winId == wl_surface*` per-Qt-minor pin is DONE:
  `tests/gpu_interaction/test_wgpu_native_wayland_pin.py`, ring 4.)

## Done (most recent first — one line each, evidence linked)

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
