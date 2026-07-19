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
| 3 | **wgpu strangler — promotion evidence** (slices (a)–(c) LANDED, see Done; ADR 0057). zoom_out full-matrix red ADJUDICATED 2026-07-18: **harness gap, closed** (`da22dad7`, see Done — the canvas repainted ~24 ms after the journey-end capture; sampler now drains presentation-draw acks before the end sample; three consecutive FULL Wayland matrices v10/v11/v12 wgpu-green). The 2026-07-18 field stalls `259-1`/`1-1` are **not** the standing 272-tile fill: both are one wgpu physical-first-pass quality drift (exact latch followed by a mixed exact+fallback snapshot), fixed in `43287f8` and recorded in [the field-stall dossier](redesign/wgpu-field-stalls-2026-07-18.md). **2026-07-19 dogfood crash (complex FFT chain, L2 view) FIXED:** a submission's own `EnsureChunkResident` commands could LRU-evict a page snapshotted for a later `DispatchHistogram` in the SAME submission; the executor's loud KeyError then aborted the whole commit mid-batch (ensures applied, present never ran). Two-layer fix: the executor pre-scans each submission and shields histogram frontier keys from its own eviction (scoped pin owner, always released per submission; pool pressure beyond the shield yields the page honestly), and a key missing at histogram time becomes `FrameReport.histogram_missing` — the view drops that evidence spec with a loud `wgpu_histogram_queue_bail reason="evicted_in_batch"` trace, the commit completes, and evidence retries via the normal re-queue machinery (design rationale in ADR 0057 status). **Dogfood overlay parity:** wgpu now draws camera-locked ROI outlines/handles, live-profile cursor geometry, and tile loading/skipped boxes + symbols natively after the tiles in the same render pass; camera-only frames update one transform uniform and never rewrite the world-anchored overlay buffer. **TEXT REMAINS MISSING:** tile-truth labels, coach marks, and any other overlay text need a glyph atlas and remain an explicit promotion gap — do not treat geometry parity as complete overlay parity. Three consecutive full real-Wayland matrices v17/v18/v19 hold that geometry on every sample and pass all five wgpu rows outright (including index 10/10); zoom-in/out remain zero-commit. Open: **(d)** promotion by evidence: perf bars vs VisPy on real data (the montage FFT-scroll 4→17 fps target is the headline number), Thomas dogfooding the explicit wgpu pin through daily use, VisPy retirement review only after a release cycle — never a flag-day switch. **Successor traps:** import rendercanvas ONLY via `import_qrenderwidget()`; every wgpu adapter probe pins `set_instance_extras(backends=["Vulkan"])` BEFORE its first request (an all-backends instance re-inits EGL during GL enumeration and SIGABRTs workers holding vispy GL state — `8c57a7bf`). | Promotion gate: journey matrix + perf bars on real data; written verdict in tensor-engine-endpoint.md |
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
- **R8 continuity gate vs document-changing stages (adjudication needed).**
  With the fill stall and entry blackout fixed, `profile_montage_workflow`'s
  `fft_full_tiled_montage` fails `presentation_continuity` on BOTH vispy
  (first tile 4.6 s) and pyqtgraph (3.4 s), offscreen 2026-07-19: applying
  the FFT pipeline is a document change, ADR 0051 forbids retaining
  old-operation pixels, so entry honestly blanks — and the gate's
  no-blank-sample rule can never pass a document-changing stage slower than
  the sampler's first tick. Either the gate learns a document-change
  transition class (blank legal, successor latency still measured), or the
  FFT successor needs its own first-pixel latency work. Pre-existing on all
  backends; raw-stage entry (same document) now passes via the montage-axis
  bridge.
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
