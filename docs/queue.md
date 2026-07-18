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
| 2 | **G6 — GPU histogram/levels.** Slices **(a)** GPU histogram evidence and **(d)** reducer-honest GPU LOD generation from resident chunks are IMPLEMENTED on wgpu (see Done 2026-07-18); real-Wayland acceptance rides the row-3 matrix. Remaining: **(b) RESOLVED — measured NO:** keep rough→refined phasing. The real 336×336×272 T2 / Intel TGL Vulkan probe found that a native-L0, exact-bounds, 500-bin/8,192-representative pass preserves the live four-source heartbeat (Wayland max 9.28 ms at 60 tiles, 4.08 ms at 272), but it is **not the phase-1 resident population**: the representative montage frontier is retained L2. Making L0 available would move target/native residency into coverage, violating the binding phase order; marking the L2 mean population `REFINED` would violate ADR 0054. The lower-bound exact GPU cost is also 110.49 ms vs 29.56 ms rough at 60 tiles and 382.99 ms vs 163.78 ms at 272. Offscreen repeated the cost direction (136.61 vs 31.70 ms; 401.14 vs 164.73 ms). Evidence: `tests/artifacts/g6b-histogram-collapse-2026-07-18/{offscreen,wayland}.json`; no collapse or scheduling change. Remaining: Known-legal micro-follow-up (recorded, not scheduled): the SINGLE-SLICE session owns native L0 in phase 1 and exact measured FASTER than rough (2.6 vs 4.4 ms GPU) — a scoped collapse there would be honest; benefit ~10 ms on the fastest case. **(c)** histogram widget: RESOLVED, keep PyQtGraph — see Done.) | Levels/histogram converge from chunk summaries; sampling measured ON the GPU; no GUI-thread aggregation; real-GL gate; journey matrix not regressed |
| 3 | **wgpu strangler — promotion evidence** (slices (a)–(c) LANDED, see Done; ADR 0057). zoom_out full-matrix red ADJUDICATED 2026-07-18: **harness gap, closed** (`da22dad7`, see Done — the canvas repainted ~24 ms after the journey-end capture; sampler now drains presentation-draw acks before the end sample; three consecutive FULL Wayland matrices v10/v11/v12 wgpu-green). The 2026-07-18 field stalls `259-1`/`1-1` are **not** the standing 272-tile fill: both are one wgpu physical-first-pass quality drift (exact latch followed by a mixed exact+fallback snapshot), fixed in `43287f8` and recorded in [the field-stall dossier](redesign/wgpu-field-stalls-2026-07-18.md). **Dogfood overlay parity:** wgpu now draws camera-locked ROI outlines/handles, live-profile cursor geometry, and tile loading/skipped boxes + symbols natively after the tiles in the same render pass; camera-only frames update one transform uniform and never rewrite the world-anchored overlay buffer. **TEXT REMAINS MISSING:** tile-truth labels, coach marks, and any other overlay text need a glyph atlas and remain an explicit promotion gap — do not treat geometry parity as complete overlay parity. Open: **(d)** promotion by evidence: perf bars vs VisPy on real data (the montage FFT-scroll 4→17 fps target is the headline number), Thomas dogfooding the explicit wgpu pin through daily use, VisPy retirement review only after a release cycle — never a flag-day switch. **Successor traps:** import rendercanvas ONLY via `import_qrenderwidget()`; every wgpu adapter probe pins `set_instance_extras(backends=["Vulkan"])` BEFORE its first request (an all-backends instance re-inits EGL during GL enumeration and SIGABRTs workers holding vispy GL state — `8c57a7bf`). | Promotion gate: journey matrix + perf bars on real data; written verdict in tensor-engine-endpoint.md |
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
- **PRE-EXISTING, BACKEND-AGNOSTIC: 272-tile raw fill stalls (offscreen-deterministic
  repro!).** `profile_montage_workflow --backend pyqtgraph --stages
  raw_full_tiled_montage` times out at 0/272 presented with the commit
  backlog gate armed and no progress (gate_no_progress climbing,
  visible_upserts=272, dirty=77), for pyqtgraph AND vispy (271-272 unsettled) on main `9a5c6465`, at
  pre-camera-reanchor `976ea275`, and on the wgpu branch (wgpu stalls in
  the same stage), offscreen AND real Wayland (2026-07-18 probes; branch
  probes both with and without `--session-fixture ""`, main probed with
  it). This blocks the `--backend all` perf-bars run and is the likely
  identity of the journey matrix's all-day pyqtgraph cold_fill red.
  G5 measured this harness green 2026-07-17 (`raw 272/272 ~11.4 s`), so
  the window is main `661b6ba5..976ea275` — or the invocation itself was
  never re-validated. Offscreen determinism makes this the first cheaply
  bisectable member of the stall family; it BLOCKS the row-3(d) perf-bars
  measurement. Hunt chip out (re-scoped).
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

<<<<<<< HEAD
<<<<<<< HEAD
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
||||||| parent of 5b18f039 (Queue row 3: zoom_out adjudicated harness-gap and closed with matrix evidence)
=======
- 2026-07-18 — **zoom_out full-matrix red adjudicated: harness gap, closed**
  (`da22dad7`): the v7 trace walk proved the canvas repainted — 172k-pixel
  screenshot delta with ZERO commits and no render_request, the
  descriptor-only camera repaint working as designed — but ~24 ms AFTER the
  journey-end capture, outside every gesture-tagged sample; under load the
  on-demand scheduler defers the one repaint of a resident zoom-out past
  gesture completion (isolated reruns pass because the draw lands inside the
  glide's event pumping). Journey-end samples now drain
  `presentationDrawPending()` — the only signal that sees a pure-camera
  `request_draw`; `_wait_for_tile_presentation_draw` counts commit-keyed
  requests only — bounded/non-raising, every gesture site, every backend;
  incumbent oracles untouched (freshness clock still starts at gesture
  start). Fault-injection proof both levels: a never-clearing pending flag
  still captures the stale sample (`presentation_drained=false` in trace),
  and an injected missed redraw on a zero-commit zoom_out stays red.
  Acceptance: three consecutive FULL real-Wayland matrices — v10 wgpu 5/5
  outright (zoom_out first_new 272 ms, commits=0), v11 4/5 + cold_fill
  attributed `reference_vispy_cold_level_convergence_standing_red` by the
  existing classification (zoom_out 446 ms), v12 5/5 outright (zoom_out
  451 ms). Incumbent reds across runs are the standing
  coverage/level-convergence stall family (present in v7 pre-change).
  Suite 2427/0.
>>>>>>> 5b18f039 (Queue row 3: zoom_out adjudicated harness-gap and closed with matrix evidence)
||||||| parent of 15fad3d7 (Measure G6b exact histogram collapse and keep phasing)
=======
- 2026-07-18 — **G6(b) rough→refined collapse rejected by measurement:**
  timestamped resident-page evidence on the real T2 dataset, offscreen and in a
  real Wayland window, found the native-L0 candidate heartbeat-safe but absent
  from the montage phase-1 L2 frontier; exact GPU work was 3.7× rough at 60
  tiles and 2.3× at 272. The existing quality machinery and phase-2 refinement stay
  unchanged; benchmark artifacts:
  `tests/artifacts/g6b-histogram-collapse-2026-07-18/`.
>>>>>>> 15fad3d7 (Measure G6b exact histogram collapse and keep phasing)
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
