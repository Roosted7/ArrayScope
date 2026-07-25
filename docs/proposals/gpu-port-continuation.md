# GPU port — continuation brief (handoff, 2026-07-16)

> **Superseded as a queue (2026-07-16):** the branch is merged; `main` is
> the single line of development and **[`../queue.md`](../queue.md) owns
> the ordered next steps** (this brief's OPEN items 1–3 landed; 4–9 were
> transferred there). The worktree named below no longer exists and the
> cwd trap with it. Kept for its done-inventory, evidence paths, and
> working practices (now codified in [`../ground-rules.md`](../ground-rules.md)).

Self-contained handoff for the session completing the GPU port. Branch:
`codex/gpu-engine` in `/home/thomas/projects/ArrayScope/.worktrees/gpu-engine`
(historical — see banner above).

## Environment facts (hard-won; trust these)

- Python: `/home/thomas/miniconda3/envs/arrayscope/bin/python` (conda env
  `arrayscope`). **The env's editable install resolves `import arrayscope`
  to THIS WORKTREE** — the user's live app runs this branch; field reports
  are branch reports.
- **The harness shell cwd resets to the MAIN repo between calls** — start
  every command chain with `cd` into the worktree, or you will silently
  read/write main's files (this caused two incidents).
- Tests offscreen: `QT_QPA_PLATFORM=offscreen … -m pytest …` (xdist default;
  `-n 0` for single files). Real display: `QT_QPA_PLATFORM=wayland`, live
  Wayland works; on-screen GPU suite: `ARRAYSCOPE_GPU_TESTS=1 … pytest
  tests/gpu_interaction -n 0`. Profile tools need cwd = MAIN repo (`data/`
  relative paths). Benchmarks: `python -m
  arrayscope.display.rendering_benchmarks --runs 3 --presented`; artifacts
  convention `tests/artifacts/<gate>-<date>/` (gitignored).
- Full suite baseline at handoff: **2072 passed, 24 skipped** (~104 s); the
  only real-display failures are 4 pre-existing `tests/gpu_interaction`
  baseline failures (P9-era, also failing at the branch base commit).
- Parallel-work rules: main worktree runs the P9 program — do not edit its
  lane (scheduler/pacing, `prepare_rung`/fanout internals) without recording
  a dedupe note; ADR numbering may collide across branches (renumber at
  integration). When running multiple agents, assign disjoint file
  ownership explicitly.

## What is DONE (do not rebuild; extend)

Read in order: `docs/decisions/0055-…`, `docs/decisions/0056-…`,
`docs/proposals/gpu-engine-plan.md` (status + stages),
`docs/proposals/tensor-engine-endpoint.md` (endpoint + renderer verdicts),
`docs/redesign/coverage-stall-2026-07-15.md` (defect dossier).

- **G1/G2**: `arrayscope/gpu/` (Qt/VisPy-free): DataChunkKey/ViewTileKey,
  anisotropic `ChunkLod` (reduction vector + semantic reducer), `ChunkGrid`
  window math, `PageTable` (generations, LRU, pins, remap), `ChunkStore`.
  The VisPy atlas pool's residency bookkeeping runs on `PageTable`.
- **G3 (window-shift fast path, real-GL verified)**: exact non-montage
  payloads carry `PayloadSourceAnchor` (content key + native rect;
  windowability per axis via `pipeline_windowable_display_axes`); the pool
  chunks eligible payloads into 256² content-keyed slots drawn as
  `TileDrawPart` UV-cropped quads. ±1 window shift uploads boundary strips
  only; FFT-along-shifted-axis correctly misses (gate tests:
  `tests/display/test_window_shift_gate.py`, `test_vispy_chunked_residency.py`,
  live `tests/ui/test_window_shift_live_path.py`).
- **G4a/G4c**: content-keyed residency without window anchoring (FFT views
  get scroll-back reuse); grow-before-evict for budgeted pools; scroll-back
  = 0 uploads live; warm prefetch of adjacent planes (pure chunk residency,
  never evicts/relocates residents — denial tests). Montage prefetch now
  crosses the same backend seam into the bounded VisPy atlas warm queue and
  biases eligible work along observed index-window motion without changing
  presentation truth. G4b (session-reuse)
  CLOSED BY MEASUREMENT: scrub step 10.2 ms mean / 13.8 p95 on the user's
  real data — rebirth is not the bottleneck.
- **G5 slice 1**: reduced-LOD planes chunk into uniform plane-pixel pages
  (native-scaled keys + LOD triple); revisit reuse = 0 uploads; honest
  limit pinned (±1 native shift at factor>1 legitimately re-uploads — bins
  move; source-anchored reduction binning is the fix, queued with the
  ladder migration).
- **Physical presentation truth (standing invariant)**: before ANY
  re-present the layer audits per-page mapping key + derived uniforms +
  levels + per-quad `a_mode` spans, repairs divergence (`physical_repairs`
  stat), never acknowledges from divergent state. Framebuffer gate: zero-
  magnitude complex must never render PAL-relaxed LUT[0] orange
  (`tests/ui/test_vispy_phase_framebuffer.py`,
  `tests/display/test_vispy_physical_presentation.py`). "Drawn ⇒ physical
  truth row exists" is enforced with self-heal.
- **Field fixes (all repro-first, failing-pre-fix gates)**: session-148
  identity-aliasing starvation stall (2026-07-16: explicit-full
  `axis_range_indices` vs `None` aliased retained payloads into permanently
  unacknowledgeable upserts; silent backend rejection + dead payloads counted
  as coverage + first-pass barrier = whole-montage stale/empty livelock;
  root-caused live, predates the G5 series — dossier
  `docs/redesign/stale-empty-tiles-2026-07-16.md`); first-pass
  histogram evidence-race stall (`fd6b77a6`); zoom/phys-None triple pool
  defect (`3c2d6520`); scrub retain-until-replace
  (`slice_only_session_transition`; doc/op changes still blank) +
  prefetch re-arm on visible drain (`ab052659`); Performance-menu prefetch
  toggle restored; post-race session-50 shared-target candidate hole (coarse
  payload labelled exact was physically valid fallback but excluded from the
  finer target pass) fixed from lifecycle settlement truth. Severed-wire
  pattern: three limbs of old deletions were
  found dangling (scheduler call, menu item, and dead presenter fallbacks)
  — grep for orphaned handlers when something "does nothing".
- **Cleanups**: legacy single-quad path deleted (~1100 LOC, resurrection
  guard), presenter single-commit-path (loud error, no fallback planning),
  montage-backend chooser deleted, R1-era shims deleted.
- **OperationClass** execution identity (coordinate-metadata /
  shader-on-read / … / opaque) in `operations/capabilities.py`.

## OPEN items (the new session's queue, in suggested order)

1. **Fresh field evidence**: session-50 montage stall from
   `arrayscope-diagnostics-20260715-235102.jsonl` /
   `/tmp/arrayscope-stall-50-1.trace.jsonl` is fixed at the shared-target
   candidate owner (details and rejected shortcut in the coverage-stall
   dossier).  The remaining `…-235200.jsonl` item was single-slice retention:
   no black flicker, but the retained stale plane lingered too long. It is now
   measured and fixed at the first-pixel lane owner (priority 3 below).
2. **Montage scroll-direction GPU warming** (task board #20, landed
   2026-07-16): montage prefetch completion now warms both persistent
   `cpu_item` and `gpu_atlas` backends. VisPy queues bounded GUI-thread atlas
   batches; gates prove that warm residency does not alter active mappings or
   physical-presentation acknowledgement. The renderer observes montage
   index-window momentum and partitions candidates ahead of motion inside
   the existing viewport priority bands. Rejected: synthesizing payloads for
   a future montage window through `_ensure_display_tile_payload`; that would
   manufacture lifecycle/presentation state rather than warm evaluated
   content. The implementation deliberately reorders only candidates already
   admitted by the standing montage-prefetch scheduler.
3. **Retention staleness perf** (landed 2026-07-16): the 23:52 diagnostics
   disproved session rebirth and GPU upload as bottlenecks. The atlas already
   had 200 warm residents and zero upload time; 235 completed successor
   admissions were initially quota-blocked because the first/only `DESIRED`
   rung was mislabeled `DISPLAY_PREPARATION`. The ladder now labels a
   first-presentable `DESIRED` as `DISPLAY_PREVIEW`; ordinary refinement stays
   preparation, and the existing interaction proof still defers cold native
   work while admitting retained-stage extraction. Diagnostics and trace
   events measure retained-transition replacement latency. Full evidence,
   exit gates, and rejected timer/session/warming shortcuts are in
   `docs/redesign/slice-retention-staleness-2026-07-16.md`.
4. **G5 landing candidate**: the live ladder/cache/producers and both backends
   now consume the canonical source-grid `DataChunkKey` page route, including
   reducer families, clipped draw geometry, exact page-set claims, physical
   ancestor truth, and complete coarse fallback. Legacy whole-plane ownership
   is structurally forbidden. The authoritative contract is
   `docs/redesign/g5-source-grid-pyramid-2026-07-16.md`; only its remaining
   focused/broad/stress and real-Wayland exit gates stand between this candidate
   and moving queue row 1 to Done.
5. **G6**: GPU histogram/levels over the chunk store with the ADR 0056
   coverage frontier (per-chunk summaries; workgroup-local bins); GPU LOD
   generation from resident chunks.
6. **Renderer protocol + Experiment A** (task #22): formalize the semantic
   command table (tensor-engine-endpoint.md maps commands → existing
   seams); wgpu-py vertical slice per the recorded study (real
   QRenderWidget; test present_method screen vs bitmap on Wayland).
   QRhiWidget+native runtime is the recorded production candidate;
   direct Vulkan the measured escape hatch.
7. **G7**: compressed transport (codec ladder, measured topology; ZFP-class
   first) — after G6.
8. **Cleanup queue**: ImageViewShell duplication (imageview2d 2723 /
   vispy_imageview2d ~1840 lines); ruff F841 at tiles.py:931; the 4
   gpu_interaction baseline failures may deserve a re-look after main
   integrates tonight's fixes.
9. **Session-level loose end** (from 3c2d6520 report):
   `_resident_source_matches_expected(source, None)` returns True —
   controller-side expected-source coverage during session switches is
   worth an audit.

## Test-suite state (consolidation landed 2026-07-16: a39a36e6/3707b813/d4110d3d)

Shared fakes live in `tests/display/vispy_test_utils.py` (FakeTexture2D/
FakeGloo/FakeVisual/payload builders) and the live-window harness in
`tests/ui/helpers.py` (backend switch, settle predicates, upload_log
fixture) — use these, don't re-roll. Current suite 2081/24 in ~124 s after
the montage-warming and retention gates; no new
test in the top-25 durations. Watch item: one observed flake in
`test_prefetch_gated_by_busy_visible_runs_after_drain` (timing-sensitive,
passed all re-runs). Follow-up: six pre-existing UI test files carry their
own backend-switch helpers and could adopt tests/ui/helpers.py.

## Working practices that made tonight work

Repro-first (failing test before fix); every fix carries its gate; live
gates offscreen AND real-GL; trace mining before code (stall ring buffers
+ diagnostics JSONLs decode completely); dossiers in `docs/redesign/` with
exit gates for other-lane work; physical truth over bookkeeping trust;
speculative work must never change visible outcomes; honest-limit tests
for known non-goals. The user tests quickly and precisely — give them
bounded field-test lists and ask for JSONLs.
