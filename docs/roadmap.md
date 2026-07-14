# Roadmap

Ordered by risk reduction, not feature excitement. An item is complete only
when its exit gate is met — "code exists" is not completion.

Completed gates N4–N7 and X1–X4 are archived with their exit criteria in
[`archive/roadmaps/completed-gates-n4-x4.md`](archive/roadmaps/completed-gates-n4-x4.md).
Y1–Y3 (one generation contract, backend de-duplication, declarative UI
sync/one cache core) completed 2026-07-02 — see git history and ADR
0045/0046. The 2026-06/07 LOD and tile-lifecycle landings are recorded in
ADR 0050/0051 and the archived
[lod-remaining-work plans](archive/plans/lod-remaining-work/README.md).

## Now — measured performance and suite truth (post-redesign)

> **[Codex 2026-07-14 — post-V4 roadmap update; linear-history correction]**
> R1–R7 and V0–V4 are rebased linearly onto `main`; no integration merge
> remains. The fixed viewer passed the V1/V2 real-Wayland
> pixel/trace scenarios on both backends and the V3 loud-stall injection.
> The final pre-integration non-GPU suite remained red at 42 failures and 2
> teardown errors; this is tracked work, not a green-suite claim.

Proceed with the redesign P-program one measured cause at a time against
the frozen T1 baseline, in the order recorded in
[`docs/redesign/marathon-salvage.md`](redesign/marathon-salvage.md):
prefetch-busy → committed `level_source` → viewport intent → background
histogram aggregation → coalesced completion drain → cadence throttle →
stage-cache snapshot/cancellation → governor policy → admission batching →
gate pacing → slot relocation. Every P commit carries before/after trace and
benchmark evidence; the real-display pixel/trace gates must stay green.

> **[Codex 2026-07-14 — P1 result]** Narrowing prefetch-busy was measured on
> both backends, produced no FFT improvement, regressed the scalar elapsed
> sample, and did not change PyQtGraph's 50/60 presentation freeze. The code
> was reverted and the rejected measurements are recorded in the redesign
> README. The active measured cause is now committed-frame `level_source`.

> **[Codex 2026-07-14 — P2 result]** Committing `level_source` removed the
> workflow's first-evidence-quality failure but regressed the real VisPy V2
> priority gate from 14/16 to 4/16 nearest first-cohort tiles and exposed a
> rough-bounds relative-window error. The code was reverted; the redesign
> README records the measurements and missing maturity rule. The active
> measured cause is now viewport-intent replay.

> **[Codex 2026-07-14 — P3 result]** Acknowledged content-extent changes now
> replay AUTO/FIT without moving USER cameras, including VisPy's hidden-bounds
> update. Focused and real-pixel gates pass on both backends. The canonical
> USER-camera workflow remained stalled at the same 7/60 presentation state,
> so P3 carries no performance credit and the stall remains open. The active
> measured cause is now background histogram aggregation.

> **[Codex 2026-07-14 — P4 result]** Aggregate histogram sampling is now
> revision-guarded kernel work; its deterministic 60-tile selector is 9.8×
> faster and the real trace moves up to 36.6 ms per aggregate off the GUI
> thread. Prepared atomic transactions now include level revision, closing a
> stale-level identity defect exposed by the new wake. Broad display/window
> coverage is 840 passed and both physical gates remain green. The unrelated
> 7/60 presentation stall persists. The active measured cause is now the
> coalesced kernel completion drain.

> **[Codex 2026-07-14 — P5 result]** Coalescing 205 completions into 42
> bridge drains bounded the observed drain callbacks, but all three capacity-
> wake variants failed the real VisPy priority gate with 36/36 exact targets
> stranded at preview quality. The runtime experiment was removed and the
> failed designs are recorded in the redesign README. The active measured
> cause is now LOD-plan cadence plus synchronous-title removal.

> **[Codex 2026-07-14 — P6 result]** Wheel/pan-derived replans now have a
> committed-frame-only 16 ms cadence, separate from programmatic replay and
> pipeline continuation; range input no longer performs synchronous title
> layout. Two real traces cut kernel submissions 39–50% and bridge drains
> 61–65%, with a +0.8% two-run first-ack midpoint. The V1/V2 physical matrix
> and 873 focused/broad tests pass. The 7/60 deadlock remains unchanged. The
> active measured cause is now the stage-cache resident snapshot,
> cancellation tokens, and `peek_many`.

> **[Codex 2026-07-14 — P7 result]** Hot stage reuse no longer acquires the
> mutation lock, preview-floor probes batch 60 potential cache locks into one,
> and cancelled render results stop between evaluation/reduction boundaries.
> The deterministic lock-contention regression, 1,297 broad tests, and all
> four physical gates pass. The workflow sample stayed within the ±10% latency
> guard but did not improve the 7/60 deadlock. The active measured cause is now
> governor lane policy.

In parallel only where it does not reorder a P-step, migrate stale tests to
the canonical `window.renderer` / `FrameSession` owners and fix the remaining
coalescer, levels, viewport/ROI, cache-rebind, and transition behavior. Do
not weaken user-visible assertions merely to make the suite green.

## Next — evidence gates (X5, after the redesign)

X5 remains the evidence-first physical-strategy gate
([ADR 0046](decisions/0046-evidence-first-performance-strategy.md)).
X5a (Linux telemetry baseline) and X5b (acknowledged residency for montage
tiled scenes) are done — see ADR 0047/0051. Remaining, re-expressed against
the post-redesign modules:

1. **X5c — Viewport-scoped tiled scenes.** Retarget scheduling keys on
   tiled-scene/storage checks (not montage-mode checks) so internally tiled
   normal images get visible-only active regions through the same
   `RenderIntent`/pipeline path.
2. **X5d — Region-first materialization + physical strategy policy.**
   Visible-region reads without a full display image first; measured
   singleton/direct vs tiled storage choice below `ImageSurface`, without a
   separate normal-image semantic path.
3. **X5e — Backend and LOD decision matrix.** Windows/macOS traces join the
   Linux ones; per-OS backend defaults and source-provided-pyramid handling
   decided from measurements. Includes P4 (per-slot mip validity before
   atlas mipmaps default on).
4. **Probe hardening.** Analytic per-tile content assertions for the
   blank-tiles-at-zoom-back report; scripted zoom-across-threshold content
   test.

Exit gates for X5c–X5e are unchanged from the pre-redesign roadmap (full
list in this file's git history): published benchmark matrix, no fixed
texture-size assumptions, acknowledged residency everywhere, no full-set
rebuilds on zoom threshold crossings, exact inspection independent of
display LOD, documented backend/LOD default evidence.

## Later — product capabilities

- **Linked windows and inspection groups** — first iteration shipped
  ([ADR 0048](decisions/0048-linked-window-sync.md)); remaining: cursor and
  viewport links, named groups.
- **Focused compare mode** — side-by-side/overlay with shared
  coordinates/levels; registration/segmentation stay out of core.
- **Rich axis metadata** — `AxisInfo` continues incrementally
  ([proposal](proposals/axis-info.md): coordinate arrays, orientation,
  physical cursor readout).
- **Out-of-core sources** — chunked (Zarr/HDF5-like) adapter behind the
  ADR 0049 protocol, lazy selectors, chunk-aligned planning hints.
- **Invocation adapters** — Jupyter/editor routes over the one semantic API
  (Julia/MATLAB wrappers exist, [`invocation.md`](invocation.md)).

## Explicitly not now

- Plugin marketplace/layer ecosystem; broad segmentation/registration/qMRI
  workbench; remote multi-user collaboration; destructive workspace
  operations.
- Re-enabling the synchronous LOD pyramid path; refuse/degrade render
  decisions; the bespoke idle stage-warmup scheduler.
- New scheduling systems beside the kernel, new pacing timers, or another
  parallel tile-state collection (ADR 0053 forbids all three).
