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

## Now — the redesign (ADR 0053)

**One queue, owned by [`docs/redesign/README.md`](redesign/README.md).**

The kernel (`arrayscope/kernel/`), the modular pipeline nucleus and unified
LOD ladder (`arrayscope/render/`), the vocabulary canonicalization, and the
first hygiene deletions are landed on the `redesign` branch. Remaining, in
order:

| plan | delivers |
|---|---|
| R1 | all execution on the kernel; 8 controllers + WorkGraph deleted |
| R2 | MontagePipeline live; frame_renderer clusters B/C/E dissolved |
| R3 | LOD ladder adoption; montage_lod deleted; ops once per rung; PyQtGraph parity decision |
| R4 | timer/governor audit: no scheduling timers, governor = telemetry + 2 knobs |
| R5 | test pruning + docs truth pass; known-red ledger emptied |

Do not start items below this line while a redesign plan is open — they all
get cheaper after it.

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
