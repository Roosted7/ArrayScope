# Roadmap

Why the current work order serves the [mission](mission.md). The ordered,
gated execution queue lives in [`queue.md`](queue.md) — this file stays at
the strategy level and must not accumulate status logs (execution records go
to dossiers and the archive; the P1–P9 log that used to fill this section is
at
[`redesign/archive/p-program-log-2026-07.md`](redesign/archive/p-program-log-2026-07.md)).

An item is complete only when its exit gate is met — "code exists" is not
completion. Completed gate history: N4–X4 in
[`archive/roadmaps/completed-gates-n4-x4.md`](archive/roadmaps/completed-gates-n4-x4.md),
Y1–Y3 in ADR 0045/0046, the R1–R7 rewrite and V0–V4 visible-truth program in
[`redesign/`](redesign/README.md), the 2026-07 P-program in the archive
above.

## Now — one architecture: the GPU tensor engine

As of 2026-07-16, `main` **is** the GPU engine (ADR 0055/0056, G1–G5 slice 1
landed and real-GL verified). The direction record is
[`proposals/tensor-engine-endpoint.md`](proposals/tensor-engine-endpoint.md):
a deadline-driven engine where the ADR 0053 kernel evolves into the resource
broker and a backend-neutral semantic command protocol separates meaning
from rendering runtime. The executable program is the G-series
([`proposals/gpu-engine-plan.md`](proposals/gpu-engine-plan.md)), continued
in [`queue.md`](queue.md).

Strategic commitments inside "Now":

- **Visible truth stays the acceptance bar** — the performance bars and
  their real-display gates are recorded in [`queue.md`](queue.md); the
  rings and their enforcement in [`testing/README.md`](testing/README.md).
- **The P-program's endpoint is absorbed, not abandoned.** P9 isolated the
  measured cost ("a one-index move prepares and acknowledges 60 slots");
  the engine's content-keyed chunked residency is the structural answer.
  Performance work resumes as measured steps on the engine substrate
  (queue step 3), not as scheduler tuning beside it.
- **De-clutter is part of the program.** Legacy single-quad path (~1100
  LOC), dead planners, and shims are already deleted with resurrection
  guards; the standing-debt lane in [`queue.md`](queue.md) continues this
  (ImageViewShell duplication, key-owner consolidation).

## Next — after the bars are green

1. **X5c–X5e — evidence gates, re-expressed post-engine** (ADR 0046):
   viewport-scoped tiled scenes for internally tiled normal images;
   region-first materialization + measured physical-strategy policy;
   Windows/macOS traces joining the Linux ones for per-OS backend/LOD
   defaults. Exit gates unchanged (see this file's git history for the full
   original list).
2. **Renderer runtime decision** — after Experiment A (wgpu-py slice) and
   the QRhiWidget study, pick the production runtime with measurements
   (queue step 5 produces the evidence).

## Later — product capabilities

- **Linked windows and inspection groups** — first iteration shipped
  (ADR 0048); remaining: cursor and viewport links, named groups.
- **Focused compare mode** — side-by-side/overlay with shared
  coordinates/levels; registration/segmentation stay out of core.
- **Rich axis metadata** — `AxisInfo` continues incrementally
  ([proposal](proposals/axis-info.md)).
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
  parallel tile-state collection (ADR 0053 — see
  [ground rules](ground-rules.md)).
- GPU op kernels (flip/crop/conjugate): operations stay on the CPU; the
  engine consumes evaluated planes. Late, evidence-gated experiment only.
