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

**Course frame (2026-07-19 review):** the engine bet is nearly won; the
risk has flipped from "engine not good enough" to "no product on top of
it". The north star is *the trusted lens for scientific arrays* — any
array-like, any size, complex-native, every displayed value traceable to
source through a reversible pipeline, extensible by users, comparable
side-by-side, with a stated and enforced responsiveness bound. Evidence
and rationale:
[`reviews/2026-07-19-course-review.md`](reviews/2026-07-19-course-review.md).
The competitive whitespace is verified: no other tool combines
complex-native values, dimension-first UI, a reversible pipeline, and
bounded progressive rendering.

## Now — Program A: finish the engine bet

**2026-07-19 update:** the renderer decision is made and implemented. The
wgpu backend (ADR 0057 command protocol) is live behind an explicit pin
with every payload shape, native GPU overlays incl. glyph text, GPU
histogram/LOD compute (G6 complete), and opt-in screen presentation
(bitmap remains default; AUTO = screen on native Wayland). The journey
matrix reached its first 15/15; wgpu leads fast-scroll. What "Now" means
until further notice: **promotion evidence** (queue row 3d — shared row-1
callback bars, dogfood hours, the FFT-scroll headline) on the VisPy
decision ladder (perf bars → AUTO backend flip at field parity → VisPy
demotion review at G7 start → removal one release later; PyQtGraph keeps
the headless/interaction-host role), then **G7 compressed transport**,
whose decompress-on-GPU form compounds only on wgpu.

As of 2026-07-16, `main` **is** the GPU engine (ADR 0055/0056). The
direction record is
[`proposals/tensor-engine-endpoint.md`](proposals/tensor-engine-endpoint.md).

Engine close-out items folded into Program A (from the 2026-07-19 review's
bug archaeology; they retire the two biggest recurring-defect roots):

- **Presentation clock** — one surface-published, commit-keyed,
  monotonic presented-generation edge; every wait, gate, camera-intent
  decision, and harness sampler keys on it instead of reconstructing
  "what is presented now" from proxies. Retires the demand-freshness /
  sampler-gap adjudication class.
- **Shutdown & cancellation contract** — timers/dialogs/deferred
  callbacks registered under kernel scopes; ops declare a cancellation
  poll boundary as a capability (shared design with plugin ops, below).
- **Policy out of backends** — continue moving admission/upsert/commit
  policy from `backends/*/tiles.py` into the protocol/model layer until a
  backend is only a command executor; this is what makes VisPy retirement
  cheap and a later QPainter software executor small (~2k LOC).

## Next — the product turn (Programs B–D)

Ordered; each starts only when its predecessor's exit gate is green or it
is proven independent. Execution rows: [`queue.md`](queue.md) Next section.

1. **B — Compare v1.** Camera/viewport facet on the existing sync bus →
   "Compare with…" launcher (second window linked on dims+camera+levels)
   → linked complex-aware cursor (mag *and* phase of A, B, A−B) →
   `CompositeArraySource(A, B, op)` difference as a derived *source*
   through the unchanged unary pipeline. Rejected: multi-input pipeline
   ops (invalidates region pull-back, stage keys, and recipes at once).
   This is the highest-signal user feature and the arrShow/Slicer parity
   point where we can lead (complex-aware, progressive, values-exact).
2. **C — Plugin operations v1 + toolbox bridges.** Entry-point group
   `arrayscope.operations`, npe2-style lazy manifest, namespaced stable
   ids. Tier 1: pure `fn(ndarray)->ndarray` + shape/dtype (OPAQUE,
   whole-array — zero engine changes). Tier 2: declared capabilities +
   region algebra, honored only after a conformance harness proves
   region-path equivalence. **BART** as an optional subprocess pack (cfl
   handoff at the stage-materialization seam, cancellation → SIGTERM, honest
   cost hints). No sigpy pack: `sigpy.fft` is `numpy.fft` underneath, so it
   duplicates the built-in centered-FFT op + the `FFTBackendChoice` setting;
   sigpy's additive value (nufft/espirit) awaits a multi-input op contract.
   Registry yes; marketplace remains a non-goal.
3. **D — Ingestion breadth.** DataWrapper-style adapter behind the ADR
   0049 protocol: zarr/dask/cupy/torch/xarray without touching the
   planner; chunk-aligned request hints for the tile engine.

## Standing programs (parallel-safe, continuous)

- **E — Test consolidation + realism:** parametrize backend-twin suites,
  split the 5–7k-LOC montage files, 4-cell fast journey subset for
  iteration, physical readback for the remaining pyqtgraph/RGB gaps,
  transition-trace + sampler-drain generalization. Details:
  [`reviews/2026-07-19-course-review.md`](reviews/2026-07-19-course-review.md) §6E.
- **F — Docs that police themselves:** queue/ledger split, row-size cap,
  CI doc-lints (index completeness, resolving step refs, graduation of
  CLOSED/FIXED rows), ADR reservation ledger. Details: review §6F.

## Later — after the product turn

- **X5c–X5e evidence gates, re-expressed post-engine** (ADR 0046):
  viewport-scoped tiled scenes; region-first materialization;
  Windows/macOS traces for per-OS defaults.
- **Rich axis metadata** (`AxisInfo`,
  [proposal](proposals/axis-info.md)) — feeds dimension presets and the
  dimension-first op UI ("reduce over coils").
- **Notebook/editor reach** — Jupyter embedding over the one semantic
  API; the recon-research workflow lives in notebooks and every
  competitor with reach (ndv, fastplotlib, itkwidgets) meets it there.
- **QPainter software tile executor** — replaces the pyqtgraph *tile*
  path once policy-out-of-backends completes; pyqtgraph remains the
  interaction/widget host. Full pyqtgraph removal is not scheduled.
- **Named link groups, cursor/viewport links beyond compare** (ADR 0048
  continuation).
- **G8 candidate — tensor ops on the GPU**
  ([proposal](proposals/tensor-ops-g8.md)): ops as compute commands over
  the ADR 0057 protocol ("upload the source once, derive on the GPU");
  T0 shader-on-read elementwise may ride earlier; T1 gated on the
  FFT-scroll headline decisively beating ~17 fps or the program stops.
  Trigger: AUTO flip + presentation clock + shutdown contract, in order.

## Explicitly not now

- Plugin *marketplace*/layer ecosystem (the typed op registry in Program
  C is the product; distribution stays pip); broad
  segmentation/registration/qMRI workbench; remote multi-user
  collaboration; destructive workspace operations.
- Multi-input pipeline operations (compare uses composite *sources*).
- Full PyQtGraph removal (interaction re-host + widget rewrite; only if
  a dedicated interaction layer is justified later).
- Re-enabling the synchronous LOD pyramid path; refuse/degrade render
  decisions; the bespoke idle stage-warmup scheduler.
- New scheduling systems beside the kernel, new pacing timers, or another
  parallel tile-state collection (ADR 0053 — see
  [ground rules](ground-rules.md)).
- GPU op kernels (flip/crop/conjugate): operations stay on the CPU; the
  engine consumes evaluated planes. Late, evidence-gated experiment only.
