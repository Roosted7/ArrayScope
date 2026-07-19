# 2026-07-19 course review — whole-project review and reshaped course

Dated assessment (Thomas's ask: step back, review the whole project, reshape
the course). Method: six parallel deep-dives — feature/architecture
inventory, bug archaeology (819 commits, graveyard, 57 ADRs, dossiers),
test-suite audit (199 files, ~75k LOC), docs-system audit, competitive
web research, and an in-code extensibility/backend assessment. Durable
direction changes land in [`roadmap.md`](../roadmap.md) and
[`queue.md`](../queue.md); this review is the evidence and rationale.

---

## 1. What ArrayScope is today

**Product (user-visible):** `arrayscope(data)` / CLI / Julia/MATLAB handoff;
dimension strip with per-axis roles, slicing, flips, fftshift; relative/
absolute/auto windowing; grouped colormaps incl. cmcrameri + designer;
complex channel modes (complex/real/imag/abs/angle, cyclic-map
auto-restriction); 13 built-in reversible operations
(`ArrayDocument` + recipes/views persistence); ROI kinds with stats +
histograms; line profiles with complex modes + CSV; progressive tiled
montage; video/frame export; linked-window sync (levels/dims/recipe/ROI
facets); loaders for npy/cfl/DICOM/NIfTI/REC + lazy sources.

**Engine:** ~101k LOC. The Qt-free core is genuinely strong — one kernel
(ADR 0053), one tile lifecycle (ADR 0051), one scheduling-phase owner,
the ADR 0055/0056 chunk/page residency engine, the ADR 0057 renderer
command protocol, wgpu with native overlays/text/screen-present that now
*leads* on perf. The legacy mass is `window/` (22k LOC; 9-mixin
`ArrayScopeWindow`, ~100-field `FrameSession`) and `display/` (30k LOC;
three per-backend view/tile stacks).

**Extensibility: effectively zero.** Ops are frozen dataclasses in a
hard-coded dict (`operations/registry.py`); no entry points, no
registration API. BART exists only as a `.cfl` *loader*; sigpy appears
nowhere. Compare is a 36-line helper (`core/compare.py`) that never
renders. This is the gap between an excellent engine and a thin product.

## 2. Mission check

The mission ("quickest trustworthy way to understand an nd-array") is
right and needs no change. But the course has been engine-inward for
months: everything in "Now" is renderer promotion and transport, while
every capability a *user* would name — compare, plugin ops, toolbox
integration, notebook reach, broader ingestion — sits in "Later" or a
parking lot. The engine bet was correct (it is the moat — see §5), but it
is close enough to won that the risk has flipped: the differentiated
engine is finished before there is a product on top that anyone else can
adopt.

## 3. Recurring pain and its architectural roots

Bug archaeology across 819 commits identifies five generative properties
(full evidence in the archaeology dossier of this review's session):

1. **Duplicated progressive state** — the same fact ("which level is
   resident", "what is visible", "tile priority") held in 3–8 places;
   every black-tile/staleness/priority family traces here. Ground rule 2
   ("fix by deleting a duplicate owner") is the correct standing answer;
   the remaining structural move is a **commit-keyed presentation clock**:
   one surface-published monotonic presented-generation edge that every
   wait, gate, camera-intent decision, and *harness sampler* keys on. The
   cold_fill demand-freshness red, the zoom_out sampler gap, and the
   `presentationDrawPending` adjudications were all consumers
   reconstructing "what is presented now" from different proxies.
2. **Deferral without an owned resume-event** — six 2026-07 stalls with
   one grammar; ground rule 11 exists; the `commit_bail` trace events make
   it visible. Continue enforcing; no new abstraction needed.
3. **Three first-class backends with divergent semantics** — every
   progressive contract proven 2–3×; the upsert-governance skip was
   pyqtgraph-only, first-pass quality drift wgpu-only; 128/52/48 backend
   commit-subject hits. The ADR 0057 command protocol is the cure; it
   arrived after each backend had re-implemented policy. **Keep pushing
   policy out of `backends/*/tiles.py` until a backend is only a command
   executor**; then backend count stops multiplying bug surface.
4. **Qt event-loop coupling** — storms, GC pauses, the still-open
   unbounded process exit. Missing abstractions: (a) every timer/dialog/
   deferred callback registered under a kernel scope so shutdown provably
   reaches it; (b) ops declare a cancellation poll boundary as a
   capability (also required for plugin/subprocess ops — land together).
5. **Counters as acceptance** — the R8 era's six weeks of false "fixed";
   now countered by pixel oracles + journey matrix. Of recently
   adjudicated journey reds, roughly half were harness gaps — the
   presentation clock (item 1) removes the biggest remaining source.

**Would a different architecture have avoided this?** Largely yes, and
the project has already discovered each answer once: the command protocol
(0057), the single phase owner, the Qt-free gpu/ package. The reshaped
course finishes those consolidations instead of starting new ones.

## 4. Backend verdict

- **Drop VisPy: yes — keep the existing evidence ladder, and treat it as
  a real program, not an afterthought.** Scope is moderate: 4 source
  files (~7.5k LOC, incl. the sneaky AUTO probe in
  `display/image_view_factory.py` importing device limits from vispy
  tiles), ~8k LOC of vispy test twins, journey rows, docs. Ladder stays:
  AUTO flip at field parity → demotion review at G7 start → removal one
  release later.
- **PyQtGraph stays, permanently, for now** — it is not just the headless
  backend: it is the app's Qt shim (68 files), the pointer/camera
  interaction owner for *all* backends, and the profile/histogram/ROI
  widget toolkit. True no-GL software rendering is a real capability wgpu
  cannot claim.
- **Extract software rendering harder: yes, as the end-state.** The
  frame-composition model is already backend-neutral
  (`display/model/commit.py`, shader-mapping CPU mirror, page table); a
  plain-QPainter tile executor is realistically 1.5–2.5k LOC. The
  sequence that makes it cheap: policy-out-of-backends first (§3.3), then
  the QPainter executor can replace pyqtgraph's *tile* path while
  pyqtgraph remains the interaction/widget host. Full pyqtgraph removal
  is a large program (interaction re-host + widget rewrites) — only if a
  dedicated interaction layer is wanted anyway; not scheduled.

## 5. Competitive position (2026-07 web research)

Feature-level: napari (npe2 plugins, dask/zarr multiscale), ndv (broad
array-protocol ingestion, pygfx), ImageJ/Fiji (plugin ecosystem, macros),
FSLeyes/3D Slicer (linked medical views; Slicer's synced crosshair +
Compare Volumes is the spatial-sync gold standard), glue-viz (brushing &
linking), fastplotlib/datoviz (raw GPU throughput), arrShow (complex-
native, synced cursor — our lineage), sigpy.plot/bartview (complex-aware
throwaway plots).

**The whitespace, verified: no tool combines (a) complex/MRI-native
values, (b) dimension-first UI, (c) a reversible op pipeline, (d)
bounded-latency progressive GPU rendering.** Each competitor has at most
two. Specific unowned features:

- **Complex-aware compare**: synced cursor reporting mag *and* phase of
  A, B, and A−B; checkerboard/difference as first-class derived views on
  progressively rendered arrays. Slicer is real-valued; glue is
  table-centric; arrShow is MATLAB/in-memory/eager.
- **Bounded-latency as a guarantee**: napari/dask stalls unboundedly on
  cold slices; nobody else even states a responsiveness contract.
- **Reversible pipeline with provenance**: napari plugins mutate
  imperatively with no undo; ImageJ macros replay but don't invert.
  sigpy's adjoint-aware linops slot naturally into ours.

Borrow deliberately: npe2's **lazy declarative manifest** (capabilities
declared in data, code imported on invocation), ndv's **DataWrapper
ingestion** (numpy/cupy/dask/zarr/torch/xarray behind one thin adapter),
Slicer/arrShow's **synced crosshair**.

## 6. The reshaped course

**North star (the moon shot):** *the trusted lens for scientific arrays* —
any array-like, any size, complex-native, every displayed value traceable
to source through a reversible pipeline, extensible by users, comparable
side-by-side, with a stated and enforced responsiveness bound. Sold on
trust and reproducibility, not feature count.

Programs, in order (execution steps + exit gates live in
[`queue.md`](../queue.md)):

- **A — Finish the engine bet (current queue rows, unchanged):** wgpu
  promotion evidence → AUTO flip; VisPy retirement ladder; G7 compressed
  transport. Includes the presentation-clock consolidation (§3.1) and the
  shutdown/cancellation capability (§3.4) as engine-close-out items.
- **B — Compare v1 (first product program):** camera/viewport facet on
  the existing sync bus (small; machinery proven) → "Compare with…"
  launcher opening B linked on dims+camera+levels → linked complex-aware
  cursor → `CompositeArraySource(A, B, op)` difference as a derived
  *source* flowing through the unchanged unary pipeline. Explicitly
  rejected: multi-input pipeline ops (invalidates region pull-back, stage
  keys, recipes at once).
- **C — Plugin operations v1 + toolbox bridges:** entry-point group
  (`arrayscope.operations`), npe2-style manifest, namespaced ids stable
  under recipe persistence. Tier 1 = pure `fn(ndarray)->ndarray` +
  shape/dtype (OPAQUE, whole-array, works with zero engine changes);
  Tier 2 = declared capabilities + region algebra, honored **only** after
  a conformance harness proves `apply(whole)[region] == region-path` (a
  wrong capability is silent-wrong-pixels). **sigpy = default in-process
  op pack** (fft/nufft/espirit; adjoint-aware linops give reverse edges);
  **BART = optional subprocess pack** via the stage-materialization seam
  (cfl temp files, cancellation → SIGTERM, honest cost hints). The
  *marketplace* stays a non-goal; the *registry* is the product.
- **D — Ingestion breadth:** DataWrapper-style adapter (ADR 0049 seam) so
  zarr/dask/cupy/torch/xarray arrive without touching the planner;
  chunk-aligned request hints for the tile engine.
- **E — Test-suite consolidation + realism (standing lane):** parametrize
  the backend-twin view tests (73 vispy / 33 wgpu / 5 pyqtgraph is
  missing coverage, not just duplication); split the two 5–7k-LOC montage
  files so xdist can spread them; a 4-cell fast real-Wayland journey
  subset for iteration (full 15 stays the merge gate); physical
  framebuffer readback for pyqtgraph scalar LUT + RGB modes;
  transition-trace events + sampler draw-ack drain generalized to all
  journey oracles.
- **F — Docs that police themselves (standing lane):** queue split
  (Now+standing ≤ ~120 lines; Done → `queue-done.md` ledger); ~60-word
  row cap; grep-level CI doc-lints (every ADR indexed, every dossier
  indexed, roadmap step refs resolve, CLOSED/FIXED bullets graduate);
  ADR reservation ledger to end integration-time renumbering surprises.

Sequencing: A completes first (it is nearly done and everything
compounds on it). B before C (compare exercises the sync/source seams C
needs, and is the highest-signal user feature). E and F run in the
standing lane continuously.

## 7. What we explicitly keep refusing

Unchanged non-goals: napari-replacement layer platform, plugin
*marketplace*, DICOM workstation, registration/segmentation suite,
remote collaboration, dashboard sprawl. New explicit refusals from this
review: multi-input pipeline ops (see B), full pyqtgraph removal (not
scheduled; see §4), GPU op kernels (unchanged, late evidence-gated
experiment).

## 8. Doc corrections carried by this review's landing

Done in the same landing: roadmap.md rewritten to this course (stale
"queue step 5" and renderer-decision duplicates removed); queue.md gained
the Next section and was split — Done history and graduated
CLOSED/FIXED bullets moved to `queue-done.md`, row 3's status essay
condensed with the narrative preserved in the ledger; ADR 0057 indexed in
`decisions/README.md` and `index.md`; the three 2026-07-18/19 dossiers
indexed in `redesign/README.md`; `ground-rules.md` double "10." fixed
(One scheduler is now rule 12; rule 11 keeps its referenced number);
`current-state.md` refreshed to 2026-07-19; `comparison.md` gained a
pointer to §5; the two missing proposals indexed in `proposals/README.md`.
Remaining for Program F: the CI doc-lints that keep all of this true.
The brainstorm follow-up is parked in `ideas.md` (2026-07-19 section);
the tensor-ops G8 ladder is `proposals/tensor-ops-g8.md`.
