# ArrayScope v28 project audit

**Audit branch:** `audit/v28-project-review`

**Baseline:** v28 development line reviewed on 2026-06-22
**Scope:** recent changes, architecture, correctness, responsiveness, tests/CI/release posture,
documentation, and comparison with ArrayShow and ArrayView.

## Executive assessment

ArrayScope has crossed from an early viewer into a capable scientific application. Its strongest work
is below the visible UI: immutable view/operation state, region-aware evaluation, stage reuse, bounded
memory policy, progressive montage rendering, semantic presentation models, and explicit stale-result
rejection. Those foundations are materially stronger than either comparison project.

The project is nevertheless not release-ready at the audited baseline. The immediate blockers are
version/release ambiguity, incomplete real-platform/backend validation, and the absence of end-to-end
frame-delivery budgets. The main engineering risk is ownership concentration in timing-sensitive
classes: recent features are individually useful, but histogram, viewport, montage, and backend
migration work now meet inside several 1,700–2,100-line classes. Adding more branches there would make
correctness increasingly dependent on callback and timer ordering.

This audit made low-risk corrections directly and reorganized the documentation. It deliberately did
not perform a broad renderer rewrite, choose an unrequested release version, or promote VisPy to the
production default without real OpenGL evidence.

## Audit method and evidence

The review used the repository's complete Git history rather than treating the archive as a source
snapshot. It examined the latest feature changes, compared changed behavior with tests and ADRs, ran
static checks and targeted/full test groups, inspected timer/callback lifecycles, and traced the normal
and montage presentation paths. The supplied ArrayView repository was inspected at its current
`v0.26.3` source, and ArrayShow was checked against the public `develop` head identified in the
reference document.

Baseline scale:

- approximately 40,098 Python source lines;
- 90 `test_*.py` files;
- four especially concentrated rendering/view classes above 1,700 lines;
- 39 ADRs covering state, evaluation, rendering, memory, and backend migration;
- package/runtime metadata still reports `0.0.1`, while release provenance needs reconciliation.

The first complete local baseline run reported **906 passed, 17 failed, and 2 skipped**. The failures
were investigated individually. They were a mixture of stale expectations after intentional behavior
changes, test isolation/order assumptions, and real product defects described below.

## Recent Change Scrutiny

### Center/hover-prioritized montage work

The scheduler direction is correct: priority is a property of semantic work value, not tile numbering.
The regression was mainly in the tests, which captured callbacks in submission order and later assumed
callback zero represented tile zero. Once priority changed, those tests could report false stale-result
failures or drive the wrong completion.

**Action:** tests now associate callbacks with tile identity and verify the current session/target. This
keeps priority free to evolve while preserving stale-result protection.

### Slice/range extension and centered defaults

Centered defaults are a better scientific default than index zero, and image-axis ranges are useful.
The change, however, changed state initialization, parsing, UI controls, and synchronization in one
step. Several tests still encoded zero as the default. The parser also now carries a product-level
ambiguity: Python syntax is preferred, MATLAB syntax is a fallback, and some three-part expressions can
reasonably mean different things. Explicit lists are bounded silently, which can turn a typo into a
plausible but wrong selection.

**Action:** stale tests were updated. Syntax mode and out-of-range feedback remain explicit roadmap
work; more inference heuristics would make the problem worse.

### Histogram/manual editing/auto-window

The feature is useful and well covered, but it added roughly 635 lines to a new controller that owns
sampling, histogram construction, level policy, mouse interception, a manual-edit popup, and signaling.
A real interaction defect occurred when auto-window was invoked while manual preview was open:
accepting the preview emitted one user-level change and auto-window emitted another, causing two render
paths for one intent.

**Action:** auto-window rejects the transient preview before requesting automatic levels. The next
structural step is to separate histogram data/level policy from popup and pointer UI.

### Viewport constraints and montage auto-fit

The viewport policy is clearer and prevents unrecoverable panning/zooming. The implementation also
added more state and delayed refresh behavior to the main image and montage classes. A delayed viewport
refresh can legitimately reschedule a tile, so assertions based on an exact timer/callback count are
brittle.

**Action:** affected tests now assert latest-session semantics rather than timer count. Future work
should keep viewport policy Qt-free and make presentation/settle timing observable through frame metrics.

## Findings and direct fixes

| Severity | Finding | Consequence | Resolution |
|---|---|---|---|
| High | Normal-image render refusal called `format_bytes` without importing it when the planner supplied no custom status text. | Oversized/budget-refused renders could crash precisely on the recovery path. | Import added with a focused regression test. |
| Medium | Replacing an evaluation group invalidated the same group hierarchy once per outstanding request and retained historical per-tile generation entries. | Cancellation cost and bookkeeping grew with both outstanding work and prior tiles. | Prefix generations make invalidation bounded; completed per-tile state is pruned; stale child results remain rejected. |
| Medium | Manual histogram preview was accepted before auto-window. | One user action emitted two semantic level changes/render requests. | Preview is rejected first; auto-window owns the single final action. |
| Medium | Montage scheduling tests coupled tile identity to queue order. | Priority improvements looked like correctness regressions and could mask real stale-result bugs. | Callbacks are keyed by tile/session identity. |
| Medium | Centered slice defaults and adaptive memory profiles changed behavior without all contracts following. | Baseline suite was red despite intended behavior; hard-ceiling semantics were confused with a target allocation. | Tests now assert centered defaults and adaptive budget `<=` hard ceiling. UI/docs state the distinction. |
| Low | Empty Inspection refresh rebuilt table/histogram on montage viewport changes. | Avoidable GUI work on a hot interaction path. | Empty-to-empty clear is now a no-op. |
| Low | A skipped VisPy viewport test could leave persistent backend settings behind. | Later tests could run under the wrong backend depending on order/environment. | Settings restoration now runs on skip and success paths. |
| Low | CLI accepted multiple paths but launched each with blocking behavior. | The second and later paths were not opened until the previous window closed. | Multi-path launches are non-blocking; single-path behavior remains blocking/compatible. |
| Release | Two overlapping full-suite workflows, incomplete declared-Python coverage, optional VisPy absent, and no tag/version guard. | Wasted CI time, blind backend gaps, and possible publication of mislabeled artifacts. | CI consolidated; Linux covers 3.10–3.14, representative macOS/Windows runs remain, strict UI/build jobs are separate, VisPy is installed, and releases enforce version consistency. |

## Performance and responsiveness assessment

### What is already good

- Evaluation, render coordination, viewport settling, ROI work, upload flushing, and diagnostics use
  bounded or idle-stopping timers; the reviewed scheduler poller is not an always-on leak.
- Materialization identity is separated from presentation identity, so levels/LUT/camera changes need
  not redefine source pixels.
- Progressive montage geometry and semantic levels remain stable while tiles arrive.
- Persistent tile residency, warm-work interruption, indexed viewport selection, and bounded atlas
  bookkeeping address the right failure modes for large montages.
- Stale work is guarded by document/view/session/generation identity rather than merely “last callback
  wins.”

### Remaining latency risk

No individual timer is obviously unreasonable in isolation, but a cold request can pass through
several stages: interaction coalescing, scheduler polling, viewport settling, evaluation, GUI presentation,
and backend upload. Measuring setter duration therefore understates visible latency. The project needs
request-to-first-usable-frame and request-to-exact-frame metrics, maximum event-loop gaps, and frame age.

The most important regression cases are:

1. warm pan/zoom causes no evaluation or upload when residency already covers the viewport;
2. levels/LUT changes update uniforms/presentation without invalidating textures;
3. a cold shared expensive stage is computed once and reused across montage tiles;
4. active-plus-latest scheduling makes visible progress during sustained input rather than starving by
   cancelling every intermediate target;
5. queue/group invalidation and GUI callbacks remain bounded as tile count grows;
6. hidden Profile/Inspection panels stop demand work and repaint work;
7. memory/GPU residency settles under repeated range expansion and contraction.

### GPU/backend direction

The current direction—shared semantic presentation and interaction, separate physical raster/tiled
strategies, native-resolution persistent tiles by default, and backend adapters—is sound. A small image
need not be forced into an atlas, but a large single plane and a montage should share semantic levels,
hover values, scheduling, and cache identity. Multi-resolution residency should use compatible pages,
texture arrays, or a virtual-texture/page-table design rather than mixing arbitrary LOD sizes in a
fixed-slot atlas.

PyQtGraph should remain the production backend. The VisPy path still inherits the PyQtGraph-oriented
viewer shell, and pointer capture/drag state is not fully backend-neutral. Promotion requires the same
semantic conformance tests plus real OpenGL context-loss, Wayland/X11, high-DPI, cursor, overlay, and
screenshot checks.

## Architecture and code-flow assessment

### Strong boundaries to preserve

- `ArrayDocument`/`ViewState` and immutable operation steps are the scientific source of truth.
- Region planning and evaluation are mostly Qt-free and do not derive state back from widgets.
- The scheduler distinguishes lanes, cost, cancellation/supersession, and memory policy.
- Presentations preserve semantic value sources separately from display textures.
- ADRs document why the current architecture exists, including accepted migration directions.

### Concentrated ownership to reduce

| Module | Approx. lines | Recommended split |
|---|---:|---|
| `window/montage_renderer.py` | 2,101 | pure session/planning progression; evaluation requests; semantic level aggregation; Qt presentation coordinator |
| `display/imageview2d.py` | 2,079 | `ImageViewShell`; PyQtGraph surface; viewport bridge; overlay/interaction bridge |
| `display/vispy_imageview2d.py` | 2,080 | compose the common shell instead of subclassing the PyQtGraph view; keep only VisPy bridge mechanics |
| `display/backends/vispy/tiles.py` | 1,736 | atlas/page allocator; residency policy; upload queue; visual/page-table binding; diagnostics |
| `display/histogram_controller.py` | 691 | sampler/bin model; level policy; popup/editor; mouse adapter |

The rule is not “split every large file.” Split when one owner currently combines semantic policy,
scheduling, toolkit lifecycle, and physical rendering. Keep compatibility modules import-only during
migration, and add architecture guards so new behavior cannot drift back into them.

### Exception handling and observability

The tree contains many broad `except Exception` handlers, concentrated around UI callbacks, export,
montage, and backend integration. User-facing recovery is appropriate, but broad catches are dangerous
when they convert programming defects into silent status messages. Strict UI mode is the right
countermeasure. Continue narrowing catches where a known operational exception exists, attach semantic
context to diagnostics, and ensure strict mode re-raises unexpected failures.

### State and lifecycle recommendations

- Implement explicit requested, materialized, resident, committed, and presented identities.
- Track active and latest targets separately; do not make “cancel everything on every input” the only
  freshness mechanism.
- Make shared interaction state own pointer capture, drag lifecycle, hit priority, and cursor intent;
  surfaces should only translate events and draw state.
- Expose a stable public session/window handle so scripts can use the same semantic commands as UI.
- Keep source mutation explicit through revision/context APIs rather than implicit observation of
  mutable NumPy memory.

## Test, CI, and release assessment

The test suite is substantial and caught several regressions once its assumptions were corrected. The
largest weakness was not test quantity but asynchronous test semantics: submission order, timer count,
and persistent settings were sometimes treated as product contracts. New tests should key on semantic
targets and use barriers/events where order matters.

Release metadata is the clearest current blocker. Both `pyproject.toml` and
`arrayscope.__version__` report `0.0.1`, while release history and package provenance are not yet
reconciled, and no project wheel is currently published on PyPI. The audit does not guess
whether the next release is `0.8.0`, `0.7.1`, or a different scheme. A maintainer must choose the
source of truth and intended package name. The release workflow now blocks a tag, package, and runtime
version mismatch.

The consolidated CI design intentionally separates:

- declared-Python compatibility on Linux;
- representative macOS/Windows Qt behavior;
- strict UI exception surfacing;
- screenshot smoke artifacts;
- package build and metadata validation;
- release-only tag/version verification and wheel installation.

## Documentation reorganization

The previous documentation contained useful evidence but mixed four levels in the same navigation
path: current architecture, phase implementation notebooks, completed checklist roadmaps, and manual
regression fragments. The 593-line roadmap alone contained hundreds of historical status boxes, making
it difficult to tell what mattered now.

The new progressive structure is:

1. `README.md`, `docs/mission.md`, `docs/project-status.md`, and `docs/roadmap.md` for product/current
   state;
2. `docs/architecture.md` for the system map, with subsystem details under `docs/architecture/`;
3. `docs/decisions/` for durable rationale and `docs/testing/` for current verification;
4. `docs/references/` and `docs/reviews/` for research/audit evidence;
5. `docs/archive/` for completed Phase 4 notes, old architecture/roadmap snapshots, and superseded
   manual checklists.

Historical material remains searchable and linked, but it no longer presents itself as the active
plan. The active roadmap uses Now/Next/Later plus measurable gates; speculative work lives in
`docs/ideas.md`.

## Comparative product and technical review

### Positioning summary

| Dimension | ArrayScope | ArrayShow | ArrayView |
|---|---|---|---|
| Primary environment | Native Python/Qt desktop | MATLAB figure/workspace | Browser canvas with Python service/launcher |
| Core promise | Bounded scientific inspection and operations on nD arrays | Immediate MATLAB workspace inspection | Fast, minimal, remote-capable multi-language viewing |
| Strongest advantage | State/evaluation/render separation; operations; memory/scheduler discipline | Mature one-call handle, complex/MRI workflows, linked viewers | Distribution/integration reach, modern minimal UI, remote/VS Code/Jupyter story |
| Main weakness | Public workflow/API and release polish lag architecture; hot classes are concentrated | Monolithic/global synchronous GUI architecture | Large frontend/launcher concentration, transport/state-reconciliation and mode-combination complexity |
| Best lesson | — | Preserve one-call scriptability and linked scientific workflows | Treat launch/integration/remote delivery and progressive disclosure as product features |
| Main trap to avoid | Over-engineering before measuring frame delivery | Global registry, fixed panels, callback-time workarounds | Copying server-rendered transport into the local path or allowing dual frontend state to drift |

### ArrayShow: what to adopt and what not to copy

ArrayShow's enduring value is workflow proximity: a MATLAB user can inspect an array with one call,
keep a handle, script visible state, link related viewers, and access complex/MRI-oriented controls.
ArrayScope should adopt a stable public session handle, linked-view groups, scriptable semantic
commands, axis-adjacent operations, and equally low-friction one-call use.

It should not adopt ArrayShow's global `asObjs` registry, one coordinating handle class, fixed panel
geometry, synchronous playback that ignores render duration, or callback-order/time workarounds.
ArrayScope's explicit state, revision, scheduling, and composition boundaries are the better long-term
foundation.

### ArrayView: what to adopt and what not to copy

ArrayView demonstrates that “works where the user already works” is a first-class feature. Its current
product story spans shell, Python/Julia/MATLAB scripts, Jupyter, local/remote use, and VS Code, with a
minimal interface that hides controls until relevant. ArrayScope should prioritize public sessions,
command-registry-driven UI/help, layout presets, excellent CLI/file opening, and a deliberate remote or
IDE integration boundary after local semantics stabilize.

ArrayView's browser/service design is not automatically the right rendering architecture for
ArrayScope. Server-rendered image transport can be excellent for remote reach, but would add encoding,
transport, reconciliation, and latency to the native local path. Its source also shows concentration in
a very large no-build frontend and launcher, coexistence of legacy/global and component state, mode
combinatorics, and a prior launcher split that had to be reverted after broad failures. The lesson is
not to reject a web frontend; it is to isolate transport behind semantic session commands, keep one
source of truth, generate mode/command behavior from registries, and prove migrations incrementally.

### Recommended synthesis

ArrayScope should remain a native, local-first scientific viewer with a backend-neutral semantic core.
Build the public session/command API next; then linked sessions, compare workflows, notebook/IDE
adapters, and optional remote transport can reuse it. This captures ArrayShow's scriptability and
ArrayView's reach without sacrificing bounded evaluation or duplicating state across widgets,
backends, and transports.

## Prioritized plan

### Now

1. Resolve authoritative versioning and complete wheel-install/release validation.
2. Add end-to-end frame delivery, event-loop-gap, no-upload-pan, presentation-only-level, and queue
   scaling benchmarks.
3. Finish active/latest frame lifecycle and typed supersession semantics.
4. Extract montage session progression and compose `ImageViewShell` with backend surfaces.
5. Decide explicit slice syntax and out-of-range-list behavior.
6. Complete real-platform PyQtGraph and VisPy manual matrices.

### Next

1. Public semantic session/window handle and command registry.
2. Linked view groups and compare state backed by sessions, not widget coupling.
3. Metadata-aware axes/coordinates and export preservation.
4. Layout presets and unified keyboard/help/menu discoverability.
5. Benchmark-backed operation scheduling and stage reuse on representative MRI workloads.

### Later, only with evidence

- multi-resolution/virtual-texture residency;
- disk-backed stage cache;
- dask/zarr execution adapters;
- plugins;
- remote/browser transport;
- GPU compute operations.

These are useful directions, but none should outrank measurable first-frame responsiveness, a stable
public semantic API, or release reliability.

## Fixes produced by this audit

| Area | Purpose |
|---|---|
| Tests and inspection refresh | Stabilize centered-slice and prioritized montage tests; remove empty inspection repaint; restore backend settings around VisPy skip. |
| Evaluation bookkeeping | Bound evaluation group invalidation and prune completed tile-generation state. |
| Render fallback and levels | Fix render-refusal fallback and single-action histogram auto-window behavior. |
| CI and release guards | Consolidate CI and guard release/package/runtime versions. |
| CLI launch | Open multiple CLI paths concurrently while retaining single-path blocking behavior. |
| Lifecycle cleanup | Close display resources and bound benchmark lifecycle. |
| Viewport recovery | Preserve viewport overlap after max-span clamping. |
| Priority hardening | Harden montage priority inputs and early hover cleanup. |
| Documentation | Archive phase-era guidance, preserve supplemental v28 notes, and install the progressive current documentation set. |

## Validation

After restoration, the complete headless/offscreen suite passed with `935 passed, 1 skipped`.
Real-platform PyQtGraph/VisPy GPU, Wayland/X11, DPI, context-loss, and interaction-feel checks remain
manual release evidence.

## Known limitations of this audit

- Headless/offscreen Qt cannot prove real Wayland/X11 compositor, cursor, DPI, focus, or window-manager
  behavior.
- The environment may not provide a usable OpenGL context; a narrowly scoped VisPy test skip is not a
  production backend pass.
- No representative private MRI dataset or target workstation/GPU performance envelope was supplied.
- The audit inspected ArrayView and ArrayShow source and product behavior but did not execute MATLAB or
  the complete ArrayView browser/IDE matrix in this environment.
