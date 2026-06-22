# ArrayScope, ArrayShow, and ArrayView

This comparison uses the ArrayScope v28 audit line, ArrayShow’s reviewed `develop` branch, and the supplied ArrayView `v0.26.3` checkout. It is a product/architecture assessment, not a checklist requiring ArrayScope to match every feature.

## Different aims

### ArrayScope

A local Python/Qt scientific array inspector with reversible operations and an explicit path from small eager arrays to bounded progressive large-data rendering. Its ambition is trustworthy semantics and responsiveness without becoming a general imaging platform.

### ArrayShow

A MATLAB-native, dimension-first workbench tightly integrated with workspace arrays and figures. It optimizes for immediate manipulation, linked windows, and scripted multi-view workflows inside one MATLAB process.

### ArrayView

A modern, minimal browser/webview product that works across many invocation environments. It optimizes for visual polish, keyboard operation, broad reach, and a large set of view/compare/analysis modes.

## Capability comparison

| Area | ArrayScope v28 | ArrayShow | ArrayView v0.26.3 |
|---|---|---|---|
| Primary environment | Python + Qt desktop/process | MATLAB figures/workspace | Browser/webview + Python service/hosts |
| Core mental model | axes + reversible document + committed frame | array object + dimensions + figure callbacks | session + viewing mode + canvas frontend |
| Basic n-D slicing | Strong | Strong | Strong |
| Dimension-local actions | Good but can become more immediate | Excellent | Mostly keyboard/tool-mode oriented |
| Reversible operation stack | Strong | Direct/manipulative, less separated | Display transforms/modes, not the same document pipeline |
| FFT/complex data | Strong semantic/cost path | Mature and immediate | Polished display modes/FFT |
| Line/profile inspection | Strong | Mature/custom cursor functions | More view/mode focused |
| ROI/statistics | Substantial | Mature/scriptable | Broad ROI/ruler/segmentation UI |
| Montage/mosaic | Progressive tiled montage | Collage/multiview utilities | Polished mosaic mode |
| Linked windows | Not yet first-class | Strong send groups/scripts | Multi-session compare, not the same typed links |
| Compare/difference | Basic core comparison helper, not productized | Multi-view scripts | Excellent first-class compare modes |
| UI polish/restraint | Improving; control/panel heavy | Functional MATLAB UI | Strongest |
| Keyboard/command discoverability | Shortcuts + palette foundation | MATLAB/menu conventions | Strong registry/palette/help model |
| Invocation reach | Python, CLI, Jupyter Qt fallback | MATLAB | Strongest: CLI/scripts/notebooks/editors/remote |
| Large-data model | Cost/region/stage/cache/montage budgets | Primarily eager MATLAB object workflows | Lazy/mmap formats and server cache; CPU render transport |
| Progressive semantic rendering | Strongest internal model | Limited | Responsive pipeline, but frontend/server state is mode-centric |
| GPU strategy | PyQtGraph default, experimental VisPy/raw shader/tile residency | MATLAB graphics | Canvas + some WebGL; much server-rendered RGBA/PNG |
| Diagnostics/admission | Explicit memory, lanes, governor, traces | Limited | Tests/audits, less explicit per-request resource policy |
| Architecture concentration | Several large transition modules | Main handle object/global workspace | 29k-line frontend + large launcher |

## Where ArrayScope is strongest

### Reversible scientific transformations

ArrayScope keeps source data, visible operation history, runtime optimization, region expansion, and reusable stages distinct. That is safer and more reproducible than a viewer whose active array object is directly transformed.

### Semantic progressive correctness

The requested/materialized/resident/presented distinction, committed geometry/value source, stale revision rejection, semantic level coverage, and separation of materialization from presentation identity are unusually strong foundations. They directly prevent common progressive-rendering bugs: wrong hover values, placeholders clearing too early, contrast causing re-upload, or old deltas mutating a new frame.

### Explicit boundedness

Memory policy, render decisions, per-cache budgets, cost declarations, lane worker policy, feedback, and resource governance make expensive behavior explainable. Neither comparator offers the same coherent internal contract.

### Testable pure architecture

Operation planning, geometry, memory policy, state, cache, and much of scheduling are Qt-free with substantial unit/property coverage and architecture guards.

## Where ArrayScope is lacking

### Product clarity and release coherence

The code has advanced faster than packaging, screenshots, user documentation, versioning, and stable release provenance. A prospective user cannot infer the real feature set or maturity from `0.0.1` plus the legacy changelog.

### Visual restraint and discoverability

ArrayScope exposes many controls/panels and has accumulated developer/performance features. ArrayView demonstrates stronger hierarchy, contextual surfaces, shortcut/help coherence, and an “array first” identity.

### Linked and comparative workflows

ArrayShow’s synchronized viewers and ArrayView’s compare/difference modes address common scientific tasks that ArrayScope does not yet productize.

### Invocation reach

ArrayScope has solid Python/CLI and process handling, but notebook/editor/remote routes are less polished. ArrayView makes opening the viewer part of the product.

### Architectural convergence

The normal/montage and PyQtGraph/VisPy paths still differ too much. Large files and timer interactions make local fixes risky until the FramePlanner/scheduler/surface migration advances.

### Hardware evidence

ArrayScope has better diagnostics concepts than both comparators, but it lacks a stable published real-GPU/Wayland benchmark matrix. An experimental backend should not become default from microbenchmarks alone.

## Lessons from ArrayShow

### Adopt

- Put common actions on the dimension being acted upon.
- Make linked viewers and batch multi-window inspection scriptable.
- Optimize the first minute of investigation: slice, component, levels, cursor, profile, transform.

### Adapt

- Express operations as reversible document steps with preview/undo.
- Replace global send groups with explicit scoped links and loop guards.
- Replace callback timing/recursion workarounds with typed targets and bounded scheduling.

### Avoid

- global viewer/workspace registries;
- one object owning every concern;
- destructive defaults;
- figure state as semantic truth;
- playback/update loops whose pace is disconnected from presented frames.

## Lessons from ArrayView

### Adopt

- Let the array dominate; reveal controls contextually.
- Use one command registry for keyboard, menus, palette, and help.
- Treat launch speed and “works where I work” as user-facing features.
- Offer a narrow, excellent compare workflow.
- Maintain a mode/platform visual-consistency matrix.

### Adapt

- Use contextual islands/pills only where they communicate state without hiding essential controls.
- Bring host integration through one session API rather than duplicating behavior.
- Use reconciliation temporarily during migration, while converging on one owner.

### Avoid

- a self-contained frontend/module that becomes the whole application;
- permanent dual writes between old and new state;
- multiplying modes faster than their cross-product can be tested;
- transporting pre-rendered RGBA/PNG for local interactions that benefit from raw textures;
- one global render worker for heterogeneous visible/analysis/speculative work.

## Recommended synthesis

### 1. Stabilize before expanding

Complete the v28 correctness/release gate and callback instrumentation. The most valuable lesson from both comparators is not “add their features”; it is “make the primary interaction coherent.”

### 2. Keep the dimension strip as the product anchor

Improve contextual operation menus, normalized range preview, axis metadata, and role clarity. This is ArrayScope’s strongest differentiator and the place to borrow ArrayShow’s immediacy.

### 3. Make the canvas quieter

Use progressive disclosure for rare controls, compact status badges for active transforms, and one command model. Borrow ArrayView’s restraint without turning ArrayScope into an animation-driven browser UI.

### 4. Finish semantic convergence

Unify normal/montage frame planning and replace backend inheritance before adding large new rendering features. This preserves the internal advantage ArrayScope already has.

### 5. Add linked views before broad modes

A scoped linked-view system unlocks image/k-space, reference/result, channels, parameter maps, and synchronized ROI with less combinatorial UI than many dedicated modes.

### 6. Add one focused compare workflow

Start with two arrays, shared or independent levels, linked viewport/slice, side-by-side plus signed/absolute difference, and ROI statistics. Do not begin with registration, segmentation, qMRI, vector field, and remote collaboration.

### 7. Improve invocation through adapters, not a second app

Define one semantic session/command boundary that CLI, Jupyter, and editor integrations call. Measure startup and lifecycle. Avoid a second frontend and transport stack until concrete users require it.

### 8. Use evidence to choose the backend

The default should be selected by parity, stability, request-to-presented latency, event-loop behavior, memory/residency, and platform coverage. PyQtGraph can remain the safe default while VisPy matures; no roadmap should assume that “GPU” automatically means faster.

## Strategic position

ArrayScope should aim to be:

> as immediate and dimension-aware as ArrayShow, as restrained and approachable as ArrayView, and more explicit than either about reversible computation, semantic progressive rendering, and resource limits.

That position is coherent. Trying to become ArrayShow’s entire MATLAB workbench plus ArrayView’s entire browser imaging suite would not be.
