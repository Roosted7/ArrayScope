# Architecture decisions

Decision records capture durable rationale. They are not a chronological backlog. Read the [architecture overview](../architecture.md) first, then the smallest relevant group below.

Status terminology:

- **Implemented**: the decision’s central contract is present and tested.
- **Partly implemented**: key pieces exist, but the stated destination is incomplete.
- **Historical foundation**: still explains the code, though later decisions refine the mechanism.
- **Experiment**: records an evaluated direction rather than a final default.

## State, slicing, operations, and UI boundary

| ADR | Topic | Current status |
|---|---|---|
| [0001](0001-view-state.md) | View state | Implemented; expanded since acceptance. |
| [0002](0002-slice-engine.md) | Slice engine | Historical foundation. |
| [0003](0003-dimension-operations.md) | Dimension operations | Implemented and generalized into the operation stack. |
| [0004](0004-ui-boundary.md) | UI/state boundary | Implemented principle; ongoing enforcement. |
| [0005](0005-operation-pipeline.md) | Operation pipeline | Implemented. |
| [0006](0006-operation-ui-and-recipes.md) | Operation UI/recipes | Implemented. |
| [0007](0007-window-level-mode.md) | Window/level mode | Historical foundation, refined by later presentation ADRs. |
| [0008](0008-live-image-profiles.md) | Live profiles | Implemented. |
| [0009](0009-profile-dock.md) | Profile panel | Implemented, later folded into managed panels. |
| [0010](0010-ui-polish-settings-theme.md) | Settings/theme | Implemented. |
| [0011](0011-progressive-interaction-polish.md) | Interaction polish | Historical foundation; parts remain roadmap work. |

## Inspection, geometry, panels, and viewport

| ADR | Topic | Current status |
|---|---|---|
| [0013](0013-roi-inspection-workflows.md) | ROI inspection | Implemented core workflow. |
| [0014](0014-profiles-and-montage.md) | Profiles/montage roles | Implemented; range behavior recently extended. |
| [0015](0015-coordinate-geometry-and-viewport.md) | Geometry/viewport | Implemented contract. |
| [0017](0017-managed-panels-and-wayland.md) | Managed panels/Wayland | Implemented with platform regression needs. |
| [0018](0018-viewport-fit-toggle.md) | Fit/1:1 behavior | Implemented and recently refined. |
| [0043](0043-file-session-restore-boundaries.md) | File-session restore boundaries | Implemented. |

## Evaluation, planning, cache, and resource policy

| ADR | Topic | Current status |
|---|---|---|
| [0012](0012-lazy-slab-evaluation.md) | Lazy slab evaluation | Implemented and superseded in detail by planner/stage work. |
| [0016](0016-evaluation-scheduler-and-memory-budget.md) | Evaluation scheduling/memory | Historical foundation. |
| [0020](0020-operation-cost-and-fft-backend.md) | Cost model/FFT backend | Implemented. |
| [0021](0021-scheduler-v2-cost-aware-rendering.md) | Cost-aware rendering | Partly implemented; visible scheduling still converging. |
| [0023](0023-memory-policy-and-developer-diagnostics.md) | Memory/diagnostics | Implemented foundation. |
| [0025](0025-operation-capabilities-and-region-contracts.md) | Operation contracts | Implemented. |
| [0026](0026-runtime-region-planner.md) | Region planner | Implemented. |
| [0027](0027-in-memory-stage-cache.md) | Stage cache | Implemented. |
| [0028](0028-runtime-operation-optimizer.md) | Optimizer | Implemented. |
| [0029](0029-stage-first-rendering.md) | Stage-first rendering | Partly implemented. |
| [0034](0034-compute-policy-and-stage-warmup.md) | Compute/stage warmup | Implemented foundation. |
| [0035](0035-resource-governor-feedback-control.md) | Resource governor | Implemented foundation; evidence/tuning ongoing. |
| [0052](0052-ui-work-pacing-governor.md) | UI-work pacing governor | Superseded by ADR 0053 R4; historical context for deleted per-controller pacing. |
| [0049](0049-out-of-core-lazy-sources.md) | Out-of-core/lazy sources | Implemented first slice: protocol, budgeted read seam, memmap adapters. |

## Presentation, montage, and backends

| ADR | Topic | Current status |
|---|---|---|
| [0019](0019-tiled-montage-renderer.md) | Tiled montage | Implemented foundation. |
| [0022](0022-stable-progressive-montage-rendering.md) | Stable progressive montage | Implemented, refined by repair work. |
| [0024](0024-canvas-preserve-transaction.md) | Canvas preservation | Implemented. |
| [0030](0030-display-presentation-boundaries.md) | Presentation boundaries | Implemented. |
| [0031](0031-render-presentation-model.md) | Presentation model | Implemented. |
| [0032](0032-semantic-montage-histograms.md) | Semantic levels/histograms | Implemented contract. |
| [0033](0033-responsive-montage-display-upload.md) | Responsive upload | Partly implemented; callback budgets remain active work. |
| [0036](0036-vispy-rendering-backend-experiment.md) | VisPy experiment | Experiment completed; led to 0037/0038. |
| [0037](0037-first-class-vispy-tiled-renderer.md) | VisPy tiled renderer | Preferred large-tiled target; CPU-side LOD portions superseded by 0041. |
| [0038](0038-render-backend-composition.md) | Backend composition | Implemented; surfaces own presentation/lifecycle and mirror shared interaction state. |
| [0039](0039-unified-image-surface-and-deadline-scheduler.md) | Unified surface/scheduler | Historical foundation; frame planner, typed tiled surface, backend composition, and pointer capture remain, while WorkGraph was superseded by the ADR 0053 kernel in R1. |
| [0040](0040-backend-aware-presentation-convergence.md) | Backend-aware presentation convergence | Implemented for tiled presentation. |
| [0041](0041-lod-selection-materialization-and-residency.md) | LOD selection/materialization/residency | Implemented via ADR 0050 (async materialization + compatible residency; `resident` default on VisPy tiled scenes). The three-way split remains the governing contract. |
| [0042](0042-montage-viewport-reflow-and-roi-ownership.md) | Montage viewport reflow and ROI ownership | Implemented for tiled montage. |
| [0044](0044-viewport-scoped-tiled-residency.md) | Viewport-scoped tiled residency | Accepted; acknowledgement repair implemented, normal-image retarget remains roadmap work. |
| [0045](0045-render-orchestrator-composition.md) | Render orchestrator composition | Implemented (v32); render state owned by one composed object instead of window mixins. |
| [0046](0046-evidence-first-performance-strategy.md) | Evidence-first performance strategy | Accepted; X5 decides physical strategies, backend defaults, and LOD from hardware traces and residency conformance. |
| [0047](0047-auto-image-backend-selection.md) | Capability-probed automatic image backend selection | Accepted; `auto` resolves to VisPy on Linux with hardware GL (X5a traces), PyQtGraph everywhere else. |
| [0048](0048-linked-window-sync.md) | Linked-window sync over local sockets | Implemented; per-facet window/level, dimension-indexing, operations, and ROI sync across separately started processes. |
| [0050](0050-async-multi-resolution-tile-residency.md) | Async multi-resolution tile residency | Implemented for VisPy tiled scenes (`resident` default on VisPy); retained preview level implemented; PyQtGraph adoption implemented opt-in (`ARRAYSCOPE_PYQTGRAPH_RESIDENT_LOD=1`, default off pending the measured gate); reduce-before-ops consumer and ops-input LOD remain roadmap work. |
| [0051](0051-single-owner-tile-lifecycle.md) | Single-owner tile lifecycle | Accepted; P1 + P2 core implemented, field-verified, and machine-derived dispatch landed (every event edge re-derives all pumps; the stall watchdog is an assertion). Remaining P2: sets as views + stage fan-in events, delta-walk cost; P3–P5 phased. |
| [0053](0053-execution-kernel-and-modular-pipeline.md) | Execution kernel and modular rendering pipeline | Accepted; kernel, pipeline, ladder, and frame control-plane landed (R1–R7). Remaining visible-truth work is queued in `docs/redesign/README.md`. |
| [0054](0054-montage-level-evidence-phasing.md) | Montage level evidence phasing | Implemented for montage level/histogram evidence ordering; rough preview, rough target, and refined stats are ranked explicitly. |
| [0055](0055-view-tiles-data-chunks-residency-pages.md) | View tiles / data chunks / residency pages | Accepted on `codex/gpu-engine`; G1–G3 and G4a implemented (chunked content-keyed residency live, real-GL verified). |
| [0056](0056-sparse-virtual-multiresolution-pyramid.md) | Sparse virtual multiresolution pyramid | Accepted; canonical page route/cache/backend cutover implemented on the G5 landing candidate; final real-Wayland/stress acceptance pending. |

## Adding or changing a decision

Create an ADR only when future contributors need to preserve a choice about architecture, public API, packaging, testing strategy, or major UX behavior. Record context, decision, alternatives, consequences, and migration. When a later ADR supersedes part of an earlier one, update this index and cross-link both records rather than rewriting history.
